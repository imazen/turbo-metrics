#![forbid(unsafe_code)]

//! `zen-metrics features-backfill` — re-derive zensim 300-feature parquet
//! sidecars from already-completed sweep TSVs.
//!
//! Background: `zen-metrics-cli` 0.3.0 shipped the `sweep` subcommand but did
//! NOT emit per-cell zensim feature parquets. Two large production sweeps
//! (`sweep-v04-full-2026-05-04`, `sweep-v05c-2026-05-04`) ran with that
//! version; the resulting per-chunk TSVs are on R2 but the parquet sidecars
//! are missing. Re-running the sweeps would mean re-encoding, which is the
//! dominant compute cost. This subcommand re-encodes locally-cheap things
//! only — encode + decode + zensim feature extract — for every row in the
//! existing TSVs and writes a per-chunk parquet that joins back to the TSV
//! by `(image_path, codec, q, knob_tuple_json)`.
//!
//! ## Modes
//!
//! ### Local mode (used by tests, ad-hoc re-runs)
//!
//! ```text
//! zen-metrics features-backfill \
//!     --input-tsv path/to/chunk.tsv \
//!     --corpus-root path/to/source-images/ \
//!     --output-parquet path/to/chunk.features.parquet \
//!     [--codec zenwebp]   # only required if the TSV is missing the codec column
//! ```
//!
//! ### R2 mode (production)
//!
//! ```text
//! zen-metrics features-backfill \
//!     --run-id sweep-v05c-2026-05-04 \
//!     --codec zenwebp \
//!     --corpus-root path/to/source-images/ \
//!     [--r2-prefix s3://zentrain/sweep-v05c-2026-05-04/zenwebp/] \
//!     [--output-r2-prefix s3://zentrain/sweep-v05c-2026-05-04/zenwebp/features/]
//! ```
//!
//! For each chunk in R2:
//! 1. Pre-skip: `s3 ls features/<chunk_id>.parquet` → present → skip.
//! 2. Pull TSV from R2 to a temp dir.
//! 3. Re-encode each row, decode, run zensim feature extract.
//! 4. Write parquet to a temp file.
//! 5. Upload to R2 at `<output-r2-prefix>/<chunk_id>.parquet`.
//!
//! TSVs are read-only — the encoded blobs and parquet are derived
//! artifacts. Idempotent at the per-chunk level: re-running the same
//! command with the same inputs is a cheap no-op once parquets exist.
//!
//! ## Image-path resolution
//!
//! The TSV's `image_path` column was written by the original sweep worker
//! and points at a worker-local staging directory (e.g.
//! `/workspace/sweep/stage-zenwebp-000/dir__sub__file.png`). We can't open
//! that path from here, so we resolve against `--corpus-root`:
//!
//! 1. Take the basename of `image_path`.
//! 2. Replace `__` with `/` to undo the staging flatten.
//! 3. Look for that relative path under `--corpus-root`.
//! 4. If not found, also try the basename as-is (some corpora are flat).
//!
//! Rows whose source image cannot be resolved are skipped with a warning;
//! the parquet still records all resolvable rows. A run summary at the
//! end reports the resolution / encode / score success rate.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::decode::{Rgb8Image, decode_image_to_rgb8};
use crate::metrics::run_zensim_with_features;
use crate::sweep::encode::{CodecKind, encode};
use crate::sweep::feature_writer::FeatureParquetWriter;

use serde_json::{Map, Value};

/// Top-level configuration for a backfill invocation. One of the two
/// modes (`Local` or `R2`) is filled in by the CLI parser.
#[derive(Debug, Clone)]
pub struct BackfillConfig {
    pub mode: BackfillMode,
    pub corpus_root: PathBuf,
    /// Codec override. Required for R2 mode (selects the chunk subset).
    /// For local mode, used only when the TSV row's `codec` column doesn't
    /// match a known codec (or the column is missing).
    pub codec: Option<CodecKind>,
}

#[derive(Debug, Clone)]
pub enum BackfillMode {
    Local {
        input_tsv: PathBuf,
        output_parquet: PathBuf,
    },
    R2 {
        run_id: String,
        /// Override for the R2 prefix to scan for chunk TSVs. Default:
        /// `s3://zentrain/<run_id>/<codec>/`.
        r2_prefix: Option<String>,
        /// Override for where to upload feature parquets. Default:
        /// `s3://zentrain/<run_id>/<codec>/features/`.
        output_r2_prefix: Option<String>,
    },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BackfillStats {
    pub chunks_total: usize,
    pub chunks_skipped: usize,
    pub chunks_processed: usize,
    pub chunks_failed: usize,
    pub rows_total: usize,
    pub rows_emitted: usize,
    pub rows_resolve_fail: usize,
    pub rows_encode_fail: usize,
    pub rows_decode_fail: usize,
    pub rows_score_fail: usize,
}

pub fn run_backfill(cfg: &BackfillConfig) -> Result<BackfillStats, Box<dyn Error>> {
    if !cfg.corpus_root.is_dir() {
        return Err(format!(
            "--corpus-root {} is not a directory",
            cfg.corpus_root.display()
        )
        .into());
    }

    match &cfg.mode {
        BackfillMode::Local {
            input_tsv,
            output_parquet,
        } => {
            if output_parquet.exists() {
                eprintln!(
                    "[backfill] {} already exists; skipping (idempotent)",
                    output_parquet.display()
                );
                return Ok(BackfillStats {
                    chunks_total: 1,
                    chunks_skipped: 1,
                    ..Default::default()
                });
            }
            let chunk_stats =
                backfill_one_tsv(input_tsv, output_parquet, &cfg.corpus_root, cfg.codec)?;
            let mut stats = BackfillStats {
                chunks_total: 1,
                chunks_processed: 1,
                ..Default::default()
            };
            accumulate(&mut stats, &chunk_stats);
            Ok(stats)
        }
        BackfillMode::R2 {
            run_id,
            r2_prefix,
            output_r2_prefix,
        } => {
            let codec = cfg
                .codec
                .ok_or("R2 mode requires --codec to select chunks")?;
            run_r2_backfill(
                run_id,
                codec,
                r2_prefix.as_deref(),
                output_r2_prefix.as_deref(),
                &cfg.corpus_root,
            )
        }
    }
}

/// Per-chunk statistics returned by [`backfill_one_tsv`].
#[derive(Debug, Default, Clone, Copy)]
struct ChunkStats {
    rows_total: usize,
    rows_emitted: usize,
    rows_resolve_fail: usize,
    rows_encode_fail: usize,
    rows_decode_fail: usize,
    rows_score_fail: usize,
}

fn accumulate(stats: &mut BackfillStats, chunk: &ChunkStats) {
    stats.rows_total += chunk.rows_total;
    stats.rows_emitted += chunk.rows_emitted;
    stats.rows_resolve_fail += chunk.rows_resolve_fail;
    stats.rows_encode_fail += chunk.rows_encode_fail;
    stats.rows_decode_fail += chunk.rows_decode_fail;
    stats.rows_score_fail += chunk.rows_score_fail;
}

/// Read every row of `input_tsv`, re-encode + decode + score, and write
/// the parquet to `output_parquet`. Writes to `<output>.tmp` first then
/// renames so a partial run never leaves a corrupt parquet under the
/// final name.
fn backfill_one_tsv(
    input_tsv: &Path,
    output_parquet: &Path,
    corpus_root: &Path,
    codec_override: Option<CodecKind>,
) -> Result<ChunkStats, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(input_tsv)
        .map_err(|e| format!("open {}: {e}", input_tsv.display()))?;
    let headers = rdr.headers()?.clone();

    let col_image = find_col(&headers, &["image_path"])?;
    let col_codec = find_col(&headers, &["codec"]).ok();
    let col_q = find_col(&headers, &["q"])?;
    let col_knobs = find_col(&headers, &["knob_tuple_json"])?;

    let tmp_path = output_parquet.with_extension("parquet.tmp");
    if let Some(parent) = tmp_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = FeatureParquetWriter::create(&tmp_path)?;

    // Cache decoded sources so we don't PNG-decode the same image once per
    // (q, knob) cell. Order-of-magnitude speedup on chunks where one image
    // covers multiple rows.
    let mut source_cache: Option<(PathBuf, Rgb8Image)> = None;

    let mut stats = ChunkStats::default();
    for record in rdr.records() {
        let record = record?;
        stats.rows_total += 1;

        let image_path_str = record.get(col_image).ok_or("missing image_path")?;
        let q_str = record.get(col_q).ok_or("missing q")?;
        let knob_json = record.get(col_knobs).ok_or("missing knob_tuple_json")?;

        let codec = match codec_override {
            Some(c) => c,
            None => match col_codec {
                // Fallback: zenwebp is the default in our sweep
                // pipeline. We don't fail hard on an unrecognised
                // codec string so a single bad row doesn't kill the
                // chunk.
                Some(idx) => {
                    parse_codec(record.get(idx).unwrap_or("")).unwrap_or(CodecKind::Zenwebp)
                }
                None => return Err("TSV has no `codec` column and --codec was not provided".into()),
            },
        };
        let q: u32 = q_str
            .trim()
            .parse()
            .map_err(|e| format!("invalid q value {q_str:?}: {e}"))?;
        let knobs: Map<String, Value> = match parse_knob_json(knob_json) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[backfill] skip: bad knob json {knob_json:?} for {image_path_str}: {e}");
                stats.rows_resolve_fail += 1;
                continue;
            }
        };

        // Resolve the image path against the corpus root.
        let resolved = match resolve_image_path(image_path_str, corpus_root) {
            Some(p) => p,
            None => {
                eprintln!(
                    "[backfill] skip: cannot resolve {image_path_str} under {}",
                    corpus_root.display()
                );
                stats.rows_resolve_fail += 1;
                continue;
            }
        };

        // Decode the source (cached across consecutive rows for the same image).
        let source: &Rgb8Image = match &source_cache {
            Some((p, img)) if p == &resolved => img,
            _ => {
                let img = match decode_image_to_rgb8(&resolved) {
                    Ok(img) => img,
                    Err(e) => {
                        eprintln!(
                            "[backfill] skip: decode source {} failed: {e}",
                            resolved.display()
                        );
                        stats.rows_resolve_fail += 1;
                        source_cache = None;
                        continue;
                    }
                };
                source_cache = Some((resolved.clone(), img));
                &source_cache.as_ref().unwrap().1
            }
        };

        // Encode → bytes.
        let cell = match encode(codec, source, q, &knobs) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "[backfill] encode failed: {} q={q} knobs={knob_json}: {e}",
                    image_path_str
                );
                stats.rows_encode_fail += 1;
                continue;
            }
        };

        // Decode-back through a tempfile so format sniff is identical to
        // the production sweep path. This is the same pattern run.rs uses.
        let decoded = match decode_encoded_bytes(&cell.bytes, codec) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "[backfill] decode-back failed: {} q={q}: {e}",
                    image_path_str
                );
                stats.rows_decode_fail += 1;
                continue;
            }
        };

        if decoded.width != source.width || decoded.height != source.height {
            eprintln!(
                "[backfill] dimension mismatch: {} q={q}: {}x{} vs {}x{}",
                image_path_str, source.width, source.height, decoded.width, decoded.height
            );
            stats.rows_decode_fail += 1;
            continue;
        }

        // Score + features.
        let (score, features) = match run_zensim_with_features(source, &decoded) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[backfill] zensim failed: {} q={q}: {e}", image_path_str);
                stats.rows_score_fail += 1;
                continue;
            }
        };

        if let Err(e) = writer.push_row(
            image_path_str,
            codec.name(),
            q,
            knob_json,
            score as f32,
            &features,
        ) {
            eprintln!(
                "[backfill] writer push failed: {} q={q}: {e}",
                image_path_str
            );
            stats.rows_score_fail += 1;
            continue;
        }
        stats.rows_emitted += 1;
    }

    writer.finish()?;
    // Atomic rename: temp → final.
    std::fs::rename(&tmp_path, output_parquet).map_err(|e| {
        format!(
            "rename {} → {}: {e}",
            tmp_path.display(),
            output_parquet.display()
        )
    })?;

    Ok(stats)
}

fn parse_codec(s: &str) -> Option<CodecKind> {
    match s.trim() {
        "zenwebp" => Some(CodecKind::Zenwebp),
        "zenavif" => Some(CodecKind::Zenavif),
        "zenjxl" => Some(CodecKind::Zenjxl),
        _ => None,
    }
}

fn parse_knob_json(s: &str) -> Result<Map<String, Value>, Box<dyn Error>> {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return Ok(Map::new());
    }
    let v: Value = serde_json::from_str(trimmed)?;
    let obj = v.as_object().ok_or("knob_tuple_json is not an object")?;
    Ok(obj.clone())
}

/// Resolve a TSV `image_path` value against `corpus_root`.
///
/// Strategy:
/// 1. If the path exists as written, return it (covers tests that pass
///    real paths).
/// 2. Take the basename. Try `corpus_root/<basename>` (flat corpus).
/// 3. Replace `__` in the basename with `/` (undo onstart staging
///    flatten) and try `corpus_root/<that>`.
/// 4. As a last resort, walk corpus_root looking for any file whose
///    name equals the basename (handles weird stage names from older
///    runs). The walk caches nothing — call sites already cache the
///    decoded image, so resolution overhead is at most once per
///    distinct image.
pub(crate) fn resolve_image_path(image_path: &str, corpus_root: &Path) -> Option<PathBuf> {
    let p = Path::new(image_path);
    if p.is_file() {
        return Some(p.to_path_buf());
    }
    let basename = p.file_name()?.to_string_lossy().to_string();

    // 2. Flat: corpus_root/basename.
    let flat = corpus_root.join(&basename);
    if flat.is_file() {
        return Some(flat);
    }

    // 3. Unflatten: replace `__` with `/`.
    if basename.contains("__") {
        let unflat = basename.replace("__", "/");
        let nested = corpus_root.join(&unflat);
        if nested.is_file() {
            return Some(nested);
        }
    }

    // 4. Walk fallback. Bounded: stop at first match. We don't index
    // the corpus up-front because in the steady state every chunk has
    // resolvable paths via (2) or (3); the walk is a defensive
    // last-ditch lookup for ad-hoc TSVs and shouldn't fire on the v04
    // / v05c corpora.
    walk_find_basename(corpus_root, &basename)
}

fn walk_find_basename(root: &Path, basename: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().map(|n| n == basename).unwrap_or(false) {
                return Some(path);
            }
        }
    }
    None
}

fn find_col(headers: &csv::StringRecord, names: &[&str]) -> Result<usize, Box<dyn Error>> {
    for (idx, h) in headers.iter().enumerate() {
        for n in names {
            if h.eq_ignore_ascii_case(n) {
                return Ok(idx);
            }
        }
    }
    Err(format!("input TSV is missing one of the expected columns: {names:?}").into())
}

fn decode_encoded_bytes(bytes: &[u8], codec: CodecKind) -> Result<Rgb8Image, Box<dyn Error>> {
    let suffix = match codec {
        CodecKind::Zenwebp => ".webp",
        CodecKind::Zenavif => ".avif",
        CodecKind::Zenjxl => ".jxl",
    };
    let tmp = tempfile::Builder::new()
        .prefix("zen-metrics-backfill-")
        .suffix(suffix)
        .tempfile()?;
    std::fs::write(tmp.path(), bytes)?;
    decode_image_to_rgb8(tmp.path())
}

// ── R2 mode ─────────────────────────────────────────────────────────────

fn run_r2_backfill(
    run_id: &str,
    codec: CodecKind,
    r2_prefix: Option<&str>,
    output_r2_prefix: Option<&str>,
    corpus_root: &Path,
) -> Result<BackfillStats, Box<dyn Error>> {
    let in_prefix = r2_prefix
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| format!("s3://zentrain/{run_id}/{name}", name = codec.name()));
    let out_prefix = output_r2_prefix
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| format!("{in_prefix}/features"));

    eprintln!("[backfill] R2 mode: input prefix = {in_prefix}");
    eprintln!("[backfill] R2 mode: output prefix = {out_prefix}");

    let chunk_keys = list_r2_chunk_tsvs(&in_prefix)?;
    if chunk_keys.is_empty() {
        return Err(format!("no chunk TSVs found under {in_prefix}/").into());
    }
    eprintln!("[backfill] found {} chunk TSVs", chunk_keys.len());

    let staging = tempfile::Builder::new()
        .prefix("zen-metrics-backfill-r2-")
        .tempdir()?;

    let mut stats = BackfillStats {
        chunks_total: chunk_keys.len(),
        ..Default::default()
    };

    for tsv_key in &chunk_keys {
        let chunk_id = extract_chunk_id(tsv_key);
        let parquet_key = format!("{out_prefix}/{chunk_id}.parquet");

        // 1. Pre-skip if already present.
        if r2_object_exists(&parquet_key)? {
            eprintln!("[backfill] skip: {parquet_key} already present");
            stats.chunks_skipped += 1;
            continue;
        }

        let local_tsv = staging.path().join(format!("{chunk_id}.tsv"));
        let local_parquet = staging.path().join(format!("{chunk_id}.parquet"));

        // 2. Pull TSV.
        if let Err(e) = s5cmd_cp(tsv_key, &local_tsv.to_string_lossy()) {
            eprintln!("[backfill] failed to pull {tsv_key}: {e}");
            stats.chunks_failed += 1;
            continue;
        }

        // 3-4. Re-encode + write parquet.
        let chunk_stats =
            match backfill_one_tsv(&local_tsv, &local_parquet, corpus_root, Some(codec)) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[backfill] {chunk_id} backfill failed: {e}");
                    stats.chunks_failed += 1;
                    let _ = std::fs::remove_file(&local_tsv);
                    continue;
                }
            };

        // 5. Upload parquet to a temp key first, then rename. R2 doesn't
        // have rename so we upload to `<key>.uploading` then issue a
        // server-side copy. If the copy fails we leave the .uploading
        // sentinel for human inspection.
        let staging_key = format!("{parquet_key}.uploading");
        if let Err(e) = s5cmd_cp(&local_parquet.to_string_lossy(), &staging_key) {
            eprintln!("[backfill] {chunk_id} upload (staging) failed: {e}");
            stats.chunks_failed += 1;
            let _ = std::fs::remove_file(&local_tsv);
            let _ = std::fs::remove_file(&local_parquet);
            continue;
        }
        if let Err(e) = s5cmd_mv(&staging_key, &parquet_key) {
            eprintln!("[backfill] {chunk_id} upload (rename) failed: {e}");
            stats.chunks_failed += 1;
            let _ = std::fs::remove_file(&local_tsv);
            let _ = std::fs::remove_file(&local_parquet);
            continue;
        }

        eprintln!(
            "[backfill] {chunk_id}: emitted {}/{} rows ({} resolve, {} encode, {} decode, {} score failures)",
            chunk_stats.rows_emitted,
            chunk_stats.rows_total,
            chunk_stats.rows_resolve_fail,
            chunk_stats.rows_encode_fail,
            chunk_stats.rows_decode_fail,
            chunk_stats.rows_score_fail,
        );
        stats.chunks_processed += 1;
        accumulate(&mut stats, &chunk_stats);
        let _ = std::fs::remove_file(&local_tsv);
        let _ = std::fs::remove_file(&local_parquet);
    }

    Ok(stats)
}

fn list_r2_chunk_tsvs(prefix: &str) -> Result<Vec<String>, Box<dyn Error>> {
    // `s5cmd ls s3://bucket/prefix/*.tsv` lists matching keys. Output is
    // size/date columns + key; we want only the key column. We use a
    // glob that matches direct children only — chunk TSVs live at
    // `<prefix>/<chunk_id>.tsv`, never deeper.
    let glob = format!("{}/*.tsv", prefix);
    let out = Command::new("s5cmd")
        .args(["ls", &glob])
        .output()
        .map_err(|e| format!("invoke s5cmd: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "s5cmd ls {glob} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut keys = Vec::new();
    for line in stdout.lines() {
        // s5cmd ls output: `<date> <time> <size> <name>` for files. The
        // name is the basename of the listed glob, NOT the full s3://
        // path. We reconstruct the full key from the prefix.
        let name = line.split_whitespace().last().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        if !name.ends_with(".tsv") {
            continue;
        }
        // Skip the concat output (`<codec>_pareto_concat.tsv`) and
        // any `_manifest`-shaped sentinels. Real chunk IDs look like
        // `zenwebp-000` etc.
        if name.contains("_concat") || name.contains("_manifest") {
            continue;
        }
        keys.push(format!("{prefix}/{name}"));
    }
    Ok(keys)
}

fn r2_object_exists(key: &str) -> Result<bool, Box<dyn Error>> {
    // `s5cmd ls <key>` exits 0 when the object exists, non-zero otherwise.
    let out = Command::new("s5cmd")
        .args(["ls", key])
        .output()
        .map_err(|e| format!("invoke s5cmd ls {key}: {e}"))?;
    if out.status.success() {
        // Defensive: ls also matches prefix expansion. Confirm we got
        // exactly one match for the same basename.
        let stdout = String::from_utf8_lossy(&out.stdout);
        let basename = key.rsplit('/').next().unwrap_or(key);
        Ok(stdout
            .lines()
            .any(|l| l.split_whitespace().last() == Some(basename)))
    } else {
        // Non-zero on miss is the documented behaviour.
        Ok(false)
    }
}

fn s5cmd_cp(src: &str, dst: &str) -> Result<(), Box<dyn Error>> {
    let out = Command::new("s5cmd")
        .args(["cp", src, dst])
        .output()
        .map_err(|e| format!("invoke s5cmd cp: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "s5cmd cp {src} → {dst} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(())
}

fn s5cmd_mv(src: &str, dst: &str) -> Result<(), Box<dyn Error>> {
    let out = Command::new("s5cmd")
        .args(["mv", src, dst])
        .output()
        .map_err(|e| format!("invoke s5cmd mv: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "s5cmd mv {src} → {dst} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(())
}

/// Extract `<chunk_id>` from a key like `s3://bucket/path/zenwebp-000.tsv`.
fn extract_chunk_id(tsv_key: &str) -> String {
    let basename = tsv_key.rsplit('/').next().unwrap_or(tsv_key);
    basename.trim_end_matches(".tsv").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_chunk_id_basic() {
        assert_eq!(
            extract_chunk_id("s3://zentrain/run/zenwebp/zenwebp-007.tsv"),
            "zenwebp-007"
        );
        assert_eq!(extract_chunk_id("zenavif-001.tsv"), "zenavif-001");
    }

    #[test]
    fn parse_codec_known() {
        assert!(matches!(parse_codec("zenwebp"), Some(CodecKind::Zenwebp)));
        assert!(matches!(
            parse_codec("  zenavif "),
            Some(CodecKind::Zenavif)
        ));
        assert!(parse_codec("nope").is_none());
    }

    #[test]
    fn resolve_flat_corpus() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("foo.png"), b"x").unwrap();
        let resolved = resolve_image_path("/some/old/staging/foo.png", dir.path());
        assert_eq!(resolved.unwrap(), dir.path().join("foo.png"));
    }

    #[test]
    fn resolve_unflatten_corpus() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub/sub2")).unwrap();
        std::fs::write(dir.path().join("sub/sub2/bar.png"), b"x").unwrap();
        let resolved = resolve_image_path("/old/staging/sub__sub2__bar.png", dir.path());
        assert_eq!(resolved.unwrap(), dir.path().join("sub/sub2/bar.png"));
    }

    #[test]
    fn resolve_walk_fallback() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::write(dir.path().join("a/b/baz.png"), b"x").unwrap();
        // Basename matches but no `__` flatten pattern — walk fallback
        // should locate it.
        let resolved = resolve_image_path("/staged/baz.png", dir.path());
        assert_eq!(resolved.unwrap(), dir.path().join("a/b/baz.png"));
    }

    #[test]
    fn resolve_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let resolved = resolve_image_path("/staged/nope.png", dir.path());
        assert!(resolved.is_none());
    }

    #[test]
    fn parse_knob_json_empty_object() {
        let m = parse_knob_json("{}").unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn parse_knob_json_with_values() {
        let m = parse_knob_json(r#"{"method": 4, "segments": 2}"#).unwrap();
        assert_eq!(m.get("method").and_then(|v| v.as_u64()), Some(4));
    }
}
