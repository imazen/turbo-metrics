#![forbid(unsafe_code)]

//! Sweep execution: walk the (image × q × knobs) Cartesian grid, encode
//! each cell, decode-back, score against the source for every selected
//! metric, and write a Pareto TSV.
//!
//! The runner is intentionally serial. CPU metrics already use rayon
//! internally where they care to; GPU metrics serialize through one
//! CubeCL stream and parallel encode would only fight them for memory.
//! Adding a `--jobs` knob to fan out per source image is a separate
//! follow-up — the harness prints a progress line per cell so callers
//! can pipe a chunk through `parallel` if they want a hand-rolled fan
//! out.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::decode::{Rgb8Image, decode_image_to_rgb8};
use crate::metrics::{GpuRuntime, MetricKind, run_metric};
use crate::sweep::encode::{CodecKind, encode};
use crate::sweep::grid::{KnobGrid, KnobTuple};

/// Runtime parameters for a sweep invocation.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub codec: CodecKind,
    pub sources: Vec<PathBuf>,
    pub q_grid: Vec<u32>,
    pub knob_grid: KnobGrid,
    pub metrics: Vec<MetricKind>,
    pub gpu_runtime: GpuRuntime,
    pub output: PathBuf,
}

/// Drive the sweep end-to-end. The function streams TSV rows to disk as
/// they are produced — even if a later cell panics or runs out of memory,
/// the rows that landed are durable.
pub fn run_sweep(cfg: &SweepConfig) -> Result<SweepStats, Box<dyn Error>> {
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&cfg.output)?;

    write_header(&mut wtr, &cfg.metrics)?;

    let mut stats = SweepStats {
        cells_total: cfg.sources.len() * cfg.q_grid.len() * cfg.knob_grid.cell_count(),
        cells_emitted: 0,
        cells_failed_encode: 0,
        cells_failed_decode: 0,
        cells_failed_score: 0,
    };

    for src_path in &cfg.sources {
        // Decode the source once per image so we don't re-PNG-decode for
        // every cell. The bytes are freed when we move to the next image.
        let source = match decode_image_to_rgb8(src_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!(
                    "[sweep] skipping {} (decode failed: {e})",
                    src_path.display()
                );
                stats.cells_failed_decode += cfg.q_grid.len() * cfg.knob_grid.cell_count();
                continue;
            }
        };

        for &q in &cfg.q_grid {
            for tuple in cfg.knob_grid.iter_tuples() {
                run_one_cell(cfg, &mut wtr, src_path, &source, q, &tuple, &mut stats);
            }
        }
    }

    wtr.flush()?;
    Ok(stats)
}

fn run_one_cell(
    cfg: &SweepConfig,
    wtr: &mut csv::Writer<std::fs::File>,
    src_path: &Path,
    source: &Rgb8Image,
    q: u32,
    tuple: &KnobTuple,
    stats: &mut SweepStats,
) {
    let knob_json = tuple.to_canonical_json();
    let mut row: Vec<String> = vec![
        src_path.display().to_string(),
        cfg.codec.name().to_string(),
        q.to_string(),
        knob_json.clone(),
    ];

    // Encode.
    let cell = match encode(cfg.codec, source, q, &tuple.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[sweep] encode failed: {} q={q} knobs={knob_json}: {e}",
                src_path.display()
            );
            stats.cells_failed_encode += 1;
            // Emit a row with blank score columns so downstream tooling
            // can see the cell was attempted.
            row.push("".to_string()); // encoded_bytes
            row.push("".to_string()); // encode_ms
            row.push("".to_string()); // decode_ms
            for _ in &cfg.metrics {
                row.push("".to_string());
            }
            let _ = wtr.write_record(&row);
            return;
        }
    };

    row.push(cell.bytes.len().to_string());
    row.push(format!("{:.3}", cell.encode_ms));

    // Decode-back. We must round-trip through the same decoder the metric
    // crates would see in production. `decode_bytes_to_rgb8` doesn't
    // exist as a public helper today, so we write to a tempfile and
    // re-use the existing path-based decoder. The tempfile pattern keeps
    // the format-sniffing logic identical.
    let decode_start = Instant::now();
    let decoded = match decode_encoded_bytes(&cell.bytes, cfg.codec) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "[sweep] decode-back failed: {} q={q} knobs={knob_json}: {e}",
                src_path.display()
            );
            stats.cells_failed_decode += 1;
            row.push("".to_string()); // decode_ms
            for _ in &cfg.metrics {
                row.push("".to_string());
            }
            let _ = wtr.write_record(&row);
            return;
        }
    };
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    row.push(format!("{decode_ms:.3}"));

    // Dimension check — codecs that pad odd dimensions should still
    // round-trip the original size; mismatch usually means a chroma
    // upsampling bug or a wrong pixel-format conversion. Either way we
    // skip scoring rather than ship a misleading number.
    if decoded.width != source.width || decoded.height != source.height {
        eprintln!(
            "[sweep] dimension mismatch: {} q={q} src={}x{} decoded={}x{}",
            src_path.display(),
            source.width,
            source.height,
            decoded.width,
            decoded.height,
        );
        stats.cells_failed_decode += 1;
        for _ in &cfg.metrics {
            row.push("".to_string());
        }
        let _ = wtr.write_record(&row);
        return;
    }

    // Score every selected metric.
    let mut any_score_failed = false;
    for &metric in &cfg.metrics {
        match run_metric(metric, source, &decoded, cfg.gpu_runtime) {
            Ok(score) => row.push(format!("{score:.6}")),
            Err(e) => {
                eprintln!(
                    "[sweep] metric {} failed on {} q={q}: {e}",
                    metric.name(),
                    src_path.display()
                );
                row.push("".to_string());
                any_score_failed = true;
            }
        }
    }
    if any_score_failed {
        stats.cells_failed_score += 1;
    }

    if let Err(e) = wtr.write_record(&row) {
        eprintln!("[sweep] write_record failed: {e}");
    } else {
        stats.cells_emitted += 1;
    }
}

fn decode_encoded_bytes(bytes: &[u8], codec: CodecKind) -> Result<Rgb8Image, Box<dyn Error>> {
    // The path-based decode_image_to_rgb8 sniffs format and dispatches
    // through the per-codec decoder. We write to a tempfile to reuse it
    // unchanged. Performance: write+read is on the order of microseconds
    // for typical encoded sizes (10 KB - 1 MB) and dominates neither
    // encode nor decode wall time, so we don't optimize this away.
    let suffix = match codec {
        CodecKind::Zenwebp => ".webp",
        CodecKind::Zenavif => ".avif",
        CodecKind::Zenjxl => ".jxl",
    };
    let tmp = tempfile::Builder::new()
        .prefix("zen-metrics-sweep-")
        .suffix(suffix)
        .tempfile()?;
    std::fs::write(tmp.path(), bytes)?;
    decode_image_to_rgb8(tmp.path())
}

fn write_header(
    wtr: &mut csv::Writer<std::fs::File>,
    metrics: &[MetricKind],
) -> Result<(), Box<dyn Error>> {
    let mut headers: Vec<&str> = vec![
        "image_path",
        "codec",
        "q",
        "knob_tuple_json",
        "encoded_bytes",
        "encode_ms",
        "decode_ms",
    ];
    let metric_cols: Vec<String> = metrics
        .iter()
        .map(|m| format!("score_{}", m.column_name()))
        .collect();
    for c in &metric_cols {
        headers.push(c.as_str());
    }
    wtr.write_record(&headers)?;
    Ok(())
}

/// Aggregate counters from a sweep run. Useful for the EOM report.
#[derive(Debug, Clone, Copy, Default)]
pub struct SweepStats {
    pub cells_total: usize,
    pub cells_emitted: usize,
    pub cells_failed_encode: usize,
    pub cells_failed_decode: usize,
    pub cells_failed_score: usize,
}
