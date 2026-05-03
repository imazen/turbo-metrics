#![forbid(unsafe_code)]

//! `compare` subcommand: run M references × N variants × K metrics in a
//! single invocation, decoding each unique image only once.
//!
//! The natural picker-evaluation workflow is: "given one reference, score
//! it against several encoded variants on several metrics simultaneously,
//! and give me one structured result." `score` does 1×1×1, `batch` does
//! N×1 from a TSV with one metric. `compare` covers the cartesian product
//! and amortises decoding.
//!
//! Per-cell failures (decode error, metric panic, GPU OOM) are recorded in
//! the result rather than aborting the whole run. The process exits non-zero
//! when any cell failed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::decode::{Rgb8Image, decode_image_to_rgb8};
use crate::metrics::{GpuRuntime, MetricKind, run_metric};
use crate::output::{CompareRow, OutputFormat, render_compare};

/// Outcome of a `compare` run: the structured rows plus a flag indicating
/// whether any individual cell failed.
pub struct CompareReport {
    pub rows: Vec<CompareRow>,
    pub had_failures: bool,
}

/// Run a cartesian comparison: every reference × every variant × every
/// metric. Decoding is cached so each unique image hits the codec exactly
/// once.
///
/// `_jobs` is currently accepted but unused — the loop is serial. CPU
/// parallelism via rayon is left as a follow-up to match `batch`'s policy
/// (zen decoders and CPU metrics mix Send + non-Send internals that need
/// per-metric review). GPU metrics serialise through one CubeCL stream
/// regardless.
pub fn run_compare(
    references: &[PathBuf],
    variants: &[PathBuf],
    metrics: &[MetricKind],
    gpu_runtime: GpuRuntime,
    _jobs: usize,
) -> CompareReport {
    // Decode each unique path exactly once. Errors are stored too — a
    // reference that fails to decode produces error rows for every (variant,
    // metric) it would have been paired with, but the rest of the run still
    // proceeds.
    let mut cache: HashMap<PathBuf, Result<Arc<Rgb8Image>, String>> = HashMap::new();
    for path in references.iter().chain(variants.iter()) {
        let key = canonical_key(path);
        cache.entry(key).or_insert_with(|| {
            decode_image_to_rgb8(path)
                .map(Arc::new)
                .map_err(|e| format!("decode {}: {e}", path.display()))
        });
    }

    let mut rows: Vec<CompareRow> = Vec::with_capacity(references.len() * variants.len());
    let mut had_failures = false;

    for reference in references {
        for variant in variants {
            let ref_key = canonical_key(reference);
            let var_key = canonical_key(variant);
            let ref_img = cache.get(&ref_key).expect("populated above");
            let var_img = cache.get(&var_key).expect("populated above");

            let mut scores: Vec<Result<f64, String>> = Vec::with_capacity(metrics.len());
            for &metric in metrics {
                let result = score_one(ref_img, var_img, reference, variant, metric, gpu_runtime);
                if let Err(ref reason) = result {
                    had_failures = true;
                    eprintln!(
                        "warning: {} vs {} [{}]: {reason}",
                        reference.display(),
                        variant.display(),
                        metric.name()
                    );
                }
                scores.push(result);
            }

            rows.push(CompareRow {
                reference: reference.display().to_string(),
                variant: variant.display().to_string(),
                scores,
            });
        }
    }

    CompareReport { rows, had_failures }
}

/// Score a single (ref, variant, metric) cell. Returns `Err(reason)` rather
/// than aborting on a decode/metric failure so the caller can record the
/// cell and keep going.
fn score_one(
    ref_img: &Result<Arc<Rgb8Image>, String>,
    var_img: &Result<Arc<Rgb8Image>, String>,
    ref_path: &Path,
    var_path: &Path,
    metric: MetricKind,
    gpu_runtime: GpuRuntime,
) -> Result<f64, String> {
    let r = ref_img.as_ref().map_err(|e| e.clone())?;
    let d = var_img.as_ref().map_err(|e| e.clone())?;
    if r.width != d.width || r.height != d.height {
        return Err(format!(
            "dimension mismatch: {} is {}x{}, {} is {}x{}",
            ref_path.display(),
            r.width,
            r.height,
            var_path.display(),
            d.width,
            d.height,
        ));
    }
    run_metric(metric, r, d, gpu_runtime).map_err(|e| e.to_string())
}

/// Render the report and stream it to stdout in the chosen format.
pub fn print_report(
    format: OutputFormat,
    metrics: &[MetricKind],
    report: &CompareReport,
) -> std::io::Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    render_compare(&mut handle, format, metrics, &report.rows)
}

/// Canonicalise the path so two spellings of the same file (e.g. `./a.png`
/// and `a.png`) share a cache slot. Falls back to the raw path if
/// canonicalisation fails (file not on disk yet, etc.) — in that case the
/// decode itself will produce the proper error.
fn canonical_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}
