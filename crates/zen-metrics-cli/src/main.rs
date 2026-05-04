#![forbid(unsafe_code)]

//! `zen-metrics` — a focused CLI for decoding zen image formats and scoring
//! perceptual quality metrics on either CPU or GPU.
//!
//! Built so encoder rate-distortion experiments and selector training runs
//! can call one binary instead of stitching together half a dozen
//! per-metric scripts. Format support and metrics are gated behind cargo
//! features; see `Cargo.toml` for the feature matrix.
//!
//! Subcommands at a glance:
//! - `score` — one (reference, distorted) pair, one metric.
//! - `batch` — N rows from a TSV, one metric, output TSV with a score column.
//! - `compare` — M references × N variants × K metrics in one invocation,
//!   with each unique image decoded only once. Use this when you have one
//!   reference and several encoded variants you want scored across multiple
//!   metrics for picker / RD-curve work.
//! - `list-metrics` / `list-formats` — environment introspection.
//!
//! See `--help` on any subcommand for full options.

mod compare;
mod decode;
mod metrics;
mod output;

#[cfg(feature = "sweep")]
mod sweep;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::process::ExitCode;

use crate::compare::{print_report, run_compare};
use crate::metrics::{GpuRuntime, MetricKind, run_metric};
use crate::output::{OutputFormat, print_score};

/// Top-level CLI parser.
#[derive(Parser, Debug)]
#[command(version, author = "Lilith River", about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Score a single (reference, distorted) image pair.
    Score(ScoreArgs),
    /// Run a metric over a TSV of image pairs.
    Batch(BatchArgs),
    /// Score every (reference, variant) pair across multiple metrics in
    /// one invocation. Decodes each unique image once, then runs all
    /// metrics back-to-back per pair.
    Compare(CompareArgs),
    /// Drive a codec across a (image × q × knob-tuple) Cartesian grid
    /// and score each encoded variant with one or more metrics. Emits
    /// a Pareto TSV. Only available when the binary is built with
    /// `--features sweep`.
    #[cfg(feature = "sweep")]
    Sweep(SweepArgs),
    /// Print available metrics and which require a GPU.
    ListMetrics,
    /// Print supported input formats.
    ListFormats,
}

#[derive(Parser, Debug)]
struct ScoreArgs {
    /// Metric to evaluate.
    #[arg(long, value_enum)]
    metric: MetricKind,
    /// Reference image path.
    #[arg(long)]
    reference: PathBuf,
    /// Distorted image path.
    #[arg(long)]
    distorted: PathBuf,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Output format. Defaults to plain text on stdout.
    #[arg(long, value_enum, default_value = "plain")]
    output: OutputFormat,
}

#[derive(Parser, Debug)]
struct BatchArgs {
    /// Metric to evaluate.
    #[arg(long, value_enum)]
    metric: MetricKind,
    /// Input TSV with at least two columns: ref_path and dist_path. The
    /// header row is required. Extra columns are passed through.
    #[arg(long)]
    pairs: PathBuf,
    /// Output TSV with the same columns plus a metric score column.
    #[arg(long)]
    output: PathBuf,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Number of CPU jobs (CPU metrics only). GPU metrics always serialize
    /// through one CubeCL stream.
    #[arg(long, default_value = "1")]
    jobs: usize,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    /// Reference image. Pass once per reference: `--reference a.png --reference b.png`.
    /// Every reference is paired with every `--variant`.
    #[arg(long = "reference", action = ArgAction::Append, required = true)]
    references: Vec<PathBuf>,
    /// Variant image. Pass once per variant. Every variant is scored
    /// against every `--reference`.
    #[arg(long = "variant", action = ArgAction::Append, required = true)]
    variants: Vec<PathBuf>,
    /// Metric to evaluate. Pass once per metric — every metric is run on
    /// every (reference, variant) pair.
    #[arg(long = "metric", value_enum, action = ArgAction::Append, required = true)]
    metrics: Vec<MetricKind>,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Output format. Plain is the default human-readable layout; JSON
    /// emits a structured object with `metrics` + `results`; TSV gives a
    /// flat table with one row per pair and one column per metric.
    #[arg(long, value_enum, default_value = "plain")]
    output: OutputFormat,
    /// Reserved for CPU parallelism. Currently serial — see `batch`.
    #[arg(long, default_value = "1")]
    jobs: usize,
}

#[cfg(feature = "sweep")]
#[derive(Parser, Debug)]
struct SweepArgs {
    /// Codec to drive.
    #[arg(long, value_enum)]
    codec: crate::sweep::CodecKind,
    /// Directory of source images. Every file the path-based decoder
    /// recognises (PNG / WebP / AVIF / JXL / JPEG when enabled) is
    /// included. Subdirectories are not walked.
    #[arg(long)]
    sources: PathBuf,
    /// Comma-separated list of integer qualities (0..=100). e.g.
    /// `5,10,15,20,...,95`.
    #[arg(long)]
    q_grid: String,
    /// JSON object `{axis: [values]}` describing the knob Cartesian
    /// product. See `crates/zen-metrics-cli/src/sweep/encode.rs` for
    /// the per-codec axis names.
    #[arg(long, default_value = "")]
    knob_grid: String,
    /// One or more metrics to score each cell with. Pass once per
    /// metric. Defaults to `zensim` if omitted.
    #[arg(long = "metric", value_enum, action = ArgAction::Append)]
    metrics: Vec<MetricKind>,
    /// Output Pareto TSV path.
    #[arg(long)]
    output: PathBuf,
    /// Optional path for a per-cell zensim feature parquet sidecar. When
    /// set, every cell that runs the `zensim` metric also persists its
    /// 300-feature extended vector to this parquet file. Joins back to
    /// `--output` (TSV) on `(image_path, codec, q, knob_tuple_json)`.
    /// The metric list must include `zensim` for any rows to be written.
    #[arg(long)]
    feature_output: Option<PathBuf>,
    /// CubeCL runtime selector for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Reserved for future fan-out. Currently serial.
    #[arg(long, default_value = "1")]
    jobs: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[allow(dead_code)]
enum FormatLabel {
    Png,
    Jpeg,
    Webp,
    Avif,
    Jxl,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Score(args) => match cmd_score(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::Batch(args) => match cmd_batch(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::Compare(args) => match cmd_compare(args) {
            Ok(success) => {
                if success {
                    ExitCode::SUCCESS
                } else {
                    // Some cells produced errors. Output is still complete
                    // and rendered, but signal a non-zero exit so callers
                    // (CI / driver scripts) can detect partial failures.
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        #[cfg(feature = "sweep")]
        Command::Sweep(args) => match cmd_sweep(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::ListMetrics => {
            print_metric_list();
            ExitCode::SUCCESS
        }
        Command::ListFormats => {
            print_format_list();
            ExitCode::SUCCESS
        }
    }
}

#[cfg(feature = "sweep")]
fn cmd_sweep(args: SweepArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::sweep::{SweepConfig, parse_knob_grid, parse_q_grid, run_sweep};

    let q_grid = parse_q_grid(&args.q_grid)?;
    let knob_grid = parse_knob_grid(&args.knob_grid)?;
    let mut metrics = args.metrics;
    if metrics.is_empty() {
        // Default metric set keeps the binary useful when invoked without
        // a `--metric` flag. zensim is the cheapest defensible default —
        // CPU-only, no GPU runtime needed, and already exposed by the
        // crate.
        metrics.push(MetricKind::Zensim);
    }

    // Walk the source directory (no recursion). Every file we can sniff
    // is included; everything else is skipped silently.
    let mut sources: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&args.sources)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() {
            sources.push(p);
        }
    }
    sources.sort();
    if sources.is_empty() {
        return Err(format!("no source files found in {}", args.sources.display()).into());
    }

    let cfg = SweepConfig {
        codec: args.codec,
        sources,
        q_grid,
        knob_grid,
        metrics,
        gpu_runtime: args.gpu_runtime,
        output: args.output,
        feature_output: args.feature_output,
    };
    let stats = run_sweep(&cfg)?;
    eprintln!(
        "[sweep] done: {emitted}/{total} cells emitted; \
         encode-fail={ef} decode-fail={df} score-fail={sf}",
        emitted = stats.cells_emitted,
        total = stats.cells_total,
        ef = stats.cells_failed_encode,
        df = stats.cells_failed_decode,
        sf = stats.cells_failed_score,
    );
    Ok(())
}

fn cmd_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    let reference = decode::decode_image_to_rgb8(&args.reference)?;
    let distorted = decode::decode_image_to_rgb8(&args.distorted)?;
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "dimension mismatch: reference is {}x{}, distorted is {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let score = run_metric(args.metric, &reference, &distorted, args.gpu_runtime)?;
    print_score(args.output, args.metric, score);
    Ok(())
}

fn cmd_batch(args: BatchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(&args.pairs)?;
    let headers = rdr.headers()?.clone();
    let ref_idx = find_col(&headers, &["ref_path", "reference"])?;
    let dist_idx = find_col(&headers, &["dist_path", "distorted"])?;

    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&args.output)?;
    let metric_col = args.metric.column_name();
    let mut new_headers: Vec<String> = headers.iter().map(String::from).collect();
    new_headers.push(metric_col.to_string());
    wtr.write_record(&new_headers)?;

    let _ = args.jobs; // Reserved: CPU parallelism. Currently serial; rayon
    // integration left for a follow-up because zen decoders and CPU metrics
    // mix Send + non-Send internals in ways that need per-metric review.

    for record in rdr.records() {
        let record = record?;
        let ref_path = PathBuf::from(record.get(ref_idx).ok_or("missing ref_path")?);
        let dist_path = PathBuf::from(record.get(dist_idx).ok_or("missing dist_path")?);
        let reference = decode::decode_image_to_rgb8(&ref_path)?;
        let distorted = decode::decode_image_to_rgb8(&dist_path)?;
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(format!(
                "dimension mismatch on row: {} ({}x{}) vs {} ({}x{})",
                ref_path.display(),
                reference.width,
                reference.height,
                dist_path.display(),
                distorted.width,
                distorted.height
            )
            .into());
        }
        let score = run_metric(args.metric, &reference, &distorted, args.gpu_runtime)?;
        let mut row: Vec<String> = record.iter().map(String::from).collect();
        row.push(format!("{score:.6}"));
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Returns `Ok(true)` when every cell succeeded and `Ok(false)` when at
/// least one cell failed (the report is still rendered in full). `Err` is
/// reserved for setup failures that prevent any output (currently: empty
/// argument lists, which clap also enforces via `required = true`).
fn cmd_compare(args: CompareArgs) -> Result<bool, Box<dyn std::error::Error>> {
    if args.references.is_empty() {
        return Err("at least one --reference is required".into());
    }
    if args.variants.is_empty() {
        return Err("at least one --variant is required".into());
    }
    if args.metrics.is_empty() {
        return Err("at least one --metric is required".into());
    }
    let report = run_compare(
        &args.references,
        &args.variants,
        &args.metrics,
        args.gpu_runtime,
        args.jobs,
    );
    print_report(args.output, &args.metrics, &report)?;
    Ok(!report.had_failures)
}

fn find_col(headers: &csv::StringRecord, names: &[&str]) -> Result<usize, String> {
    for (idx, h) in headers.iter().enumerate() {
        for n in names {
            if h.eq_ignore_ascii_case(n) {
                return Ok(idx);
            }
        }
    }
    Err(format!(
        "input TSV is missing one of the expected columns: {names:?}"
    ))
}

fn print_metric_list() {
    println!("name                 backend  requires_gpu");
    for m in MetricKind::all() {
        println!(
            "{:<20} {:<8} {}",
            m.name(),
            m.backend(),
            if m.requires_gpu() { "yes" } else { "no" }
        );
    }
    println!();
    println!("GPU runtimes (--gpu-runtime): auto, cuda, wgpu, hip, cpu");
}

fn print_format_list() {
    let mut formats: Vec<&str> = Vec::new();
    if cfg!(feature = "png") {
        formats.push("png  (zenpng)");
    }
    if cfg!(feature = "jpeg") {
        formats.push("jpeg (zenjpeg)");
    }
    if cfg!(feature = "webp") {
        formats.push("webp (zenwebp)");
    }
    if cfg!(feature = "avif") {
        formats.push("avif (zenavif)");
    }
    if cfg!(feature = "jxl") {
        formats.push("jxl  (zenjxl)");
    }
    if formats.is_empty() {
        println!("(no decoders enabled — rebuild with `--features png,jpeg,webp,avif,jxl`)");
    } else {
        for f in formats {
            println!("{f}");
        }
    }
}
