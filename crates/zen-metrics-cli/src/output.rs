#![forbid(unsafe_code)]

//! Output formatting for `score` and `compare` subcommands.

use clap::ValueEnum;
use std::io::{self, Write};

use crate::metrics::MetricKind;

/// CLI output format.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Plain text: `metric=<name> score=<value>`.
    Plain,
    /// JSON object on a single line.
    Json,
    /// Two-row TSV with headers on the first row.
    Tsv,
}

pub fn print_score(format: OutputFormat, kind: MetricKind, score: f64) {
    match format {
        OutputFormat::Plain => println!("metric={} score={:.6}", kind.name(), score),
        OutputFormat::Json => {
            let v = serde_json::json!({"metric": kind.name(), "score": score});
            println!("{v}");
        }
        OutputFormat::Tsv => {
            println!("metric\tscore");
            println!("{}\t{:.6}", kind.name(), score);
        }
    }
}

/// One (reference, variant) row of a `compare` run, with one score per metric
/// in `metrics_order`. `Err(reason)` slots are recorded as `null` / `NaN` /
/// `ERROR: <reason>` depending on the output format.
pub struct CompareRow {
    pub reference: String,
    pub variant: String,
    /// Parallel to `metrics_order` — one entry per metric, in the same order.
    pub scores: Vec<Result<f64, String>>,
}

/// Render the full result set of a `compare` invocation to `w`.
pub fn render_compare(
    w: &mut dyn Write,
    format: OutputFormat,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
) -> io::Result<()> {
    match format {
        OutputFormat::Plain => render_compare_plain(w, metrics_order, rows),
        OutputFormat::Json => render_compare_json(w, metrics_order, rows),
        OutputFormat::Tsv => render_compare_tsv(w, metrics_order, rows),
    }
}

fn render_compare_plain(
    w: &mut dyn Write,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
) -> io::Result<()> {
    // Width of the longest metric name, for column alignment on the score column.
    let name_width = metrics_order
        .iter()
        .map(|m| m.name().len())
        .max()
        .unwrap_or(0);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            writeln!(w)?;
        }
        writeln!(w, "{} vs {}:", row.reference, row.variant)?;
        for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
            match score {
                Ok(v) => writeln!(w, "  {:<width$}  {:.6}", m.name(), v, width = name_width)?,
                Err(reason) => writeln!(
                    w,
                    "  {:<width$}  ERROR: {}",
                    m.name(),
                    reason,
                    width = name_width
                )?,
            }
        }
    }
    Ok(())
}

fn render_compare_json(
    w: &mut dyn Write,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
) -> io::Result<()> {
    let metric_names: Vec<&str> = metrics_order.iter().map(|m| m.name()).collect();
    let results: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            // Preserve metric column order in the emitted JSON object.
            let mut scores = serde_json::Map::with_capacity(metrics_order.len());
            for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
                let v = match score {
                    Ok(v) => serde_json::Value::from(*v),
                    Err(_) => serde_json::Value::Null,
                };
                scores.insert(m.name().to_string(), v);
            }
            // Field order: reference, variant, scores (matches the
            // `compare --help` documented schema). `preserve_order` on
            // serde_json keeps this insertion order in the rendered JSON.
            let mut row_obj = serde_json::Map::with_capacity(3);
            row_obj.insert("reference".into(), row.reference.clone().into());
            row_obj.insert("variant".into(), row.variant.clone().into());
            row_obj.insert("scores".into(), serde_json::Value::Object(scores));
            serde_json::Value::Object(row_obj)
        })
        .collect();
    let doc = serde_json::json!({
        "metrics": metric_names,
        "results": results,
    });
    serde_json::to_writer_pretty(&mut *w, &doc)?;
    writeln!(w)?;
    Ok(())
}

fn render_compare_tsv(
    w: &mut dyn Write,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
) -> io::Result<()> {
    // Header: reference, variant, then one column per metric (using the CLI
    // metric name verbatim, not the friendlier batch column_name).
    write!(w, "reference\tvariant")?;
    for m in metrics_order {
        write!(w, "\t{}", m.name())?;
    }
    writeln!(w)?;
    for row in rows {
        write!(w, "{}\t{}", row.reference, row.variant)?;
        for score in &row.scores {
            match score {
                Ok(v) => write!(w, "\t{v:.6}")?,
                // NaN signals a failed cell in TSV mode — distinguishable
                // from any real metric value because metrics never produce
                // NaN on success (we explicitly reject non-finite scores).
                Err(_) => write!(w, "\tNaN")?,
            }
        }
        writeln!(w)?;
    }
    Ok(())
}
