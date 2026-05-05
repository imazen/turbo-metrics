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

/// Render the score(s) returned by a single [`crate::metrics::run_metric`]
/// invocation. `scores` is the `(column_name, value)` list — one entry for
/// most metrics, two for butteraugli (max + 3-norm).
///
/// TSV layout matches the `batch` / `sweep` convention: one column per
/// score with the column name as the header. Plain and JSON layouts emit
/// every column too — JSON nests the values under `"scores"` keyed by
/// column name so callers can read individual fields without parsing the
/// metric out.
pub fn print_score(format: OutputFormat, kind: MetricKind, scores: &[(&'static str, f64)]) {
    match format {
        OutputFormat::Plain => {
            print!("metric={}", kind.name());
            for (col, value) in scores {
                print!(" {col}={value:.6}");
            }
            println!();
        }
        OutputFormat::Json => {
            let mut score_map = serde_json::Map::with_capacity(scores.len());
            for (col, value) in scores {
                score_map.insert((*col).to_string(), serde_json::Value::from(*value));
            }
            let v = serde_json::json!({
                "metric": kind.name(),
                "scores": serde_json::Value::Object(score_map),
            });
            println!("{v}");
        }
        OutputFormat::Tsv => {
            // Two-row TSV: columns are the score names; the metric name is
            // not a column anymore because a single metric can emit several
            // columns (butteraugli emits both max + pnorm3).
            let header: Vec<&str> = scores.iter().map(|(col, _)| *col).collect();
            println!("{}", header.join("\t"));
            let row: Vec<String> = scores.iter().map(|(_, v)| format!("{v:.6}")).collect();
            println!("{}", row.join("\t"));
        }
    }
}

/// One (reference, variant) row of a `compare` run, with one score per metric
/// in `metrics_order`. `Err(reason)` slots are recorded as `null` / `NaN` /
/// `ERROR: <reason>` depending on the output format.
///
/// Each metric may contribute multiple columns (butteraugli emits two), so
/// `scores[i]` is itself a `Vec<(column_name, value)>` parallel to
/// `metrics_order[i].column_names()`. When a metric fails the whole entry
/// is `Err` — we don't track per-column success because the metric backend
/// is what fails, not individual aggregations.
pub struct CompareRow {
    pub reference: String,
    pub variant: String,
    /// Parallel to `metrics_order` — one entry per metric, in the same
    /// order. Each entry is the metric's full `(column_name, value)` list.
    pub scores: Vec<Result<Vec<(&'static str, f64)>, String>>,
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
    // Width of the longest column name, for alignment of the score column.
    // We use column_names rather than metric names so butteraugli's two
    // columns line up.
    let name_width = metrics_order
        .iter()
        .flat_map(|m| m.column_names().iter().copied())
        .map(str::len)
        .max()
        .unwrap_or(0);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            writeln!(w)?;
        }
        writeln!(w, "{} vs {}:", row.reference, row.variant)?;
        for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
            match score {
                Ok(values) => {
                    for (col, v) in values {
                        writeln!(w, "  {col:<name_width$}  {v:.6}")?;
                    }
                }
                Err(reason) => {
                    // Print an error line per column the metric would have
                    // produced, so callers diff'ing TSV-shaped output still
                    // see one entry per column they expected.
                    for col in m.column_names() {
                        writeln!(w, "  {col:<name_width$}  ERROR: {reason}")?;
                    }
                }
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
            // `scores` is keyed by column name, not metric name. For
            // single-column metrics that's identical to the previous
            // behaviour; for butteraugli it lets callers pull
            // `scores.butteraugli_max` and `scores.butteraugli_pnorm3`
            // independently. Failed metrics produce nulls for every
            // column the metric would have emitted.
            let total_cols: usize = metrics_order.iter().map(|m| m.column_names().len()).sum();
            let mut scores = serde_json::Map::with_capacity(total_cols);
            for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
                match score {
                    Ok(values) => {
                        for (col, v) in values {
                            scores.insert((*col).to_string(), serde_json::Value::from(*v));
                        }
                    }
                    Err(_) => {
                        for col in m.column_names() {
                            scores.insert((*col).to_string(), serde_json::Value::Null);
                        }
                    }
                }
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
    // Header: reference, variant, then one column per (metric × emitted
    // column). Single-metric butteraugli expands to two columns; everything
    // else is one. Column names use the friendly underscore form (matches
    // the sweep TSV `score_<column>` convention without the `score_`
    // prefix).
    write!(w, "reference\tvariant")?;
    for m in metrics_order {
        for col in m.column_names() {
            write!(w, "\t{col}")?;
        }
    }
    writeln!(w)?;
    for row in rows {
        write!(w, "{}\t{}", row.reference, row.variant)?;
        for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
            match score {
                Ok(values) => {
                    for (_, v) in values {
                        write!(w, "\t{v:.6}")?;
                    }
                }
                // NaN signals a failed cell in TSV mode — distinguishable
                // from any real metric value because metrics never produce
                // NaN on success (we explicitly reject non-finite scores).
                // Emit one NaN per column the metric would have produced.
                Err(_) => {
                    for _ in m.column_names() {
                        write!(w, "\tNaN")?;
                    }
                }
            }
        }
        writeln!(w)?;
    }
    Ok(())
}
