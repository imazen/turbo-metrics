#![forbid(unsafe_code)]

//! Output formatting for the `score` subcommand.

use clap::ValueEnum;

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
