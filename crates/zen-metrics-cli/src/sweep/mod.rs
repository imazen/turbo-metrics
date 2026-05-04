#![forbid(unsafe_code)]

//! `zen-metrics sweep` — encode a corpus of source images across a Cartesian
//! grid of (codec, quality, knob-tuple) triples and score every encoded
//! variant with one or more perceptual metrics.
//!
//! The driver lives entirely outside any codec source tree: codecs stay
//! "dumb" (no in-codec picker glue, no `.bin` shipped alongside the encoder)
//! and we orchestrate science from this binary by poking each codec's
//! `__expert` / public encode API. One encoded variant per cell is held in
//! memory just long enough to (a) decode it back to RGB8 for scoring and
//! (b) write a Pareto row, then dropped.
//!
//! Output schema (TSV, one row per cell):
//! ```text
//! image_path codec q knob_tuple_json encoded_bytes encode_ms decode_ms \
//!     score_<metric_1> score_<metric_2> ...
//! ```
//!
//! The metric column names match `MetricKind::column_name()` so existing
//! Pareto / picker tooling already knows how to consume them. Cells that
//! fail to encode or score are emitted with `score_*` columns blank but the
//! `encode_ms` / `decode_ms` columns still populated where possible, so a
//! downstream filter step can drop or quarantine partial rows without
//! having to re-run the sweep.

pub mod encode;
pub mod feature_writer;
pub mod grid;
pub mod run;

pub use encode::CodecKind;
#[allow(unused_imports)]
pub use grid::{KnobGrid, KnobTuple, parse_knob_grid, parse_q_grid};
pub use run::{SweepConfig, run_sweep};
