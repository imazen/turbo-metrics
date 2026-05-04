#![forbid(unsafe_code)]

//! Per-cell zensim feature vector → Parquet sidecar.
//!
//! When the sweep subcommand is invoked with `--feature-output <path.parquet>`,
//! every cell that runs the zensim metric also persists its 300-feature
//! extended vector here. The TSV row continues to carry the human-readable
//! `score_*` columns (ssim2, butteraugli, zensim, etc.) and the parquet sidecar
//! is joined back to the TSV by `(image_path, codec, q, knob_tuple_json)`.
//!
//! Schema (one row per encoded cell):
//! ```text
//! image_path       : utf8
//! codec            : utf8
//! q                : uint32
//! knob_tuple_json  : utf8
//! zensim_score     : float32
//! feat_0..feat_299 : float32   (300 columns)
//! ```
//!
//! Each `run_sweep` invocation owns one writer and produces one parquet file.
//! For chunked / distributed sweeps each worker writes its own file
//! (`features-<chunk_id>.parquet`); the upstream finalize step concatenates
//! them. We don't try to "append" to an existing parquet — the format isn't
//! row-appendable without a row-group rewrite, and per-chunk files are simpler
//! to reason about.
//!
//! Rows are buffered into Arrow batches and flushed at a fixed batch size
//! (`FLUSH_EVERY`) so the in-memory footprint stays bounded even for
//! million-cell sweeps.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// Number of features in the extended vector (4 scales × 3 channels × 25
/// features/channel at the default profile).
pub const NUM_FEATURES: usize = 300;

/// Flush an Arrow record batch to disk every `FLUSH_EVERY` rows. 256 keeps
/// memory bounded (≈ 256 × 304 floats × 4 B = 311 KiB per batch) while
/// amortising the parquet column-encoding fixed cost.
const FLUSH_EVERY: usize = 256;

/// Buffered Parquet writer for the per-cell feature sidecar.
///
/// Append rows with [`FeatureParquetWriter::push_row`]; call [`finish`] (or
/// drop) to flush and close the file. Dropping without `finish()` does its
/// best to flush — but a panic between the last `push_row` and the implicit
/// drop will leave a truncated file. Prefer the explicit `finish()` at the
/// end of a sweep.
pub struct FeatureParquetWriter {
    schema: Arc<Schema>,
    writer: ArrowWriter<File>,
    buf: RowBuffer,
}

struct RowBuffer {
    image_path: Vec<String>,
    codec: Vec<String>,
    q: Vec<u32>,
    knob_tuple_json: Vec<String>,
    zensim_score: Vec<f32>,
    /// `feature_columns[i]` collects the values for `feat_i` across rows.
    feature_columns: Vec<Vec<f32>>,
    rows: usize,
}

impl RowBuffer {
    fn new() -> Self {
        Self {
            image_path: Vec::with_capacity(FLUSH_EVERY),
            codec: Vec::with_capacity(FLUSH_EVERY),
            q: Vec::with_capacity(FLUSH_EVERY),
            knob_tuple_json: Vec::with_capacity(FLUSH_EVERY),
            zensim_score: Vec::with_capacity(FLUSH_EVERY),
            feature_columns: (0..NUM_FEATURES)
                .map(|_| Vec::with_capacity(FLUSH_EVERY))
                .collect(),
            rows: 0,
        }
    }

    fn clear(&mut self) {
        self.image_path.clear();
        self.codec.clear();
        self.q.clear();
        self.knob_tuple_json.clear();
        self.zensim_score.clear();
        for col in &mut self.feature_columns {
            col.clear();
        }
        self.rows = 0;
    }
}

impl FeatureParquetWriter {
    /// Create a new parquet writer at `path`. Overwrites if the file exists.
    pub fn create(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = Arc::new(build_schema());
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            // zstd is a compromise: smaller than snappy, slower than lz4 but
            // not as slow as gzip. We're producing GB-class sweeps where disk
            // and bandwidth dominate; the writer cost per cell is a rounding
            // error next to the zensim compute itself.
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        Ok(Self {
            schema,
            writer,
            buf: RowBuffer::new(),
        })
    }

    /// Append one cell to the buffer. `features` must have length
    /// [`NUM_FEATURES`]; mismatch returns an error rather than silently
    /// truncating or padding.
    pub fn push_row(
        &mut self,
        image_path: &str,
        codec: &str,
        q: u32,
        knob_tuple_json: &str,
        zensim_score: f32,
        features: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if features.len() != NUM_FEATURES {
            return Err(format!(
                "feature_writer: expected {NUM_FEATURES} features, got {}",
                features.len()
            )
            .into());
        }
        self.buf.image_path.push(image_path.to_string());
        self.buf.codec.push(codec.to_string());
        self.buf.q.push(q);
        self.buf.knob_tuple_json.push(knob_tuple_json.to_string());
        self.buf.zensim_score.push(zensim_score);
        for (i, &v) in features.iter().enumerate() {
            self.buf.feature_columns[i].push(v as f32);
        }
        self.buf.rows += 1;
        if self.buf.rows >= FLUSH_EVERY {
            self.flush_buffer()?;
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.buf.rows == 0 {
            return Ok(());
        }
        let batch = build_record_batch(&self.schema, &self.buf)?;
        self.writer.write(&batch)?;
        self.buf.clear();
        Ok(())
    }

    /// Flush any pending rows and close the writer. Always call this at the
    /// end of a sweep; dropping without `finish()` may leave a partial file.
    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.flush_buffer()?;
        self.writer.close()?;
        Ok(())
    }
}

fn build_schema() -> Schema {
    let mut fields: Vec<Field> = Vec::with_capacity(5 + NUM_FEATURES);
    fields.push(Field::new("image_path", DataType::Utf8, false));
    fields.push(Field::new("codec", DataType::Utf8, false));
    fields.push(Field::new("q", DataType::UInt32, false));
    fields.push(Field::new("knob_tuple_json", DataType::Utf8, false));
    fields.push(Field::new("zensim_score", DataType::Float32, false));
    for i in 0..NUM_FEATURES {
        fields.push(Field::new(format!("feat_{i}"), DataType::Float32, false));
    }
    Schema::new(fields)
}

fn build_record_batch(
    schema: &Arc<Schema>,
    buf: &RowBuffer,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(5 + NUM_FEATURES);
    cols.push(Arc::new(StringArray::from(buf.image_path.clone())));
    cols.push(Arc::new(StringArray::from(buf.codec.clone())));
    cols.push(Arc::new(UInt32Array::from(buf.q.clone())));
    cols.push(Arc::new(StringArray::from(buf.knob_tuple_json.clone())));
    cols.push(Arc::new(Float32Array::from(buf.zensim_score.clone())));
    for col in &buf.feature_columns {
        cols.push(Arc::new(Float32Array::from(col.clone())));
    }
    Ok(RecordBatch::try_new(schema.clone(), cols)?)
}
