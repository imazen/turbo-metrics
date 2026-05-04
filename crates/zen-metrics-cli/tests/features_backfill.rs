//! End-to-end test for the `features-backfill` subcommand.
//!
//! Covers the local single-TSV mode (the R2 mode shells out to `s5cmd` and
//! is exercised manually against a sandbox bucket — not included in CI).
//!
//! Verifies:
//! 1. A backfill on a fresh TSV produces a parquet with the expected row
//!    count and 305-column schema.
//! 2. Re-running the same command is a no-op (parquet already present).
//! 3. Deleting the parquet and re-running regenerates it.

#![cfg(feature = "sweep")]

use std::path::PathBuf;
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn cli() -> Command {
    let bin = env!("CARGO_BIN_EXE_zen-metrics");
    Command::new(bin)
}

/// Generate a tiny TSV mimicking what `zen-metrics sweep` would have
/// produced in the v04 / v05c sweeps. Only the columns the backfill
/// reads (`image_path`, `codec`, `q`, `knob_tuple_json`) need accurate
/// values — the others are passed through opaquely.
fn write_tiny_tsv(tsv_path: &std::path::Path, image_path: &str, codec: &str) {
    let mut body = String::new();
    body.push_str("image_path\tcodec\tq\tknob_tuple_json\tencoded_bytes\tencode_ms\tdecode_ms\tscore_zensim\n");
    // Two cells: q=50 and q=90, both with empty knob tuple.
    for q in [50, 90] {
        body.push_str(&format!(
            "{image_path}\t{codec}\t{q}\t{{}}\t1234\t1.0\t1.0\t90.0\n"
        ));
    }
    std::fs::write(tsv_path, body).unwrap();
}

#[test]
fn local_backfill_produces_parquet() {
    let staged = tempfile::tempdir().expect("tmp");
    let corpus_root = staged.path().join("corpus");
    std::fs::create_dir_all(&corpus_root).unwrap();
    let src_image = corpus_root.join("ref.png");
    std::fs::copy(fixtures_dir().join("ref_64.png"), &src_image).unwrap();

    let tsv_path = staged.path().join("chunk-zenwebp-000.tsv");
    let pq_path = staged.path().join("chunk-zenwebp-000.parquet");
    write_tiny_tsv(&tsv_path, src_image.to_str().unwrap(), "zenwebp");

    let result = cli()
        .args([
            "features-backfill",
            "--input-tsv",
            tsv_path.to_str().unwrap(),
            "--output-parquet",
            pq_path.to_str().unwrap(),
            "--corpus-root",
            corpus_root.to_str().unwrap(),
            "--codec",
            "zenwebp",
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "features-backfill failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let pq_meta = std::fs::metadata(&pq_path).expect("parquet exists");
    assert!(
        pq_meta.len() > 1024,
        "feature parquet is suspiciously small: {} bytes",
        pq_meta.len()
    );

    use parquet::file::reader::FileReader;
    let file = std::fs::File::open(&pq_path).expect("open pq");
    let reader = parquet::file::reader::SerializedFileReader::new(file).expect("parquet reader");
    let meta = reader.metadata();
    let schema_descr = meta.file_metadata().schema_descr();
    // 5 ID columns + 300 features = 305
    assert_eq!(
        schema_descr.num_columns(),
        305,
        "expected 305 columns, got {}",
        schema_descr.num_columns()
    );
    let num_rows = meta.file_metadata().num_rows();
    assert_eq!(num_rows, 2, "expected 2 rows in parquet, got {num_rows}");

    // Schema sanity check.
    let names: Vec<String> = (0..schema_descr.num_columns())
        .map(|i| schema_descr.column(i).name().to_string())
        .collect();
    assert_eq!(names[0], "image_path");
    assert_eq!(names[1], "codec");
    assert_eq!(names[2], "q");
    assert_eq!(names[3], "knob_tuple_json");
    assert_eq!(names[4], "zensim_score");
    assert_eq!(names[5], "feat_0");
    assert_eq!(names[304], "feat_299");
}

#[test]
fn local_backfill_is_idempotent() {
    let staged = tempfile::tempdir().expect("tmp");
    let corpus_root = staged.path().join("corpus");
    std::fs::create_dir_all(&corpus_root).unwrap();
    let src_image = corpus_root.join("ref.png");
    std::fs::copy(fixtures_dir().join("ref_64.png"), &src_image).unwrap();

    let tsv_path = staged.path().join("chunk-zenwebp-001.tsv");
    let pq_path = staged.path().join("chunk-zenwebp-001.parquet");
    write_tiny_tsv(&tsv_path, src_image.to_str().unwrap(), "zenwebp");

    // First run — produces parquet.
    let r1 = cli()
        .args([
            "features-backfill",
            "--input-tsv",
            tsv_path.to_str().unwrap(),
            "--output-parquet",
            pq_path.to_str().unwrap(),
            "--corpus-root",
            corpus_root.to_str().unwrap(),
            "--codec",
            "zenwebp",
        ])
        .output()
        .expect("run cli");
    assert!(r1.status.success(), "first run failed");
    let mtime1 = std::fs::metadata(&pq_path).unwrap().modified().unwrap();

    // Sleep a moment so mtime resolution can register a re-write if it
    // were to happen erroneously. fs filetime granularity on most
    // filesystems is 1ms, but be defensive.
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Second run — must be a no-op.
    let r2 = cli()
        .args([
            "features-backfill",
            "--input-tsv",
            tsv_path.to_str().unwrap(),
            "--output-parquet",
            pq_path.to_str().unwrap(),
            "--corpus-root",
            corpus_root.to_str().unwrap(),
            "--codec",
            "zenwebp",
        ])
        .output()
        .expect("run cli");
    assert!(r2.status.success(), "second run failed");
    let mtime2 = std::fs::metadata(&pq_path).unwrap().modified().unwrap();
    assert_eq!(
        mtime1, mtime2,
        "second run rewrote the parquet (idempotence broken)"
    );

    let stderr = String::from_utf8_lossy(&r2.stderr);
    assert!(
        stderr.contains("already exists; skipping"),
        "expected skip notice on stderr, got: {stderr}"
    );

    // Delete and re-run — should regenerate.
    std::fs::remove_file(&pq_path).unwrap();
    let r3 = cli()
        .args([
            "features-backfill",
            "--input-tsv",
            tsv_path.to_str().unwrap(),
            "--output-parquet",
            pq_path.to_str().unwrap(),
            "--corpus-root",
            corpus_root.to_str().unwrap(),
            "--codec",
            "zenwebp",
        ])
        .output()
        .expect("run cli");
    assert!(r3.status.success(), "third run (after delete) failed");
    assert!(pq_path.exists(), "parquet should be re-created");
}

#[test]
fn local_backfill_resolves_unflattened_paths() {
    // Stage looks like the production v04 / v05c worker layout: the TSV
    // has paths like `/workspace/sweep/stage-foo/dir__sub__file.png`,
    // and the corpus has the original `dir/sub/file.png` layout.
    let staged = tempfile::tempdir().expect("tmp");
    let corpus_root = staged.path().join("corpus");
    std::fs::create_dir_all(corpus_root.join("dir/sub")).unwrap();
    std::fs::copy(
        fixtures_dir().join("ref_64.png"),
        corpus_root.join("dir/sub/file.png"),
    )
    .unwrap();

    let tsv_path = staged.path().join("chunk.tsv");
    let pq_path = staged.path().join("chunk.parquet");
    write_tiny_tsv(
        &tsv_path,
        "/workspace/sweep/stage-zenwebp-000/dir__sub__file.png",
        "zenwebp",
    );

    let result = cli()
        .args([
            "features-backfill",
            "--input-tsv",
            tsv_path.to_str().unwrap(),
            "--output-parquet",
            pq_path.to_str().unwrap(),
            "--corpus-root",
            corpus_root.to_str().unwrap(),
            "--codec",
            "zenwebp",
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "backfill failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    use parquet::file::reader::FileReader;
    let file = std::fs::File::open(&pq_path).expect("open pq");
    let reader = parquet::file::reader::SerializedFileReader::new(file).expect("parquet reader");
    let num_rows = reader.metadata().file_metadata().num_rows();
    assert_eq!(
        num_rows, 2,
        "expected 2 rows after unflatten resolution, got {num_rows}"
    );
}

#[test]
fn r2_mode_requires_codec() {
    // R2 mode without --codec must error out before any S3 traffic.
    let staged = tempfile::tempdir().expect("tmp");
    let corpus_root = staged.path();

    let result = cli()
        .args([
            "features-backfill",
            "--run-id",
            "no-such-run",
            "--corpus-root",
            corpus_root.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        !result.status.success(),
        "expected failure on R2-mode-without-codec, got success"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("codec"),
        "stderr did not mention codec: {stderr}"
    );
}

#[test]
fn requires_one_of_input_or_run_id() {
    let staged = tempfile::tempdir().expect("tmp");
    let result = cli()
        .args([
            "features-backfill",
            "--corpus-root",
            staged.path().to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        !result.status.success(),
        "expected failure when neither --input-tsv nor --run-id is set"
    );
}
