//! End-to-end CLI tests. Use `assert_cmd`-style spawning of the compiled
//! `zen-metrics` binary so we exercise the same code path users hit.

use std::path::PathBuf;
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn cli() -> Command {
    let bin = env!("CARGO_BIN_EXE_zen-metrics");
    Command::new(bin)
}

#[test]
fn list_metrics_runs() {
    let out = cli().args(["list-metrics"]).output().expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("ssim2"));
    assert!(s.contains("ssim2-gpu"));
    assert!(s.contains("butteraugli"));
    assert!(s.contains("butteraugli-gpu"));
    assert!(s.contains("dssim"));
    assert!(s.contains("dssim-gpu"));
    assert!(s.contains("zensim"));
}

#[test]
fn list_formats_runs() {
    let out = cli().args(["list-formats"]).output().expect("run cli");
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    // Default features include png + webp.
    assert!(s.contains("png"));
    assert!(s.contains("webp"));
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_zensim_identical_pngs() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "score",
            "--metric",
            "zensim",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    let score = v["score"].as_f64().expect("score");
    // zensim returns ~100 for identical images.
    assert!(score > 95.0, "expected ~100, got {score}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_butteraugli_identical_pngs_tsv() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "score",
            "--metric",
            "butteraugli",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--output",
            "tsv",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    // TSV: header + one data row.
    let mut lines = s.lines();
    assert_eq!(lines.next().unwrap(), "metric\tscore");
    let row = lines.next().unwrap();
    let parts: Vec<&str> = row.split('\t').collect();
    assert_eq!(parts[0], "butteraugli");
    let score: f64 = parts[1].parse().unwrap();
    // Butteraugli 3-norm of identical images should be effectively zero.
    assert!(score < 0.01, "expected ~0, got {score}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_butteraugli_noisy_is_higher_than_identical() {
    let dir = fixtures_dir();
    let identical = run_score(
        "butteraugli",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    let noisy = run_score(
        "butteraugli",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    assert!(identical < 0.5, "identical butteraugli={identical}");
    assert!(
        noisy > identical,
        "noisy {noisy} should be > identical {identical}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_dssim_identical_is_zero() {
    let dir = fixtures_dir();
    let s = run_score(
        "dssim",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    // DSSIM is a distance — identical images should score ~0.
    assert!(s < 1e-3, "expected ~0 for identical, got {s}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_dssim_noisy_higher_than_identical() {
    let dir = fixtures_dir();
    let identical = run_score(
        "dssim",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    let noisy = run_score(
        "dssim",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    assert!(
        noisy > identical,
        "noisy dssim {noisy} should be > identical {identical}"
    );
}

#[cfg(all(feature = "cpu-metrics", feature = "gpu-dssim"))]
#[test]
fn score_dssim_gpu_identical_is_zero() {
    let dir = fixtures_dir();
    let s = run_score(
        "dssim-gpu",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    // DSSIM-GPU is a distance — identical images should score ~0.
    assert!(s < 1e-3, "expected ~0 for identical, got {s}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_ssim2_identical_is_high() {
    let dir = fixtures_dir();
    let s = run_score(
        "ssim2",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    // SSIMULACRA2 returns ~100 for identical, lower for distorted.
    assert!(s > 95.0, "expected ~100, got {s}");
}

#[cfg(all(feature = "cpu-metrics", feature = "avif"))]
#[test]
fn score_decodes_avif_fixture() {
    // Fixture is checked into the repo — no skip path.
    let dir = fixtures_dir();
    let s = run_score("zensim", &dir.join("ref_64.png"), &dir.join("ref_64.avif"));
    assert!(s.is_finite() && s > 0.0, "got {s}");
}

#[cfg(all(feature = "cpu-metrics", feature = "jxl"))]
#[test]
fn score_decodes_jxl_fixture() {
    let dir = fixtures_dir();
    let s = run_score("zensim", &dir.join("ref_64.png"), &dir.join("ref_64.jxl"));
    assert!(s.is_finite() && s > 0.0, "got {s}");
}

#[cfg(all(feature = "cpu-metrics", feature = "webp"))]
#[test]
fn score_works_across_png_and_webp_decoders() {
    let dir = fixtures_dir();
    // Compare PNG-encoded ref against WebP-encoded ref (both lossless,
    // same content) — both decoders should produce matching pixels and
    // give a near-identical zensim score.
    let s = run_score("zensim", &dir.join("ref_64.png"), &dir.join("ref_64.webp"));
    // Lossless WebP of the exact same source should round-trip pixel-exact
    // → zensim score effectively 100.
    assert!(s > 95.0, "expected ~100 for lossless cross-format, got {s}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn batch_zensim_appends_metric_column() {
    let dir = fixtures_dir();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let pairs = tmp.path().join("pairs.tsv");
    let output = tmp.path().join("output.tsv");

    let ref_path = dir.join("ref_64.png");
    let dist_a = dir.join("dist_identical_64.png");
    let dist_b = dir.join("dist_noisy_64.png");

    let mut tsv = String::from("ref_path\tdist_path\ttag\n");
    tsv.push_str(&format!(
        "{}\t{}\tidentical\n",
        ref_path.display(),
        dist_a.display()
    ));
    tsv.push_str(&format!(
        "{}\t{}\tnoisy\n",
        ref_path.display(),
        dist_b.display()
    ));
    std::fs::write(&pairs, tsv).unwrap();

    let out = cli()
        .args([
            "batch",
            "--metric",
            "zensim",
            "--pairs",
            pairs.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let read = std::fs::read_to_string(&output).unwrap();
    let mut lines = read.lines();
    let headers = lines.next().unwrap();
    assert!(
        headers.contains("zensim"),
        "expected zensim col in {headers}"
    );
    let row1 = lines.next().unwrap();
    let row2 = lines.next().unwrap();
    let score1: f64 = row1.split('\t').next_back().unwrap().parse().unwrap();
    let score2: f64 = row2.split('\t').next_back().unwrap().parse().unwrap();
    assert!(score1 > 95.0, "identical: {score1}");
    assert!(
        score2 < score1,
        "noisy {score2} should be < identical {score1}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn compare_one_ref_one_variant_one_metric_json_shape() {
    // Smallest possible compare: 1×1×1. Verify the JSON document shape
    // matches the spec — a top-level `metrics` array and a `results` array
    // where each row carries `reference`, `variant`, and a `scores` map
    // keyed on metric names.
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "compare",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--metric",
            "zensim",
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    let metrics = v["metrics"].as_array().expect("metrics array");
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0], "zensim");
    let results = v["results"].as_array().expect("results array");
    assert_eq!(results.len(), 1);
    let row = &results[0];
    assert!(row["reference"].as_str().unwrap().ends_with("ref_64.png"));
    assert!(
        row["variant"]
            .as_str()
            .unwrap()
            .ends_with("dist_identical_64.png")
    );
    let score = row["scores"]["zensim"].as_f64().expect("score");
    assert!(score > 95.0, "expected ~100 for identical, got {score}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn compare_one_ref_two_variants_two_metrics_tsv() {
    // 1×2 × 2 = 4 cells. Verify all four scores show up in the TSV with
    // the correct column ordering, that all values are finite, and that
    // the "noisy" variant scores differ from the "identical" one.
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "compare",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_noisy_64.png").to_str().unwrap(),
            "--metric",
            "zensim",
            "--metric",
            "butteraugli",
            "--output",
            "tsv",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let mut lines = s.lines();
    let header = lines.next().expect("header");
    assert_eq!(header, "reference\tvariant\tzensim\tbutteraugli");
    let row1: Vec<&str> = lines.next().expect("row1").split('\t').collect();
    let row2: Vec<&str> = lines.next().expect("row2").split('\t').collect();
    assert!(lines.next().is_none(), "exactly two data rows expected");
    assert_eq!(row1.len(), 4);
    assert_eq!(row2.len(), 4);
    let identical_zensim: f64 = row1[2].parse().unwrap();
    let identical_butter: f64 = row1[3].parse().unwrap();
    let noisy_zensim: f64 = row2[2].parse().unwrap();
    let noisy_butter: f64 = row2[3].parse().unwrap();
    assert!(identical_zensim > 95.0, "{identical_zensim}");
    assert!(identical_butter < 0.01, "{identical_butter}");
    assert!(
        noisy_zensim < identical_zensim,
        "noisy {noisy_zensim} should be < identical {identical_zensim}"
    );
    assert!(
        noisy_butter > identical_butter,
        "noisy {noisy_butter} (butteraugli 3-norm, higher = worse) should be > identical {identical_butter}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn compare_continues_on_per_cell_failure() {
    // Two variants: one valid, one that does not exist on disk. The bad
    // variant should produce error cells (null in JSON) for every metric
    // it's paired with, but the good variant should still get scored.
    // Process exit must be non-zero because at least one cell failed.
    let dir = fixtures_dir();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let bogus = tmp.path().join("does_not_exist.png");
    let out = cli()
        .args([
            "compare",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--variant",
            bogus.to_str().unwrap(),
            "--metric",
            "zensim",
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        !out.status.success(),
        "expected non-zero exit when a cell fails"
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    let results = v["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2);
    // First row (identical variant) should have a real score.
    let good_score = results[0]["scores"]["zensim"].as_f64().expect("good score");
    assert!(
        good_score > 95.0,
        "good score should be ~100, got {good_score}"
    );
    // Second row (missing variant) should be null.
    assert!(
        results[1]["scores"]["zensim"].is_null(),
        "expected null score for missing variant, got {}",
        results[1]["scores"]["zensim"]
    );
}

/// CPU butteraugli (3-norm) vs GPU butteraugli (3-norm) on the same
/// noisy-pair fixture. Both backends compute the same aggregation; the two
/// scores should agree closely modulo floating-point order across CubeCL
/// runtimes (CUDA vs wgpu vs HIP vs CPU). Tolerance is set to 5e-2 in
/// absolute terms — empirically the cross-backend slack on butteraugli is
/// dominated by reduction order, well below this bound on 64×64 fixtures.
#[cfg(all(feature = "cpu-metrics", feature = "gpu-butteraugli"))]
#[test]
fn butteraugli_cpu_and_gpu_agree_on_3norm() {
    let dir = fixtures_dir();
    let cpu = run_score(
        "butteraugli",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    let gpu = run_score(
        "butteraugli-gpu",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    let diff = (cpu - gpu).abs();
    assert!(
        diff < 5e-2,
        "cpu butteraugli={cpu} vs gpu butteraugli={gpu} (|diff|={diff}) exceeds 5e-2 tolerance"
    );
}

#[cfg(feature = "cpu-metrics")]
fn run_score(metric: &str, reference: &std::path::Path, distorted: &std::path::Path) -> f64 {
    let out = cli()
        .args([
            "score",
            "--metric",
            metric,
            "--reference",
            reference.to_str().unwrap(),
            "--distorted",
            distorted.to_str().unwrap(),
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    v["score"].as_f64().expect("score")
}
