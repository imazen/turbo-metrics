//! Comprehensive integration tests for ssim2-gpu.
//!
//! Categories:
//! - **Parity**: GPU vs CPU `ssimulacra2` v0.5.1 (the published reference).
//! - **Equivalence**: cached / batched paths agree with the direct path.
//! - **Error paths**: every public API's `Result` is exercised.
//! - **Dimensions**: pyramid behaviour at odd / small / non-256 sizes.
//! - **Lifecycle**: clear_reference round-trips, repeated batch calls.
//!
//! Backend is selected at compile time:
//! - `cuda` feature → cubecl-cuda (RTX 5070 + CUDA 13.2 reference setup).
//! - `wgpu` feature (no `cuda`) → cubecl-wgpu (Metal on macOS, DX12 on
//!   Windows, Vulkan on Linux when an ICD is available).
//!
//! The cubecl JIT cache is per-process, so all tests in this file share
//! kernel compile cost.

use cubecl::Runtime;
use ssim2_gpu::{Error, Ssim2, Ssim2Batch};

// Backend selection — picks the first available cubecl runtime in
// order of preference. CUDA preferred locally; macOS / Windows / WSL2
// CI fall back to wgpu (Metal / DX12 / Vulkan respectively).
#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "ssim2-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, Xyb};

const CORPUS_DIR: &str = "../dssim-cuda/test_data";

// ───────────────────────── helpers ─────────────────────────

fn load_rgb8(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width(), rgb8.height());
    (rgb8.into_raw(), w, h)
}

fn srgb_u8_to_xyb(bytes: &[u8], w: usize, h: usize) -> Xyb {
    let pixels: Vec<[f32; 3]> = bytes
        .chunks_exact(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();
    Xyb::try_from(
        Rgb::new(
            pixels,
            w,
            h,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap(),
    )
    .unwrap()
}

fn corpus_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIR)
}

/// Build a synthetic reference + distorted pair with deterministic content.
/// Useful for tests that don't depend on a real PNG corpus.
fn synthetic_pair(width: usize, height: usize, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let mut a = vec![0u8; width * height * 3];
    let mut b = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 220 / width.max(1)) & 0xff) as u8;
            let g = ((y * 220 / height.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 200 / (width + height).max(1)) & 0xff) as u8;
            let i = (y * width + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 { mag as i32 } else { -(mag as i32) };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

// ───────────────────────── parity: CPU reference ─────────────────────────

#[test]
fn parity_jpeg_corpus() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    for q in [5u32, 20, 45, 70, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, _, _) = load_rgb8(&path);

        let cpu = ssimulacra2::compute_frame_ssimulacra2(
            srgb_u8_to_xyb(&src_bytes, w as usize, h as usize),
            srgb_u8_to_xyb(&dis_bytes, w as usize, h as usize),
        )
        .expect("cpu");
        let gpu = s.compute(&src_bytes, &dis_bytes).expect("gpu").score;
        let d = (gpu - cpu).abs();
        let rel = if cpu.abs() > 1e-3 {
            d / cpu.abs() * 100.0
        } else {
            0.0
        };
        assert!(
            d < 0.1 || rel < 0.5,
            "q{q}: cpu={cpu:.4}, gpu={gpu:.4}, Δ={d:.5}, rel={rel:.3}%"
        );
    }
}

#[test]
fn identical_image_scores_100() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let result = s.compute(&src_bytes, &src_bytes).expect("identical");
    // Tolerance is backend-dependent. CUDA's FMA / atomic-add ordering
    // converges essentially exactly to 100; the cross-vendor wgpu /
    // Metal path accumulates ulp-level FP rounding through the IIR
    // blur and product chain that, even for bit-identical inputs,
    // leaves residual error stats in the 1e-3 range. After the
    // sigmoid remap that's ~0.4 absolute. We keep the tolerance tight
    // enough to detect real regressions (score < 99 means error stats
    // are no longer near zero) without burning on FP-rounding noise.
    assert!(
        result.score >= 99.0 && result.score <= 100.05,
        "identical-image score {} outside [99.0, 100.05] — error stats not converging to zero",
        result.score
    );
}

#[test]
fn cached_reference_matches_direct() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client_a = Backend::client(&Default::default());
    let client_b = Backend::client(&Default::default());
    let mut s_direct = Ssim2::<Backend>::new(client_a, w, h).expect("direct");
    let mut s_cached = Ssim2::<Backend>::new(client_b, w, h).expect("cached");
    s_cached.set_reference(&src_bytes).expect("set_reference");

    for q in [5u32, 45, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, _, _) = load_rgb8(&path);
        let direct = s_direct.compute(&src_bytes, &dis_bytes).expect("direct").score;
        let cached = s_cached
            .compute_with_reference(&dis_bytes)
            .expect("cached")
            .score;
        let d = (direct - cached).abs();
        assert!(d < 1e-4, "q{q}: direct={direct}, cached={cached}, Δ={d}");
    }
}

// ───────────────────── batched-path parity / lock ─────────────────────

#[test]
fn batch_matches_single_image() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let qs = [5u32, 20, 45, 70, 90];
    let dis: Vec<Vec<u8>> = qs
        .iter()
        .map(|q| load_rgb8(&dir.join(format!("q{q}.jpg"))).0)
        .collect();

    // Single-image path.
    let client = Backend::client(&Default::default());
    let mut single = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let single_results: Vec<f64> = dis
        .iter()
        .map(|d| single.compute(&src_bytes, d).expect("compute").score)
        .collect();

    // Batched path.
    let client = Backend::client(&Default::default());
    let mut batch =
        Ssim2Batch::<Backend>::new(client, w, h, dis.len() as u32).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("set_reference");
    let batch_results = batch.compute_batch(&dis).expect("compute_batch");

    assert_eq!(single_results.len(), batch_results.len());
    for (i, (s, b)) in single_results.iter().zip(batch_results.iter()).enumerate() {
        let d = (*s - b.score).abs();
        assert!(d < 1e-4, "q{}: single={s}, batch={}, Δ={d}", qs[i], b.score);
    }
}

#[test]
fn batch_partial_fill() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, w, h, 8).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("set_reference");

    // Pass 3 images even though batch_size = 8.
    let dis_one = load_rgb8(&dir.join("q45.jpg")).0;
    let dis: Vec<Vec<u8>> = (0..3).map(|_| dis_one.clone()).collect();
    let results = batch.compute_batch(&dis).expect("partial batch");
    assert_eq!(results.len(), 3, "partial-batch should return n_in results");

    // Three identical inputs → three identical scores (within atomic-add noise).
    let s0 = results[0].score;
    for r in &results[1..] {
        let d = (r.score - s0).abs();
        assert!(d < 1e-4, "partial-batch slots disagree: {s0} vs {}", r.score);
    }
}

#[test]
fn batch_repeated_calls_reset_sums() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, w, h, 4).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("set_reference");

    let dis = vec![load_rgb8(&dir.join("q45.jpg")).0; 4];

    // Call compute_batch 3 times — sums must be reset between calls
    // or scores would drift each time.
    let r0 = batch.compute_batch(&dis).expect("call 0");
    let r1 = batch.compute_batch(&dis).expect("call 1");
    let r2 = batch.compute_batch(&dis).expect("call 2");

    for i in 0..4 {
        let s0 = r0[i].score;
        let s1 = r1[i].score;
        let s2 = r2[i].score;
        let d_01 = (s0 - s1).abs();
        let d_02 = (s0 - s2).abs();
        assert!(
            d_01 < 1e-4 && d_02 < 1e-4,
            "slot {i} drifted across repeated calls: {s0} {s1} {s2}"
        );
    }
}

#[test]
fn batch_empty_input_returns_empty() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, w, h, 4).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("set_reference");
    let r = batch.compute_batch(&[]).expect("empty input ok");
    assert_eq!(r.len(), 0);
}

// ───────────────────────── error paths ─────────────────────────

#[test]
fn err_invalid_image_size() {
    let client = Backend::client(&Default::default());
    let r = Ssim2::<Backend>::new(client, 7, 7);
    assert!(matches!(r, Err(Error::InvalidImageSize)));

    let client = Backend::client(&Default::default());
    let r = Ssim2Batch::<Backend>::new(client, 7, 7, 4);
    assert!(matches!(r, Err(Error::InvalidImageSize)));
}

#[test]
fn err_invalid_batch_size_zero() {
    let client = Backend::client(&Default::default());
    let r = Ssim2Batch::<Backend>::new(client, 16, 16, 0);
    assert!(matches!(r, Err(Error::InvalidBatchSize { .. })));
}

#[test]
fn err_dim_mismatch_compute() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 16, 16).expect("Ssim2::new");
    let too_small = vec![0_u8; 16 * 16 * 3 - 1];
    let ok_size = vec![0_u8; 16 * 16 * 3];
    let r = s.compute(&too_small, &ok_size);
    assert!(matches!(r, Err(Error::DimensionMismatch { .. })));
    let r = s.compute(&ok_size, &too_small);
    assert!(matches!(r, Err(Error::DimensionMismatch { .. })));
}

#[test]
fn err_dim_mismatch_set_reference() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 16, 16).expect("Ssim2::new");
    let too_small = vec![0_u8; 16 * 16 * 3 - 1];
    let r = s.set_reference(&too_small);
    assert!(matches!(r, Err(Error::DimensionMismatch { .. })));
    assert!(!s.has_cached_reference());
}

#[test]
fn err_no_cached_reference() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 16, 16).expect("Ssim2::new");
    let dis = vec![0_u8; 16 * 16 * 3];
    let r = s.compute_with_reference(&dis);
    assert!(matches!(r, Err(Error::NoCachedReference)));

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, 16, 16, 4).expect("Ssim2Batch::new");
    let r = batch.compute_batch(&[dis]);
    assert!(matches!(r, Err(Error::NoCachedReference)));
}

#[test]
fn err_dim_mismatch_compute_with_reference() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 16, 16).expect("Ssim2::new");
    let ref_ok = vec![0_u8; 16 * 16 * 3];
    s.set_reference(&ref_ok).expect("set_reference");
    let too_small = vec![0_u8; 16 * 16 * 3 - 1];
    let r = s.compute_with_reference(&too_small);
    assert!(matches!(r, Err(Error::DimensionMismatch { .. })));
}

#[test]
fn err_dim_mismatch_compute_batch() {
    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, 16, 16, 4).expect("Ssim2Batch::new");
    let ref_ok = vec![0_u8; 16 * 16 * 3];
    batch.set_reference(&ref_ok).expect("set_reference");
    let too_small = vec![0_u8; 16 * 16 * 3 - 1];
    let r = batch.compute_batch(&[too_small]);
    assert!(matches!(r, Err(Error::DimensionMismatch { .. })));
}

#[test]
fn err_invalid_batch_size_too_many() {
    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, 16, 16, 2).expect("Ssim2Batch::new");
    let ref_ok = vec![0_u8; 16 * 16 * 3];
    batch.set_reference(&ref_ok).expect("set_reference");
    let dis = vec![ref_ok.clone(); 5]; // 5 > batch_size = 2
    let r = batch.compute_batch(&dis);
    assert!(matches!(r, Err(Error::InvalidBatchSize { .. })));
}

// ───────────────────────── lifecycle ─────────────────────────

#[test]
fn clear_reference_then_set_again() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    s.set_reference(&src_bytes).expect("first set");
    assert!(s.has_cached_reference());

    s.clear_reference();
    assert!(!s.has_cached_reference());
    let dis = load_rgb8(&dir.join("q45.jpg")).0;
    let r = s.compute_with_reference(&dis);
    assert!(matches!(r, Err(Error::NoCachedReference)));

    // Re-arm.
    s.set_reference(&src_bytes).expect("second set");
    assert!(s.has_cached_reference());
    let r = s.compute_with_reference(&dis).expect("compute after re-set");
    // Score should be a sane value, not 0/inf.
    assert!(r.score.is_finite());
    assert!(r.score > -1000.0 && r.score < 200.0, "score = {}", r.score);
}

#[test]
fn batch_clear_reference_round_trip() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));
    let dis = load_rgb8(&dir.join("q45.jpg")).0;

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, w, h, 2).expect("Ssim2Batch::new");
    batch.set_reference(&src_bytes).expect("first");
    let r0 = batch.compute_batch(&[dis.clone()]).expect("first call");

    batch.clear_reference();
    let r = batch.compute_batch(&[dis.clone()]);
    assert!(matches!(r, Err(Error::NoCachedReference)));

    batch.set_reference(&src_bytes).expect("second");
    let r1 = batch.compute_batch(&[dis.clone()]).expect("after re-set");
    let d = (r0[0].score - r1[0].score).abs();
    assert!(
        d < 1e-4,
        "clear+set should give identical score: {} vs {}",
        r0[0].score,
        r1[0].score
    );
}

// ───────────────────────── varied dimensions ─────────────────────────

#[test]
fn dim_odd_non_square() {
    // 200×150 — non-power-of-2, non-square. Pyramid shrinks via div_ceil.
    let (w, h): (u32, u32) = (200, 150);
    let (a, b) = synthetic_pair(w as usize, h as usize, 8);
    let cpu = ssimulacra2::compute_frame_ssimulacra2(
        srgb_u8_to_xyb(&a, w as usize, h as usize),
        srgb_u8_to_xyb(&b, w as usize, h as usize),
    )
    .expect("cpu");

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let gpu = s.compute(&a, &b).expect("compute").score;
    let d = (gpu - cpu).abs();
    let rel = if cpu.abs() > 1e-3 {
        d / cpu.abs() * 100.0
    } else {
        0.0
    };
    assert!(
        d < 0.1 || rel < 0.5,
        "200×150: cpu={cpu:.4}, gpu={gpu:.4}, Δ={d:.5}, rel={rel:.3}%"
    );
}

#[test]
fn dim_minimum_supported() {
    // Smallest supported by the published `ssimulacra2` (>=8×8). At
    // 16×16 we get only ~2 pyramid levels before everything shrinks
    // below 8×8 — tests the early-break path.
    let (w, h): (u32, u32) = (16, 16);
    let (a, b) = synthetic_pair(w as usize, h as usize, 4);
    let cpu = ssimulacra2::compute_frame_ssimulacra2(
        srgb_u8_to_xyb(&a, w as usize, h as usize),
        srgb_u8_to_xyb(&b, w as usize, h as usize),
    )
    .expect("cpu");

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let gpu = s.compute(&a, &b).expect("compute").score;
    let d = (gpu - cpu).abs();
    let rel = if cpu.abs() > 1e-3 {
        d / cpu.abs() * 100.0
    } else {
        0.0
    };
    assert!(
        d < 0.5 || rel < 5.0,
        "16×16: cpu={cpu:.4}, gpu={gpu:.4}, Δ={d:.5}, rel={rel:.3}%"
    );
    // Only 2 active scales (16, 8) — confirm.
    assert_eq!(s.n_scales(), 2);
}

#[test]
fn dim_larger_512x384() {
    // Validate parity holds at a more realistic resolution.
    let (w, h): (u32, u32) = (512, 384);
    let (a, b) = synthetic_pair(w as usize, h as usize, 6);
    let cpu = ssimulacra2::compute_frame_ssimulacra2(
        srgb_u8_to_xyb(&a, w as usize, h as usize),
        srgb_u8_to_xyb(&b, w as usize, h as usize),
    )
    .expect("cpu");

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let gpu = s.compute(&a, &b).expect("compute").score;
    let d = (gpu - cpu).abs();
    let rel = if cpu.abs() > 1e-3 {
        d / cpu.abs() * 100.0
    } else {
        0.0
    };
    assert!(
        d < 0.1 || rel < 0.5,
        "512×384: cpu={cpu:.4}, gpu={gpu:.4}, Δ={d:.5}, rel={rel:.3}%"
    );
}

#[test]
fn dim_odd_with_batch() {
    // Odd dimensions through Ssim2Batch — the batched downscale's
    // div_ceil pyramid-shrinking and per-image-strided buffer layouts
    // must both work for non-power-of-2 shapes.
    let (w, h): (u32, u32) = (123, 45); // intentionally weird
    let (a, b) = synthetic_pair(w as usize, h as usize, 4);

    let client = Backend::client(&Default::default());
    let mut single = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
    let single_score = single.compute(&a, &b).expect("single").score;

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, w, h, 3).expect("Ssim2Batch::new");
    batch.set_reference(&a).expect("set_reference");
    let dis = vec![b.clone(); 3];
    let results = batch.compute_batch(&dis).expect("compute_batch");
    // Cross-check vs the single-image path.
    for r in &results {
        let d = (r.score - single_score).abs();
        assert!(
            d < 1e-4,
            "odd-dim batch vs single mismatch: single={single_score}, batch={}, Δ={d}",
            r.score
        );
    }
    assert_eq!(results.len(), 3);
    let s0 = results[0].score;
    for r in &results[1..] {
        assert!(
            (r.score - s0).abs() < 1e-4,
            "odd-dim batch slots disagree: {s0} vs {}",
            r.score
        );
    }
}
