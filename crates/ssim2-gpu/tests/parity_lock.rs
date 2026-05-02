//! CI-friendly parity lock: scoring the JPEG corpus against
//! `source.png` should produce values within 0.5% relative (or 0.1
//! absolute, whichever is looser) of the published `ssimulacra2` CPU
//! crate. Catches regressions in any kernel that lands later.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::Ssim2;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, Xyb};

const CORPUS_DIR: &str = "../dssim-cuda/test_data";

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

#[test]
fn parity_jpeg_corpus() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join(CORPUS_DIR);
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, w, h).expect("Ssim2::new");

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
fn cached_reference_matches_direct() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join(CORPUS_DIR);
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client_a = CudaRuntime::client(&Default::default());
    let client_b = CudaRuntime::client(&Default::default());
    let mut s_direct = Ssim2::<CudaRuntime>::new(client_a, w, h).expect("direct");
    let mut s_cached = Ssim2::<CudaRuntime>::new(client_b, w, h).expect("cached");
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

#[test]
fn identical_image_scores_100() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join(CORPUS_DIR);
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, w, h).expect("Ssim2::new");
    let result = s.compute(&src_bytes, &src_bytes).expect("identical");
    // Float round-off in the 30+ kernel pipeline keeps us within 0.05
    // of the analytical 100. The CPU crate hits exact 100; we're
    // within 1e-2 typically.
    assert!(
        (result.score - 100.0).abs() < 0.05,
        "identical-image score {} too far from 100",
        result.score
    );
}
