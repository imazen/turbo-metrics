//! Real-image parity vs CPU `ssimulacra2` using the JPEG quality
//! corpus from `dssim-cuda/test_data`.
//!
//! Loads `source.png` as the reference and each of `q{1,5,20,45,70,90}.jpg`
//! as a distorted variant (decoded back to RGB u8). Both pipelines see
//! the same byte buffers, so any score difference is purely from the
//! GPU pipeline implementation.

use cubecl::Runtime;
#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
use ssim2_gpu::Ssim2;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, Xyb};

const CORPUS_DIR: &str = "../dssim-cuda/test_data";

fn load_rgb8(path: &str) -> (Vec<u8>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"));
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width(), rgb8.height());
    (rgb8.into_raw(), w, h)
}

fn srgb_u8_to_xyb(bytes: &[u8], width: usize, height: usize) -> Xyb {
    let pixels: Vec<[f32; 3]> = bytes
        .chunks_exact(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();
    Xyb::try_from(
        Rgb::new(
            pixels,
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap(),
    )
    .unwrap()
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join(CORPUS_DIR);

    let (src_bytes, w, h) = load_rgb8(dir.join("source.png").to_str().unwrap());
    println!("source: {}×{}", w, h);

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    println!("{:>4}  {:>10}  {:>10}  {:>9}  {:>7}", "q", "cpu", "gpu", "Δ", "rel");
    let mut all_ok = true;
    for q in [1u32, 5, 20, 45, 70, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, dw, dh) = load_rgb8(path.to_str().unwrap());
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert_eq!(dis_bytes.len(), src_bytes.len());

        let cpu = ssimulacra2::compute_frame_ssimulacra2(
            srgb_u8_to_xyb(&src_bytes, w as usize, h as usize),
            srgb_u8_to_xyb(&dis_bytes, w as usize, h as usize),
        )
        .expect("cpu");
        let gpu = s.compute(&src_bytes, &dis_bytes).expect("gpu").score;
        let d = (gpu - cpu).abs();
        let rel = if cpu.abs() > 1e-3 { d / cpu.abs() * 100.0 } else { 0.0 };
        // Absolute tolerance dominates near zero (q=1 territory), where
        // 0.029 absolute drift is the f32-reduction-vs-f64 noise floor.
        let ok = d < 0.1 || rel < 0.5;
        if !ok {
            all_ok = false;
        }
        println!(
            "{:>4}  {:>10.4}  {:>10.4}  {:>9.5}  {:>6.3}%{}",
            q,
            cpu,
            gpu,
            d,
            rel,
            if ok { "" } else { "  ❌ FAIL" }
        );
    }
    if !all_ok {
        std::process::exit(1);
    }
    println!("All JPEG-corpus parity cases pass (rel ≤ 0.5%)");
}
