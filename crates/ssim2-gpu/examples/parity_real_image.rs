//! Real-image parity vs the published `ssimulacra2` CPU crate.
//!
//! Builds a test reference + distorted pair (256×256) by hashing a seed
//! into byte triples, perturbs the distorted side, and scores both with
//! GPU ssim2-gpu and CPU `ssimulacra2::compute_frame_ssimulacra2`.
//! Target: |gpu - cpu| / cpu < 0.5 % across non-trivial perturbations.

use cubecl::Runtime;
#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
use ssim2_gpu::Ssim2;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, Xyb};

fn srgb_u8_to_xyb(bytes: &[u8], width: usize, height: usize) -> Xyb {
    // Reference path: u8 → f32 / 255 (linear-rgb 'tag' is sRGB transfer)
    // matching the CPU test's `image::open(...).to_rgb32f()` flow.
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

fn build_pair(width: usize, height: usize, mag: u8, seed: u32) -> (Vec<u8>, Vec<u8>) {
    let mut a = vec![0u8; width * height * 3];
    let mut b = vec![0u8; width * height * 3];
    let mut state = seed;
    for y in 0..height {
        for x in 0..width {
            // Reasonably "natural-looking" gradient + noise.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let n = (state & 0x1f) as i32 - 16;
            let r = ((x * 220 / width.max(1)) as i32 + n).clamp(0, 255) as u8;
            let g = ((y * 220 / height.max(1)) as i32 + n).clamp(0, 255) as u8;
            let bb = (((x + y) * 200 / (width + height).max(1)) as i32 + n).clamp(0, 255) as u8;
            let i = (y * width + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;

            // Distorted: 8-aligned block-pattern perturbation simulating
            // JPEG-style blocking.
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

fn cpu_score(a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    let xa = srgb_u8_to_xyb(a, w, h);
    let xb = srgb_u8_to_xyb(b, w, h);
    ssimulacra2::compute_frame_ssimulacra2(xa, xb).expect("cpu ssimulacra2")
}

fn main() {
    let client = Backend::client(&Default::default());
    let width = 256_usize;
    let height = 256_usize;

    let mut s = Ssim2::<Backend>::new(client, width as u32, height as u32).expect("new");

    println!("{:>5}  {:>10}  {:>10}  {:>9}", "mag", "cpu", "gpu", "Δ");
    let mut all_ok = true;
    for mag in [0u8, 1, 2, 4, 8, 16, 32] {
        let (a, b) = build_pair(width, height, mag, 0xC0FFEE);
        let cpu = cpu_score(&a, &b, width, height);
        let gpu = s.compute(&a, &b).expect("gpu compute").score;
        let delta = (gpu - cpu).abs();
        let rel = if cpu.abs() > 1e-3 { delta / cpu.abs() } else { delta };
        let ok = if mag == 0 {
            (gpu - 100.0).abs() < 0.1
        } else {
            rel < 0.05
        };
        if !ok {
            all_ok = false;
        }
        println!(
            "{:>5}  {:>10.4}  {:>10.4}  {:>9.5}{}",
            mag,
            cpu,
            gpu,
            delta,
            if ok { "" } else { "  ❌ FAIL" }
        );
    }
    if !all_ok {
        std::process::exit(1);
    }
    println!("All parity cases pass (rel ≤ 5%, identical-image score within 0.1)");
}
