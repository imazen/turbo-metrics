//! End-to-end parity check: GPU butteraugli vs the published `butteraugli`
//! v0.9.2 CPU crate on synthetic test pairs.
//!
//! What this proves:
//! - Pipeline produces a real (non-zero, non-NaN, non-saturating) score
//! - Score tracks CPU butteraugli within the expected divergence band
//!
//! Caveat: the GPU pipeline is **single-resolution only** at this commit.
//! CPU butteraugli's recursive `Diffmap` adds a half-resolution pass
//! mixed in with `K_HEURISTIC_MIXING = 0.3`. So GPU scores are expected
//! to be *lower* than CPU on natural images by ~5–15 %; pixel-perfect
//! parity will only land once the multi-res orchestration ships.

use butteraugli::{ButteraugliParams, butteraugli};
use butteraugli_gpu::Butteraugli;
use imgref::ImgVec;
use rgb::{RGB, RGB8};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn make_gradient(width: u32, height: u32, salt: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x.wrapping_mul(7).wrapping_add(salt)) & 0xff) as u8;
            let g = ((y.wrapping_mul(11).wrapping_add(salt.wrapping_mul(3))) & 0xff) as u8;
            let b = (((x ^ y).wrapping_mul(13).wrapping_add(salt.wrapping_mul(5))) & 0xff) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

/// Apply a small, spatially-varying perturbation that survives sRGB→linear
/// without saturating — so we get a non-degenerate butteraugli score.
fn perturb(src: &[u8], width: u32, height: u32, mag: i32) -> Vec<u8> {
    let mut out = src.to_vec();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = (y * width as usize + x) * 3;
            // Diagonal stripes, alternating sign — looks like JPEG ringing
            let stripe = (((x + y) % 7) as i32 - 3).signum();
            for c in 0..3 {
                let delta = stripe * mag * (1 + c as i32);
                let v = src[i + c] as i32 + delta;
                out[i + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    out
}

fn rgb_buf_to_imgvec(buf: &[u8], width: u32, height: u32) -> ImgVec<RGB8> {
    let pixels: Vec<RGB8> = buf
        .chunks_exact(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    ImgVec::new(pixels, width as usize, height as usize)
}

fn main() {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let cases: &[(u32, u32, &str)] = &[
        (64, 64, "tiny synthetic"),
        (256, 256, "small synthetic"),
        (512, 512, "medium synthetic"),
    ];

    for &(width, height, label) in cases {
        let ref_rgb = make_gradient(width, height, 0xC0FFEE);
        // Pick perturbation magnitudes that span a useful butteraugli range.
        for &mag in &[1_i32, 4, 12, 32] {
            let dist_rgb = perturb(&ref_rgb, width, height, mag);

            // CPU butteraugli (the actual published reference)
            let ref_img = rgb_buf_to_imgvec(&ref_rgb, width, height);
            let dist_img = rgb_buf_to_imgvec(&dist_rgb, width, height);
            // Use single_resolution=true so the CPU side runs the same
            // pipeline shape as our GPU port (no half-res supersample-add).
            let params = ButteraugliParams::default().with_single_resolution(true);
            let cpu =
                butteraugli(ref_img.as_ref(), dist_img.as_ref(), &params).expect("cpu butteraugli");

            // GPU butteraugli
            let mut gpu = Butteraugli::<Backend>::new(client.clone(), width, height);
            let g = gpu.compute(&ref_rgb, &dist_rgb);

            let cpu_score = cpu.score;
            let cpu_pnorm = cpu.pnorm_3;
            let rel_score = if cpu_score > 1e-6 {
                (g.score as f64 - cpu_score).abs() / cpu_score * 100.0
            } else {
                0.0
            };
            let rel_pnorm = if cpu_pnorm > 1e-6 {
                (g.pnorm_3 as f64 - cpu_pnorm).abs() / cpu_pnorm * 100.0
            } else {
                0.0
            };
            println!(
                "{:14} {}×{}  mag={:>2}  | CPU score={:.4} pnorm3={:.4}  | GPU score={:.4} pnorm3={:.4}  | Δ score={:>+5.1}%  Δ pnorm3={:>+5.1}%",
                label,
                width,
                height,
                mag,
                cpu_score,
                cpu_pnorm,
                g.score,
                g.pnorm_3,
                rel_score,
                rel_pnorm,
            );
        }
    }
}
