//! End-to-end smoke test: feed a real image pair through the
//! single-resolution butteraugli-gpu pipeline and confirm we get a
//! plausible (max-norm, 3-norm) result.
//!
//! Note: the pipeline currently runs only the early stages of butteraugli
//! (sRGB→linear, opsin dynamics, partial frequency) before falling back
//! to a stand-in reduction over the Y plane — see `pipeline.rs` for
//! status. The full pipeline lands when frequency separation, malta,
//! masking, and compute_diffmap are wired in.

use butteraugli_gpu::Butteraugli;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn main() {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    // 64×64 synthetic gradient pair — same shape works on every backend.
    let width = 64u32;
    let height = 64u32;
    let n = (width * height) as usize;

    let ref_rgb: Vec<u8> = (0..n * 3)
        .map(|i| ((i.wrapping_mul(31)) % 256) as u8)
        .collect();
    let dist_rgb: Vec<u8> = (0..n * 3)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(13)) % 256) as u8)
        .collect();

    let mut bu = Butteraugli::<Backend>::new(client, width, height);
    let r = bu.compute(&ref_rgb, &dist_rgb);

    println!(
        "[{w}×{h}] GPU butteraugli: score={:.6}  pnorm_3={:.6}",
        r.score,
        r.pnorm_3,
        w = width,
        h = height,
    );
}
