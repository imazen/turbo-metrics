//! End-to-end validation: feed the fused reduction kernel a synthetic
//! diffmap and confirm `(score, pnorm_3)` match the CPU butteraugli
//! crate's own `pnorm_slice`-equivalent computation.
//!
//! Run with:
//!   `CUDA_PATH=/usr/local/cuda-12 cargo run --example reduction_parity --release`
//!
//! The CUDA backend is the default. Other backends can be selected by
//! editing `Backend` below. The point of this example is just toolchain
//! validation — the real cross-backend test is in `tests/`.

use butteraugli_gpu::reduce_diffmap_to_score;

// Backend selection: CUDA → wgpu → CPU. CPU is the most-portable validator
// (no GPU drivers required); CUDA needs a matching toolkit (12.x for
// pre-Blackwell, 13.x for RTX 50-series), and wgpu's Vulkan path needs an
// ICD that's reachable from the host (WSL2 typically isn't, without the
// NVIDIA WSL Vulkan driver).
//
// Force a specific one with: `--no-default-features --features cpu` etc.
#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "cpu")))]
compile_error!("Enable at least one of `cuda`, `wgpu`, or `cpu` features");

/// CPU reference: same formula as `butteraugli::pnorm_slice` for p=3,
/// inlined here so we don't pull a hard dep on the CPU crate's internals.
fn cpu_reference(diffmap: &[f32]) -> (f32, f32) {
    let mut max = 0.0f32;
    let mut sum_p3 = 0.0f64;
    let mut sum_p6 = 0.0f64;
    let mut sum_p12 = 0.0f64;
    for &v in diffmap {
        if v > max {
            max = v;
        }
        let d = v as f64;
        let d3 = d * d * d;
        sum_p3 += d3;
        let d6 = d3 * d3;
        sum_p6 += d6;
        sum_p12 += d6 * d6;
    }
    let n_inv = 1.0 / diffmap.len() as f64;
    let v0 = (n_inv * sum_p3).powf(1.0 / 3.0);
    let v1 = (n_inv * sum_p6).powf(1.0 / 6.0);
    let v2 = (n_inv * sum_p12).powf(1.0 / 12.0);
    (max, ((v0 + v1 + v2) / 3.0) as f32)
}

fn run_case(name: &str, diffmap: Vec<f32>) {
    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    let n = diffmap.len();
    let bytes = bytemuck::cast_slice::<f32, u8>(&diffmap);
    let handle = client.create_from_slice(bytes);

    let gpu = reduce_diffmap_to_score::<Backend>(&client, handle, n);
    let (cpu_score, cpu_pnorm_3) = cpu_reference(&diffmap);

    let max_rel = if cpu_score.abs() > 1e-9 {
        ((gpu.score - cpu_score).abs() / cpu_score.abs()) as f64
    } else {
        (gpu.score - cpu_score).abs() as f64
    };
    let pnorm_rel = if cpu_pnorm_3.abs() > 1e-9 {
        ((gpu.pnorm_3 - cpu_pnorm_3).abs() / cpu_pnorm_3.abs()) as f64
    } else {
        (gpu.pnorm_3 - cpu_pnorm_3).abs() as f64
    };

    println!(
        "[{name}] n={n}\n  GPU: score={:.6} pnorm_3={:.6}\n  CPU: score={:.6} pnorm_3={:.6}\n  rel: max={:.2e}  pnorm_3={:.2e}",
        gpu.score, gpu.pnorm_3, cpu_score, cpu_pnorm_3, max_rel, pnorm_rel
    );
}

fn main() {
    // Case 1: uniform — pnorm of value v should equal v.
    run_case("uniform_2.5_64x64", vec![2.5; 64 * 64]);

    // Case 2: zero — both should be 0.
    run_case("zero_64x64", vec![0.0; 64 * 64]);

    // Case 3: gradient — what a real diffmap looks like.
    let n = 512 * 512;
    let dm: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 5.0).collect();
    run_case("gradient_512x512", dm);

    // Case 4: 4K — stress the f32 sum precision.
    let n = 3840 * 2160;
    let dm: Vec<f32> = (0..n)
        .map(|i| ((i as f32).sin() * 0.5 + 1.5).max(0.0))
        .collect();
    run_case("sine_3840x2160", dm);
}
