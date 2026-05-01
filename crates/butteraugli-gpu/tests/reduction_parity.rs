//! Cross-check the GPU reduction kernel against a CPU reference computation
//! over synthetic diffmaps. Runs whatever backend the dev-features select.
//!
//! The CPU reference matches `butteraugli::pnorm_slice` semantics for p=3.

use butteraugli_gpu::reduce_diffmap_to_score;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

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

fn run_case(name: &str, diffmap: Vec<f32>, max_rel_tol: f64) {
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

    assert!(
        max_rel <= max_rel_tol,
        "[{name}] max-norm rel diff {max_rel:.2e} > tol {max_rel_tol:.2e}\n  GPU {} CPU {}",
        gpu.score,
        cpu_score
    );
    assert!(
        pnorm_rel <= max_rel_tol,
        "[{name}] 3-norm  rel diff {pnorm_rel:.2e} > tol {max_rel_tol:.2e}\n  GPU {} CPU {}",
        gpu.pnorm_3,
        cpu_pnorm_3
    );
}

#[test]
fn uniform_value_round_trips() {
    // Uniform diffmap of value v: pnorm_3 should equal v exactly (modulo
    // f32 rounding) for any p — bulk reduction sanity check.
    run_case("uniform_2.5", vec![2.5f32; 64 * 64], 1e-5);
}

#[test]
fn zero_diffmap_yields_zero() {
    run_case("zero", vec![0.0f32; 64 * 64], 1e-9);
}

#[test]
fn gradient_512_matches_cpu() {
    // Gradient 0..5 over 256K pixels — exercises the sum precision.
    let n = 512 * 512;
    let dm: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 5.0).collect();
    run_case("gradient_512", dm, 5e-3);
}

#[test]
fn sine_4k_matches_cpu() {
    // 4K-scale stress test for f32 sum precision.
    let n = 3840 * 2160;
    let dm: Vec<f32> = (0..n)
        .map(|i| ((i as f32).sin() * 0.5 + 1.5).max(0.0))
        .collect();
    run_case("sine_4k", dm, 5e-3);
}
