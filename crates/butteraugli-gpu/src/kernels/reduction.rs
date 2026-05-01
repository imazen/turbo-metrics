//! Fused max-norm + libjxl 3-norm aggregation over a butteraugli diffmap.
//!
//! Single grid-strided pass writes:
//!   * max(d) → output_max  via `Atomic<u32>::fetch_max` on the f32 bit
//!     pattern. Diffmap values are non-negative, so f32 bit-pattern
//!     ordering matches f32 value ordering. CUDA doesn't expose
//!     `atomicMax` for f32; this is the same trick the existing
//!     `butteraugli-cuda` PTX kernel uses.
//!   * Σd³  → output_sums[0]  via `Atomic<f32>::fetch_add`
//!   * Σd⁶  → output_sums[1]
//!   * Σd¹² → output_sums[2]
//!
//! Host then folds the sums into the final 3-norm:
//! `((Σd³/n)^(1/3) + (Σd⁶/n)^(1/6) + (Σd¹²/n)^(1/12)) / 3`
//!
//! Matches `lib/extras/metrics.cc:ComputeDistanceP` from libjxl at p=3.
//!
//! ## Precision note
//!
//! Sums use `f32` accumulation (not the f64 used by the CUDA-only
//! `butteraugli-cuda` crate). `Atomic<f32>::fetch_add` is the lowest
//! common denominator across CubeCL backends — CUDA, WGPU/Vulkan/Metal,
//! HIP all support it; `Atomic<f64>::fetch_add` is CUDA-only. f32 sums
//! hold adequate precision for diffmap aggregation: at 33 MP (8K) with
//! diffmap values ≤ ~10, relative error in Σd¹² is below 5e-3, well
//! within the algorithmic 1% noise floor between GPU and CPU
//! implementations.

use cubecl::prelude::*;

/// One grid-strided thread reads its slice of `diffmap` and accumulates
/// (local_max, local_p3, local_p6, local_p12), then issues four atomics:
/// one `Atomic<u32>::fetch_max` on the f32 bit pattern (CUDA doesn't have
/// f32 atomicMax), and three `Atomic<f32>::fetch_add` for the sums.
///
/// Caller zeroes both buffers before launch.
#[cube(launch_unchecked)]
fn fused_max_pnorm_sums_kernel(
    diffmap: &Array<f32>,
    output_max_bits: &mut Array<Atomic<u32>>,
    output_sums: &mut Array<Atomic<f32>>,
) {
    let tid = ABSOLUTE_POS;
    // CUBE_COUNT is usize; CUBE_DIM_X is u32 — cast to compose.
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let n = diffmap.len();

    let mut local_max = 0.0f32;
    let mut local_p3 = 0.0f32;
    let mut local_p6 = 0.0f32;
    let mut local_p12 = 0.0f32;

    let mut i = tid;
    while i < n {
        let v = diffmap[i];
        if v > local_max {
            local_max = v;
        }
        let d3 = v * v * v;
        local_p3 += d3;
        let d6 = d3 * d3;
        local_p6 += d6;
        local_p12 += d6 * d6;
        i += stride;
    }

    // Bit-cast non-negative f32 max to u32, atomicMax on bits, then
    // bit-cast back at host. f32 IEEE-754 bit-pattern ordering matches
    // f32 value ordering for non-negative values.
    let max_bits = u32::reinterpret(local_max);
    output_max_bits[0].fetch_max(max_bits);

    // Three f32 atomic adds for the sums.
    output_sums[0].fetch_add(local_p3);
    output_sums[1].fetch_add(local_p6);
    output_sums[2].fetch_add(local_p12);
}

/// Launch the fused reduction and fold the sums into the final libjxl
/// 3-norm aggregation.
pub fn reduce<R: Runtime>(
    client: &ComputeClient<R>,
    diffmap_handle: cubecl::server::Handle,
    n_pixels: usize,
) -> crate::GpuButteraugliResult {
    // Two output buffers: a 4-byte u32 for max bits, and a 3×f32 buffer
    // for the sums. Both zero-initialized.
    let max_bits_handle = client.create_from_slice(u32::as_bytes(&[0_u32]));
    let sums_handle = client.create_from_slice(f32::as_bytes(&[0.0_f32; 3]));

    // Same launch geometry as the butteraugli-cuda PTX path: 16 blocks ×
    // 256 threads = 4096 grid-strided workers.
    const BLOCKS: u32 = 16;
    const THREADS: u32 = 256;

    let cube_count = CubeCount::Static(BLOCKS, 1, 1);
    let cube_dim = CubeDim::new_1d(THREADS);

    unsafe {
        fused_max_pnorm_sums_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(diffmap_handle, n_pixels),
            ArrayArg::from_raw_parts(max_bits_handle.clone(), 1),
            ArrayArg::from_raw_parts(sums_handle.clone(), 3),
        );
    }

    let max_raw = client.read_one(max_bits_handle).expect("read_one max");
    let max_bits = u32::from_bytes(&max_raw)[0];
    let max = f32::from_bits(max_bits);

    let sums_raw = client.read_one(sums_handle).expect("read_one sums");
    let sums = f32::from_bytes(&sums_raw);
    let sum_p3 = sums[0] as f64;
    let sum_p6 = sums[1] as f64;
    let sum_p12 = sums[2] as f64;

    let n_inv = 1.0_f64 / (n_pixels as f64);
    let v0 = (n_inv * sum_p3).powf(1.0 / 3.0);
    let v1 = (n_inv * sum_p6).powf(1.0 / 6.0);
    let v2 = (n_inv * sum_p12).powf(1.0 / 12.0);
    let pnorm_3 = ((v0 + v1 + v2) / 3.0) as f32;

    crate::GpuButteraugliResult {
        score: max,
        pnorm_3,
    }
}
