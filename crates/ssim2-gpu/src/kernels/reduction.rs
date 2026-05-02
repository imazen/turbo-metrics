//! Per-plane fused (Σd, Σd⁴) reduction.
//!
//! For each error-map plane the SSIMULACRA2 score wants two numbers:
//! - mean: `(1/N) · Σ d`
//! - p4 norm: `((1/N) · Σ d⁴)^(1/4)`
//!
//! See `ssimulacra2::ssim_map` (sum + sum-of-fourth-powers per plane,
//! per channel). The CUDA reference splits this into two NPP `Sum`
//! launches; here we fuse them into a single grid-strided kernel,
//! matching butteraugli-gpu's reduction pattern.
//!
//! ## Atomics across backends
//!
//! `Atomic<f32>::fetch_add` is supported on all CubeCL backends
//! (CUDA/WGPU/Metal/HIP). f64 atomics are CUDA-only, so we accumulate
//! in f32 and host-side fold to f64. Error-map values are in [0, 1] for
//! the typical SSIMULACRA2 input ranges; at 33 MP the relative
//! round-off in `Σ d⁴` stays below 1e-3, well below the 0.1 % score
//! tolerance.

use cubecl::prelude::*;

/// Output convention: `output_sums[2*plane_idx]   = Σ d`
///                    `output_sums[2*plane_idx+1] = Σ d⁴`
///
/// Caller zeroes the buffer; we fetch_add into both slots.
#[cube(launch_unchecked)]
fn fused_sum_p4_kernel(
    plane: &Array<f32>,
    output_sums: &mut Array<Atomic<f32>>,
    out_offset: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let n = plane.len();

    let mut local_sum = 0.0_f32;
    let mut local_p4 = 0.0_f32;

    let mut i = tid;
    while i < n {
        let v = plane[i];
        local_sum += v;
        let v2 = v * v;
        local_p4 += v2 * v2;
        i += stride;
    }

    let off = out_offset as usize;
    output_sums[off].fetch_add(local_sum);
    output_sums[off + 1].fetch_add(local_p4);
}

/// Run the fused (Σ, Σ⁴) reduction for one plane. Caller manages
/// `output_sums_handle` lifetime — typically allocated once per
/// (scale × channel × error-map) and read back at the end.
///
/// `out_offset` indexes into a flat sums buffer that may hold many
/// reductions packed together (so the host can issue one read at the
/// end of the pipeline).
pub fn launch_sum_p4<R: Runtime>(
    client: &ComputeClient<R>,
    plane_handle: cubecl::server::Handle,
    n_pixels: usize,
    output_sums_handle: cubecl::server::Handle,
    output_sums_len: usize,
    out_offset: u32,
) {
    const BLOCKS: u32 = 16;
    const THREADS: u32 = 256;
    let cube_count = CubeCount::Static(BLOCKS, 1, 1);
    let cube_dim = CubeDim::new_1d(THREADS);

    unsafe {
        fused_sum_p4_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(plane_handle, n_pixels),
            ArrayArg::from_raw_parts(output_sums_handle, output_sums_len),
            out_offset,
        );
    }
}

/// Batched per-image reduction kernel.
///
/// Each cube grouping (`CUBE_POS_Y`) handles one image's plane in the
/// batch and reduces it grid-strided over `CUBE_POS_X`. Output layout:
///   `output_sums[2 * (batch_idx * stats_per_image + slot)]`     = Σ
///   `output_sums[2 * (batch_idx * stats_per_image + slot) + 1]` = Σ⁴
///
/// Caller zeroes the output. `slot` selects which (channel × map_type)
/// stat slot inside an image's stats block this reduction writes.
#[cube(launch_unchecked)]
fn fused_sum_p4_batched_kernel(
    plane: &Array<f32>,
    output_sums: &mut Array<Atomic<f32>>,
    plane_stride: u32,
    stats_per_image: u32,
    slot: u32,
) {
    let batch_idx = CUBE_POS_Y;
    let tid_in_plane = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
    let stride_per_plane = CUBE_COUNT_X as u32 * (CUBE_DIM_X as u32);
    let plane_off = (batch_idx * plane_stride) as usize;
    let plane_us = plane_stride as usize;

    let mut local_sum = 0.0_f32;
    let mut local_p4 = 0.0_f32;

    let mut i = tid_in_plane as usize;
    while i < plane_us {
        let v = plane[plane_off + i];
        local_sum += v;
        let v2 = v * v;
        local_p4 += v2 * v2;
        i += stride_per_plane as usize;
    }

    let off = ((batch_idx * stats_per_image + slot) * 2) as usize;
    output_sums[off].fetch_add(local_sum);
    output_sums[off + 1].fetch_add(local_p4);
}

/// Run the batched (Σ, Σ⁴) reduction for one (channel × error-map)
/// across `batch_size` images. `output_sums_handle` covers the entire
/// (NUM_SCALES × 3 channels × 3 map types × 2 stats × batch_size)
/// flat sums region; `slot` selects which (scale × channel × map_type)
/// triple inside the per-image stats block this reduction writes.
pub fn launch_sum_p4_batched<R: Runtime>(
    client: &ComputeClient<R>,
    plane_handle: cubecl::server::Handle,
    plane_stride: u32,
    batch_size: u32,
    output_sums_handle: cubecl::server::Handle,
    output_sums_len: usize,
    stats_per_image: u32,
    slot: u32,
) {
    const CUBES_PER_IMAGE: u32 = 8;
    const THREADS: u32 = 256;
    let cube_count = CubeCount::Static(CUBES_PER_IMAGE, batch_size, 1);
    let cube_dim = CubeDim::new_1d(THREADS);

    unsafe {
        fused_sum_p4_batched_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(
                plane_handle,
                (plane_stride as usize) * (batch_size as usize),
            ),
            ArrayArg::from_raw_parts(output_sums_handle, output_sums_len),
            plane_stride,
            stats_per_image,
            slot,
        );
    }
}
