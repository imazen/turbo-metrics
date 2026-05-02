//! Per-plane (Σd, Σd⁴) reduction via per-thread partials + host-side
//! aggregation.
//!
//! For each error-map plane the SSIMULACRA2 score wants two numbers:
//! - mean: `(1/N) · Σ d`
//! - p4 norm: `((1/N) · Σ d⁴)^(1/4)`
//!
//! ## Why not `Atomic<f32>::fetch_add`?
//!
//! CUDA exposes `atomicAdd(float*, float)` as a hardware primitive and
//! cubecl-cuda lowers `Atomic<f32>::fetch_add` to it directly. cubecl-
//! wgpu only registers the same op when the device exposes
//! `SHADER_FLOAT32_ATOMIC` (Vulkan `VK_EXT_shader_atomic_float` /
//! Metal 3.0+ atomic_float). The GitHub-hosted Metal runners don't
//! enable it (or the codegen silently drops the op), and our
//! reductions returned all zeros — every score collapsed to ~100.
//!
//! Rather than depend on that, each grid-strided thread writes its
//! own (sum, p4) partial to a per-thread output slot; the host sums
//! `NUM_THREADS_TOTAL` partials per slot at f64 precision. Slightly
//! more bandwidth than the atomic path (≈ 1.7 MB read per
//! `compute()` instead of 432 B), but works on every cubecl backend
//! without feature negotiation.

use cubecl::prelude::*;

/// Threads per cube. 256 keeps occupancy high without exceeding any
/// backend's max workgroup size.
pub const BLOCK_SIZE: u32 = 256;

/// Cubes per reduction. 16 × 256 = 4096 grid-strided workers — same
/// total worker count the atomic-based path used.
pub const NUM_BLOCKS: u32 = 16;

/// Total threads per reduction (= number of (sum, p4) partials a
/// single reduction emits).
pub const THREADS_PER_REDUCTION: usize = (NUM_BLOCKS * BLOCK_SIZE) as usize;

/// Floats per reduction in the output buffer: each thread writes 2.
pub const PARTIALS_PER_REDUCTION: usize = THREADS_PER_REDUCTION * 2;

/// Per-thread (Σd, Σd⁴) emit. Each grid-strided thread emits its own
/// partial; no atomics, no shared memory, no cross-thread sync. The
/// host folds NUM_THREADS_TOTAL partials per slot.
#[cube(launch_unchecked)]
fn thread_sum_p4_kernel(
    plane: &Array<f32>,
    output: &mut Array<f32>,
    slot_offset: u32,
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

    let off = (slot_offset as usize) + tid * 2;
    output[off] = local_sum;
    output[off + 1] = local_p4;
}

/// Run the reduction for one plane. Caller passes the byte-offset
/// (in floats) where this slot's `PARTIALS_PER_REDUCTION` floats live
/// inside `output_sums_handle`.
pub fn launch_sum_p4<R: Runtime>(
    client: &ComputeClient<R>,
    plane_handle: cubecl::server::Handle,
    n_pixels: usize,
    output_sums_handle: cubecl::server::Handle,
    output_sums_len: usize,
    slot: u32,
) {
    let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
    let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
    let slot_offset = slot * (PARTIALS_PER_REDUCTION as u32);

    unsafe {
        thread_sum_p4_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(plane_handle, n_pixels),
            ArrayArg::from_raw_parts(output_sums_handle, output_sums_len),
            slot_offset,
        );
    }
}

/// Per-image reduction kernel — `CUBE_POS_Y` selects the image; each
/// thread within (block, image) emits its own partial. Output region
/// per (image, slot) is `PARTIALS_PER_REDUCTION` floats.
#[cube(launch_unchecked)]
fn thread_sum_p4_batched_kernel(
    plane: &Array<f32>,
    output: &mut Array<f32>,
    plane_stride: u32,
    image_stride: u32,
    slot_offset: u32,
) {
    let tid_in_plane = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
    let stride_per_image = CUBE_COUNT_X as u32 * (CUBE_DIM_X as u32);
    let batch_idx = CUBE_POS_Y;
    let plane_us = plane_stride as usize;
    let plane_off = (batch_idx * plane_stride) as usize;
    let img_off = (batch_idx * image_stride + slot_offset) as usize;

    let mut local_sum = 0.0_f32;
    let mut local_p4 = 0.0_f32;
    let mut i = tid_in_plane as usize;
    while i < plane_us {
        let v = plane[plane_off + i];
        local_sum += v;
        let v2 = v * v;
        local_p4 += v2 * v2;
        i += stride_per_image as usize;
    }

    let out_idx = img_off + (tid_in_plane as usize) * 2;
    output[out_idx] = local_sum;
    output[out_idx + 1] = local_p4;
}

/// Run the batched reduction. `image_stride` is the per-image stride
/// in the output buffer (= `num_slots × PARTIALS_PER_REDUCTION`),
/// `slot_offset` selects which slot (in floats) inside the image's
/// stats region this reduction writes to.
pub fn launch_sum_p4_batched<R: Runtime>(
    client: &ComputeClient<R>,
    plane_handle: cubecl::server::Handle,
    plane_stride: u32,
    batch_size: u32,
    output_sums_handle: cubecl::server::Handle,
    output_sums_len: usize,
    num_slots: u32,
    slot: u32,
) {
    let cube_count = CubeCount::Static(NUM_BLOCKS, batch_size, 1);
    let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
    let image_stride = num_slots * (PARTIALS_PER_REDUCTION as u32);
    let slot_offset = slot * (PARTIALS_PER_REDUCTION as u32);

    unsafe {
        thread_sum_p4_batched_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(
                plane_handle,
                (plane_stride as usize) * (batch_size as usize),
            ),
            ArrayArg::from_raw_parts(output_sums_handle, output_sums_len),
            plane_stride,
            image_stride,
            slot_offset,
        );
    }
}
