//! Slotted Σ reductions for DSSIM.
//!
//! DSSIM needs two scalars per scale: Σ ssim (for `mean_ssim`) and
//! Σ |ssim - avg| (for the per-scale MAD that becomes `1 - mad`).
//! We use `NUM_SLOTS = 2 × NUM_SCALES` and write each scale's sum
//! into a fixed slot.
//!
//! ## Two reduction implementations
//!
//! Selected at compile time via the `fast-reduction` feature flag:
//!
//! - **`fast-reduction` (default-on)** — single-pass
//!   `Atomic<f32>::fetch_add`. Works on CUDA / HIP / DX12 wgpu;
//!   silently no-ops on Metal (per `ssim2-gpu`'s reality check).
//! - **`fast-reduction` off** — two-stage path: each grid-strided
//!   thread writes its own partial to a scratch region, then a
//!   one-cube-per-slot finalizer kernel folds the partials into the
//!   small sums buffer. Slower but works everywhere.
//!
//! Build for cross-vendor (wgpu / Metal):
//! ```bash
//! cargo build -p dssim-gpu --no-default-features --features wgpu
//! ```

use cubecl::prelude::*;

/// Threads per cube for the reduction kernels.
pub const BLOCK_SIZE: u32 = 256;
/// Cubes per reduction (16 × 256 = 4096 grid-strided workers).
pub const NUM_BLOCKS: u32 = 16;
/// Total threads per reduction.
pub const THREADS_PER_REDUCTION: usize = (NUM_BLOCKS * BLOCK_SIZE) as usize;

/// Floats this slot consumes in the on-device `partials` buffer.
/// In fast mode this is the single final scalar (atomic writes go
/// straight there); in portable mode it's per-thread partials that a
/// finalizer kernel folds into the small sums buffer.
#[cfg(feature = "fast-reduction")]
pub const PARTIALS_PER_REDUCTION: usize = 1;
#[cfg(not(feature = "fast-reduction"))]
pub const PARTIALS_PER_REDUCTION: usize = THREADS_PER_REDUCTION;

// =====================================================================
// Fast path — single-pass Atomic<f32>::fetch_add.
// =====================================================================

#[cfg(feature = "fast-reduction")]
mod fast {
    use super::*;

    #[cube(launch_unchecked)]
    fn fused_sum_kernel(plane: &Array<f32>, output_sums: &mut Array<Atomic<f32>>, slot: u32) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = plane.len();

        let mut local_sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            local_sum += plane[i];
            i += stride;
        }
        output_sums[slot as usize].fetch_add(local_sum);
    }

    pub fn launch_sum<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        n_pixels: usize,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        slot: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        unsafe {
            fused_sum_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(plane_handle, n_pixels),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                slot,
            );
        }
    }

    /// In fast mode the `partials` buffer already holds the final
    /// per-slot sums. The "finalizer" is a copy so the rest of the
    /// pipeline is symmetric with the portable path.
    #[cube(launch_unchecked)]
    fn copy_kernel(src: &Array<f32>, dst: &mut Array<f32>) {
        let i = ABSOLUTE_POS;
        if i >= dst.len() {
            terminate!();
        }
        dst[i] = src[i];
    }

    pub fn launch_finalize<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        output_handle: cubecl::server::Handle,
        output_len: usize,
        _num_slots: u32,
    ) {
        const TPB: u32 = 64;
        let cubes = ((output_len as u32) + TPB - 1) / TPB;
        unsafe {
            copy_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes.max(1), 1, 1),
                CubeDim::new_1d(TPB),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                ArrayArg::from_raw_parts(output_handle, output_len),
            );
        }
    }
}

// =====================================================================
// Portable path — per-thread partials + on-device finalizer.
// =====================================================================

#[cfg(not(feature = "fast-reduction"))]
mod portable {
    use super::*;

    /// u32 mirror for use inside `#[cube]` bodies (cubecl 0.10's macro
    /// dislikes mixed `usize` const arithmetic in kernel scope).
    const THREADS_PER_REDUCTION_U32: u32 = NUM_BLOCKS * BLOCK_SIZE;

    #[cube(launch_unchecked)]
    fn thread_sum_kernel(plane: &Array<f32>, output: &mut Array<f32>, slot_offset: u32) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = plane.len();

        let mut local_sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            local_sum += plane[i];
            i += stride;
        }
        output[(slot_offset as usize) + tid] = local_sum;
    }

    #[cube(launch_unchecked)]
    fn finalize_sum_kernel(partials: &Array<f32>, output: &mut Array<f32>) {
        let slot = CUBE_POS_X;
        let in_off = (slot * THREADS_PER_REDUCTION_U32) as usize;
        let mut sum = 0.0_f32;
        let mut k: u32 = 0;
        while k < THREADS_PER_REDUCTION_U32 {
            sum += partials[in_off + (k as usize)];
            k += 1;
        }
        output[slot as usize] = sum;
    }

    pub fn launch_sum<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        n_pixels: usize,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        slot: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        let slot_offset = slot * (PARTIALS_PER_REDUCTION as u32);
        unsafe {
            thread_sum_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(plane_handle, n_pixels),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                slot_offset,
            );
        }
    }

    pub fn launch_finalize<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        output_handle: cubecl::server::Handle,
        output_len: usize,
        num_slots: u32,
    ) {
        let cube_count = CubeCount::Static(num_slots, 1, 1);
        let cube_dim = CubeDim::new_1d(1);
        unsafe {
            finalize_sum_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                ArrayArg::from_raw_parts(output_handle, output_len),
            );
        }
    }
}

#[cfg(feature = "fast-reduction")]
pub use fast::{launch_finalize, launch_sum};

#[cfg(not(feature = "fast-reduction"))]
pub use portable::{launch_finalize, launch_sum};
