//! Fused max-norm + libjxl 3-norm aggregation over a butteraugli diffmap.
//!
//! Single grid-strided pass writes:
//!   * max(d) → max-bits buffer  via `Atomic<u32>::fetch_max` on the
//!     f32 bit pattern. Diffmap values are non-negative, so f32 IEEE-754
//!     bit-pattern ordering matches f32 value ordering. Works on every
//!     cubecl backend (CUDA / DX12 / Metal / Vulkan / HIP).
//!   * Σd³  → sums[0]  ┐
//!   * Σd⁶  → sums[1]  ├─ via two paths, selected by the `fast-reduction`
//!   * Σd¹² → sums[2]  ┘  feature flag.
//!
//! Host then folds the sums into the final 3-norm:
//! `((Σd³/n)^(1/3) + (Σd⁶/n)^(1/6) + (Σd¹²/n)^(1/12)) / 3`
//!
//! Matches `lib/extras/metrics.cc:ComputeDistanceP` from libjxl at p=3.
//!
//! ## Two reduction implementations for the f32 sums
//!
//! - **`fast-reduction` (default-on)** — single-pass `Atomic<f32>::fetch_add`
//!   per (p3, p6, p12). Verified on CUDA + Windows DX12 + HIP.
//! - **`fast-reduction` off** — two-stage path: per-thread partials then
//!   an on-device finalizer kernel folds them. Required for Metal —
//!   cubecl-wgpu's Metal backend reports `Atomic<f32> = LoadStore|Add`
//!   as supported but the codegen silently no-ops at execution time and
//!   the sums return zero, collapsing every score's pnorm_3 to 0.
//!
//! Cross-vendor build:
//! ```bash
//! cargo build -p butteraugli-gpu --no-default-features --features wgpu
//! ```

use cubecl::prelude::*;

/// Threads per cube and cubes per reduction. Same launch geometry as
/// the existing `butteraugli-cuda` PTX path.
const BLOCKS: u32 = 16;
const THREADS: u32 = 256;
const THREADS_PER_REDUCTION: usize = (BLOCKS * THREADS) as usize;

// Per-cube launch dim for the batched paths.
const BATCHED_CUBES_PER_IMAGE: u32 = 8;
const BATCHED_THREADS_PER_REDUCTION: usize = (BATCHED_CUBES_PER_IMAGE * THREADS) as usize;

// =====================================================================
// Fast path — single-pass Atomic<f32>::fetch_add. Works on CUDA, DX12,
// HIP. Broken on Metal (silently no-ops).
// =====================================================================

#[cfg(feature = "fast-reduction")]
mod fast {
    use super::*;

    #[cube(launch_unchecked)]
    fn fused_max_pnorm_sums_kernel(
        diffmap: &Array<f32>,
        output_max_bits: &mut Array<Atomic<u32>>,
        output_sums: &mut Array<Atomic<f32>>,
    ) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = diffmap.len();

        let mut local_max = 0.0_f32;
        let mut local_p3 = 0.0_f32;
        let mut local_p6 = 0.0_f32;
        let mut local_p12 = 0.0_f32;

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

        let max_bits = u32::reinterpret(local_max);
        output_max_bits[0].fetch_max(max_bits);
        output_sums[0].fetch_add(local_p3);
        output_sums[1].fetch_add(local_p6);
        output_sums[2].fetch_add(local_p12);
    }

    #[cube(launch_unchecked)]
    fn batched_max_pnorm_sums_kernel(
        diffmap: &Array<f32>,
        output_max_bits: &mut Array<Atomic<u32>>,
        output_sums: &mut Array<Atomic<f32>>,
        plane_stride: u32,
    ) {
        let batch_idx = CUBE_POS_Y;
        let tid_in_plane = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
        let stride_per_plane = CUBE_COUNT_X as u32 * (CUBE_DIM_X as u32);
        let plane_off = (batch_idx * plane_stride) as usize;
        let plane_us = plane_stride as usize;

        let mut local_max = 0.0_f32;
        let mut local_p3 = 0.0_f32;
        let mut local_p6 = 0.0_f32;
        let mut local_p12 = 0.0_f32;

        let mut i = tid_in_plane as usize;
        while i < plane_us {
            let v = diffmap[plane_off + i];
            if v > local_max {
                local_max = v;
            }
            let d3 = v * v * v;
            local_p3 += d3;
            let d6 = d3 * d3;
            local_p6 += d6;
            local_p12 += d6 * d6;
            i += stride_per_plane as usize;
        }

        let bits = u32::reinterpret(local_max);
        output_max_bits[batch_idx as usize].fetch_max(bits);
        let sum_off = (batch_idx * 3) as usize;
        output_sums[sum_off].fetch_add(local_p3);
        output_sums[sum_off + 1].fetch_add(local_p6);
        output_sums[sum_off + 2].fetch_add(local_p12);
    }

    pub fn reduce<R: Runtime>(
        client: &ComputeClient<R>,
        diffmap_handle: cubecl::server::Handle,
        n_pixels: usize,
    ) -> crate::GpuButteraugliResult {
        let max_bits_handle = client.create_from_slice(u32::as_bytes(&[0_u32]));
        let sums_handle = client.create_from_slice(f32::as_bytes(&[0.0_f32; 3]));

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

        finalize_single::<R>(client, max_bits_handle, sums_handle, n_pixels)
    }

    pub fn reduce_batched_with_pnorm<R: Runtime>(
        client: &ComputeClient<R>,
        diffmap_handle: cubecl::server::Handle,
        plane_stride: u32,
        batch_size: u32,
    ) -> Vec<crate::GpuButteraugliResult> {
        let max_bits_handle =
            client.create_from_slice(u32::as_bytes(&vec![0_u32; batch_size as usize]));
        let sums_handle =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; (batch_size * 3) as usize]));

        let cube_count = CubeCount::Static(BATCHED_CUBES_PER_IMAGE, batch_size, 1);
        let cube_dim = CubeDim::new_1d(THREADS);

        unsafe {
            batched_max_pnorm_sums_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(
                    diffmap_handle,
                    (plane_stride as usize) * (batch_size as usize),
                ),
                ArrayArg::from_raw_parts(max_bits_handle.clone(), batch_size as usize),
                ArrayArg::from_raw_parts(sums_handle.clone(), (batch_size * 3) as usize),
                plane_stride,
            );
        }

        finalize_batched::<R>(
            client,
            max_bits_handle,
            sums_handle,
            plane_stride as usize,
            batch_size as usize,
        )
    }
}

// =====================================================================
// Portable path — Atomic<u32>::fetch_max for max (works everywhere)
// + per-thread partials for sums + on-device finalizer kernel.
// =====================================================================

#[cfg(not(feature = "fast-reduction"))]
mod portable {
    use super::*;

    /// Per-thread partials kernel (single-image). Each thread writes
    /// `(p3, p6, p12)` at `partials[tid*3..tid*3+3]`. Max goes via
    /// `Atomic<u32>::fetch_max` on the bit-cast just like the fast path.
    #[cube(launch_unchecked)]
    fn thread_max_pnorm_sums_kernel(
        diffmap: &Array<f32>,
        output_max_bits: &mut Array<Atomic<u32>>,
        partials: &mut Array<f32>,
    ) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = diffmap.len();

        let mut local_max = 0.0_f32;
        let mut local_p3 = 0.0_f32;
        let mut local_p6 = 0.0_f32;
        let mut local_p12 = 0.0_f32;

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

        let max_bits = u32::reinterpret(local_max);
        output_max_bits[0].fetch_max(max_bits);

        let off = tid * 3;
        partials[off] = local_p3;
        partials[off + 1] = local_p6;
        partials[off + 2] = local_p12;
    }

    /// Finalizer (single-image). One cube, one thread, sums all
    /// `THREADS_PER_REDUCTION` partials into the 3-element output.
    #[cube(launch_unchecked)]
    fn finalize_kernel(partials: &Array<f32>, output_sums: &mut Array<f32>) {
        let mut sum_p3 = 0.0_f32;
        let mut sum_p6 = 0.0_f32;
        let mut sum_p12 = 0.0_f32;
        let mut k: u32 = 0;
        while k < THREADS_PER_REDUCTION_U32 {
            let off = (k as usize) * 3;
            sum_p3 += partials[off];
            sum_p6 += partials[off + 1];
            sum_p12 += partials[off + 2];
            k += 1;
        }
        output_sums[0] = sum_p3;
        output_sums[1] = sum_p6;
        output_sums[2] = sum_p12;
    }

    /// Per-thread batched partials kernel. Each (cube_y=batch_idx,
    /// cube_x=block_id, tid_x=thread) emits 3 partials at
    /// `partials[(batch_idx * THREADS_PER_IMAGE + thread_in_plane) * 3..]`.
    #[cube(launch_unchecked)]
    fn thread_max_pnorm_sums_batched_kernel(
        diffmap: &Array<f32>,
        output_max_bits: &mut Array<Atomic<u32>>,
        partials: &mut Array<f32>,
        plane_stride: u32,
        threads_per_image: u32,
    ) {
        let batch_idx = CUBE_POS_Y;
        let tid_in_plane = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
        let stride_per_plane = CUBE_COUNT_X as u32 * (CUBE_DIM_X as u32);
        let plane_off = (batch_idx * plane_stride) as usize;
        let plane_us = plane_stride as usize;

        let mut local_max = 0.0_f32;
        let mut local_p3 = 0.0_f32;
        let mut local_p6 = 0.0_f32;
        let mut local_p12 = 0.0_f32;

        let mut i = tid_in_plane as usize;
        while i < plane_us {
            let v = diffmap[plane_off + i];
            if v > local_max {
                local_max = v;
            }
            let d3 = v * v * v;
            local_p3 += d3;
            let d6 = d3 * d3;
            local_p6 += d6;
            local_p12 += d6 * d6;
            i += stride_per_plane as usize;
        }

        let bits = u32::reinterpret(local_max);
        output_max_bits[batch_idx as usize].fetch_max(bits);

        let off = ((batch_idx * threads_per_image + tid_in_plane) * 3) as usize;
        partials[off] = local_p3;
        partials[off + 1] = local_p6;
        partials[off + 2] = local_p12;
    }

    /// Batched finalizer. One cube per image, single thread per cube,
    /// reads `BATCHED_THREADS_PER_REDUCTION` partial triples and emits
    /// (Σ p3, Σ p6, Σ p12) per image.
    #[cube(launch_unchecked)]
    fn finalize_batched_kernel(
        partials: &Array<f32>,
        output_sums: &mut Array<f32>,
        threads_per_image: u32,
    ) {
        let batch_idx = CUBE_POS_X;
        let in_off = (batch_idx * threads_per_image * 3) as usize;
        let out_off = (batch_idx * 3) as usize;

        let mut sum_p3 = 0.0_f32;
        let mut sum_p6 = 0.0_f32;
        let mut sum_p12 = 0.0_f32;
        let mut k: u32 = 0;
        while k < threads_per_image {
            let off = in_off + (k as usize) * 3;
            sum_p3 += partials[off];
            sum_p6 += partials[off + 1];
            sum_p12 += partials[off + 2];
            k += 1;
        }
        output_sums[out_off] = sum_p3;
        output_sums[out_off + 1] = sum_p6;
        output_sums[out_off + 2] = sum_p12;
    }

    const THREADS_PER_REDUCTION_U32: u32 = BLOCKS * THREADS;

    pub fn reduce<R: Runtime>(
        client: &ComputeClient<R>,
        diffmap_handle: cubecl::server::Handle,
        n_pixels: usize,
    ) -> crate::GpuButteraugliResult {
        let max_bits_handle = client.create_from_slice(u32::as_bytes(&[0_u32]));
        let partials_handle = client.create_from_slice(f32::as_bytes(&vec![
            0.0_f32;
            THREADS_PER_REDUCTION * 3
        ]));
        let sums_handle = client.create_from_slice(f32::as_bytes(&[0.0_f32; 3]));

        let cube_count = CubeCount::Static(BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(THREADS);

        unsafe {
            thread_max_pnorm_sums_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(diffmap_handle, n_pixels),
                ArrayArg::from_raw_parts(max_bits_handle.clone(), 1),
                ArrayArg::from_raw_parts(partials_handle.clone(), THREADS_PER_REDUCTION * 3),
            );
            finalize_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(partials_handle, THREADS_PER_REDUCTION * 3),
                ArrayArg::from_raw_parts(sums_handle.clone(), 3),
            );
        }

        finalize_single::<R>(client, max_bits_handle, sums_handle, n_pixels)
    }

    pub fn reduce_batched_with_pnorm<R: Runtime>(
        client: &ComputeClient<R>,
        diffmap_handle: cubecl::server::Handle,
        plane_stride: u32,
        batch_size: u32,
    ) -> Vec<crate::GpuButteraugliResult> {
        let threads_per_image = BATCHED_THREADS_PER_REDUCTION;
        let max_bits_handle =
            client.create_from_slice(u32::as_bytes(&vec![0_u32; batch_size as usize]));
        let partials_handle = client.create_from_slice(f32::as_bytes(&vec![
            0.0_f32;
            threads_per_image
                * (batch_size as usize)
                * 3
        ]));
        let sums_handle =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; (batch_size * 3) as usize]));

        let cube_count = CubeCount::Static(BATCHED_CUBES_PER_IMAGE, batch_size, 1);
        let cube_dim = CubeDim::new_1d(THREADS);

        unsafe {
            thread_max_pnorm_sums_batched_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(
                    diffmap_handle,
                    (plane_stride as usize) * (batch_size as usize),
                ),
                ArrayArg::from_raw_parts(max_bits_handle.clone(), batch_size as usize),
                ArrayArg::from_raw_parts(
                    partials_handle.clone(),
                    threads_per_image * (batch_size as usize) * 3,
                ),
                plane_stride,
                threads_per_image as u32,
            );
            finalize_batched_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(batch_size, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(
                    partials_handle,
                    threads_per_image * (batch_size as usize) * 3,
                ),
                ArrayArg::from_raw_parts(sums_handle.clone(), (batch_size * 3) as usize),
                threads_per_image as u32,
            );
        }

        finalize_batched::<R>(
            client,
            max_bits_handle,
            sums_handle,
            plane_stride as usize,
            batch_size as usize,
        )
    }
}

// Shared host-side fold helpers used by both fast and portable paths.

fn finalize_single<R: Runtime>(
    client: &ComputeClient<R>,
    max_bits_handle: cubecl::server::Handle,
    sums_handle: cubecl::server::Handle,
    n_pixels: usize,
) -> crate::GpuButteraugliResult {
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

fn finalize_batched<R: Runtime>(
    client: &ComputeClient<R>,
    max_bits_handle: cubecl::server::Handle,
    sums_handle: cubecl::server::Handle,
    plane_stride: usize,
    batch_size: usize,
) -> Vec<crate::GpuButteraugliResult> {
    let max_bytes = client.read_one(max_bits_handle).expect("read batched max");
    let sums_bytes = client.read_one(sums_handle).expect("read batched sums");
    let max_bits = u32::from_bytes(&max_bytes);
    let sums = f32::from_bytes(&sums_bytes);

    let n_inv = 1.0_f64 / (plane_stride as f64);
    (0..batch_size)
        .map(|i| {
            let max = f32::from_bits(max_bits[i]);
            let sum_p3 = sums[i * 3] as f64;
            let sum_p6 = sums[i * 3 + 1] as f64;
            let sum_p12 = sums[i * 3 + 2] as f64;
            let v0 = (n_inv * sum_p3).powf(1.0 / 3.0);
            let v1 = (n_inv * sum_p6).powf(1.0 / 6.0);
            let v2 = (n_inv * sum_p12).powf(1.0 / 12.0);
            let pnorm_3 = ((v0 + v1 + v2) / 3.0) as f32;
            crate::GpuButteraugliResult {
                score: max,
                pnorm_3,
            }
        })
        .collect()
}

// Max-only batched reduction. Uses Atomic<u32>::fetch_max which works
// on every backend regardless of the float-atomic situation, so no
// cfg-gating needed.

#[cube(launch_unchecked)]
fn batched_max_reduce_kernel(
    diffmap: &Array<f32>,
    output_max_bits: &mut Array<Atomic<u32>>,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Y;
    let tid_in_plane = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
    let stride_per_plane = CUBE_COUNT_X as u32 * (CUBE_DIM_X as u32);
    let plane_off = (batch_idx * plane_stride) as usize;
    let plane_us = plane_stride as usize;

    let mut local_max = 0.0_f32;
    let mut i = tid_in_plane as usize;
    while i < plane_us {
        let v = diffmap[plane_off + i];
        if v > local_max {
            local_max = v;
        }
        i += stride_per_plane as usize;
    }
    let bits = u32::reinterpret(local_max);
    output_max_bits[batch_idx as usize].fetch_max(bits);
}

/// Run batched max reduction. Returns `Vec<f32>` of length `batch_size`.
pub fn reduce_batched<R: Runtime>(
    client: &ComputeClient<R>,
    diffmap_handle: cubecl::server::Handle,
    plane_stride: u32,
    batch_size: u32,
) -> Vec<f32> {
    let max_bits_handle =
        client.create_from_slice(u32::as_bytes(&vec![0_u32; batch_size as usize]));
    let cube_count = CubeCount::Static(BATCHED_CUBES_PER_IMAGE, batch_size, 1);
    let cube_dim = CubeDim::new_1d(THREADS);

    unsafe {
        batched_max_reduce_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(
                diffmap_handle,
                (plane_stride as usize) * (batch_size as usize),
            ),
            ArrayArg::from_raw_parts(max_bits_handle.clone(), batch_size as usize),
            plane_stride,
        );
    }
    let bytes = client
        .read_one(max_bits_handle)
        .expect("read_one batched max");
    u32::from_bytes(&bytes)
        .iter()
        .map(|&b| f32::from_bits(b))
        .collect()
}

#[cfg(feature = "fast-reduction")]
pub use fast::{reduce, reduce_batched_with_pnorm};

#[cfg(not(feature = "fast-reduction"))]
pub use portable::{reduce, reduce_batched_with_pnorm};
