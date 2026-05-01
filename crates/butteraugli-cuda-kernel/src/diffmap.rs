//! Final diffmap computation kernels for Butteraugli
//!
//! Combines AC and DC differences with masking to produce the final difference map.

use nvptx_std::math::StdMathExt;

/// MaskY: compute mask value for AC differences
/// Uses f64 internally like CPU for precision in division
#[inline]
fn mask_y(delta: f32) -> f32 {
    const OFFSET: f64 = 0.829591754942;
    const SCALER: f64 = 0.451936922203;
    const MUL: f64 = 2.5485944793;
    // GLOBAL_SCALE = 1.0 / (0.79079917404 * 17.83)
    const GLOBAL_SCALE: f64 = 0.07093654424083289;

    let delta = delta as f64;
    let c = MUL / (SCALER * delta + OFFSET);
    let retval = GLOBAL_SCALE * (1.0 + c);
    (retval * retval) as f32
}

/// MaskDcY: compute mask value for DC differences
/// Uses f64 internally like CPU for precision in division
#[inline]
fn mask_dc_y(delta: f32) -> f32 {
    const OFFSET: f64 = 0.20025578522;
    const SCALER: f64 = 3.87449418804;
    const MUL: f64 = 0.505054525019;
    // GLOBAL_SCALE = 1.0 / (0.79079917404 * 17.83)
    const GLOBAL_SCALE: f64 = 0.07093654424083289;

    let delta = delta as f64;
    let c = MUL / (SCALER * delta + OFFSET);
    let retval = GLOBAL_SCALE * (1.0 + c);
    (retval * retval) as f32
}

/// Compute final diffmap from masked AC and DC differences
///
/// diffmap = sqrt(maskY * (ac0 + ac1 + ac2) + maskDcY * (dc0 + dc1 + dc2))
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn compute_diffmap_kernel(
    mask: *const f32,
    block_diff_dc0: *const f32,
    block_diff_dc1: *const f32,
    block_diff_dc2: *const f32,
    block_diff_ac0: *const f32,
    block_diff_ac1: *const f32,
    block_diff_ac2: *const f32,
    dst: *mut f32,
    size: usize,
) {
    let idx = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;

    if idx >= size {
        return;
    }

    let mask_val = *mask.add(idx);
    let maskval_ac = mask_y(mask_val);
    let maskval_dc = mask_dc_y(mask_val);

    let ac_sum = *block_diff_ac0.add(idx) + *block_diff_ac1.add(idx) + *block_diff_ac2.add(idx);
    let dc_sum = *block_diff_dc0.add(idx) + *block_diff_dc1.add(idx) + *block_diff_dc2.add(idx);

    *dst.add(idx) = (maskval_ac * ac_sum + maskval_dc * dc_sum).sqrt();
}

/// Batched compute_diffmap. `size` is the per-image work size (plane or
/// half_plane). `plane_stride` is the per-image slot stride in the
/// concatenated buffers (usually full-plane even at half-res, because
/// the same buffers host both passes). gridDim.z = batch_size.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn compute_diffmap_batch_kernel(
    mask: *const f32,
    block_diff_dc0: *const f32,
    block_diff_dc1: *const f32,
    block_diff_dc2: *const f32,
    block_diff_ac0: *const f32,
    block_diff_ac1: *const f32,
    block_diff_ac2: *const f32,
    dst: *mut f32,
    size: usize,
    plane_stride: usize,
) {
    let idx = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let b = core::arch::nvptx::_block_idx_z() as usize;
    if idx >= size {
        return;
    }
    let off = b * plane_stride;
    let mask = mask.add(off);
    let dc0 = block_diff_dc0.add(off);
    let dc1 = block_diff_dc1.add(off);
    let dc2 = block_diff_dc2.add(off);
    let ac0 = block_diff_ac0.add(off);
    let ac1 = block_diff_ac1.add(off);
    let ac2 = block_diff_ac2.add(off);
    let dst = dst.add(off);

    let mask_val = *mask.add(idx);
    let maskval_ac = mask_y(mask_val);
    let maskval_dc = mask_dc_y(mask_val);

    let ac_sum = *ac0.add(idx) + *ac1.add(idx) + *ac2.add(idx);
    let dc_sum = *dc0.add(idx) + *dc1.add(idx) + *dc2.add(idx);

    *dst.add(idx) = (maskval_ac * ac_sum + maskval_dc * dc_sum).sqrt();
}

/// L2 difference: accumulate squared difference with weight
/// Used for simple difference metrics
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn l2_diff_kernel(
    src1: *const f32,
    src2: *const f32,
    dst: *mut f32,
    size: usize,
    weight: f32,
) {
    let idx = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;

    if idx >= size {
        return;
    }

    let diff = *src1.add(idx) - *src2.add(idx);
    *dst.add(idx) = *dst.add(idx) + weight * diff * diff;
}

/// L2 asymmetric difference
/// Matches CPU butteraugli L2DiffAsymmetric exactly:
/// - Primary symmetric quadratic: diff^2 * w_0gt1 * 0.8
/// - Secondary half-open quadratic: v^2 * w_0lt1 * 0.8
///   where v captures values outside "acceptable" range
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn l2_asym_diff_kernel(
    src1: *const f32,
    src2: *const f32,
    dst: *mut f32,
    size: usize,
    weight_gt: f32,
    weight_lt: f32,
) {
    let idx = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;

    if idx >= size {
        return;
    }

    let val0 = *src1.add(idx);
    let val1 = *src2.add(idx);

    // CPU multiplies weights by 0.8
    let vw_0gt1 = weight_gt * 0.8;
    let vw_0lt1 = weight_lt * 0.8;

    // Primary symmetric quadratic objective
    let diff = val0 - val1;
    let mut total = *dst.add(idx) + diff * diff * vw_0gt1;

    // Secondary half-open quadratic objectives
    let fabs0 = val0.abs();
    let too_small = 0.4 * fabs0;
    let too_big = fabs0;

    let v = if val0 < 0.0 {
        if val1 > -too_small {
            val1 + too_small
        } else if val1 < -too_big {
            -val1 - too_big
        } else {
            0.0
        }
    } else {
        if val1 < too_small {
            too_small - val1
        } else if val1 > too_big {
            val1 - too_big
        } else {
            0.0
        }
    };

    total += vw_0lt1 * v * v;
    *dst.add(idx) = total;
}

/// Atomic max reduction on a u32 result. For non-negative f32 values
/// the IEEE-754 bit pattern compares the same as the float value, so
/// we reinterpret as u32 and use CAS-based max.
///
/// Butteraugli's diffmap is derived from a sqrt of sums of squares —
/// always >= 0 — so this is safe.
///
/// Caller MUST zero `result_u32` on the same stream before the launch.
/// (This kernel doesn't zero it to avoid an extra launch; the callsite
/// has a cuMemsetD32Async helper.)
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn max_reduce_f32_to_u32_kernel(
    src: *const f32,
    result_u32: *mut u32,
    size: usize,
) {
    let tid = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let stride =
        core::arch::nvptx::_grid_dim_x() as usize * core::arch::nvptx::_block_dim_x() as usize;

    // Per-thread local max over the grid-strided slice of `src`.
    let mut local: u32 = 0;
    let mut i = tid;
    while i < size {
        let v = *src.add(i);
        let bits = v.to_bits();
        if bits > local {
            local = bits;
        }
        i += stride;
    }

    if local == 0 {
        return;
    }

    // Single atomicMax.u32 on the final result. Emit inline PTX directly
    // since nvptx targets don't expose core::intrinsics::atomic_umax.
    // For non-negative f32 values the bit pattern compares the same as
    // the float, so this correctly finds the maximum float.
    let mut _discard: u32;
    core::arch::asm!(
        "atom.global.max.u32 {d}, [{p}], {v};",
        d = out(reg32) _discard,
        p = in(reg64) result_u32,
        v = in(reg32) local,
        options(nostack, preserves_flags),
    );
}

/// Batched max reduction: one f32 result per image in `result_u32[b]`.
/// Zero result_u32[0..batch_size] on host before launch.
/// Grid: (16, 1, batch_size) x (256, 1, 1) threads; each block reduces
/// a grid-strided slice of image `b`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn max_reduce_f32_to_u32_batch_kernel(
    src: *const f32,
    result_u32: *mut u32,
    size: usize,         // per-image element count
    plane_stride: usize, // f32 elements per image (usually == size)
) {
    let tid_in_z = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let stride =
        core::arch::nvptx::_grid_dim_x() as usize * core::arch::nvptx::_block_dim_x() as usize;
    let b = core::arch::nvptx::_block_idx_z() as usize;

    let src_b = src.add(b * plane_stride);

    let mut local: u32 = 0;
    let mut i = tid_in_z;
    while i < size {
        let v = *src_b.add(i);
        let bits = v.to_bits();
        if bits > local {
            local = bits;
        }
        i += stride;
    }

    if local == 0 {
        return;
    }

    let target = result_u32.add(b);
    let mut _discard: u32;
    core::arch::asm!(
        "atom.global.max.u32 {d}, [{p}], {v};",
        d = out(reg32) _discard,
        p = in(reg64) target,
        v = in(reg32) local,
        options(nostack, preserves_flags),
    );
}

/// Compute x^q for each element (for norm calculation)
/// The actual reduction/sum will be done on host side using NPP
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn power_elements_kernel(
    src: *const f32,
    dst: *mut f32,
    size: usize,
    q: f32,
) {
    let idx = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;

    if idx >= size {
        return;
    }

    *dst.add(idx) = (*src.add(idx)).powf(q);
}

/// Fused max + libjxl 3-norm sums reduction over a Butteraugli diffmap.
///
/// In a single grid-strided pass over `src`, accumulates:
///   * max(src) into `result_max_u32` (via atomicMax.u32 on f32 bit pattern)
///   * Σ src³ into `sum_p3`   (f64, via atomicAdd.f64)
///   * Σ src⁶ into `sum_p6`   (f64, via atomicAdd.f64)
///   * Σ src¹² into `sum_p12` (f64, via atomicAdd.f64)
///
/// f64 sums match the libjxl `lib/extras/metrics.cc:ComputeDistanceP`
/// HWY_CAP_FLOAT64 path — at 8K (33 MP) with diffmap values ≤ ~10, d¹²
/// summed in f32 loses precision; f64 has the headroom.
///
/// Each thread does its own atomics (one per output) per loop iteration —
/// straightforward but contended. For 4096 threads × 4 atomics it's still
/// ~16K atomics total, dominated by D-D bandwidth on the diffmap read,
/// not atomic contention. If contention becomes a hotspot, switch to a
/// per-block shared-memory reduction with one atomic per output per block.
///
/// **Caller must zero `result_max_u32`, `sum_p3`, `sum_p6`, `sum_p12` on
/// the same stream before launch.**
///
/// Requires SM 6.0+ for `atom.global.add.f64`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn max_and_pnorm_sums_reduce_kernel(
    src: *const f32,
    result_max_u32: *mut u32,
    sum_p3: *mut f64,
    sum_p6: *mut f64,
    sum_p12: *mut f64,
    size: usize,
) {
    let tid = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let stride =
        core::arch::nvptx::_grid_dim_x() as usize * core::arch::nvptx::_block_dim_x() as usize;

    let mut local_max_bits: u32 = 0;
    let mut local_p3: f64 = 0.0;
    let mut local_p6: f64 = 0.0;
    let mut local_p12: f64 = 0.0;

    let mut i = tid;
    while i < size {
        let v = *src.add(i);
        let bits = v.to_bits();
        if bits > local_max_bits {
            local_max_bits = bits;
        }
        let d = v as f64;
        let d3 = d * d * d;
        local_p3 += d3;
        let d6 = d3 * d3;
        local_p6 += d6;
        local_p12 += d6 * d6;
        i += stride;
    }

    // Max: skip atomic if local is zero (matches max_reduce_f32_to_u32_kernel).
    if local_max_bits != 0 {
        let mut _discard: u32;
        core::arch::asm!(
            "atom.global.max.u32 {d}, [{p}], {v};",
            d = out(reg32) _discard,
            p = in(reg64) result_max_u32,
            v = in(reg32) local_max_bits,
            options(nostack, preserves_flags),
        );
    }

    // Three f64 atomic adds. Always issue (skipping for zero local sums
    // would only matter for all-zero diffmaps, where the cost is negligible).
    let mut _d1: f64;
    let mut _d2: f64;
    let mut _d3: f64;
    core::arch::asm!(
        "atom.global.add.f64 {d}, [{p}], {v};",
        d = out(reg64) _d1,
        p = in(reg64) sum_p3,
        v = in(reg64) local_p3,
        options(nostack, preserves_flags),
    );
    core::arch::asm!(
        "atom.global.add.f64 {d}, [{p}], {v};",
        d = out(reg64) _d2,
        p = in(reg64) sum_p6,
        v = in(reg64) local_p6,
        options(nostack, preserves_flags),
    );
    core::arch::asm!(
        "atom.global.add.f64 {d}, [{p}], {v};",
        d = out(reg64) _d3,
        p = in(reg64) sum_p12,
        v = in(reg64) local_p12,
        options(nostack, preserves_flags),
    );
}
