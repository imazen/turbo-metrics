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
#[no_mangle]
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

/// L2 difference: accumulate squared difference with weight
/// Used for simple difference metrics
#[no_mangle]
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
#[no_mangle]
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

/// Compute x^q for each element (for norm calculation)
/// The actual reduction/sum will be done on host side using NPP
#[no_mangle]
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
