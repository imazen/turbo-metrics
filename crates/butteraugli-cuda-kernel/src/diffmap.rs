//! Final diffmap computation kernels for Butteraugli
//!
//! Combines AC and DC differences with masking to produce the final difference map.

use nvptx_std::math::StdMathExt;

/// MaskY: compute mask value for AC differences
#[inline]
fn mask_y(delta: f32) -> f32 {
    const OFFSET: f32 = 0.829591754942;
    const SCALER: f32 = 0.451936922203;
    const MUL: f32 = 2.5485944793;
    const NORM: f32 = 1.0 / (0.79079917404 * 17.83);

    let c = MUL / (SCALER * delta + OFFSET);
    let retval = (1.0 + c) * NORM;
    retval * retval
}

/// MaskDcY: compute mask value for DC differences
#[inline]
fn mask_dc_y(delta: f32) -> f32 {
    const OFFSET: f32 = 0.20025578522;
    const SCALER: f32 = 3.87449418804;
    const MUL: f32 = 0.505054525019;
    const NORM: f32 = 1.0 / (0.79079917404 * 17.83);

    let c = MUL / (SCALER * delta + OFFSET);
    let retval = (1.0 + c) * NORM;
    retval * retval
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
/// Different weights for when src1 > src2 vs src1 < src2
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

    let v1 = *src1.add(idx);
    let v2 = *src2.add(idx);
    let diff = v1 - v2;
    let weight = if v1 > v2 { weight_gt } else { weight_lt };

    *dst.add(idx) = *dst.add(idx) + weight * diff * diff;
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
