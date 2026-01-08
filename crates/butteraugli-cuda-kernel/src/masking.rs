//! Psychovisual masking kernels for Butteraugli
//!
//! Implements the masking model that determines how visible differences are
//! based on local image content. Areas with high texture/detail mask errors better.

use nvptx_std::math::StdMathExt;

// Masking constants
const MASK_MUL_X: f32 = 2.5;
const MASK_MUL_Y_UHF: f32 = 0.4;
const MASK_MUL_Y_HF: f32 = 0.4;

const DIFF_PRECOMPUTE_MUL: f32 = 6.19424080439;
const DIFF_PRECOMPUTE_BIAS: f32 = 12.61050594197;

const EROSION_STEP: usize = 3;

/// Initialize mask from HF and UHF bands
/// Combines X and Y channels with different weights
#[no_mangle]
pub unsafe extern "ptx-kernel" fn mask_init_kernel(
    hf_x: *const f32,
    uhf_x: *const f32,
    hf_y: *const f32,
    uhf_y: *const f32,
    dst: *mut f32,
    size: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let x_diff = (*uhf_x.add(idx) + *hf_x.add(idx)) * MASK_MUL_X;
    let y_diff = *uhf_y.add(idx) * MASK_MUL_Y_UHF + *hf_y.add(idx) * MASK_MUL_Y_HF;

    *dst.add(idx) = (x_diff * x_diff + y_diff * y_diff).sqrt();
}

/// Precompute diff values for masking
/// Applies sqrt(mul * |x| + bias) - sqrt(bias)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn diff_precompute_kernel(
    src: *const f32,
    dst: *mut f32,
    size: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let bias = DIFF_PRECOMPUTE_MUL * DIFF_PRECOMPUTE_BIAS;
    let val = *src.add(idx);

    *dst.add(idx) = (DIFF_PRECOMPUTE_MUL * val.abs() + bias).sqrt() - bias.sqrt();
}

/// Store minimum 3 values in sorted order
#[inline]
fn store_min3(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32) {
    if v < *min2 {
        if v < *min0 {
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min2 = *min1;
            *min1 = v;
        } else {
            *min2 = v;
        }
    }
}

/// Fuzzy erosion - morphological operation for mask refinement
/// Uses weighted average of 3 minimum values in 3x3 neighborhood (step 3)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fuzzy_erosion_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    let x = idx % width;
    let y = idx / width;

    let mut min0 = *src.add(idx);
    let mut min1 = 2.0 * min0;
    let mut min2 = min1;

    // Check 8 neighbors at distance EROSION_STEP
    if x >= EROSION_STEP {
        store_min3(
            *src.add(y * width + x - EROSION_STEP),
            &mut min0,
            &mut min1,
            &mut min2,
        );
        if y >= EROSION_STEP {
            store_min3(
                *src.add((y - EROSION_STEP) * width + x - EROSION_STEP),
                &mut min0,
                &mut min1,
                &mut min2,
            );
        }
        if y + EROSION_STEP < height {
            store_min3(
                *src.add((y + EROSION_STEP) * width + x - EROSION_STEP),
                &mut min0,
                &mut min1,
                &mut min2,
            );
        }
    }

    if x + EROSION_STEP < width {
        store_min3(
            *src.add(y * width + x + EROSION_STEP),
            &mut min0,
            &mut min1,
            &mut min2,
        );
        if y >= EROSION_STEP {
            store_min3(
                *src.add((y - EROSION_STEP) * width + x + EROSION_STEP),
                &mut min0,
                &mut min1,
                &mut min2,
            );
        }
        if y + EROSION_STEP < height {
            store_min3(
                *src.add((y + EROSION_STEP) * width + x + EROSION_STEP),
                &mut min0,
                &mut min1,
                &mut min2,
            );
        }
    }

    if y >= EROSION_STEP {
        store_min3(
            *src.add((y - EROSION_STEP) * width + x),
            &mut min0,
            &mut min1,
            &mut min2,
        );
    }

    if y + EROSION_STEP < height {
        store_min3(
            *src.add((y + EROSION_STEP) * width + x),
            &mut min0,
            &mut min1,
            &mut min2,
        );
    }

    *dst.add(idx) = 0.45 * min0 + 0.3 * min1 + 0.25 * min2;
}
