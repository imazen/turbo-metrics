//! Frequency separation kernels for Butteraugli
//!
//! Separates XYB image into low, medium, high, and ultra-high frequency bands.

use nvptx_std::prelude::*;

// Constants from Vship implementation
const XMULI: f32 = 33.832837186260;
const YMULI: f32 = 14.458268100570;
const BMULI: f32 = 49.87984651440;
const Y_TO_B_MULI: f32 = -0.362267051518;

const KMAXCLAMP_HF: f32 = 28.4691806922;
const KMAXCLAMP_UHF: f32 = 5.19175294647;
const UHF_MUL: f32 = 2.69313763794;
const HF_MUL: f32 = 2.155;
const HF_AMPLIFY: f32 = 0.132;

const SUPRESS_S: f32 = 0.653020556257;

/// Convert XYB low-frequency values to weighted values
#[no_mangle]
pub unsafe extern "ptx-kernel" fn xyb_low_freq_to_vals_kernel(
    x_plane: *mut f32,
    y_plane: *mut f32,
    b_plane: *mut f32,
    size: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let x = *x_plane.add(idx);
    let y = *y_plane.add(idx);
    let mut b = *b_plane.add(idx);

    // Apply cross-channel mixing
    b = b + Y_TO_B_MULI * y;
    b = b * BMULI;

    *x_plane.add(idx) = x * XMULI;
    *y_plane.add(idx) = y * YMULI;
    *b_plane.add(idx) = b;
}

/// Subtract arrays: second[i] -= first[i]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn subtract_arrays_kernel(
    first: *const f32,
    second: *mut f32,
    size: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    *second.add(idx) = *second.add(idx) - *first.add(idx);
}

/// Remove range around zero: values close to zero become zero
#[inline]
fn remove_range_around_zero(x: f32, w: f32) -> f32 {
    if x > w {
        x - w
    } else if x < -w {
        x + w
    } else {
        0.0
    }
}

/// Amplify range around zero: values close to zero are doubled
#[inline]
fn amplify_range_around_zero(x: f32, w: f32) -> f32 {
    if x > w {
        x + w
    } else if x < -w {
        x - w
    } else {
        2.0 * x
    }
}

/// Maximum clamp: soft limit values above maxval
#[inline]
fn maximum_clamp(v: f32, maxval: f32) -> f32 {
    const KMUL: f32 = 0.688059627878;
    if v >= maxval {
        (v - maxval) * KMUL + maxval
    } else if v < -maxval {
        (v + maxval) * KMUL - maxval
    } else {
        v
    }
}

/// Subtract and remove range around zero
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sub_remove_range_kernel(
    first: *mut f32,
    second: *mut f32,
    size: usize,
    w: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let f = *first.add(idx);
    *second.add(idx) = *second.add(idx) - f;
    *first.add(idx) = remove_range_around_zero(f, w);
}

/// Subtract and amplify range around zero
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sub_amplify_range_kernel(
    first: *mut f32,
    second: *mut f32,
    size: usize,
    w: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let f = *first.add(idx);
    *second.add(idx) = *second.add(idx) - f;
    *first.add(idx) = amplify_range_around_zero(f, w);
}

/// Suppress X by Y: cross-channel masking
#[no_mangle]
pub unsafe extern "ptx-kernel" fn suppress_x_by_y_kernel(
    x: *mut f32,
    y: *const f32,
    size: usize,
    yw: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let yval = *y.add(idx);
    let scaler = SUPRESS_S + (yw * (1.0 - SUPRESS_S)) / (yw + yval * yval);
    *x.add(idx) = *x.add(idx) * scaler;
}

/// Separate HF from UHF
#[no_mangle]
pub unsafe extern "ptx-kernel" fn separate_hf_uhf_kernel(hf: *mut f32, uhf: *mut f32, size: usize) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    let mut h = *hf.add(idx);
    let mut u = *uhf.add(idx);

    h = maximum_clamp(h, KMAXCLAMP_HF);
    u = u - h;
    u = maximum_clamp(u, KMAXCLAMP_UHF);
    u = u * UHF_MUL;
    h = h * HF_MUL;
    h = amplify_range_around_zero(h, HF_AMPLIFY);

    *hf.add(idx) = h;
    *uhf.add(idx) = u;
}

/// Remove range around zero (standalone)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn remove_range_kernel(arr: *mut f32, size: usize, w: f32) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= size {
        return;
    }

    *arr.add(idx) = remove_range_around_zero(*arr.add(idx), w);
}
