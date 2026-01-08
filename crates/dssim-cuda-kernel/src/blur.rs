//! 3x3 Gaussian blur kernel for DSSIM
//!
//! DSSIM uses a fixed 3x3 Gaussian blur kernel for computing local statistics.
//! This is much simpler than ssimulacra2's recursive Gaussian blur.
//!
//! The kernel is symmetric and normalized to sum to 1.0.

use nvptx_std::prelude::*;

// 3x3 Gaussian kernel (from dssim-core blur.rs)
// This is a discrete approximation of Gaussian with sigma ≈ 0.85
#[rustfmt::skip]
const KERNEL: [[f32; 3]; 3] = [
    [0.095332, 0.118095, 0.095332],
    [0.118095, 0.146293, 0.118095],
    [0.095332, 0.118095, 0.095332],
];

// For separable version (optional optimization):
// 1D kernel: [0.25, 0.5, 0.25] approximately
// But the actual kernel above is slightly different, so we use the 2D version

/// Apply 3x3 Gaussian blur to a single plane.
///
/// Uses edge clamping (replicate boundary pixels).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_3x3(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let mut sum = 0.0f32;

        // 3x3 convolution with edge clamping
        for ky in 0..3 {
            for kx in 0..3 {
                // Offset from center (-1, 0, +1)
                let ox = kx as isize - 1;
                let oy = ky as isize - 1;

                // Source coordinates with clamping
                let sx = ((x as isize + ox).max(0) as usize).min(width - 1);
                let sy = ((y as isize + oy).max(0) as usize).min(height - 1);

                let val = *src.byte_add(sy * src_pitch).add(sx);
                sum += val * KERNEL[ky][kx];
            }
        }

        *dst.byte_add(y * dst_pitch).add(x) = sum;
    }
}

/// Compute element-wise square: dst = src * src
///
/// Used before blurring to compute blur(x^2) for variance.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn square(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let val = *src.byte_add(y * src_pitch).add(x);
        *dst.byte_add(y * dst_pitch).add(x) = val * val;
    }
}

/// Compute element-wise product: dst = src1 * src2
///
/// Used to compute img1 * img2 before blurring for covariance.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multiply(
    src1: *const f32,
    src1_pitch: usize,
    src2: *const f32,
    src2_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let v1 = *src1.byte_add(y * src1_pitch).add(x);
        let v2 = *src2.byte_add(y * src2_pitch).add(x);
        *dst.byte_add(y * dst_pitch).add(x) = v1 * v2;
    }
}

/// Fused blur and square: dst = blur(src * src)
///
/// More efficient than separate square + blur as it reads src only once
/// and doesn't need intermediate storage.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_squared(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let mut sum = 0.0f32;

        for ky in 0..3 {
            for kx in 0..3 {
                let ox = kx as isize - 1;
                let oy = ky as isize - 1;

                let sx = ((x as isize + ox).max(0) as usize).min(width - 1);
                let sy = ((y as isize + oy).max(0) as usize).min(height - 1);

                let val = *src.byte_add(sy * src_pitch).add(sx);
                sum += val * val * KERNEL[ky][kx];
            }
        }

        *dst.byte_add(y * dst_pitch).add(x) = sum;
    }
}

/// Fused blur of product: dst = blur(src1 * src2)
///
/// Used for covariance: blur(img1 * img2)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_product(
    src1: *const f32,
    src1_pitch: usize,
    src2: *const f32,
    src2_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let mut sum = 0.0f32;

        for ky in 0..3 {
            for kx in 0..3 {
                let ox = kx as isize - 1;
                let oy = ky as isize - 1;

                let sx = ((x as isize + ox).max(0) as usize).min(width - 1);
                let sy = ((y as isize + oy).max(0) as usize).min(height - 1);

                let v1 = *src1.byte_add(sy * src1_pitch).add(sx);
                let v2 = *src2.byte_add(sy * src2_pitch).add(sx);
                sum += v1 * v2 * KERNEL[ky][kx];
            }
        }

        *dst.byte_add(y * dst_pitch).add(x) = sum;
    }
}
