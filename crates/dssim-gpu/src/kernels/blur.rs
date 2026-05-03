//! 3×3 fixed Gaussian convolution with edge clamping.
//!
//! DSSIM uses a tiny fixed-coefficient kernel (sigma ≈ 0.85) applied
//! as a 9-tap 2-D convolution. dssim-core applies the same filter
//! twice in succession to approximate a wider effective Gaussian — we
//! match that by launching `blur_3x3` twice in the host-side
//! orchestration.
//!
//! Three kernel variants:
//! - `blur_3x3_kernel`        — `dst = blur(src)`
//! - `blur_squared_kernel`    — `dst = blur(src²)` (one pass; the
//!   pipeline runs another `blur_3x3` over the result for the second pass)
//! - `blur_product_kernel`    — `dst = blur(src1 · src2)` (same pattern)
//!
//! All three use clamp-to-edge at the image boundary (replicate the
//! nearest valid pixel), matching `dssim-core/src/blur.rs`.

use cubecl::prelude::*;

// 3×3 Gaussian kernel coefficients — verbatim from `dssim-core` /
// `dssim-cuda-kernel/src/blur.rs`. Sums to 1 by construction.
const K00: f32 = 0.095_332;
const K01: f32 = 0.118_095;
const K11: f32 = 0.146_293;
// Diagonal symmetry: K00 = K02 = K20 = K22, K01 = K10 = K12 = K21.

#[cube(launch_unchecked)]
pub fn blur_3x3_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let y_us = idx / w;
    let x_us = idx - y_us * w;
    dst[idx] = sample_blur(src, x_us as u32, y_us as u32, width, height);
}

#[cube(launch_unchecked)]
pub fn blur_squared_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let y_us = idx / w;
    let x_us = idx - y_us * w;
    dst[idx] = sample_blur_squared(src, x_us as u32, y_us as u32, width, height);
}

#[cube(launch_unchecked)]
pub fn blur_product_kernel(
    src1: &Array<f32>,
    src2: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let y_us = idx / w;
    let x_us = idx - y_us * w;
    dst[idx] = sample_blur_product(src1, src2, x_us as u32, y_us as u32, width, height);
}

/// 9-tap convolution with replicate-clamp at the boundary.
#[cube]
fn sample_blur(src: &Array<f32>, x: u32, y: u32, width: u32, height: u32) -> f32 {
    let w = width as usize;
    let lx = u32::saturating_sub(x, 1) as usize;
    let cx = x as usize;
    let rx = u32::min(x + 1, width - 1) as usize;
    let ty = u32::saturating_sub(y, 1) as usize;
    let cy = y as usize;
    let by = u32::min(y + 1, height - 1) as usize;

    let p00 = src[ty * w + lx];
    let p01 = src[ty * w + cx];
    let p02 = src[ty * w + rx];
    let p10 = src[cy * w + lx];
    let p11 = src[cy * w + cx];
    let p12 = src[cy * w + rx];
    let p20 = src[by * w + lx];
    let p21 = src[by * w + cx];
    let p22 = src[by * w + rx];

    (p00 + p02 + p20 + p22) * K00 + (p01 + p10 + p12 + p21) * K01 + p11 * K11
}

/// `blur(src²)` — same kernel layout, squares each tap before
/// convolving.
#[cube]
fn sample_blur_squared(src: &Array<f32>, x: u32, y: u32, width: u32, height: u32) -> f32 {
    let w = width as usize;
    let lx = u32::saturating_sub(x, 1) as usize;
    let cx = x as usize;
    let rx = u32::min(x + 1, width - 1) as usize;
    let ty = u32::saturating_sub(y, 1) as usize;
    let cy = y as usize;
    let by = u32::min(y + 1, height - 1) as usize;

    let p00 = src[ty * w + lx];
    let p01 = src[ty * w + cx];
    let p02 = src[ty * w + rx];
    let p10 = src[cy * w + lx];
    let p11 = src[cy * w + cx];
    let p12 = src[cy * w + rx];
    let p20 = src[by * w + lx];
    let p21 = src[by * w + cx];
    let p22 = src[by * w + rx];

    (p00 * p00 + p02 * p02 + p20 * p20 + p22 * p22) * K00
        + (p01 * p01 + p10 * p10 + p12 * p12 + p21 * p21) * K01
        + p11 * p11 * K11
}

/// `blur(src1 · src2)` — fused for covariance computation.
#[cube]
fn sample_blur_product(
    src1: &Array<f32>,
    src2: &Array<f32>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> f32 {
    let w = width as usize;
    let lx = u32::saturating_sub(x, 1) as usize;
    let cx = x as usize;
    let rx = u32::min(x + 1, width - 1) as usize;
    let ty = u32::saturating_sub(y, 1) as usize;
    let cy = y as usize;
    let by = u32::min(y + 1, height - 1) as usize;

    let a00 = src1[ty * w + lx];
    let a01 = src1[ty * w + cx];
    let a02 = src1[ty * w + rx];
    let a10 = src1[cy * w + lx];
    let a11 = src1[cy * w + cx];
    let a12 = src1[cy * w + rx];
    let a20 = src1[by * w + lx];
    let a21 = src1[by * w + cx];
    let a22 = src1[by * w + rx];

    let b00 = src2[ty * w + lx];
    let b01 = src2[ty * w + cx];
    let b02 = src2[ty * w + rx];
    let b10 = src2[cy * w + lx];
    let b11 = src2[cy * w + cx];
    let b12 = src2[cy * w + rx];
    let b20 = src2[by * w + lx];
    let b21 = src2[by * w + cx];
    let b22 = src2[by * w + rx];

    (a00 * b00 + a02 * b02 + a20 * b20 + a22 * b22) * K00
        + (a01 * b01 + a10 * b10 + a12 * b12 + a21 * b21) * K01
        + a11 * b11 * K11
}
