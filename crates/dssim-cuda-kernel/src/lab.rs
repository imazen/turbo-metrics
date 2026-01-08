//! Linear RGB to LAB color space conversion
//!
//! Implements the dssim-core LAB conversion algorithm:
//! 1. Linear RGB → XYZ (D65 illuminant)
//! 2. XYZ → Lab (cube root transform)
//! 3. Scale to 0-1 range with custom offsets
//!
//! The output LAB values are scaled differently from standard LAB:
//! - L is boosted by 1.05 to increase luminance importance
//! - a and b are shifted to stay positive (0-1 range)

use nvptx_std::prelude::*;

// D65 illuminant reference white
const D65_X: f32 = 0.9505;
const D65_Y: f32 = 1.0;
const D65_Z: f32 = 1.089;

// XYZ cube root threshold (from CIE standard)
const EPSILON: f32 = 216.0 / 24389.0; // ~0.008856
const K: f32 = 24389.0 / (27.0 * 116.0); // ~7.787

// RGB to XYZ matrix (sRGB/D65)
const M_R_X: f32 = 0.4124;
const M_G_X: f32 = 0.3576;
const M_B_X: f32 = 0.1805;

const M_R_Y: f32 = 0.2126;
const M_G_Y: f32 = 0.7152;
const M_B_Y: f32 = 0.0722;

const M_R_Z: f32 = 0.0193;
const M_G_Z: f32 = 0.1192;
const M_B_Z: f32 = 0.9505;

/// Apply the Lab nonlinear transform (cube root with linear fallback)
#[inline]
unsafe fn lab_f(t: f32) -> f32 {
    if t > EPSILON {
        t.cbrt()
    } else {
        K.mul_add(t, 16.0 / 116.0)
    }
}

/// Convert a single pixel from linear RGB to LAB.
///
/// Returns (L, a, b) in dssim's custom 0-1 scaled format.
#[inline]
unsafe fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // RGB to XYZ
    let x = M_R_X.mul_add(r, M_G_X.mul_add(g, M_B_X * b)) / D65_X;
    let y = M_R_Y.mul_add(r, M_G_Y.mul_add(g, M_B_Y * b)) / D65_Y;
    let z = M_R_Z.mul_add(r, M_G_Z.mul_add(g, M_B_Z * b)) / D65_Z;

    // XYZ to Lab (using f function)
    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);

    // Standard Lab formula
    // L* = 116 * fy - 16
    // a* = 500 * (fx - fy)
    // b* = 200 * (fy - fz)

    // dssim's custom scaling to keep values in ~0-1 range:
    // L = (116 * fy - 16) / 100 * 1.05 ≈ fy * 1.05 (roughly)
    // Actually dssim uses: L = Y * 1.05 where Y = fy - 16/116
    // a = (500/220) * (fx - fy) + 86.2/220
    // b = (200/220) * (fy - fz) + 107.9/220

    // Simplified from dssim-core tolab.rs:
    let l_raw = fy - 16.0 / 116.0;
    let l = l_raw * 1.05;

    let a = (fx - fy).mul_add(500.0 / 220.0, 86.2 / 220.0);
    let b_out = (fy - fz).mul_add(200.0 / 220.0, 107.9 / 220.0);

    (l, a, b_out)
}

/// Convert packed linear RGB to planar LAB.
///
/// Input: Packed RGB (RGBRGBRGB...) in linear light
/// Output: Three separate planes (L, a, b)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_lab_planar(
    src: *const f32,
    src_pitch: usize,
    dst_l: *mut f32,
    dst_a: *mut f32,
    dst_b: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (col, row) = coords_2d();

    if col < width && row < height {
        let src_row = src.byte_add(row * src_pitch);
        let r = *src_row.add(col * 3);
        let g = *src_row.add(col * 3 + 1);
        let b = *src_row.add(col * 3 + 2);

        let (l, a, b_out) = rgb_to_lab(r, g, b);

        let dst_offset = row * dst_pitch;
        *dst_l.byte_add(dst_offset).add(col) = l;
        *dst_a.byte_add(dst_offset).add(col) = a;
        *dst_b.byte_add(dst_offset).add(col) = b_out;
    }
}

/// Convert packed linear RGB to packed LAB (interleaved).
///
/// Useful when we want to keep data packed for memory efficiency.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_lab_packed(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (col, row) = coords_2d();

    if col < width && row < height {
        let src_row = src.byte_add(row * src_pitch);
        let r = *src_row.add(col * 3);
        let g = *src_row.add(col * 3 + 1);
        let b = *src_row.add(col * 3 + 2);

        let (l, a, b_out) = rgb_to_lab(r, g, b);

        let dst_row = dst.byte_add(row * dst_pitch);
        *dst_row.add(col * 3) = l;
        *dst_row.add(col * 3 + 1) = a;
        *dst_row.add(col * 3 + 2) = b_out;
    }
}
