//! Linear RGB → custom-scaled Lab conversion (planar input, planar output).
//!
//! Pointwise translation of `dssim-cuda-kernel/src/lab.rs`. The Lab
//! variant matches dssim-core's `tolab.rs`:
//! 1. Linear RGB → XYZ via the sRGB / D65 matrix.
//! 2. XYZ / D65-white → cube-root nonlinearity (linear toe under
//!    EPSILON ≈ 0.008856).
//! 3. Pack into custom 0-1-ish scaled L, a, b (luminance is multiplied
//!    by 1.05 to weight it higher; chroma components are normalised by
//!    220 with a positive offset so they stay non-negative).
//!
//! cubecl 0.10 has no `f32::cbrt`; we substitute `f32::powf(_, 1.0/3.0)`
//! the same way `ssim2-gpu` did. At byte-precision inputs the
//! difference vs the published `cbrt` is below 1 ulp; verified by the
//! parity tests against `dssim-core`.

use cubecl::prelude::*;

// D65 illuminant reference white.
const D65_X: f32 = 0.9505;
const D65_Y: f32 = 1.0;
const D65_Z: f32 = 1.089;

// Cube-root threshold and linear-toe slope from CIE.
const EPSILON: f32 = 216.0 / 24389.0; // ≈ 0.008856
const K_TOE: f32 = 24389.0 / (27.0 * 116.0); // ≈ 7.787

// sRGB → XYZ (D65) row matrix.
const M_R_X: f32 = 0.4124;
const M_G_X: f32 = 0.3576;
const M_B_X: f32 = 0.1805;
const M_R_Y: f32 = 0.2126;
const M_G_Y: f32 = 0.7152;
const M_B_Y: f32 = 0.0722;
const M_R_Z: f32 = 0.0193;
const M_G_Z: f32 = 0.1192;
const M_B_Z: f32 = 0.9505;

/// Take three planar linear-RGB f32 buffers and write three planar
/// custom-Lab f32 buffers.
#[cube(launch_unchecked)]
pub fn linear_to_lab_planar_kernel(
    src_r: &Array<f32>,
    src_g: &Array<f32>,
    src_b: &Array<f32>,
    dst_l: &mut Array<f32>,
    dst_a: &mut Array<f32>,
    dst_b_chan: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = dst_l.len();
    if idx >= n {
        terminate!();
    }
    let r = src_r[idx];
    let g = src_g[idx];
    let b = src_b[idx];

    // RGB → XYZ, then divide by white-point.
    let x = (M_R_X * r + M_G_X * g + M_B_X * b) / D65_X;
    let y = (M_R_Y * r + M_G_Y * g + M_B_Y * b) / D65_Y;
    let z = (M_R_Z * r + M_G_Z * g + M_B_Z * b) / D65_Z;

    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);

    // dssim-core's custom scaling:
    //   L = (fy - 16/116) * 1.05
    //   a = (fx - fy) * (500/220) + 86.2/220
    //   b = (fy - fz) * (200/220) + 107.9/220
    let l = (fy - 16.0 / 116.0) * 1.05;
    let a = (fx - fy) * (500.0 / 220.0) + 86.2 / 220.0;
    let b_out = (fy - fz) * (200.0 / 220.0) + 107.9 / 220.0;

    dst_l[idx] = l;
    dst_a[idx] = a;
    dst_b_chan[idx] = b_out;
}

/// Lab cube-root with linear toe. Substitutes `powf(_, 1/3)` for
/// `cbrt` (cubecl 0.10 has no f32 cbrt op — same workaround as
/// ssim2-gpu).
#[cube]
fn lab_f(t: f32) -> f32 {
    if t > EPSILON {
        f32::powf(t, 1.0 / 3.0)
    } else {
        K_TOE * t + 16.0 / 116.0
    }
}
