//! Per-pixel SSIM map and MAD-step kernels.
//!
//! `ssim_lab_kernel` fuses the dssim-core `compute_ssim_lab` reference:
//! 15 inputs (5 statistics × 3 channels), the Lab terms are averaged
//! across channels, then plugged into the standard SSIM formula. One
//! output plane.
//!
//! `abs_diff_scalar_kernel` is the per-scale MAD step — `|src - scalar|`
//! pointwise, used after computing the per-scale mean SSIM on the host.

use cubecl::prelude::*;

// SSIM constants (`(0.01)²`, `(0.03)²`) — verbatim from dssim-core.
const C1: f32 = 0.0001;
const C2: f32 = 0.0009;

#[cube(launch_unchecked)]
pub fn ssim_lab_kernel(
    mu1_l: &Array<f32>,
    mu1_a: &Array<f32>,
    mu1_b: &Array<f32>,
    mu2_l: &Array<f32>,
    mu2_a: &Array<f32>,
    mu2_b: &Array<f32>,
    sq1_l: &Array<f32>,
    sq1_a: &Array<f32>,
    sq1_b: &Array<f32>,
    sq2_l: &Array<f32>,
    sq2_a: &Array<f32>,
    sq2_b: &Array<f32>,
    cross_l: &Array<f32>,
    cross_a: &Array<f32>,
    cross_b: &Array<f32>,
    ssim_out: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = ssim_out.len();
    if idx >= n {
        terminate!();
    }

    let m1l = mu1_l[idx];
    let m1a = mu1_a[idx];
    let m1b = mu1_b[idx];
    let m2l = mu2_l[idx];
    let m2a = mu2_a[idx];
    let m2b = mu2_b[idx];
    let s1l = sq1_l[idx];
    let s1a = sq1_a[idx];
    let s1b = sq1_b[idx];
    let s2l = sq2_l[idx];
    let s2a = sq2_a[idx];
    let s2b = sq2_b[idx];
    let cl = cross_l[idx];
    let ca = cross_a[idx];
    let cb = cross_b[idx];

    // Average mu products across L/a/b (matches dssim-core).
    let mu1_sq = (m1l * m1l + m1a * m1a + m1b * m1b) * (1.0 / 3.0);
    let mu2_sq = (m2l * m2l + m2a * m2a + m2b * m2b) * (1.0 / 3.0);
    let mu1_mu2 = (m1l * m2l + m1a * m2a + m1b * m2b) * (1.0 / 3.0);

    // Average blur(img²) and blur(img1·img2) across channels.
    let img1_sq_blur = (s1l + s1a + s1b) * (1.0 / 3.0);
    let img2_sq_blur = (s2l + s2a + s2b) * (1.0 / 3.0);
    let img12_blur = (cl + ca + cb) * (1.0 / 3.0);

    let sigma1_sq = img1_sq_blur - mu1_sq;
    let sigma2_sq = img2_sq_blur - mu2_sq;
    let sigma12 = img12_blur - mu1_mu2;

    let num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2);
    let den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);

    ssim_out[idx] = num / den;
}

/// Pointwise `|src[i] - scalar|`. Output buffer can be the same as
/// `src` only if the runtime tolerates aliasing — we always pass a
/// distinct scratch in the pipeline.
#[cube(launch_unchecked)]
pub fn abs_diff_scalar_kernel(src: &Array<f32>, dst: &mut Array<f32>, scalar: f32) {
    let idx = ABSOLUTE_POS;
    let n = dst.len();
    if idx >= n {
        terminate!();
    }
    let v = src[idx];
    let d = scalar - v;
    dst[idx] = if d < 0.0 { -d } else { d };
}
