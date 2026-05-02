//! SSIMULACRA2 per-pixel error maps.
//!
//! Pointwise kernel; 7 inputs in, 3 outputs out (all single-plane f32):
//!
//! - **ssim** =  `max(0, 1 − num_m · num_s / denom_s)`
//!   - `num_m  = 1 − (mu1 − mu2)²`
//!   - `num_s  = 2·(sigma12 − mu1·mu2) + C2`
//!   - `denom_s = (sigma11 − mu1²) + (sigma22 − mu2²) + C2`
//! - **artifact**     = `max(0,  d1)`
//! - **detail_loss**  = `max(0, −d1)`, where
//!   `d1 = (1 + |dist − mu2|) / (1 + |src − mu1|) − 1`
//!
//! Matches `ssimulacra2::ssim_map` and `edge_diff_map` (both pointwise,
//! same constants, no boundary handling needed). Verbatim from
//! `ssimulacra2-cuda-kernel/src/error_maps.rs`.
//!
//! C2 = 0.0009.

use cubecl::prelude::*;

const C2: f32 = 0.0009;

/// Pointwise broadcast multiply: `out = a_broadcast · b_batched`.
///
/// `a` is a single plane (length = `plane_stride`) shared across the
/// batch; `b` and `out` are batched buffers (length =
/// `plane_stride · batch_size`). Used by `Ssim2Batch::compute_batch`
/// to compute `sigma12 = ref_xyb · dis_xyb_batched`.
#[cube(launch_unchecked)]
pub fn pointwise_mul_broadcast_batched_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<f32>,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out.len() {
        terminate!();
    }
    let pl = plane_stride as usize;
    let local = idx - (idx / pl) * pl;
    out[idx] = a[local] * b[idx];
}

/// Compute the three SSIMULACRA2 per-pixel error maps for one channel.
/// All input/output buffers are single-plane f32 of length `n_pixels`.
#[cube(launch_unchecked)]
pub fn error_maps_kernel(
    source: &Array<f32>,
    distorted: &Array<f32>,
    mu1: &Array<f32>,
    mu2: &Array<f32>,
    sigma11: &Array<f32>,
    sigma22: &Array<f32>,
    sigma12: &Array<f32>,
    out_ssim: &mut Array<f32>,
    out_artifact: &mut Array<f32>,
    out_detail: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = out_ssim.len();
    if idx >= n {
        terminate!();
    }
    let m1 = mu1[idx];
    let m2 = mu2[idx];
    let s11 = sigma11[idx];
    let s22 = sigma22[idx];
    let s12 = sigma12[idx];
    let src = source[idx];
    let dis = distorted[idx];

    let mu11 = m1 * m1;
    let mu22 = m2 * m2;
    let mu12 = m1 * m2;
    let mu_diff = m1 - m2;
    let num_m = 1.0 - mu_diff * mu_diff;
    let num_s = 2.0 * (s12 - mu12) + C2;
    let denom_s = (s11 - mu11) + (s22 - mu22) + C2;
    let mut d_ssim = 1.0 - (num_m * num_s) / denom_s;
    if d_ssim < 0.0 {
        d_ssim = 0.0;
    }
    out_ssim[idx] = d_ssim;

    let denom = 1.0 / (1.0 + f32::abs(src - m1));
    let numer = 1.0 + f32::abs(dis - m2);
    let d1 = numer * denom - 1.0;

    let art = if d1 > 0.0 { d1 } else { f32::new(0.0) };
    let dl = if d1 < 0.0 { -d1 } else { f32::new(0.0) };
    out_artifact[idx] = art;
    out_detail[idx] = dl;
}

/// Broadcast-batched error_maps for `Ssim2Batch`.
///
/// Reference-side inputs (`source`, `mu1`, `sigma11`) are single
/// planes shared across the batch and indexed at `idx % plane_stride`.
/// Distorted-side inputs (`distorted`, `mu2`, `sigma22`, `sigma12`)
/// and the three outputs are per-image batched buffers indexed at
/// `idx`. Each plane is `plane_stride` floats; the kernel processes
/// `plane_stride · batch_size` total pixels.
#[cube(launch_unchecked)]
pub fn error_maps_broadcast_batched_kernel(
    source: &Array<f32>,
    distorted: &Array<f32>,
    mu1: &Array<f32>,
    mu2: &Array<f32>,
    sigma11: &Array<f32>,
    sigma22: &Array<f32>,
    sigma12: &Array<f32>,
    out_ssim: &mut Array<f32>,
    out_artifact: &mut Array<f32>,
    out_detail: &mut Array<f32>,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = out_ssim.len();
    if idx >= n {
        terminate!();
    }
    let pl = plane_stride as usize;
    let local = idx - (idx / pl) * pl;
    let m1 = mu1[local];
    let m2 = mu2[idx];
    let s11 = sigma11[local];
    let s22 = sigma22[idx];
    let s12 = sigma12[idx];
    let src = source[local];
    let dis = distorted[idx];

    let mu11 = m1 * m1;
    let mu22 = m2 * m2;
    let mu12 = m1 * m2;
    let mu_diff = m1 - m2;
    let num_m = 1.0 - mu_diff * mu_diff;
    let num_s = 2.0 * (s12 - mu12) + C2;
    let denom_s = (s11 - mu11) + (s22 - mu22) + C2;
    let mut d_ssim = 1.0 - (num_m * num_s) / denom_s;
    if d_ssim < 0.0 {
        d_ssim = 0.0;
    }
    out_ssim[idx] = d_ssim;

    let denom = 1.0 / (1.0 + f32::abs(src - m1));
    let numer = 1.0 + f32::abs(dis - m2);
    let d1 = numer * denom - 1.0;

    let art = if d1 > 0.0 { d1 } else { f32::new(0.0) };
    let dl = if d1 < 0.0 { -d1 } else { f32::new(0.0) };
    out_artifact[idx] = art;
    out_detail[idx] = dl;
}
