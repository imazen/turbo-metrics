//! Box blur kernels for zensim.
//!
//! zensim's CPU path uses a single-pass fused H-blur that outputs 4
//! planes from 2 inputs (src, dst): `h_mu1`, `h_mu2`, `h_sigma_sq`,
//! `h_sigma12`. Each is a horizontal running-sum box blur at `diam =
//! 2*radius + 1` taps. This kernel mirrors that with mirrored boundary.
//!
//! Radius is a runtime parameter. The CPU path commonly uses radius=3
//! at scale 0 (diam=7) with `blur_passes=1` — matching here.

use nvptx_std::math::StdMathExt;

/// Box running-sum H-blur producing 4 outputs per pixel.
///
/// For each pixel (x, y):
///   h_mu1[x,y]      = sum_k src[x+k-r, y]                 / diam
///   h_mu2[x,y]      = sum_k dst[x+k-r, y]                 / diam
///   h_sigma_sq[x,y] = sum_k (src[x+k-r,y]² + dst[x+k-r,y]²) / diam
///   h_sigma12[x,y]  = sum_k src[x+k-r,y] * dst[x+k-r,y]   / diam
///
/// NB: `h_sigma_sq` carries the sum of *both* E[s²]+E[d²] so the V-blur
/// SSIM denominator `ssq - mu1² - mu2² + C2 = var_s + var_d + C2` is
/// non-negative. Accumulating only `s*s` here makes denom_s negative
/// whenever mu2 is dominant, exploding the ratio.
///
/// Boundary uses mirror reflection (CPU zensim convention).
/// One thread per (x, y). Grid = ceil(width / 32) × ceil(height / 8).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fused_blur_h_ssim_kernel(
    src: *const f32,
    dst: *const f32,
    src_pitch: usize,
    h_mu1: *mut f32,
    h_mu2: *mut f32,
    h_sigma_sq: *mut f32,
    h_sigma12: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
    radius: usize,
) {
    let x = core::arch::nvptx::_block_idx_x() as usize * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let y = core::arch::nvptx::_block_idx_y() as usize * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize;
    if x >= width || y >= height {
        return;
    }

    let diam = 2 * radius + 1;
    let inv = 1.0_f32 / diam as f32;
    let src_row = src.byte_add(y * src_pitch);
    let dst_row = dst.byte_add(y * src_pitch);

    // Mirror indexing helper. Matches zensim's `mirror_idx`.
    // period = 2 * (width - 1); clamps to [0, width).
    let mirror = |i: isize| -> usize {
        let w = width as isize;
        if w <= 1 {
            return 0;
        }
        let period = 2 * (w - 1);
        let m = ((i % period) + period) % period;
        if m < w {
            m as usize
        } else {
            (period - m) as usize
        }
    };

    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;
    for k in 0..diam {
        let ix = mirror((x as isize) + (k as isize) - (radius as isize));
        let s = *src_row.add(ix);
        let d = *dst_row.add(ix);
        sum_m1 += s;
        sum_m2 += d;
        // Sum of squared *both* planes — matches CPU
        // `fused_blur_h_ssim_inner` (zen/zensim/src/blur.rs line 1765):
        //   sum_sq = s.mul_add(s, d.mul_add(d, sum_sq))
        sum_sq = sum_sq + s * s + d * d;
        sum_s12 = sum_s12 + s * d;
    }

    let row_m1 = h_mu1.byte_add(y * dst_pitch);
    let row_m2 = h_mu2.byte_add(y * dst_pitch);
    let row_sq = h_sigma_sq.byte_add(y * dst_pitch);
    let row_s12 = h_sigma12.byte_add(y * dst_pitch);
    *row_m1.add(x) = sum_m1 * inv;
    *row_m2.add(x) = sum_m2 * inv;
    *row_sq.add(x) = sum_sq * inv;
    *row_s12.add(x) = sum_s12 * inv;
}
