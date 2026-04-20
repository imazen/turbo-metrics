//! Fused V-blur + feature-extraction kernel for zensim.
//!
//! Mirrors the CPU scalar path in `zen/zensim/zensim/src/fused.rs ::
//! fused_vblur_ssim_inner` (the scalar remainder loop). Constants and
//! math are bit-exact to the f32 scalar CPU form; FMA contraction may
//! cause sub-ULP differences vs the AVX2/AVX-512 SIMD paths.
//!
//! One thread per output column. Each thread walks y sequentially,
//! maintaining 4 running sums over a `diam=2*radius+1`-tap box V-blur
//! window, computing 14 feature accumulators + 3 peak maxes per
//! column. At the end of the column the accumulators are atomic-added
//! to 17 global f64 slots (+ 3 global u32 max slots for peaks).

use nvptx_std::prelude::*;

const C2: f32 = 0.0009;

/// Bit-pattern atomic max over non-negative f32 values. Same trick as
/// butteraugli's max_reduce: for float ≥ 0, `f.to_bits()` compares
/// identically to the float under unsigned integer ordering.
#[inline(always)]
unsafe fn atomic_max_nonneg_f32(ptr: *mut u32, value: f32) {
    let bits = value.to_bits();
    let mut _discard: u32;
    core::arch::asm!(
        "atom.global.max.u32 {d}, [{p}], {v};",
        d = out(reg32) _discard,
        p = in(reg64) ptr,
        v = in(reg32) bits,
        options(nostack, preserves_flags),
    );
}

/// Mirror-reflect index into `[0, height)` matching CPU's `mirror_idx`.
#[inline(always)]
unsafe fn mirror_y(i: isize, height: usize) -> usize {
    let h = height as isize;
    if h <= 1 {
        return 0;
    }
    let period = 2 * (h - 1);
    let m = ((i % period) + period) % period;
    if m < h {
        m as usize
    } else {
        (period - m) as usize
    }
}

/// Fused V-blur + SSIM/edge/HF/MSE feature extraction.
///
/// Grid: one block of up to 512 threads per X-tile. Each thread
/// processes a single column. There is no cross-column shared memory
/// — each thread atomic-adds its column's sums directly into the
/// 17 global f64 slots (+ 3 global u32 peak slots).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn fused_vblur_features_ssim_kernel(
    h_mu1: *const f32,
    h_mu2: *const f32,
    h_sigma_sq: *const f32,
    h_sigma12: *const f32,
    src: *const f32,
    dst: *const f32,
    pitch: usize, // bytes between rows — shared by all input planes
    width: usize,
    height: usize,
    radius: usize,
    accum_f64: *mut f64, // 17 slots, zero-initialised by caller
    peak_u32: *mut u32,  // 3 slots, zero-initialised by caller
) {
    let x = core::arch::nvptx::_block_idx_x() as usize * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    if x >= width {
        return;
    }

    let diam = 2 * radius + 1;
    let inv = 1.0_f32 / diam as f32;
    let r = radius as isize;

    // Initialise running sums from the mirrored prefix.
    let mut sum_m1 = 0.0_f32;
    let mut sum_m2 = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut sum_s12 = 0.0_f32;
    for k in 0..diam {
        let row_i = mirror_y((k as isize) - r, height);
        let row_m1 = h_mu1.byte_add(row_i * pitch);
        let row_m2 = h_mu2.byte_add(row_i * pitch);
        let row_sq = h_sigma_sq.byte_add(row_i * pitch);
        let row_s12 = h_sigma12.byte_add(row_i * pitch);
        sum_m1 += *row_m1.add(x);
        sum_m2 += *row_m2.add(x);
        sum_sq += *row_sq.add(x);
        sum_s12 += *row_s12.add(x);
    }

    // Per-thread accumulators.
    let mut a = [0.0_f64; 17];
    let mut peak = [0.0_f32; 3];

    for y in 0..height {
        let mu1 = sum_m1 * inv;
        let mu2 = sum_m2 * inv;
        let ssq = sum_sq * inv;
        let s12 = sum_s12 * inv;

        let row_src = src.byte_add(y * pitch);
        let row_dst = dst.byte_add(y * pitch);
        let sv = *row_src.add(x);
        let dv = *row_dst.add(x);

        // SSIM (ssimulacra2 variant: drops C1, uses `1 - (mu1-mu2)²`).
        let mu_diff = mu1 - mu2;
        let num_m = mu_diff.mul_add(-mu_diff, 1.0);
        let num_s = 2.0_f32.mul_add((-mu1).mul_add(mu2, s12), C2);
        let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
        let sd_raw = 1.0 - (num_m * num_s) / denom_s;
        let sd = if sd_raw > 0.0 { sd_raw } else { 0.0 };
        let sd2 = sd * sd;
        let sd4 = sd2 * sd2;
        a[0] += sd as f64;
        a[1] += sd4 as f64;
        a[2] += sd2 as f64;
        a[14] += (sd4 * sd4) as f64;
        if sd > peak[0] {
            peak[0] = sd;
        }

        // Edge artifact / detail-lost.
        let diff1 = (sv - mu1).abs();
        let diff2 = (dv - mu2).abs();
        let ed = (1.0 + diff2) / (1.0 + diff1) - 1.0;
        let artifact = if ed > 0.0 { ed } else { 0.0 };
        let detail_lost = if ed < 0.0 { -ed } else { 0.0 };
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        let a4 = a2 * a2;
        let dl4 = dl2 * dl2;
        a[3] += artifact as f64;
        a[4] += a4 as f64;
        a[5] += a2 as f64;
        a[6] += detail_lost as f64;
        a[7] += dl4 as f64;
        a[8] += dl2 as f64;
        a[15] += (a4 * a4) as f64;
        a[16] += (dl4 * dl4) as f64;
        if artifact > peak[1] {
            peak[1] = artifact;
        }
        if detail_lost > peak[2] {
            peak[2] = detail_lost;
        }

        // HF variance + texture magnitude.
        let vs = sv - mu1;
        let vd = dv - mu2;
        a[10] += (vs * vs) as f64;
        a[11] += (vd * vd) as f64;
        a[12] += diff1 as f64;
        a[13] += diff2 as f64;

        // MSE.
        let pd = sv - dv;
        a[9] += (pd * pd) as f64;

        // Slide V-blur window.
        let add_idx = mirror_y((y as isize) + r + 1, height);
        let rem_idx = mirror_y((y as isize) - r, height);
        let a_m1 = *h_mu1.byte_add(add_idx * pitch).add(x);
        let r_m1 = *h_mu1.byte_add(rem_idx * pitch).add(x);
        let a_m2 = *h_mu2.byte_add(add_idx * pitch).add(x);
        let r_m2 = *h_mu2.byte_add(rem_idx * pitch).add(x);
        let a_sq = *h_sigma_sq.byte_add(add_idx * pitch).add(x);
        let r_sq = *h_sigma_sq.byte_add(rem_idx * pitch).add(x);
        let a_s12 = *h_sigma12.byte_add(add_idx * pitch).add(x);
        let r_s12 = *h_sigma12.byte_add(rem_idx * pitch).add(x);
        sum_m1 = sum_m1 + a_m1 - r_m1;
        sum_m2 = sum_m2 + a_m2 - r_m2;
        sum_sq = sum_sq + a_sq - r_sq;
        sum_s12 = sum_s12 + a_s12 - r_s12;
    }

    // Atomic-add this column's totals into the 17 global accumulators.
    for i in 0..17 {
        atomic_add_global_f64(accum_f64.add(i), a[i]);
    }
    for i in 0..3 {
        atomic_max_nonneg_f32(peak_u32.add(i), peak[i]);
    }
}
