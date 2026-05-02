//! Visual masking + fuzzy erosion morphology.

use cubecl::prelude::*;

const COMBINE_MUL_X: f32 = 2.5;
const COMBINE_MUL_Y_UHF: f32 = 0.4;
const COMBINE_MUL_Y_HF: f32 = 0.4;
const DIFF_PRECOMPUTE_MUL: f32 = 6.194_240_8;
const DIFF_PRECOMPUTE_BIAS: f32 = 12.610_506;
const EROSION_STEP: usize = 3;
const MASK_TO_ERROR_MUL: f32 = 10.0;

/// `dst = sqrt(((uhf_x + hf_x) · 2.5)² + (uhf_y · 0.4 + hf_y · 0.4)²)`.
#[cube(launch_unchecked)]
pub fn combine_channels_for_masking_kernel(
    hf_x: &Array<f32>,
    uhf_x: &Array<f32>,
    hf_y: &Array<f32>,
    uhf_y: &Array<f32>,
    dst: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let xdiff = (uhf_x[idx] + hf_x[idx]) * COMBINE_MUL_X;
    let ydiff = uhf_y[idx] * COMBINE_MUL_Y_UHF + hf_y[idx] * COMBINE_MUL_Y_HF;
    dst[idx] = f32::sqrt(xdiff * xdiff + ydiff * ydiff);
}

/// Add the squared diff of blurred-UHF Y planes (× 10) into the AC-Y diff accumulator.
#[cube(launch_unchecked)]
pub fn mask_to_error_mul_kernel(
    blurred1: &Array<f32>,
    blurred2: &Array<f32>,
    block_diff_ac: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= block_diff_ac.len() {
        terminate!();
    }
    let diff = blurred1[idx] - blurred2[idx];
    block_diff_ac[idx] = block_diff_ac[idx] + MASK_TO_ERROR_MUL * diff * diff;
}

/// Batched mask_to_error_mul: `blurred1` is a single broadcast plane
/// (cached reference, indexed `idx % plane_stride`), `blurred2` and
/// `block_diff_ac` are `batch_size` planes packed contiguously.
#[cube(launch_unchecked)]
pub fn mask_to_error_mul_batched_kernel(
    blurred1: &Array<f32>,
    blurred2: &Array<f32>,
    block_diff_ac: &mut Array<f32>,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= block_diff_ac.len() {
        terminate!();
    }
    let local = idx - (idx / (plane_stride as usize)) * (plane_stride as usize);
    let diff = blurred1[local] - blurred2[idx];
    block_diff_ac[idx] = block_diff_ac[idx] + MASK_TO_ERROR_MUL * diff * diff;
}

/// `dst = sqrt(MUL · |x| + MUL·BIAS) − sqrt(MUL·BIAS)` — used to feed the masking blur.
#[cube(launch_unchecked)]
pub fn diff_precompute_kernel(src: &Array<f32>, dst: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let bias = DIFF_PRECOMPUTE_MUL * DIFF_PRECOMPUTE_BIAS;
    let val = src[idx];
    dst[idx] = f32::sqrt(DIFF_PRECOMPUTE_MUL * f32::abs(val) + bias) - f32::sqrt(bias);
}

/// Insertion-sort `v` into the front of a 3-element ascending min array.
/// Returns the updated `(min0, min1, min2)` tuple.
#[cube]
fn update_min3(v: f32, min0: f32, min1: f32, min2: f32) -> (f32, f32, f32) {
    let (m0, m1, m2) = (min0, min1, min2);
    if v < m2 {
        if v < m0 {
            (v, m0, m1)
        } else if v < m1 {
            (m0, v, m1)
        } else {
            (m0, m1, v)
        }
    } else {
        (m0, m1, m2)
    }
}

/// Fuzzy erosion morphology — weighted average of the 3 minimum values
/// in an 8-neighbour ring at distance EROSION_STEP. Output:
/// `0.45·min₀ + 0.3·min₁ + 0.25·min₂`.
#[cube(launch_unchecked)]
pub fn fuzzy_erosion_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let x = idx - (idx / w) * w;
    let y = idx / w;
    let step = EROSION_STEP;

    let center = src[idx];
    let mut m0 = center;
    let mut m1 = 2.0 * center;
    let mut m2 = m1;

    if x >= step {
        let v = src[y * w + (x - step)];
        let (a, b, c) = update_min3(v, m0, m1, m2);
        m0 = a;
        m1 = b;
        m2 = c;
        if y >= step {
            let v = src[(y - step) * w + (x - step)];
            let (a, b, c) = update_min3(v, m0, m1, m2);
            m0 = a;
            m1 = b;
            m2 = c;
        }
        if y + step < h {
            let v = src[(y + step) * w + (x - step)];
            let (a, b, c) = update_min3(v, m0, m1, m2);
            m0 = a;
            m1 = b;
            m2 = c;
        }
    }
    if x + step < w {
        let v = src[y * w + (x + step)];
        let (a, b, c) = update_min3(v, m0, m1, m2);
        m0 = a;
        m1 = b;
        m2 = c;
        if y >= step {
            let v = src[(y - step) * w + (x + step)];
            let (a, b, c) = update_min3(v, m0, m1, m2);
            m0 = a;
            m1 = b;
            m2 = c;
        }
        if y + step < h {
            let v = src[(y + step) * w + (x + step)];
            let (a, b, c) = update_min3(v, m0, m1, m2);
            m0 = a;
            m1 = b;
            m2 = c;
        }
    }
    if y >= step {
        let v = src[(y - step) * w + x];
        let (a, b, c) = update_min3(v, m0, m1, m2);
        m0 = a;
        m1 = b;
        m2 = c;
    }
    if y + step < h {
        let v = src[(y + step) * w + x];
        let (a, b, c) = update_min3(v, m0, m1, m2);
        m0 = a;
        m1 = b;
        m2 = c;
    }

    dst[idx] = 0.45 * m0 + 0.3 * m1 + 0.25 * m2;
}
