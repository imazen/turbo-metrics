//! Frequency-band separation: pointwise XYB ↔ frequency-weighted ops.
//!
//! Translated from `butteraugli-cuda-kernel/src/frequency.rs`. All
//! kernels are pure pointwise (no cross-thread comm).

use cubecl::prelude::*;

const XMULI: f32 = 33.832_837;
const YMULI: f32 = 14.458_268;
const BMULI: f32 = 49.879_845;
const Y_TO_B_MULI: f32 = -0.362_267_05;

const KMAXCLAMP_HF: f32 = 28.469_181;
const KMAXCLAMP_UHF: f32 = 5.191_753;
const UHF_MUL: f32 = 2.693_137_6;
const HF_MUL: f32 = 2.155;
const HF_AMPLIFY: f32 = 0.132;

const SUPRESS_S: f32 = 0.653_020_56;

/// Soft clamp around `±maxval`. Matches CPU butteraugli `psycho.rs::maximum_clamp`.
#[cube]
fn maximum_clamp(v: f32, maxval: f32) -> f32 {
    const KMUL: f32 = 0.724_216_15;
    if v >= maxval {
        (v - maxval) * KMUL + maxval
    } else if v < -maxval {
        (v + maxval) * KMUL - maxval
    } else {
        v
    }
}

/// Drop values within `±w` to zero, shifting by `w` outside that band.
#[cube]
fn remove_range_around_zero(x: f32, w: f32) -> f32 {
    if x > w {
        x - w
    } else if x < -w {
        x + w
    } else {
        f32::new(0.0)
    }
}

/// Push values within `±w` outward by `w` (shift inside doubles their magnitude).
#[cube]
fn amplify_range_around_zero(x: f32, w: f32) -> f32 {
    if x > w {
        x + w
    } else if x < -w {
        x - w
    } else {
        2.0 * x
    }
}

/// Apply the XYB low-frequency multipliers (X, Y, B mixed) in place.
#[cube(launch_unchecked)]
pub fn xyb_low_freq_to_vals_kernel(
    x_plane: &mut Array<f32>,
    y_plane: &mut Array<f32>,
    b_plane: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= x_plane.len() {
        terminate!();
    }
    let x = x_plane[idx];
    let y = y_plane[idx];
    let mut b = b_plane[idx];
    b = b + Y_TO_B_MULI * y;
    b = b * BMULI;
    x_plane[idx] = x * XMULI;
    y_plane[idx] = y * YMULI;
    b_plane[idx] = b;
}

/// `dst = src1 - src2` element-wise.
#[cube(launch_unchecked)]
pub fn subtract_arrays_kernel(src1: &Array<f32>, src2: &Array<f32>, dst: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    dst[idx] = src1[idx] - src2[idx];
}

/// `second -= first; first = remove_range_around_zero(first, w)`.
#[cube(launch_unchecked)]
pub fn sub_remove_range_kernel(first: &mut Array<f32>, second: &mut Array<f32>, w: f32) {
    let idx = ABSOLUTE_POS;
    if idx >= first.len() {
        terminate!();
    }
    let f = first[idx];
    second[idx] = second[idx] - f;
    first[idx] = remove_range_around_zero(f, w);
}

/// `second -= first; first = amplify_range_around_zero(first, w)`.
#[cube(launch_unchecked)]
pub fn sub_amplify_range_kernel(first: &mut Array<f32>, second: &mut Array<f32>, w: f32) {
    let idx = ABSOLUTE_POS;
    if idx >= first.len() {
        terminate!();
    }
    let f = first[idx];
    second[idx] = second[idx] - f;
    first[idx] = amplify_range_around_zero(f, w);
}

/// Suppress X via Y² masking — cross-channel attenuation.
#[cube(launch_unchecked)]
pub fn suppress_x_by_y_kernel(x: &mut Array<f32>, y: &Array<f32>, yw: f32) {
    let idx = ABSOLUTE_POS;
    if idx >= x.len() {
        terminate!();
    }
    let yv = y[idx];
    let scaler = SUPRESS_S + (yw * (1.0 - SUPRESS_S)) / (yw + yv * yv);
    x[idx] = x[idx] * scaler;
}

/// Split HF and UHF — clamp HF, subtract from UHF, clamp + scale UHF, scale + amplify HF.
#[cube(launch_unchecked)]
pub fn separate_hf_uhf_kernel(hf: &mut Array<f32>, uhf: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= hf.len() {
        terminate!();
    }
    let mut h = hf[idx];
    let mut u = uhf[idx];
    h = maximum_clamp(h, KMAXCLAMP_HF);
    u = u - h;
    u = maximum_clamp(u, KMAXCLAMP_UHF);
    u = u * UHF_MUL;
    h = h * HF_MUL;
    h = amplify_range_around_zero(h, HF_AMPLIFY);
    hf[idx] = h;
    uhf[idx] = u;
}

#[cube(launch_unchecked)]
pub fn remove_range_kernel(arr: &mut Array<f32>, w: f32) {
    let idx = ABSOLUTE_POS;
    if idx >= arr.len() {
        terminate!();
    }
    arr[idx] = remove_range_around_zero(arr[idx], w);
}

#[cube(launch_unchecked)]
pub fn amplify_range_kernel(arr: &mut Array<f32>, w: f32) {
    let idx = ABSOLUTE_POS;
    if idx >= arr.len() {
        terminate!();
    }
    arr[idx] = amplify_range_around_zero(arr[idx], w);
}
