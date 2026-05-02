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

/// Split a band by `orig` and a precomputed Gaussian-blurred copy.
/// Mirrors CPU `separate_mf_hf_channel` for X channel:
///   `out_high[i] = orig[i] − blurred[i]`
///   `orig[i]    = remove_range_around_zero(blurred[i], w)`
#[cube(launch_unchecked)]
pub fn split_band_remove_inplace_kernel(
    orig: &mut Array<f32>,
    blurred: &Array<f32>,
    out_high: &mut Array<f32>,
    w: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= orig.len() {
        terminate!();
    }
    let o = orig[idx];
    let b = blurred[idx];
    out_high[idx] = o - b;
    orig[idx] = remove_range_around_zero(b, w);
}

/// Same as `split_band_remove_inplace_kernel`, but uses
/// `amplify_range_around_zero` on the blurred low-band — for Y channel.
#[cube(launch_unchecked)]
pub fn split_band_amplify_inplace_kernel(
    orig: &mut Array<f32>,
    blurred: &Array<f32>,
    out_high: &mut Array<f32>,
    w: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= orig.len() {
        terminate!();
    }
    let o = orig[idx];
    let b = blurred[idx];
    out_high[idx] = o - b;
    orig[idx] = amplify_range_around_zero(b, w);
}

/// HF→UHF split for the X (chroma) channel — matches CPU
/// `process_uhf_hf_x`.
///   `out_uhf[i] = remove_range(orig − blurred, uhf_range)`
///   `out_hf[i]  = remove_range(blurred,        hf_range)`
#[cube(launch_unchecked)]
pub fn split_uhf_hf_x_kernel(
    hf_orig: &Array<f32>,
    blurred: &Array<f32>,
    out_uhf: &mut Array<f32>,
    out_hf: &mut Array<f32>,
    uhf_range: f32,
    hf_range: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out_uhf.len() {
        terminate!();
    }
    let o = hf_orig[idx];
    let b = blurred[idx];
    out_uhf[idx] = remove_range_around_zero(o - b, uhf_range);
    out_hf[idx] = remove_range_around_zero(b, hf_range);
}

/// HF→UHF split for the Y (luminance) channel — matches CPU
/// `process_uhf_hf_y`. Uses MAXCLAMP + amplify-range with the f32-baked
/// constants from `consts.rs`.
///   hf_clamped = maximum_clamp(blurred, MAXCLAMP_HF)
///   uhf_val    = orig − hf_clamped
///   out_uhf    = maximum_clamp(uhf_val, MAXCLAMP_UHF) · MUL_Y_UHF
///   out_hf     = amplify_range(hf_clamped · MUL_Y_HF, ADD_HF_RANGE)
#[cube(launch_unchecked)]
pub fn split_uhf_hf_y_kernel(
    hf_orig: &Array<f32>,
    blurred: &Array<f32>,
    out_uhf: &mut Array<f32>,
    out_hf: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out_uhf.len() {
        terminate!();
    }
    let orig = hf_orig[idx];
    let b = blurred[idx];
    let hf_clamped = maximum_clamp(b, KMAXCLAMP_HF);
    let uhf_val = orig - hf_clamped;
    let uhf_clamped = maximum_clamp(uhf_val, KMAXCLAMP_UHF);
    out_uhf[idx] = uhf_clamped * UHF_MUL;
    let scaled = hf_clamped * HF_MUL;
    out_hf[idx] = amplify_range_around_zero(scaled, HF_AMPLIFY);
}

/// `out[i] = 0.0` — used to clear AC/DC accumulators between calls
/// since pipeline reuses pre-allocated buffers.
#[cube(launch_unchecked)]
pub fn zero_plane_kernel(dst: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    dst[idx] = f32::new(0.0);
}

/// `dst[i] = src[i]` — used to relocate intermediate results that the
/// non-aliasing split kernels write to a scratch buffer.
#[cube(launch_unchecked)]
pub fn copy_plane_kernel(src: &Array<f32>, dst: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    dst[idx] = src[idx];
}

/// Broadcast a single `plane_stride`-sized source into `batch_size`
/// contiguous slots of `dst`. Used to give the batched compute_diffmap
/// kernel `batch_size` copies of the cached reference mask.
#[cube(launch_unchecked)]
pub fn broadcast_plane_kernel(src: &Array<f32>, dst: &mut Array<f32>, plane_stride: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let local = idx - (idx / (plane_stride as usize)) * (plane_stride as usize);
    dst[idx] = src[local];
}
