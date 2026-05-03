//! sRGB byte → linear-f32 conversion (planar output).
//!
//! Pointwise translation of `dssim-cuda-kernel/src/srgb.rs`. Like
//! `ssim2-gpu`'s equivalent, we widen each input byte to `u32` on the
//! host because WGSL has no `u8` storage type — `Array<u8>` reads zero
//! on Metal. CUDA tolerates either; storing as u32 costs 4× the
//! staging bandwidth, still trivial for typical DSSIM inputs.
//!
//! Output: three planar `f32` arrays (R, G, B), each `n_pixels` long,
//! values in [0, 1].
//!
//! The transfer function is the standard sRGB EOTF with a linear toe
//! (gamma 2.4 in the higher region — matches what `dssim-core`'s
//! `to_rgblu` produces).

use cubecl::prelude::*;

/// Adjusted for continuity of first derivative — verbatim from the CUDA
/// kernel.
const SRGB_ALPHA: f32 = 1.055_010_7;
const SRGB_BETA: f32 = 0.003_041_282_5;

#[cube(launch_unchecked)]
pub fn srgb_u8_to_linear_planar_kernel(
    src: &Array<u32>,
    dst_r: &mut Array<f32>,
    dst_g: &mut Array<f32>,
    dst_b: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = dst_r.len();
    if idx >= n {
        terminate!();
    }
    let i3 = idx * 3;
    dst_r[idx] = srgb_byte_to_linear(src[i3]);
    dst_g[idx] = srgb_byte_to_linear(src[i3 + 1]);
    dst_b[idx] = srgb_byte_to_linear(src[i3 + 2]);
}

#[cube]
fn srgb_byte_to_linear(v: u32) -> f32 {
    let f = (v as f32) * (1.0 / 255.0);
    if f < 12.92 * SRGB_BETA {
        f / 12.92
    } else {
        f32::powf((f + (SRGB_ALPHA - 1.0)) / SRGB_ALPHA, 2.4)
    }
}
