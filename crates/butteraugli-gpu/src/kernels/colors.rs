//! Color-space kernels: sRGB → linear, opsin dynamics, linear → XYB.
//!
//! All pointwise (no cross-thread comm). Translated from
//! `butteraugli-cuda-kernel/src/colors.rs` to CubeCL.
//!
//! Constants and the formulas they feed are copied verbatim from the
//! butteraugli reference (libjxl `butteraugli.cc`); see comments in the
//! cuda-kernel crate for derivations.

use cubecl::prelude::*;

const OPSIN_BIAS_X: f32 = 1.7557483643287353;
const OPSIN_BIAS_Y: f32 = 1.7557483643287353;
const OPSIN_BIAS_B: f32 = 12.226454707163354;
const GAMMA_MUL: f32 = 19.245013259874995;
const GAMMA_ADD: f32 = 9.971063576929914;
const GAMMA_SUB: f32 = 23.16046239805755;

/// Per-element sRGB → linear RGB. `src` holds `n_pixels × 3` bytes
/// widened to `u32` on the host (WGSL has no `u8` storage type, so
/// `Array<u8>` reads zero on cubecl-wgpu's Metal backend). Output:
/// planar f32 RGB — `dst_r[idx]`, `dst_g[idx]`, `dst_b[idx]` per pixel.
///
/// Each thread handles one pixel; launch with `n_pixels` total units.
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
    let r = srgb_byte_to_linear(src[i3]);
    let g = srgb_byte_to_linear(src[i3 + 1]);
    let b = srgb_byte_to_linear(src[i3 + 2]);
    dst_r[idx] = r;
    dst_g[idx] = g;
    dst_b[idx] = b;
}

/// sRGB transfer function with linear toe — matches the CPU butteraugli
/// implementation. Input is a u32 holding a single byte value 0..=255.
#[cube]
fn srgb_byte_to_linear(v: u32) -> f32 {
    let f = (v as f32) * (1.0 / 255.0);
    if f <= 0.04045 {
        f / 12.92
    } else {
        f32::powf((f + 0.055) / 1.055, 2.4)
    }
}

/// Opsin dynamics: take linear RGB + a blurred copy and apply a
/// spatially-varying sensitivity, producing planar XYB.
///
/// Kernel mutates `src_*` in place — input planes hold linear RGB at
/// entry, planar XYB at exit. `blur_*` is read-only.
///
/// XYB output convention (matches butteraugli, NOT JPEG XL):
/// - X plane: `gamma_x − gamma_y`  (opponent)
/// - Y plane: `gamma_x + gamma_y`  (luminance-like)
/// - B plane: `gamma_z`            (blue)
#[cube(launch_unchecked)]
pub fn opsin_dynamics_planar_kernel(
    src_r: &mut Array<f32>,
    src_g: &mut Array<f32>,
    src_b: &mut Array<f32>,
    blur_r: &Array<f32>,
    blur_g: &Array<f32>,
    blur_b: &Array<f32>,
    intensity_multiplier: f32,
) {
    let idx = ABSOLUTE_POS;
    let n = src_r.len();
    if idx >= n {
        terminate!();
    }

    let r = src_r[idx] * intensity_multiplier;
    let g = src_g[idx] * intensity_multiplier;
    let b = src_b[idx] * intensity_multiplier;
    let br = blur_r[idx] * intensity_multiplier;
    let bg = blur_g[idx] * intensity_multiplier;
    let bb = blur_b[idx] * intensity_multiplier;

    // Sensitivity from the blurred values (clamped to bias).
    let (bx, by, bz) = opsin_absorbance(br, bg, bb, true);
    let bx = f32::max(bx, 1e-4);
    let by = f32::max(by, 1e-4);
    let bz = f32::max(bz, 1e-4);
    let sens_x = f32::max(gamma(bx) / bx, 1e-4);
    let sens_y = f32::max(gamma(by) / by, 1e-4);
    let sens_z = f32::max(gamma(bz) / bz, 1e-4);

    // Apply sensitivity to the source.
    let (mut sx, mut sy, mut sz) = opsin_absorbance(r, g, b, false);
    sx *= sens_x;
    sy *= sens_y;
    sz *= sens_z;
    sx = f32::max(sx, OPSIN_BIAS_X);
    sy = f32::max(sy, OPSIN_BIAS_Y);
    sz = f32::max(sz, OPSIN_BIAS_B);

    src_r[idx] = sx - sy;
    src_g[idx] = sx + sy;
    src_b[idx] = sz;
}

/// Butteraugli gamma — log2-based, derived to match the CPU `fast_log2f`
/// path. Identity:
/// `K_RET_MUL · log2(v + K_BIAS) + K_RET_ADD ≡ GAMMA_MUL · ln(v + GAMMA_ADD) − GAMMA_SUB`
#[cube]
fn gamma(v: f32) -> f32 {
    // CubeCL spells natural-log as `ln` (its `log` is binary log-base).
    GAMMA_MUL * f32::ln(v + GAMMA_ADD) - GAMMA_SUB
}

/// Butteraugli's opsin absorbance matrix. Returns `(x, y, z)` pre-bias-
/// added; if `clamp` is set, each component is clamped to its bias floor.
#[cube]
fn opsin_absorbance(r: f32, g: f32, b: f32, clamp: bool) -> (f32, f32, f32) {
    let mut x = 0.299565503400583_19 * r
        + 0.633730878338259_36 * g
        + 0.077705617820981_97 * b
        + OPSIN_BIAS_X;

    let mut y =
        0.221586911045747_74 * r + 0.693913880441161_42 * g + 0.0987313588422 * b + OPSIN_BIAS_Y;

    let mut z = 0.02 * r + 0.02 * g + 0.204801290410261_29 * b + OPSIN_BIAS_B;

    if clamp {
        x = f32::max(x, OPSIN_BIAS_X);
        y = f32::max(y, OPSIN_BIAS_Y);
        z = f32::max(z, OPSIN_BIAS_B);
    }

    (x, y, z)
}
