//! Separable Gaussian blur kernels.
//!
//! Two passes (horizontal then vertical) over a planar f32 image,
//! computing weights on the fly from the supplied sigma. Boundary
//! handling: clamp-to-edge (matches libjxl's `RectifiedConvolution`).
//!
//! Translated from `butteraugli-cuda-kernel/src/blur.rs`. Each thread
//! handles one output pixel; launch with `n_pixels` total units.
//!
//! ## Why no fast Exp
//!
//! CubeCL 0.10's `Exp` unary op is not registered for f32 (only f16,
//! bf16, flex32, tf32, f64). For f32 inputs we substitute the identity
//! `exp(x) = 2^(x · log₂(e))` and use `f32::powf(2, …)` which IS
//! registered. The CUDA backend lowers `powf` to its `powf` intrinsic,
//! same hardware path as a direct `exp` would have been.

use cubecl::prelude::*;

/// Kernel-extent multiplier — matches libjxl's `M = 2.25`.
const M: f32 = 2.25;
/// log₂(e) for the powf-based exp substitution.
const LOG2_E: f32 = 1.442_695_040_888_963_4;

/// `exp(x)` for f32 via `2^(x · log₂(e))`. See module docs.
#[cube]
fn exp_f32(x: f32) -> f32 {
    f32::powf(2.0, x * LOG2_E)
}

/// Gaussian kernel weight at distance `d`, sigma `s`.
#[cube]
fn gauss(d: f32, s: f32) -> f32 {
    let inv = 1.0 / s;
    let z = d * inv;
    exp_f32(-0.5 * z * z)
}

/// Horizontal Gaussian blur, planar f32, clamp-to-edge boundaries.
/// Each thread produces one output pixel; weights are recomputed on the
/// fly because the radius depends on `sigma` which is a runtime scalar.
#[cube(launch_unchecked)]
pub fn horizontal_blur_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    sigma: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let row = idx / w;
    let x = idx % w;

    // Radius = max(1, floor(M * sigma)). Use Min as a u32 op since the
    // cube context wraps native casts; the comparison-style works there.
    let raw = u32::cast_from(M * sigma);
    let radius_us = u32::max(raw, 1u32) as usize;

    let begin = usize::saturating_sub(x, radius_us);
    let end = u32::min((x + radius_us) as u32, (w - 1) as u32) as usize;

    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    let row_off = row * w;

    let mut i = begin;
    while i <= end {
        let i32_ = i as u32;
        let x32 = x as u32;
        let dist = (u32::saturating_sub(i32_, x32) + u32::saturating_sub(x32, i32_)) as f32;
        let weight = gauss(dist, sigma);
        sum += src[row_off + i] * weight;
        wsum += weight;
        i += 1;
    }

    dst[idx] = sum / wsum;
}

/// Vertical Gaussian blur, planar f32, clamp-to-edge.
#[cube(launch_unchecked)]
pub fn vertical_blur_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    sigma: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx % w;

    let raw = u32::cast_from(M * sigma);
    let radius_us = u32::max(raw, 1u32) as usize;

    let begin = usize::saturating_sub(y, radius_us);
    let end = u32::min((y + radius_us) as u32, (h - 1) as u32) as usize;

    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;

    let mut i = begin;
    while i <= end {
        let i32_ = i as u32;
        let y32 = y as u32;
        let dist = (u32::saturating_sub(i32_, y32) + u32::saturating_sub(y32, i32_)) as f32;
        let weight = gauss(dist, sigma);
        sum += src[i * w + x] * weight;
        wsum += weight;
        i += 1;
    }

    dst[y * w + x] = sum / wsum;
}

/// Batched vertical blur — `batch_size` independent `width × height`
/// planes packed contiguously into `src` and `dst`. Each thread
/// handles one pixel; per-image y boundaries are clamped within each
/// plane (so the blur never bleeds across image boundaries).
#[cube(launch_unchecked)]
pub fn vertical_blur_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    sigma: f32,
    plane_stride: u32,
    batch_size: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (plane_stride * batch_size) as usize;
    if idx >= total {
        terminate!();
    }
    let plane_us = plane_stride as usize;
    let w = width as usize;
    let h = height as usize;
    let batch_idx = idx / plane_us;
    let local_idx = idx - batch_idx * plane_us;
    if local_idx >= w * h {
        // Padding within a plane (when plane_stride > w*h) — leave as 0.
        terminate!();
    }
    let y = local_idx / w;
    let x = local_idx - y * w;
    let plane_off = batch_idx * plane_us;

    let raw = u32::cast_from(M * sigma);
    let radius_us = u32::max(raw, 1u32) as usize;
    let begin = usize::saturating_sub(y, radius_us);
    let end = u32::min((y + radius_us) as u32, (h - 1) as u32) as usize;

    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    let mut i = begin;
    while i <= end {
        let i32_ = i as u32;
        let y32 = y as u32;
        let dist = (u32::saturating_sub(i32_, y32) + u32::saturating_sub(y32, i32_)) as f32;
        let weight = gauss(dist, sigma);
        sum += src[plane_off + i * w + x] * weight;
        wsum += weight;
        i += 1;
    }
    dst[plane_off + y * w + x] = sum / wsum;
}
