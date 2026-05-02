//! 2× downsample (averaging) and supersample-add (multi-resolution mix).

use cubecl::prelude::*;

/// Average 2×2 source pixels into one destination pixel; clamp at edges.
#[cube(launch_unchecked)]
pub fn downsample_2x_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_width * dst_height) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_width as usize;
    let dh = dst_height as usize;
    let sw = src_width as usize;
    let sh = src_height as usize;
    let _ = dh;

    let y = idx / dw;
    let x = idx - y * dw;
    let sx = x * 2;
    let sy = y * 2;

    let mut sum = src[sy * sw + sx];
    let mut count = 1.0f32;

    if sx + 1 < sw {
        sum += src[sy * sw + sx + 1];
        count += 1.0;
    }
    if sy + 1 < sh {
        sum += src[(sy + 1) * sw + sx];
        count += 1.0;
    }
    if sx + 1 < sw && sy + 1 < sh {
        sum += src[(sy + 1) * sw + sx + 1];
        count += 1.0;
    }

    dst[idx] = sum / count;
}

/// Add 2× nearest-neighbour-upsampled `src` into `dst` with libjxl's
/// `K_HEURISTIC_MIXING = 0.3` blend:
///   dst[i] = dst[i] · (1 − 0.3·scale) + scale · src[upsampled]
#[cube(launch_unchecked)]
pub fn add_upsample_2x_kernel(
    dst: &mut Array<f32>,
    src: &Array<f32>,
    dst_width: u32,
    dst_height: u32,
    src_width: u32,
    scale: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_width * dst_height) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_width as usize;
    let sw = src_width as usize;
    let y = idx / dw;
    let x = idx - y * dw;

    let sx = x / 2;
    let sy = y / 2;

    let prev = dst[idx];
    let s = src[sy * sw + sx];
    const KMIX: f32 = 0.3;
    dst[idx] = prev * (1.0 - KMIX * scale) + scale * s;
}

/// Batched 2× downsample. Both `src` and `dst` are `N` planes packed
/// contiguously; each thread handles one destination pixel. Same
/// boundary handling as the single-plane variant.
#[cube(launch_unchecked)]
pub fn downsample_2x_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    src_plane_stride: u32,
    dst_plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let dst_plane_us = dst_plane_stride as usize;
    let batch_idx = idx / dst_plane_us;
    let local = idx - batch_idx * dst_plane_us;
    let dw = dst_width as usize;
    let dh = dst_height as usize;
    if local >= dw * dh {
        terminate!();
    }
    let y = local / dw;
    let x = local - y * dw;
    let sw = src_width as usize;
    let sh = src_height as usize;
    let src_off = batch_idx * (src_plane_stride as usize);
    let sx = x * 2;
    let sy = y * 2;
    let _ = dst_height;

    let mut sum = src[src_off + sy * sw + sx];
    let mut count = 1.0f32;
    if sx + 1 < sw {
        sum += src[src_off + sy * sw + sx + 1];
        count += 1.0;
    }
    if sy + 1 < sh {
        sum += src[src_off + (sy + 1) * sw + sx];
        count += 1.0;
    }
    if sx + 1 < sw && sy + 1 < sh {
        sum += src[src_off + (sy + 1) * sw + sx + 1];
        count += 1.0;
    }
    dst[idx] = sum / count;
}

/// Batched supersample-add. `src` is half-res (N planes packed),
/// `dst` is full-res (N planes packed). Same K_HEURISTIC blend as the
/// single-plane variant.
#[cube(launch_unchecked)]
pub fn add_upsample_2x_batched_kernel(
    dst: &mut Array<f32>,
    src: &Array<f32>,
    dst_width: u32,
    dst_height: u32,
    src_width: u32,
    src_plane_stride: u32,
    dst_plane_stride: u32,
    scale: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let dst_plane_us = dst_plane_stride as usize;
    let batch_idx = idx / dst_plane_us;
    let local = idx - batch_idx * dst_plane_us;
    let dw = dst_width as usize;
    let dh = dst_height as usize;
    if local >= dw * dh {
        terminate!();
    }
    let y = local / dw;
    let x = local - y * dw;
    let sw = src_width as usize;
    let src_off = batch_idx * (src_plane_stride as usize);
    let sx = x / 2;
    let sy = y / 2;

    let prev = dst[idx];
    let s = src[src_off + sy * sw + sx];
    const KMIX: f32 = 0.3;
    dst[idx] = prev * (1.0 - KMIX * scale) + scale * s;
}
