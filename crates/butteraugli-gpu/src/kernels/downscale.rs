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
