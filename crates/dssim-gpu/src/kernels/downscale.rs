//! 2× downscale via 2×2 box-filter average.
//!
//! Pointwise translation of `dssim-cuda-kernel/src/downscale.rs`. We
//! only ship the planar variant — the pipeline keeps R/G/B as separate
//! buffers from sRGB-conversion onwards (matches `ssim2-gpu`'s shape
//! and avoids the packed-RGB indirection of the CUDA reference).
//!
//! The 4-pixel patch is clamped at the source-buffer boundary so odd
//! source dimensions degrade gracefully (matches dssim-core).

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn downscale_2x_plane_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total_dst = (dst_w * dst_h) as usize;
    if idx >= total_dst {
        terminate!();
    }

    let dst_w_us = dst_w as usize;
    let oy = idx / dst_w_us;
    let ox = idx - oy * dst_w_us;

    // Source coordinates (top-left of 2×2 patch).
    let sx = (ox * 2) as u32;
    let sy = (oy * 2) as u32;

    let last_x = src_w - 1;
    let last_y = src_h - 1;
    let x0 = u32::min(sx, last_x) as usize;
    let x1 = u32::min(sx + 1, last_x) as usize;
    let y0 = u32::min(sy, last_y) as usize;
    let y1 = u32::min(sy + 1, last_y) as usize;

    let src_w_us = src_w as usize;
    let v00 = src[y0 * src_w_us + x0];
    let v10 = src[y0 * src_w_us + x1];
    let v01 = src[y1 * src_w_us + x0];
    let v11 = src[y1 * src_w_us + x1];

    dst[idx] = (v00 + v10 + v01 + v11) * 0.25;
}
