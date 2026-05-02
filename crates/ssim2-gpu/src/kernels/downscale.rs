//! 2× linear-RGB downscale (per-plane, simple 2×2 average).
//!
//! Translates `ssimulacra2-cuda-kernel/src/downscale.rs::downscale_by_2`,
//! which mirrors `ssimulacra2::downscale_by_2`. CPU and CUDA both clamp
//! source coordinates to `[0, src_w-1]`/`[0, src_h-1]` (i.e. last-row/-
//! column repeats when the source has odd dimensions); we follow.
//!
//! Single-plane variant — caller launches it three times for a planar
//! 3-channel image. (The CUDA crate has a packed-3-channel variant for
//! NPP; we don't need that here.) The warp-shuffle plane variant from
//! the CUDA crate is intentionally skipped because cubecl 0.10 has no
//! portable warp-shuffle abstraction.

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
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let _ = dst_h;
    let sw = src_w as usize;
    let sh = src_h as usize;

    let oy = idx / dw;
    let ox = idx - oy * dw;

    // Mirror the CUDA `min(src - 1)` clamp behaviour.
    let sx0 = ox * 2;
    let sy0 = oy * 2;
    let sx1 = u32::min((sx0 + 1) as u32, (sw as u32).saturating_sub(1)) as usize;
    let sy1 = u32::min((sy0 + 1) as u32, (sh as u32).saturating_sub(1)) as usize;
    let sx0c = u32::min(sx0 as u32, (sw as u32).saturating_sub(1)) as usize;
    let sy0c = u32::min(sy0 as u32, (sh as u32).saturating_sub(1)) as usize;

    let v00 = src[sy0c * sw + sx0c];
    let v01 = src[sy0c * sw + sx1];
    let v10 = src[sy1 * sw + sx0c];
    let v11 = src[sy1 * sw + sx1];

    dst[idx] = (v00 + v01 + v10 + v11) * 0.25;
}

/// Batched 2× downscale. `src` is `batch_size` planes of `src_w × src_h`
/// packed contiguously (stride `src_plane_stride` floats); `dst` is
/// `batch_size` planes of `dst_w × dst_h` packed contiguously (stride
/// `dst_plane_stride`). Each thread handles one destination pixel
/// across the whole batch.
#[cube(launch_unchecked)]
pub fn downscale_2x_plane_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    src_plane_stride: u32,
    dst_plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let dst_pl = dst_plane_stride as usize;
    let src_pl = src_plane_stride as usize;
    let batch_idx = idx / dst_pl;
    let local = idx - batch_idx * dst_pl;
    let dw = dst_w as usize;
    let _ = dst_h;
    let sw = src_w as usize;
    let sh = src_h as usize;
    if local >= dw * (dst_h as usize) {
        terminate!();
    }
    let oy = local / dw;
    let ox = local - oy * dw;
    let sx0 = ox * 2;
    let sy0 = oy * 2;
    let sx1 = u32::min((sx0 + 1) as u32, (sw as u32).saturating_sub(1)) as usize;
    let sy1 = u32::min((sy0 + 1) as u32, (sh as u32).saturating_sub(1)) as usize;
    let sx0c = u32::min(sx0 as u32, (sw as u32).saturating_sub(1)) as usize;
    let sy0c = u32::min(sy0 as u32, (sh as u32).saturating_sub(1)) as usize;
    let off = batch_idx * src_pl;
    let v00 = src[off + sy0c * sw + sx0c];
    let v01 = src[off + sy0c * sw + sx1];
    let v10 = src[off + sy1 * sw + sx0c];
    let v11 = src[off + sy1 * sw + sx1];
    dst[idx] = (v00 + v01 + v10 + v11) * 0.25;
}
