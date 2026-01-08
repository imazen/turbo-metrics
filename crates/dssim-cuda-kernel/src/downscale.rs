//! Image downsampling kernels
//!
//! DSSIM uses simple 2x2 box filter averaging for multi-scale pyramid.

use nvptx_std::prelude::*;

/// Downscale a single plane by 2x using 2x2 box filter average.
///
/// This is the primary downscale kernel for DSSIM's multi-scale processing.
/// Each output pixel is the average of 4 input pixels.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn downscale_plane_by_2(
    src: *const f32,
    src_w: usize,
    src_h: usize,
    src_pitch: usize,
    dst: *mut f32,
    dst_w: usize,
    dst_h: usize,
    dst_pitch: usize,
) {
    let (ox, oy) = coords_2d();

    if ox < dst_w && oy < dst_h {
        // Source coordinates (top-left of 2x2 patch)
        let sx = ox * 2;
        let sy = oy * 2;

        // Clamp to valid range
        let x0 = sx.min(src_w - 1);
        let x1 = (sx + 1).min(src_w - 1);
        let y0 = sy.min(src_h - 1);
        let y1 = (sy + 1).min(src_h - 1);

        // Average 4 pixels
        let v00 = *src.byte_add(y0 * src_pitch).add(x0);
        let v10 = *src.byte_add(y0 * src_pitch).add(x1);
        let v01 = *src.byte_add(y1 * src_pitch).add(x0);
        let v11 = *src.byte_add(y1 * src_pitch).add(x1);

        let avg = (v00 + v10 + v01 + v11) * 0.25;
        *dst.byte_add(oy * dst_pitch).add(ox) = avg;
    }
}

/// Downscale packed RGB (3 channels interleaved) by 2x.
///
/// Used for initial downscale before LAB conversion if needed.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn downscale_rgb_by_2(
    src: *const f32,
    src_w: usize,
    src_h: usize,
    src_pitch: usize,
    dst: *mut f32,
    dst_w: usize,
    dst_h: usize,
    dst_pitch: usize,
) {
    const C: usize = 3;

    let (ox, oy) = coords_2d();

    if ox < dst_w && oy < dst_h {
        let sx = ox * 2;
        let sy = oy * 2;

        let x0 = sx.min(src_w - 1);
        let x1 = (sx + 1).min(src_w - 1);
        let y0 = sy.min(src_h - 1);
        let y1 = (sy + 1).min(src_h - 1);

        for c in 0..C {
            let v00 = *src.byte_add(y0 * src_pitch).add(x0 * C + c);
            let v10 = *src.byte_add(y0 * src_pitch).add(x1 * C + c);
            let v01 = *src.byte_add(y1 * src_pitch).add(x0 * C + c);
            let v11 = *src.byte_add(y1 * src_pitch).add(x1 * C + c);

            let avg = (v00 + v10 + v01 + v11) * 0.25;
            *dst.byte_add(oy * dst_pitch).add(ox * C + c) = avg;
        }
    }
}
