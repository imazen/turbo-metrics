//! 2× box downscale for the zensim multi-scale pyramid.
//!
//! CPU zensim uses `downscale_2x_inplace` on a planar f32 buffer with
//! a simple 2x2 box average. This kernel reads 4 pixels from a source
//! plane and writes the mean to a destination plane at half dimensions.
//! Per-pixel work is trivial; we use one thread per destination pixel.

/// 2×2 box downsample on a single planar f32 image.
/// Source dimensions: (src_w, src_h). Dest dimensions: (src_w/2, src_h/2).
/// Odd trailing row/column at the source are handled by clamping.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn downscale_2x_plane_kernel(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let x = core::arch::nvptx::_block_idx_x() as usize * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let y = core::arch::nvptx::_block_idx_y() as usize * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize;
    if x >= dst_width || y >= dst_height {
        return;
    }

    let sx0 = x * 2;
    let sy0 = y * 2;
    let sx1 = if sx0 + 1 < src_width { sx0 + 1 } else { sx0 };
    let sy1 = if sy0 + 1 < src_height { sy0 + 1 } else { sy0 };

    let r0 = src.byte_add(sy0 * src_pitch);
    let r1 = src.byte_add(sy1 * src_pitch);
    let a = *r0.add(sx0);
    let b = *r0.add(sx1);
    let c = *r1.add(sx0);
    let d = *r1.add(sx1);

    let out = dst.byte_add(y * dst_pitch);
    *out.add(x) = (a + b + c + d) * 0.25;
}
