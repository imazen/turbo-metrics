//! Plane transpose.
//!
//! Used between the two recursive-Gaussian passes so the IIR walks
//! columns both times — exactly how the CUDA reference structures its
//! 2D blur. A naive transpose is fine for correctness; we don't bother
//! with a shared-memory tile-and-pad version because the blur is the
//! algorithmic bottleneck, not the transpose.

use cubecl::prelude::*;

/// Transpose `src` (`width × height`, row-major) into `dst`
/// (`height × width`, row-major). Each thread handles one output pixel.
#[cube(launch_unchecked)]
pub fn transpose_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let h = height as usize;
    let w = width as usize;
    // dst is (h × w); compute its row-major (yt, xt).
    let yt = idx / h;
    let xt = idx - yt * h;
    // Source coordinate is the swap.
    let src_idx = xt * w + yt;
    let _ = h;
    dst[idx] = src[src_idx];
}

/// Per-image transpose for batched buffers. Each plane is
/// `width × height`, stored at `plane_stride` floats apart in `src`
/// and `dst`.
#[cube(launch_unchecked)]
pub fn transpose_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let pl = plane_stride as usize;
    let batch_idx = idx / pl;
    let local = idx - batch_idx * pl;
    let h = height as usize;
    let w = width as usize;
    if local >= w * h {
        terminate!();
    }
    let yt = local / h;
    let xt = local - yt * h;
    let plane_off = batch_idx * pl;
    dst[idx] = src[plane_off + xt * w + yt];
}
