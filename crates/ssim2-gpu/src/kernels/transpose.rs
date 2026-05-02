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
pub fn transpose_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
) {
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
