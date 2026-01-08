//! Gaussian blur kernels for Butteraugli
//!
//! Butteraugli uses multiple blur sigmas:
//! - sigma=1.2: used in separateFrequencies for initial blur
//! - sigma=1.56416327805: used for HF separation
//! - sigma=2.7: used for mask computation
//! - sigma=3.22489901262: used for MF separation
//! - sigma=7.15593339443: used for LF separation (main opsin dynamics blur)

use nvptx_std::math::StdMathExt;

/// Compute Gaussian weight for a given distance and sigma
#[inline]
fn gauss(x: f32, sigma: f32) -> f32 {
    let inv_sigma = 1.0 / sigma;
    let x_sigma = x * inv_sigma;
    (-0.5 * x_sigma * x_sigma).exp()
}

/// Horizontal blur pass using separable Gaussian
/// Computes Gaussian weights on-the-fly from sigma
#[no_mangle]
pub unsafe extern "ptx-kernel" fn horizontal_blur_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
    sigma: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    let row = idx / width;
    let x = idx % width;

    // Window size: 3 * sigma, minimum 1
    let radius = (sigma * 3.0) as usize;
    let radius = if radius < 1 { 1 } else { radius };

    // Compute bounds (clamp to row)
    let begin = if x >= radius { x - radius } else { 0 };
    let end = if x + radius < width { x + radius } else { width - 1 };

    // Accumulate weighted sum
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for i in begin..=end {
        let dist = if i >= x { i - x } else { x - i };
        let w = gauss(dist as f32, sigma);
        sum += *src.add(row * width + i) * w;
        weight_sum += w;
    }

    *dst.add(idx) = sum / weight_sum;
}

/// Vertical blur pass using separable Gaussian
/// Computes Gaussian weights on-the-fly from sigma
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vertical_blur_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
    sigma: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    let y = idx / width;
    let x = idx % width;

    // Window size: 3 * sigma, minimum 1
    let radius = (sigma * 3.0) as usize;
    let radius = if radius < 1 { 1 } else { radius };

    // Compute bounds (clamp to column)
    let begin = if y >= radius { y - radius } else { 0 };
    let end = if y + radius < height { y + radius } else { height - 1 };

    // Accumulate weighted sum
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for i in begin..=end {
        let dist = if i >= y { i - y } else { y - i };
        let w = gauss(dist as f32, sigma);
        sum += *src.add(i * width + x) * w;
        weight_sum += w;
    }

    *dst.add(y * width + x) = sum / weight_sum;
}

/// Tiled Gaussian blur - simple fallback implementation
/// TODO: Implement optimized shared memory version
#[no_mangle]
pub unsafe extern "ptx-kernel" fn tiled_blur_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    // Just copy for now - tiled blur should be optimized later
    *dst.add(idx) = *src.add(idx);
}
