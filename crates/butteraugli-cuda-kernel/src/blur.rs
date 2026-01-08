//! Gaussian blur kernels for Butteraugli
//!
//! Butteraugli uses multiple blur sigmas:
//! - sigma=1.2: used in separateFrequencies for initial blur
//! - sigma=1.56416327805: used for HF separation
//! - sigma=2.7: used for mask computation
//! - sigma=3.22489901262: used for MF separation
//! - sigma=7.15593339443: used for LF separation (main opsin dynamics blur)

use nvptx_std::math::StdMathExt;

// Kernel extent multiplier (matches libjxl: M=2.25 for accuracy)
const M: f32 = 2.25;

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

    // Window size: M * sigma, minimum 1 (matches libjxl's M=2.25)
    let radius = (M * sigma) as usize;
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

    // Window size: M * sigma, minimum 1 (matches libjxl's M=2.25)
    let radius = (M * sigma) as usize;
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

/// Mirror a coordinate outside image bounds.
/// Matches libjxl's Mirror function - mirror is placed outside the last pixel.
/// For x < 0: x = -x - 1 (so -1 → 0, -2 → 1)
/// For x >= size: x = 2*size - 1 - x (so size → size-1, size+1 → size-2)
#[inline]
fn mirror(mut x: i32, size: i32) -> usize {
    while x < 0 || x >= size {
        if x < 0 {
            x = -x - 1;
        } else {
            x = 2 * size - 1 - x;
        }
    }
    x as usize
}

/// Horizontal pass of 5x5 mirrored blur.
/// Uses mirrored boundary handling (matching CPU blur_mirrored_5x5).
/// Output is transposed for cache-friendly vertical pass.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_mirrored_5x5_horizontal_kernel(
    src: *const f32,
    dst: *mut f32,  // Output is transposed: dst[x][y] = result
    width: usize,
    height: usize,
    w0: f32,  // Center weight
    w1: f32,  // 1-pixel offset weight
    w2: f32,  // 2-pixel offset weight
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    let y = idx / width;
    let x = idx % width;
    let iwidth = width as i32;

    let row = src.add(y * width);

    // Sample with mirrored boundaries
    let v_m2 = *row.add(mirror(x as i32 - 2, iwidth));
    let v_m1 = *row.add(mirror(x as i32 - 1, iwidth));
    let v_0 = *row.add(x);
    let v_p1 = *row.add(mirror(x as i32 + 1, iwidth));
    let v_p2 = *row.add(mirror(x as i32 + 2, iwidth));

    let sum = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;

    // Write transposed: dst[x * height + y]
    *dst.add(x * height + y) = sum;
}

/// Vertical pass of 5x5 mirrored blur.
/// Input is transposed (from horizontal pass), output is in original orientation.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_mirrored_5x5_vertical_kernel(
    src: *const f32,  // Transposed input: src[x][y]
    dst: *mut f32,    // Output in original orientation: dst[y][x]
    width: usize,     // Original width
    height: usize,    // Original height
    w0: f32,
    w1: f32,
    w2: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    // In transposed input, iterate as if original y is column index
    let x = idx / height;  // Original x coordinate
    let y = idx % height;  // Original y coordinate
    let iheight = height as i32;

    // In transposed buffer, column x (original) is at src[x * height + ...]
    let col = src.add(x * height);

    // Sample with mirrored boundaries
    let v_m2 = *col.add(mirror(y as i32 - 2, iheight));
    let v_m1 = *col.add(mirror(y as i32 - 1, iheight));
    let v_0 = *col.add(y);
    let v_p1 = *col.add(mirror(y as i32 + 1, iheight));
    let v_p2 = *col.add(mirror(y as i32 + 2, iheight));

    let sum = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;

    // Write in original orientation
    *dst.add(y * width + x) = sum;
}
