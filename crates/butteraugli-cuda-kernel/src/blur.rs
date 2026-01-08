//! Gaussian blur kernels for Butteraugli
//!
//! Butteraugli uses multiple blur sigmas:
//! - sigma=1.2 (index 0): used in separateFrequencies for initial blur
//! - sigma=1.56416327805 (index 1): used for HF separation
//! - sigma=2.7 (index 2): used for mask computation
//! - sigma=3.22489901262 (index 3): used for MF separation
//! - sigma=7.15593339443 (index 4): used for LF separation (main opsin dynamics blur)
//!
//! Window sizes (3*sigma, min 8):
//! - sigma=1.2 -> window=8
//! - sigma=1.56 -> window=8
//! - sigma=2.7 -> window=8
//! - sigma=3.22 -> window=9
//! - sigma=7.16 -> window=21

use nvptx_std::prelude::*;

/// Precomputed Gaussian kernel constants for sigma=1.5 (used by ssimulacra2)
/// These are recomputed at build time for the correct sigma values.
mod consts {
    // For now, using ssimulacra2's sigma=1.5 coefficients
    // TODO: Generate these at build time for all 5 butteraugli sigmas
    pub const RADIUS: usize = 5;

    pub const MUL_IN_1: f32 = -0.08333333_f32;
    pub const MUL_IN_3: f32 = 0.08333333_f32;
    pub const MUL_IN_5: f32 = -0.08333333_f32;

    pub const MUL_PREV_1: f32 = 1.7320508_f32;
    pub const MUL_PREV_3: f32 = -1.0_f32;
    pub const MUL_PREV_5: f32 = -1.7320508_f32;

    pub const MUL_PREV2_1: f32 = -1.0_f32;
    pub const MUL_PREV2_3: f32 = -1.0_f32;
    pub const MUL_PREV2_5: f32 = -1.0_f32;
}

/// Horizontal blur pass using separable Gaussian
/// Each thread processes one pixel
#[no_mangle]
pub unsafe extern "ptx-kernel" fn horizontal_blur_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
    kernel: *const f32,
    kernel_integral: *const f32,
    kernel_size: usize,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    let row = idx / width;
    let x = idx % width;

    // Compute blur window bounds (clamp to row)
    let begin = if x >= kernel_size {
        x - kernel_size
    } else {
        0
    };
    let end = (x + kernel_size).min(width - 1);

    // Compute normalization weight from kernel integral
    let weight = *kernel_integral.add(kernel_size + end + 1 - x)
        - *kernel_integral.add(kernel_size + begin - x);

    // Accumulate weighted sum
    let mut sum = 0.0f32;
    for i in begin..=end {
        sum += *src.add(row * width + i) * *kernel.add(kernel_size + i - x);
    }

    *dst.add(idx) = sum / weight;
}

/// Vertical blur pass using separable Gaussian
/// 2D thread layout for better memory access patterns
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vertical_blur_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
    kernel: *const f32,
    kernel_integral: *const f32,
    kernel_size: usize,
) {
    let x = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);
    let y = (core::arch::nvptx::_block_idx_y() as usize
        * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize);

    if x >= width || y >= height {
        return;
    }

    // Compute blur window bounds (clamp to column)
    let begin = if y >= kernel_size {
        y - kernel_size
    } else {
        0
    };
    let end = (y + kernel_size).min(height - 1);

    // Compute normalization weight from kernel integral
    let weight = *kernel_integral.add(kernel_size + end + 1 - y)
        - *kernel_integral.add(kernel_size + begin - y);

    // Accumulate weighted sum
    let mut sum = 0.0f32;
    for i in begin..=end {
        sum += *src.add(i * width + x) * *kernel.add(kernel_size + i - y);
    }

    *dst.add(y * width + x) = sum / weight;
}

/// Tiled Gaussian blur - performs both horizontal and vertical in shared memory
/// Optimized for sigma=2.7 (window size 8)
/// Uses 16x16 thread blocks processing 32x32 output tiles with 8-pixel borders
#[no_mangle]
pub unsafe extern "ptx-kernel" fn tiled_blur_kernel(
    src: *const f32,
    dst: *mut f32,
    width: usize,
    height: usize,
    kernel: *const f32,
    kernel_integral: *const f32,
) {
    // This kernel processes a 32x32 tile per block
    // Each 16x16 thread block loads 48x48 shared memory (with 8-pixel halo)
    // Not implementing full shared memory version in initial port
    // TODO: Implement full tiled blur with shared memory

    let block_x = core::arch::nvptx::_block_idx_x() as usize;
    let blocks_per_row = (width + 31) / 32;

    let tile_x = (block_x % blocks_per_row) * 32;
    let tile_y = (block_x / blocks_per_row) * 32;

    let tx = core::arch::nvptx::_thread_idx_x() as usize;
    let ty = core::arch::nvptx::_thread_idx_y() as usize;

    // Each thread processes 2x2 pixels
    for dy in 0..2 {
        for dx in 0..2 {
            let x = tile_x + tx + dx * 16;
            let y = tile_y + ty + dy * 16;

            if x >= width || y >= height {
                continue;
            }

            // Simple fallback: just copy for now
            // TODO: Implement actual tiled blur
            *dst.add(y * width + x) = *src.add(y * width + x);
        }
    }
}
