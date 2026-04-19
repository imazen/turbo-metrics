//! Downscaling and upscaling kernels for Butteraugli multi-resolution pipeline

use nvptx_std::prelude::*;

/// Downsample image by 2x using simple averaging
/// Output dimensions: ((width-1)/2+1, (height-1)/2+1)
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn downsample_2x_kernel(
    src: *const f32,
    dst: *mut f32,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let x = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);
    let y = (core::arch::nvptx::_block_idx_y() as usize
        * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize);

    if x >= dst_width || y >= dst_height {
        return;
    }

    let src_x = x * 2;
    let src_y = y * 2;

    // Average 2x2 block (handle edge cases)
    let mut sum = 0.0f32;
    let mut count = 0.0f32;

    // Top-left
    sum += *src.add(src_y * src_width + src_x);
    count += 1.0;

    // Top-right
    if src_x + 1 < src_width {
        sum += *src.add(src_y * src_width + src_x + 1);
        count += 1.0;
    }

    // Bottom-left
    if src_y + 1 < src_height {
        sum += *src.add((src_y + 1) * src_width + src_x);
        count += 1.0;
    }

    // Bottom-right
    if src_x + 1 < src_width && src_y + 1 < src_height {
        sum += *src.add((src_y + 1) * src_width + src_x + 1);
        count += 1.0;
    }

    *dst.add(y * dst_width + x) = sum / count;
}

/// Add upsampled 2x image to destination with heuristic mixing
/// Used for combining multi-resolution diffmaps
///
/// Matches libjxl/butteraugli's add_supersampled_2x heuristic:
///   mixed = prev * (1 - K_HEURISTIC_MIXING * weight) + weight * src
/// where K_HEURISTIC_MIXING = 0.3
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn add_upsample_2x_kernel(
    dst: *mut f32,
    src: *const f32,
    dst_width: usize,
    dst_height: usize,
    src_width: usize,
    _src_height: usize,
    scale: f32,
) {
    let x = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);
    let y = (core::arch::nvptx::_block_idx_y() as usize
        * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize);

    if x >= dst_width || y >= dst_height {
        return;
    }

    // Heuristic from C++: lower resolution images have less error
    const K_HEURISTIC_MIXING_VALUE: f32 = 0.3;

    // Nearest-neighbor upsampling
    let src_x = x / 2;
    let src_y = y / 2;

    let src_val = *src.add(src_y * src_width + src_x);
    let dst_idx = y * dst_width + x;
    let prev = *dst.add(dst_idx);

    // Apply heuristic mixing: prev * (1 - 0.3 * weight) + weight * src_val
    *dst.add(dst_idx) = prev * (1.0 - K_HEURISTIC_MIXING_VALUE * scale) + scale * src_val;
}

/// Batched downsample 2x. `src` is N contiguous source planes of
/// `src_plane_stride` f32 elements; `dst` is N contiguous destination
/// planes of `dst_plane_stride` elements. Launch with gridDim.z = N.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn downsample_2x_batch_kernel(
    src: *const f32,
    dst: *mut f32,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    src_plane_stride: usize,
    dst_plane_stride: usize,
) {
    let x = core::arch::nvptx::_block_idx_x() as usize * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let y = core::arch::nvptx::_block_idx_y() as usize * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize;
    let b = core::arch::nvptx::_block_idx_z() as usize;

    if x >= dst_width || y >= dst_height {
        return;
    }

    let src_b = src.add(b * src_plane_stride);
    let dst_b = dst.add(b * dst_plane_stride);

    let src_x = x * 2;
    let src_y = y * 2;
    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    sum += *src_b.add(src_y * src_width + src_x);
    count += 1.0;
    if src_x + 1 < src_width {
        sum += *src_b.add(src_y * src_width + src_x + 1);
        count += 1.0;
    }
    if src_y + 1 < src_height {
        sum += *src_b.add((src_y + 1) * src_width + src_x);
        count += 1.0;
    }
    if src_x + 1 < src_width && src_y + 1 < src_height {
        sum += *src_b.add((src_y + 1) * src_width + src_x + 1);
        count += 1.0;
    }
    *dst_b.add(y * dst_width + x) = sum / count;
}

/// Batched add_upsample_2x. Same heuristic mixing as the unbatched
/// version; `src_plane_stride` and `dst_plane_stride` are the f32
/// counts per image.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn add_upsample_2x_batch_kernel(
    dst: *mut f32,
    src: *const f32,
    dst_width: usize,
    dst_height: usize,
    src_width: usize,
    _src_height: usize,
    src_plane_stride: usize,
    dst_plane_stride: usize,
    scale: f32,
) {
    let x = core::arch::nvptx::_block_idx_x() as usize * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let y = core::arch::nvptx::_block_idx_y() as usize * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize;
    let b = core::arch::nvptx::_block_idx_z() as usize;
    if x >= dst_width || y >= dst_height {
        return;
    }
    const K_HEURISTIC_MIXING_VALUE: f32 = 0.3;
    let src_b = src.add(b * src_plane_stride);
    let dst_b = dst.add(b * dst_plane_stride);
    let src_x = x / 2;
    let src_y = y / 2;
    let src_val = *src_b.add(src_y * src_width + src_x);
    let dst_idx = y * dst_width + x;
    let prev = *dst_b.add(dst_idx);
    *dst_b.add(dst_idx) = prev * (1.0 - K_HEURISTIC_MIXING_VALUE * scale) + scale * src_val;
}
