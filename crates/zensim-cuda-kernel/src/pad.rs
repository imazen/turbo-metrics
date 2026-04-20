//! Fill SIMD padding columns of a planar f32 image with mirror-reflected
//! copies of the real columns, matching CPU zensim's `convert_source_to_xyb`
//! post-processing (streaming.rs line 859-876).
//!
//! For pad_col = 0..(padded_w - logical_w):
//!   plane[y, logical_w + pad_col] = plane[y, mirror_offset[pad_col]]
//!
//! where mirror_offset is computed host-side to match the CPU formula
//!   m = (logical_w + pad_col) % (2 * (logical_w - 1))
//!   offset = if m < logical_w { m } else { 2 * (logical_w - 1) - m }
//!
//! One thread per (pad_col, y). Grid = ceil(pad_count / 16) × ceil(height / 16).
//! `mirror_offsets` is a device pointer to `pad_count` u32s (one per padding col).

/// Fill padding columns with mirror-reflected values from real columns.
///
/// # Safety
/// * `plane` must be a valid f32 image at stride `pitch` bytes, at least
///   `padded_w` cols wide and `height` rows tall.
/// * `mirror_offsets` must be a valid device pointer to `pad_count` u32
///   values, where `pad_count = padded_w - logical_w`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn pad_mirror_plane_kernel(
    plane: *mut f32,
    pitch: usize,
    logical_w: usize,
    padded_w: usize,
    height: usize,
    mirror_offsets: *const u32,
) {
    let pad_col = core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize;
    let y = core::arch::nvptx::_block_idx_y() as usize
        * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize;
    let pad_count = padded_w - logical_w;
    if pad_col >= pad_count || y >= height {
        return;
    }
    let src_col = *mirror_offsets.add(pad_col) as usize;
    let row = plane.byte_add(y * pitch);
    let v = *row.add(src_col);
    *row.add(logical_w + pad_col) = v;
}
