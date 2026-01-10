//! Malta filter kernels for Butteraugli
//!
//! The Malta filter is the core perceptual difference calculation in Butteraugli.
//! It uses 16 directional patterns to detect edges at various orientations.
//!
//! Two variants:
//! - MaltaUnit: 9 samples per direction (for high frequency)
//! - MaltaUnitLF: 5 samples per direction (for low/medium frequency)
//!
//! The kernels use 16x16 thread blocks with 24x24 shared memory tiles (4-pixel halo).

use nvptx_std::math::StdMathExt;

// Malta filter constants
const MALTA_LEN: f32 = 3.75;
const MALTA_MULLI_HF: f32 = 0.39905817637;
const MALTA_MULLI_LF: f32 = 0.611612573796;
const KWEIGHT0: f32 = 0.5;
const KWEIGHT1: f32 = 0.33;

// Shared memory dimensions for 16x16 thread blocks
const TILE_SIZE: usize = 16;
const HALO: usize = 4;
const SHARED_SIZE: usize = TILE_SIZE + 2 * HALO; // 24
const SHARED_TOTAL: usize = SHARED_SIZE * SHARED_SIZE; // 576

// We can't allocate shared memory from Rust. This value is defined in shared.ll
extern "C" {
    /// Shared memory for Malta filter - 24x24 float array flattened
    static mut MALTA_DIFFS: [f32; SHARED_TOTAL];
}

/// Malta unit for high-frequency bands (9 samples per direction, 16 directions)
/// Input: pointer to center pixel in shared memory diff array
/// Returns: sum of squared directional responses
#[inline]
unsafe fn malta_unit(d: *const f32, xs: isize) -> f32 {
    let xs3 = 3 * xs;
    let mut retval = 0.0f32;

    // Direction 1: x grows, y constant
    {
        let sum = *d.offset(-4)
            + *d.offset(-3)
            + *d.offset(-2)
            + *d.offset(-1)
            + *d
            + *d.offset(1)
            + *d.offset(2)
            + *d.offset(3)
            + *d.offset(4);
        retval += sum * sum;
    }

    // Direction 2: y grows, x constant
    {
        let sum = *d.offset(-xs3 - xs)
            + *d.offset(-xs3)
            + *d.offset(-xs - xs)
            + *d.offset(-xs)
            + *d
            + *d.offset(xs)
            + *d.offset(xs + xs)
            + *d.offset(xs3)
            + *d.offset(xs3 + xs);
        retval += sum * sum;
    }

    // Direction 3: both grow (diagonal)
    {
        let sum = *d.offset(-xs3 - 3)
            + *d.offset(-xs - xs - 2)
            + *d.offset(-xs - 1)
            + *d
            + *d.offset(xs + 1)
            + *d.offset(xs + xs + 2)
            + *d.offset(xs3 + 3);
        retval += sum * sum;
    }

    // Direction 4: y grows, x shrinks
    {
        let sum = *d.offset(-xs3 + 3)
            + *d.offset(-xs - xs + 2)
            + *d.offset(-xs + 1)
            + *d
            + *d.offset(xs - 1)
            + *d.offset(xs + xs - 2)
            + *d.offset(xs3 - 3);
        retval += sum * sum;
    }

    // Direction 5: y grows -4 to 4, x shrinks 1 -> -1
    {
        let sum = *d.offset(-xs3 - xs + 1)
            + *d.offset(-xs3 + 1)
            + *d.offset(-xs - xs + 1)
            + *d.offset(-xs)
            + *d
            + *d.offset(xs)
            + *d.offset(xs + xs - 1)
            + *d.offset(xs3 - 1)
            + *d.offset(xs3 + xs - 1);
        retval += sum * sum;
    }

    // Direction 6: y grows -4 to 4, x grows -1 -> 1
    {
        let sum = *d.offset(-xs3 - xs - 1)
            + *d.offset(-xs3 - 1)
            + *d.offset(-xs - xs - 1)
            + *d.offset(-xs)
            + *d
            + *d.offset(xs)
            + *d.offset(xs + xs + 1)
            + *d.offset(xs3 + 1)
            + *d.offset(xs3 + xs + 1);
        retval += sum * sum;
    }

    // Direction 7: x grows -4 to 4, y grows -1 to 1
    {
        let sum = *d.offset(-4 - xs)
            + *d.offset(-3 - xs)
            + *d.offset(-2 - xs)
            + *d.offset(-1)
            + *d
            + *d.offset(1)
            + *d.offset(2 + xs)
            + *d.offset(3 + xs)
            + *d.offset(4 + xs);
        retval += sum * sum;
    }

    // Direction 8: x grows -4 to 4, y shrinks 1 to -1
    {
        let sum = *d.offset(-4 + xs)
            + *d.offset(-3 + xs)
            + *d.offset(-2 + xs)
            + *d.offset(-1)
            + *d
            + *d.offset(1)
            + *d.offset(2 - xs)
            + *d.offset(3 - xs)
            + *d.offset(4 - xs);
        retval += sum * sum;
    }

    // Direction 9: steep diagonal top-left to bottom-right
    {
        let sum = *d.offset(-xs3 - 2)
            + *d.offset(-xs - xs - 1)
            + *d.offset(-xs - 1)
            + *d
            + *d.offset(xs + 1)
            + *d.offset(xs + xs + 1)
            + *d.offset(xs3 + 2);
        retval += sum * sum;
    }

    // Direction 10: steep diagonal top-right to bottom-left
    {
        let sum = *d.offset(-xs3 + 2)
            + *d.offset(-xs - xs + 1)
            + *d.offset(-xs + 1)
            + *d
            + *d.offset(xs - 1)
            + *d.offset(xs + xs - 1)
            + *d.offset(xs3 - 2);
        retval += sum * sum;
    }

    // Direction 11
    {
        let sum = *d.offset(-xs - xs - 3)
            + *d.offset(-xs - 2)
            + *d.offset(-xs - 1)
            + *d
            + *d.offset(xs + 1)
            + *d.offset(xs + 2)
            + *d.offset(xs + xs + 3);
        retval += sum * sum;
    }

    // Direction 12
    {
        let sum = *d.offset(-xs - xs + 3)
            + *d.offset(-xs + 2)
            + *d.offset(-xs + 1)
            + *d
            + *d.offset(xs - 1)
            + *d.offset(xs - 2)
            + *d.offset(xs + xs - 3);
        retval += sum * sum;
    }

    // Direction 13: same as 8 (curved line pattern)
    // CPU intentionally duplicates patterns 5-8 as 13-16
    {
        let sum = *d.offset(-4 + xs)
            + *d.offset(-3 + xs)
            + *d.offset(-2 + xs)
            + *d.offset(-1)
            + *d
            + *d.offset(1)
            + *d.offset(2 - xs)
            + *d.offset(3 - xs)
            + *d.offset(4 - xs);
        retval += sum * sum;
    }

    // Direction 14: same as 7 (curved line other direction)
    {
        let sum = *d.offset(-4 - xs)
            + *d.offset(-3 - xs)
            + *d.offset(-2 - xs)
            + *d.offset(-1)
            + *d
            + *d.offset(1)
            + *d.offset(2 + xs)
            + *d.offset(3 + xs)
            + *d.offset(4 + xs);
        retval += sum * sum;
    }

    // Direction 15: same as 6 (very shallow curve)
    {
        let sum = *d.offset(-xs3 - xs - 1)
            + *d.offset(-xs3 - 1)
            + *d.offset(-xs - xs - 1)
            + *d.offset(-xs)
            + *d
            + *d.offset(xs)
            + *d.offset(xs + xs + 1)
            + *d.offset(xs3 + 1)
            + *d.offset(xs3 + xs + 1);
        retval += sum * sum;
    }

    // Direction 16: same as 5 (very shallow curve other direction)
    {
        let sum = *d.offset(-xs3 - xs + 1)
            + *d.offset(-xs3 + 1)
            + *d.offset(-xs - xs + 1)
            + *d.offset(-xs)
            + *d
            + *d.offset(xs)
            + *d.offset(xs + xs - 1)
            + *d.offset(xs3 - 1)
            + *d.offset(xs3 + xs - 1);
        retval += sum * sum;
    }

    retval
}

/// Malta unit for low/medium frequency bands (5 samples per direction, 16 directions)
#[inline]
unsafe fn malta_unit_lf(d: *const f32, xs: isize) -> f32 {
    let xs3 = 3 * xs;
    let mut retval = 0.0f32;

    // Direction 1: x grows, y constant
    {
        let sum = *d.offset(-4) + *d.offset(-2) + *d + *d.offset(2) + *d.offset(4);
        retval += sum * sum;
    }

    // Direction 2: y grows, x constant
    {
        let sum = *d.offset(-xs3 - xs)
            + *d.offset(-xs - xs)
            + *d
            + *d.offset(xs + xs)
            + *d.offset(xs3 + xs);
        retval += sum * sum;
    }

    // Direction 3: both grow
    {
        let sum = *d.offset(-xs3 - 3)
            + *d.offset(-xs - xs - 2)
            + *d
            + *d.offset(xs + xs + 2)
            + *d.offset(xs3 + 3);
        retval += sum * sum;
    }

    // Direction 4: y grows, x shrinks
    {
        let sum = *d.offset(-xs3 + 3)
            + *d.offset(-xs - xs + 2)
            + *d
            + *d.offset(xs + xs - 2)
            + *d.offset(xs3 - 3);
        retval += sum * sum;
    }

    // Direction 5
    {
        let sum = *d.offset(-xs3 - xs + 1)
            + *d.offset(-xs - xs + 1)
            + *d
            + *d.offset(xs + xs - 1)
            + *d.offset(xs3 + xs - 1);
        retval += sum * sum;
    }

    // Direction 6
    {
        let sum = *d.offset(-xs3 - xs - 1)
            + *d.offset(-xs - xs - 1)
            + *d
            + *d.offset(xs + xs + 1)
            + *d.offset(xs3 + xs + 1);
        retval += sum * sum;
    }

    // Direction 7
    {
        let sum =
            *d.offset(-4 - xs) + *d.offset(-2 - xs) + *d + *d.offset(2 + xs) + *d.offset(4 + xs);
        retval += sum * sum;
    }

    // Direction 8
    {
        let sum =
            *d.offset(-4 + xs) + *d.offset(-2 + xs) + *d + *d.offset(2 - xs) + *d.offset(4 - xs);
        retval += sum * sum;
    }

    // Direction 9
    {
        let sum = *d.offset(-xs3 - 2)
            + *d.offset(-xs - xs - 1)
            + *d
            + *d.offset(xs + xs + 1)
            + *d.offset(xs3 + 2);
        retval += sum * sum;
    }

    // Direction 10
    {
        let sum = *d.offset(-xs3 + 2)
            + *d.offset(-xs - xs + 1)
            + *d
            + *d.offset(xs + xs - 1)
            + *d.offset(xs3 - 2);
        retval += sum * sum;
    }

    // Direction 11
    {
        let sum = *d.offset(-xs - xs - 3)
            + *d.offset(-xs - 2)
            + *d
            + *d.offset(xs + 2)
            + *d.offset(xs + xs + 3);
        retval += sum * sum;
    }

    // Direction 12
    {
        let sum = *d.offset(-xs - xs + 3)
            + *d.offset(-xs + 2)
            + *d
            + *d.offset(xs - 2)
            + *d.offset(xs + xs - 3);
        retval += sum * sum;
    }

    // Direction 13
    {
        let sum = *d.offset(xs + xs - 4)
            + *d.offset(xs - 2)
            + *d
            + *d.offset(-xs + 2)
            + *d.offset(-xs - xs + 4);
        retval += sum * sum;
    }

    // Direction 14
    {
        let sum = *d.offset(-xs - xs - 4)
            + *d.offset(-xs - 2)
            + *d
            + *d.offset(xs + 2)
            + *d.offset(xs + xs + 4);
        retval += sum * sum;
    }

    // Direction 15
    {
        let sum = *d.offset(-xs3 - xs - 2)
            + *d.offset(-xs - xs - 1)
            + *d
            + *d.offset(xs + xs + 1)
            + *d.offset(xs3 + xs + 2);
        retval += sum * sum;
    }

    // Direction 16
    {
        let sum = *d.offset(-xs3 - xs + 2)
            + *d.offset(-xs - xs + 1)
            + *d
            + *d.offset(xs + xs - 1)
            + *d.offset(xs3 + xs - 2);
        retval += sum * sum;
    }

    retval
}

/// Compute scaled difference value for Malta filter
/// Handles asymmetric scaling based on whether reference is larger or smaller
#[inline]
fn compute_diff(lum0: f32, lum1: f32, norm1: f32, norm2_0gt1: f32, norm2_0lt1: f32) -> f32 {
    let absval = 0.5 * lum0.abs() + 0.5 * lum1.abs();
    let diff = lum0 - lum1;
    let scaler = norm2_0gt1 / (norm1 + absval);

    // Primary symmetric quadratic objective
    let mut result = scaler * diff;

    // Secondary half-open quadratic objectives (asymmetric)
    let scaler2 = norm2_0lt1 / (norm1 + absval);
    let fabs0 = lum0.abs();
    let too_small = 0.55 * fabs0;
    let too_big = 1.05 * fabs0;

    let mut impact = 0.0f32;

    if lum0 < 0.0 {
        if lum1 > -too_small {
            impact = lum1 + too_small;
        } else if lum1 < -too_big {
            impact = -lum1 - too_big;
        }
    } else {
        if lum1 < too_small {
            impact = -lum1 + too_small;
        } else if lum1 > too_big {
            impact = lum1 - too_big;
        }
    }
    impact *= scaler2;

    if diff < 0.0 {
        result -= impact;
    } else {
        result += impact;
    }

    result
}

/// Malta difference map kernel (high frequency version)
///
/// Uses 16x16 thread blocks with 24x24 shared memory (4-pixel halo).
/// Each thread computes Malta filter response for one pixel.
///
/// Parameters norm2_0gt1 and norm2_0lt1 should be pre-computed on host with f64 precision:
///   let mulli = 0.39905817637_f64;
///   let k_weight0 = 0.5_f64;
///   let k_weight1 = 0.33_f64;
///   let len2 = 3.75_f64 * 2.0 + 1.0;
///   let norm2_0gt1 = (mulli * (k_weight0 * w_0gt1).sqrt() / len2 * norm1) as f32;
///   let norm2_0lt1 = (mulli * (k_weight1 * w_0lt1).sqrt() / len2 * norm1) as f32;
#[no_mangle]
pub unsafe extern "ptx-kernel" fn malta_diff_map_kernel(
    lum0: *const f32,
    lum1: *const f32,
    block_diff_ac: *mut f32,
    width: usize,
    height: usize,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1: f32,
) {
    // Thread and block indices
    let tx = core::arch::nvptx::_thread_idx_x() as usize;
    let ty = core::arch::nvptx::_thread_idx_y() as usize;
    let bx = core::arch::nvptx::_block_idx_x() as usize;
    let by = core::arch::nvptx::_block_idx_y() as usize;

    let x = bx * TILE_SIZE + tx;
    let y = by * TILE_SIZE + ty;

    // norm2_0gt1 and norm2_0lt1 are now pre-computed on host with f64 precision

    // Compute tile origin (top-left corner with halo)
    let topleftx = (bx * TILE_SIZE) as isize - HALO as isize;
    let toplefty = (by * TILE_SIZE) as isize - HALO as isize;

    // Serial index for cooperative loading
    let serial_idx = tx + ty * TILE_SIZE;
    let serial_stride = TILE_SIZE * TILE_SIZE; // 256

    // Cooperatively load 24x24 tile into shared memory
    // Each of the 256 threads loads ~2.25 elements
    let mut i = serial_idx;
    while i < SHARED_TOTAL {
        let work_x = topleftx + (i % SHARED_SIZE) as isize;
        let work_y = toplefty + (i / SHARED_SIZE) as isize;

        if work_x < 0 || work_x >= width as isize || work_y < 0 || work_y >= height as isize {
            // Out of bounds - use zero padding
            *MALTA_DIFFS.as_mut_ptr().add(i) = 0.0;
        } else {
            let global_idx = work_y as usize * width + work_x as usize;
            let l0 = *lum0.add(global_idx);
            let l1 = *lum1.add(global_idx);
            *MALTA_DIFFS.as_mut_ptr().add(i) = compute_diff(l0, l1, norm1, norm2_0gt1, norm2_0lt1);
        }
        i += serial_stride;
    }

    // Sync threads after loading shared memory
    core::arch::nvptx::_syncthreads();

    // Bounds check for output
    if x >= width || y >= height {
        return;
    }

    // Compute Malta filter at thread's position
    // Position in shared memory: (ty + HALO) * SHARED_SIZE + (tx + HALO)
    let shared_pos = (ty + HALO) * SHARED_SIZE + (tx + HALO);
    let result = malta_unit(MALTA_DIFFS.as_ptr().add(shared_pos), SHARED_SIZE as isize);

    // Add to output (accumulate)
    let out_idx = y * width + x;
    *block_diff_ac.add(out_idx) += result;
}

/// Malta difference map kernel (low frequency version)
///
/// Parameters norm2_0gt1 and norm2_0lt1 should be pre-computed on host with f64 precision:
///   let mulli = 0.611612573796_f64;  // LF variant uses different mulli
///   let k_weight0 = 0.5_f64;
///   let k_weight1 = 0.33_f64;
///   let len2 = 3.75_f64 * 2.0 + 1.0;
///   let norm2_0gt1 = (mulli * (k_weight0 * w_0gt1).sqrt() / len2 * norm1) as f32;
///   let norm2_0lt1 = (mulli * (k_weight1 * w_0lt1).sqrt() / len2 * norm1) as f32;
#[no_mangle]
pub unsafe extern "ptx-kernel" fn malta_diff_map_lf_kernel(
    lum0: *const f32,
    lum1: *const f32,
    block_diff_ac: *mut f32,
    width: usize,
    height: usize,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1: f32,
) {
    // Thread and block indices
    let tx = core::arch::nvptx::_thread_idx_x() as usize;
    let ty = core::arch::nvptx::_thread_idx_y() as usize;
    let bx = core::arch::nvptx::_block_idx_x() as usize;
    let by = core::arch::nvptx::_block_idx_y() as usize;

    let x = bx * TILE_SIZE + tx;
    let y = by * TILE_SIZE + ty;

    // norm2_0gt1 and norm2_0lt1 are now pre-computed on host with f64 precision

    // Compute tile origin
    let topleftx = (bx * TILE_SIZE) as isize - HALO as isize;
    let toplefty = (by * TILE_SIZE) as isize - HALO as isize;

    // Serial index for cooperative loading
    let serial_idx = tx + ty * TILE_SIZE;
    let serial_stride = TILE_SIZE * TILE_SIZE;

    // Cooperatively load 24x24 tile into shared memory
    let mut i = serial_idx;
    while i < SHARED_TOTAL {
        let work_x = topleftx + (i % SHARED_SIZE) as isize;
        let work_y = toplefty + (i / SHARED_SIZE) as isize;

        if work_x < 0 || work_x >= width as isize || work_y < 0 || work_y >= height as isize {
            *MALTA_DIFFS.as_mut_ptr().add(i) = 0.0;
        } else {
            let global_idx = work_y as usize * width + work_x as usize;
            let l0 = *lum0.add(global_idx);
            let l1 = *lum1.add(global_idx);
            *MALTA_DIFFS.as_mut_ptr().add(i) = compute_diff(l0, l1, norm1, norm2_0gt1, norm2_0lt1);
        }
        i += serial_stride;
    }

    // Sync threads after loading shared memory
    core::arch::nvptx::_syncthreads();

    // Bounds check for output
    if x >= width || y >= height {
        return;
    }

    // Compute Malta LF filter at thread's position
    let shared_pos = (ty + HALO) * SHARED_SIZE + (tx + HALO);
    let result = malta_unit_lf(
        (MALTA_DIFFS.as_ptr() as *const f32).add(shared_pos),
        SHARED_SIZE as isize,
    );

    // Add to output (accumulate)
    let out_idx = y * width + x;
    *block_diff_ac.add(out_idx) += result;
}
