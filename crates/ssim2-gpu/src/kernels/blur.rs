//! Recursive Gaussian blur — Charalampidis [2016] truncated cosine IIR.
//!
//! This is the algorithmic centrepiece of SSIMULACRA2's blur (the
//! `Blur::blur` / `RecursiveGaussian` types in the published `ssimulacra2`
//! crate). The IIR carries six floats of state per column-walker
//! (`prev_{1,3,5}`, `prev2_{1,3,5}`) and a `2·N + 1` ring buffer for
//! the lookback term `y − N − 1`.
//!
//! ## Two passes, one direction
//!
//! Each kernel invocation does a single top-to-bottom column walk
//! ("vertical pass"). To get a 2D blur the host runs:
//!   1. `blur_pass_kernel` on the source (walks columns) → `_v`.
//!   2. Transpose `_v` → `_vt`.
//!   3. `blur_pass_kernel` on `_vt` (walks columns of the transposed —
//!      i.e. rows of the original).
//! The output stays in transposed coords, which suits the rest of the
//! pipeline (compute_error_maps + reduction are orientation-agnostic).
//!
//! That's exactly the structure `ssimulacra2-cuda/src/lib.rs::process_scale`
//! uses (the CUDA crate just folds 5 source planes into one launch via
//! `block_idx_y`; we drop that fanout — cubecl's grid_y for "pick which
//! src/dst pair" is awkward — and launch once per plane).
//!
//! ## Coefficients
//!
//! Generated at build time by `build.rs` (Charalampidis recurrence on
//! `SIGMA = 1.5`); identical values to the CPU `ssimulacra2` and CUDA
//! `ssimulacra2-cuda-kernel` build scripts. Imported via `consts::*`.
//!
//! ## Recurrence form
//!
//! The kernel uses the "single-step" formulation that matches the
//! CUDA `blur_plane_pass_fused`:
//!
//! ```text
//! out_k = sum * MUL_IN_k + MUL_PREV2_k * prev2_k + MUL_PREV_k * prev_k
//! prev2_k = prev_k
//! prev_k  = out_k
//! ```
//!
//! The "vertical" pass in the CPU `RecursiveGaussian::vertical_pass`
//! uses an algebraically-equivalent variant with VERT_MUL_* constants
//! (sign flipped: `out = sum*VERT_MUL_IN - VERT_MUL_PREV*prev - prev2`).
//! Either form yields the same f32 sequence at single-step granularity;
//! we use the MUL_IN_*/MUL_PREV*_* form so we share build.rs constants
//! 1:1 with the CUDA crate.

use cubecl::prelude::*;

mod consts {
    #![allow(clippy::unreadable_literal, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

pub use consts::RADIUS;
pub const RADIUS_U32: u32 = consts::RADIUS as u32;

const RING_SIZE: u32 = RADIUS_U32 * 2 + 1;
const RING_SIZE_USIZE: usize = consts::RADIUS * 2 + 1;
const TWO_N: u32 = 2 * RADIUS_U32;

/// One thread = one column. Walks `y` from `−N + 1` to `height − 1` and
/// emits one output per non-negative `y`.
///
/// `src` and `dst` are single-plane f32 of length `width × height`,
/// row-major. Coefficients come from `consts` (build.rs).
#[cube(launch_unchecked)]
pub fn blur_pass_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
) {
    // x = absolute column index across the launch grid.
    let x = ABSOLUTE_POS;
    if x >= width as usize {
        terminate!();
    }

    // Per-thread ring buffer. 1 row per thread × RING_SIZE columns;
    // flatten into a 1D shared array indexed by `tx * RING_SIZE + slot`.
    // (cubecl SharedMemory is 1D and `usize`-indexed.)
    let mut ring = SharedMemory::<f32>::new(BLOCK_TIMES_RING_USIZE);
    let tx = UNIT_POS_X as usize;
    let ring_base = tx * RING_SIZE_USIZE;

    // Zero the ring slots for this thread.
    let mut k: u32 = 0;
    while k < RING_SIZE {
        ring[ring_base + (k as usize)] = f32::new(0.0);
        k += 1;
    }

    let mut prev_1 = 0.0_f32;
    let mut prev_3 = 0.0_f32;
    let mut prev_5 = 0.0_f32;
    let mut prev2_1 = 0.0_f32;
    let mut prev2_3 = 0.0_f32;
    let mut prev2_5 = 0.0_f32;

    let h = height;
    let w = width as usize;

    // y ranges over `(-N + 1) ..= (height - 1)`. Use u32 with a phase
    // offset of `N - 1` so the loop variable is non-negative; the
    // mathematical y is `i - (N - 1)`.
    let span = h + RADIUS_U32 - 1; // = height + N - 1 iterations
    let mut i: u32 = 0;
    while i < span {
        // Mathematical y = i - (N - 1); we stay in u32 by tracking
        // `right = y + N - 1 = i` and `left = y - N - 1 = i - 2N`.
        let right = i; // = y + N - 1
        let left_present = i >= TWO_N; // y - N - 1 >= 0  <=>  i >= 2N
        let y_emit = i + 1 >= RADIUS_U32; // y >= 0  <=>  i + 1 >= N

        let right_val = if right < h {
            src[(right as usize) * w + (x as usize)]
        } else {
            f32::new(0.0)
        };

        let left_val = if left_present {
            // left = i - 2N
            let slot = (i - TWO_N) % RING_SIZE;
            ring[ring_base + (slot as usize)]
        } else {
            f32::new(0.0)
        };

        let sum = left_val + right_val;

        // Three IIR taps; parallel; six FMA-able terms.
        let mut out_1 = sum * consts::MUL_IN_1;
        let mut out_3 = sum * consts::MUL_IN_3;
        let mut out_5 = sum * consts::MUL_IN_5;

        out_1 = consts::MUL_PREV2_1 * prev2_1 + out_1;
        out_3 = consts::MUL_PREV2_3 * prev2_3 + out_3;
        out_5 = consts::MUL_PREV2_5 * prev2_5 + out_5;
        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 = consts::MUL_PREV_1 * prev_1 + out_1;
        out_3 = consts::MUL_PREV_3 * prev_3 + out_3;
        out_5 = consts::MUL_PREV_5 * prev_5 + out_5;
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if y_emit {
            // y = i - (N - 1), and y_emit means y >= 0; we still need to
            // mask `y < height` since the IIR keeps stepping for N-1
            // extra iterations after the last input row.
            let y = i + 1 - RADIUS_U32;
            if y < h {
                dst[(y as usize) * w + (x as usize)] = out_1 + out_3 + out_5;
            }
        }

        // Push right_val into the ring at slot `right % RING_SIZE`.
        let slot = right % RING_SIZE;
        ring[ring_base + (slot as usize)] = right_val;

        i += 1;
    }
}

/// Threads-per-block for the blur kernel. Must match the launch dim
/// chosen on the host. 96 = 3 warps of 32, the same value the CUDA
/// reference uses (`BLOCK_WIDTH = 3 * 32`).
pub const BLOCK_WIDTH: u32 = 96;
const BLOCK_WIDTH_USIZE: usize = 96;
const BLOCK_TIMES_RING_USIZE: usize = BLOCK_WIDTH_USIZE * RING_SIZE_USIZE;

/// Batched recursive Gaussian column walk.
///
/// `src` and `dst` hold `batch_size` planes of `width × height` packed
/// contiguously at `plane_stride` floats apart. Launch geometry:
/// `cube_count = (ceil(width / BLOCK_WIDTH), batch_size, 1)`,
/// `cube_dim = (BLOCK_WIDTH, 1, 1)`. `CUBE_POS_Y` selects which image
/// in the batch this cube walks; the IIR's height clamp stays within
/// each plane's local height — see G4.8 in CUBECL_GOTCHAS.md for why
/// we can't just reuse the unbatched kernel on a tall stacked buffer
/// (the column walk would bleed across image boundaries).
#[cube(launch_unchecked)]
pub fn blur_pass_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Y;
    let x = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
    if x >= width {
        terminate!();
    }
    let plane_off = (batch_idx * plane_stride) as usize;

    let mut ring = SharedMemory::<f32>::new(BLOCK_TIMES_RING_USIZE);
    let tx = UNIT_POS_X as usize;
    let ring_base = tx * RING_SIZE_USIZE;

    let mut k: u32 = 0;
    while k < RING_SIZE {
        ring[ring_base + (k as usize)] = f32::new(0.0);
        k += 1;
    }

    let mut prev_1 = 0.0_f32;
    let mut prev_3 = 0.0_f32;
    let mut prev_5 = 0.0_f32;
    let mut prev2_1 = 0.0_f32;
    let mut prev2_3 = 0.0_f32;
    let mut prev2_5 = 0.0_f32;

    let h = height;
    let w = width as usize;

    let span = h + RADIUS_U32 - 1;
    let mut i: u32 = 0;
    while i < span {
        let right = i;
        let left_present = i >= TWO_N;
        let y_emit = i + 1 >= RADIUS_U32;

        let right_val = if right < h {
            src[plane_off + (right as usize) * w + (x as usize)]
        } else {
            f32::new(0.0)
        };

        let left_val = if left_present {
            let slot = (i - TWO_N) % RING_SIZE;
            ring[ring_base + (slot as usize)]
        } else {
            f32::new(0.0)
        };

        let sum = left_val + right_val;

        let mut out_1 = sum * consts::MUL_IN_1;
        let mut out_3 = sum * consts::MUL_IN_3;
        let mut out_5 = sum * consts::MUL_IN_5;

        out_1 = consts::MUL_PREV2_1 * prev2_1 + out_1;
        out_3 = consts::MUL_PREV2_3 * prev2_3 + out_3;
        out_5 = consts::MUL_PREV2_5 * prev2_5 + out_5;
        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 = consts::MUL_PREV_1 * prev_1 + out_1;
        out_3 = consts::MUL_PREV_3 * prev_3 + out_3;
        out_5 = consts::MUL_PREV_5 * prev_5 + out_5;
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if y_emit {
            let y = i + 1 - RADIUS_U32;
            if y < h {
                dst[plane_off + (y as usize) * w + (x as usize)] = out_1 + out_3 + out_5;
            }
        }

        let slot = right % RING_SIZE;
        ring[ring_base + (slot as usize)] = right_val;

        i += 1;
    }
}
