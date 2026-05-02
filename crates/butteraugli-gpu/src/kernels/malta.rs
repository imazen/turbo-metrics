//! Malta filter — 16-direction perceptual contrast detection.
//!
//! Two variants: `_hf` uses 9 samples per direction, `_lf` uses 5 samples.
//! Both use a 24×24 shared-memory tile (16×16 work + 4-pixel halo) loaded
//! cooperatively by all 256 threads in the cube.
//!
//! Translated from `butteraugli-cuda-kernel/src/malta.rs`. Shared memory
//! in CubeCL is declared inline with `SharedMemory::<f32>::new(N)`;
//! `sync_cube()` replaces `__syncthreads()`.

use cubecl::prelude::*;

const TILE_SIZE: u32 = 16;
const HALO: u32 = 4;
const SHARED_SIZE: u32 = TILE_SIZE + 2 * HALO; // 24
const SHARED_TOTAL: u32 = SHARED_SIZE * SHARED_SIZE; // 576
const SHARED_TOTAL_USIZE: usize = (SHARED_SIZE * SHARED_SIZE) as usize; // 576

/// Per-pixel directional-error scaler used by both Malta variants.
/// Asymmetric: differences in the >ref and <ref half-spaces use
/// different `norm2_*` weights.
#[cube]
fn compute_diff(lum0: f32, lum1: f32, norm1: f32, norm2_0gt1: f32, norm2_0lt1: f32) -> f32 {
    let absval = 0.5 * f32::abs(lum0) + 0.5 * f32::abs(lum1);
    let diff = lum0 - lum1;
    let scaler = norm2_0gt1 / (norm1 + absval);
    let mut result = scaler * diff;

    let scaler2 = norm2_0lt1 / (norm1 + absval);
    let fabs0 = f32::abs(lum0);
    let too_small = 0.55 * fabs0;
    let too_big = 1.05 * fabs0;
    let mut impact = 0.0f32;

    if lum0 < 0.0 {
        if lum1 > -too_small {
            impact = lum1 + too_small;
        } else if lum1 < -too_big {
            impact = -lum1 - too_big;
        }
    } else if lum1 < too_small {
        impact = -lum1 + too_small;
    } else if lum1 > too_big {
        impact = lum1 - too_big;
    }
    impact *= scaler2;

    if diff < 0.0 {
        result -= impact;
    } else {
        result += impact;
    }
    result
}

/// HF Malta — 9 samples per direction × 16 directions.
/// `s` is the shared-memory tile (24×24 = 576 elements), `pos` is the
/// linearised index of the center pixel within the tile.
#[cube]
fn malta_unit_hf(s: &SharedMemory<f32>, pos: i32, xs: i32) -> f32 {
    let xs3 = 3 * xs;
    let mut acc = 0.0f32;

    // 16 directions; each is a 9-tap symmetric pattern.
    let p = pos;

    // 1: horizontal
    let s1 = s[(p - 4) as usize]
        + s[(p - 3) as usize]
        + s[(p - 2) as usize]
        + s[(p - 1) as usize]
        + s[p as usize]
        + s[(p + 1) as usize]
        + s[(p + 2) as usize]
        + s[(p + 3) as usize]
        + s[(p + 4) as usize];
    acc += s1 * s1;

    // 2: vertical
    let s2 = s[(p - xs3 - xs) as usize]
        + s[(p - xs3) as usize]
        + s[(p - xs - xs) as usize]
        + s[(p - xs) as usize]
        + s[p as usize]
        + s[(p + xs) as usize]
        + s[(p + xs + xs) as usize]
        + s[(p + xs3) as usize]
        + s[(p + xs3 + xs) as usize];
    acc += s2 * s2;

    // 3: diagonal both grow
    let s3 = s[(p - xs3 - 3) as usize]
        + s[(p - xs - xs - 2) as usize]
        + s[(p - xs - 1) as usize]
        + s[p as usize]
        + s[(p + xs + 1) as usize]
        + s[(p + xs + xs + 2) as usize]
        + s[(p + xs3 + 3) as usize];
    acc += s3 * s3;

    // 4: anti-diagonal
    let s4 = s[(p - xs3 + 3) as usize]
        + s[(p - xs - xs + 2) as usize]
        + s[(p - xs + 1) as usize]
        + s[p as usize]
        + s[(p + xs - 1) as usize]
        + s[(p + xs + xs - 2) as usize]
        + s[(p + xs3 - 3) as usize];
    acc += s4 * s4;

    // 5: shallow curve (y -4..4, x 1..-1)
    let s5 = s[(p - xs3 - xs + 1) as usize]
        + s[(p - xs3 + 1) as usize]
        + s[(p - xs - xs + 1) as usize]
        + s[(p - xs) as usize]
        + s[p as usize]
        + s[(p + xs) as usize]
        + s[(p + xs + xs - 1) as usize]
        + s[(p + xs3 - 1) as usize]
        + s[(p + xs3 + xs - 1) as usize];
    acc += s5 * s5;

    // 6: shallow curve (y -4..4, x -1..1)
    let s6 = s[(p - xs3 - xs - 1) as usize]
        + s[(p - xs3 - 1) as usize]
        + s[(p - xs - xs - 1) as usize]
        + s[(p - xs) as usize]
        + s[p as usize]
        + s[(p + xs) as usize]
        + s[(p + xs + xs + 1) as usize]
        + s[(p + xs3 + 1) as usize]
        + s[(p + xs3 + xs + 1) as usize];
    acc += s6 * s6;

    // 7: x grows -4..4, y -1..1
    let s7 = s[(p - 4 - xs) as usize]
        + s[(p - 3 - xs) as usize]
        + s[(p - 2 - xs) as usize]
        + s[(p - 1) as usize]
        + s[p as usize]
        + s[(p + 1) as usize]
        + s[(p + 2 + xs) as usize]
        + s[(p + 3 + xs) as usize]
        + s[(p + 4 + xs) as usize];
    acc += s7 * s7;

    // 8: x grows -4..4, y 1..-1
    let s8 = s[(p - 4 + xs) as usize]
        + s[(p - 3 + xs) as usize]
        + s[(p - 2 + xs) as usize]
        + s[(p - 1) as usize]
        + s[p as usize]
        + s[(p + 1) as usize]
        + s[(p + 2 - xs) as usize]
        + s[(p + 3 - xs) as usize]
        + s[(p + 4 - xs) as usize];
    acc += s8 * s8;

    // 9: steep diagonal TL→BR
    let s9 = s[(p - xs3 - 2) as usize]
        + s[(p - xs - xs - 1) as usize]
        + s[(p - xs - 1) as usize]
        + s[p as usize]
        + s[(p + xs + 1) as usize]
        + s[(p + xs + xs + 1) as usize]
        + s[(p + xs3 + 2) as usize];
    acc += s9 * s9;

    // 10: steep diagonal TR→BL
    let s10 = s[(p - xs3 + 2) as usize]
        + s[(p - xs - xs + 1) as usize]
        + s[(p - xs + 1) as usize]
        + s[p as usize]
        + s[(p + xs - 1) as usize]
        + s[(p + xs + xs - 1) as usize]
        + s[(p + xs3 - 2) as usize];
    acc += s10 * s10;

    // 11
    let s11 = s[(p - xs - xs - 3) as usize]
        + s[(p - xs - 2) as usize]
        + s[(p - xs - 1) as usize]
        + s[p as usize]
        + s[(p + xs + 1) as usize]
        + s[(p + xs + 2) as usize]
        + s[(p + xs + xs + 3) as usize];
    acc += s11 * s11;

    // 12
    let s12 = s[(p - xs - xs + 3) as usize]
        + s[(p - xs + 2) as usize]
        + s[(p - xs + 1) as usize]
        + s[p as usize]
        + s[(p + xs - 1) as usize]
        + s[(p + xs - 2) as usize]
        + s[(p + xs + xs - 3) as usize];
    acc += s12 * s12;

    // 13 (= 8) — duplicated by spec
    let s13 = s[(p - 4 + xs) as usize]
        + s[(p - 3 + xs) as usize]
        + s[(p - 2 + xs) as usize]
        + s[(p - 1) as usize]
        + s[p as usize]
        + s[(p + 1) as usize]
        + s[(p + 2 - xs) as usize]
        + s[(p + 3 - xs) as usize]
        + s[(p + 4 - xs) as usize];
    acc += s13 * s13;

    // 14 (= 7)
    let s14 = s[(p - 4 - xs) as usize]
        + s[(p - 3 - xs) as usize]
        + s[(p - 2 - xs) as usize]
        + s[(p - 1) as usize]
        + s[p as usize]
        + s[(p + 1) as usize]
        + s[(p + 2 + xs) as usize]
        + s[(p + 3 + xs) as usize]
        + s[(p + 4 + xs) as usize];
    acc += s14 * s14;

    // 15 (= 6)
    let s15 = s[(p - xs3 - xs - 1) as usize]
        + s[(p - xs3 - 1) as usize]
        + s[(p - xs - xs - 1) as usize]
        + s[(p - xs) as usize]
        + s[p as usize]
        + s[(p + xs) as usize]
        + s[(p + xs + xs + 1) as usize]
        + s[(p + xs3 + 1) as usize]
        + s[(p + xs3 + xs + 1) as usize];
    acc += s15 * s15;

    // 16 (= 5)
    let s16 = s[(p - xs3 - xs + 1) as usize]
        + s[(p - xs3 + 1) as usize]
        + s[(p - xs - xs + 1) as usize]
        + s[(p - xs) as usize]
        + s[p as usize]
        + s[(p + xs) as usize]
        + s[(p + xs + xs - 1) as usize]
        + s[(p + xs3 - 1) as usize]
        + s[(p + xs3 + xs - 1) as usize];
    acc += s16 * s16;

    acc
}

/// LF Malta — 5 samples per direction × 16 directions.
#[cube]
fn malta_unit_lf(s: &SharedMemory<f32>, pos: i32, xs: i32) -> f32 {
    let xs3 = 3 * xs;
    let mut acc = 0.0f32;
    let p = pos;

    let s1 = s[(p - 4) as usize]
        + s[(p - 2) as usize]
        + s[p as usize]
        + s[(p + 2) as usize]
        + s[(p + 4) as usize];
    acc += s1 * s1;

    let s2 = s[(p - xs3 - xs) as usize]
        + s[(p - xs - xs) as usize]
        + s[p as usize]
        + s[(p + xs + xs) as usize]
        + s[(p + xs3 + xs) as usize];
    acc += s2 * s2;

    let s3 = s[(p - xs3 - 3) as usize]
        + s[(p - xs - xs - 2) as usize]
        + s[p as usize]
        + s[(p + xs + xs + 2) as usize]
        + s[(p + xs3 + 3) as usize];
    acc += s3 * s3;

    let s4 = s[(p - xs3 + 3) as usize]
        + s[(p - xs - xs + 2) as usize]
        + s[p as usize]
        + s[(p + xs + xs - 2) as usize]
        + s[(p + xs3 - 3) as usize];
    acc += s4 * s4;

    let s5 = s[(p - xs3 - xs + 1) as usize]
        + s[(p - xs - xs + 1) as usize]
        + s[p as usize]
        + s[(p + xs + xs - 1) as usize]
        + s[(p + xs3 + xs - 1) as usize];
    acc += s5 * s5;

    let s6 = s[(p - xs3 - xs - 1) as usize]
        + s[(p - xs - xs - 1) as usize]
        + s[p as usize]
        + s[(p + xs + xs + 1) as usize]
        + s[(p + xs3 + xs + 1) as usize];
    acc += s6 * s6;

    let s7 = s[(p - 4 - xs) as usize]
        + s[(p - 2 - xs) as usize]
        + s[p as usize]
        + s[(p + 2 + xs) as usize]
        + s[(p + 4 + xs) as usize];
    acc += s7 * s7;

    let s8 = s[(p - 4 + xs) as usize]
        + s[(p - 2 + xs) as usize]
        + s[p as usize]
        + s[(p + 2 - xs) as usize]
        + s[(p + 4 - xs) as usize];
    acc += s8 * s8;

    let s9 = s[(p - xs3 - 2) as usize]
        + s[(p - xs - xs - 1) as usize]
        + s[p as usize]
        + s[(p + xs + xs + 1) as usize]
        + s[(p + xs3 + 2) as usize];
    acc += s9 * s9;

    let s10 = s[(p - xs3 + 2) as usize]
        + s[(p - xs - xs + 1) as usize]
        + s[p as usize]
        + s[(p + xs + xs - 1) as usize]
        + s[(p + xs3 - 2) as usize];
    acc += s10 * s10;

    let s11 = s[(p - xs - xs - 3) as usize]
        + s[(p - xs - 2) as usize]
        + s[p as usize]
        + s[(p + xs + 2) as usize]
        + s[(p + xs + xs + 3) as usize];
    acc += s11 * s11;

    let s12 = s[(p - xs - xs + 3) as usize]
        + s[(p - xs + 2) as usize]
        + s[p as usize]
        + s[(p + xs - 2) as usize]
        + s[(p + xs + xs - 3) as usize];
    acc += s12 * s12;

    let s13 = s[(p + xs + xs - 4) as usize]
        + s[(p + xs - 2) as usize]
        + s[p as usize]
        + s[(p - xs + 2) as usize]
        + s[(p - xs - xs + 4) as usize];
    acc += s13 * s13;

    let s14 = s[(p - xs - xs - 4) as usize]
        + s[(p - xs - 2) as usize]
        + s[p as usize]
        + s[(p + xs + 2) as usize]
        + s[(p + xs + xs + 4) as usize];
    acc += s14 * s14;

    let s15 = s[(p - xs3 - xs - 2) as usize]
        + s[(p - xs - xs - 1) as usize]
        + s[p as usize]
        + s[(p + xs + xs + 1) as usize]
        + s[(p + xs3 + xs + 2) as usize];
    acc += s15 * s15;

    let s16 = s[(p - xs3 - xs + 2) as usize]
        + s[(p - xs - xs + 1) as usize]
        + s[p as usize]
        + s[(p + xs + xs - 1) as usize]
        + s[(p + xs3 + xs - 2) as usize];
    acc += s16 * s16;

    acc
}

/// Cooperatively load the 24×24 diff tile into shared memory.
/// Caller must call `sync_cube()` before reading.
#[cube]
fn load_tile(
    lum0: &Array<f32>,
    lum1: &Array<f32>,
    width: u32,
    height: u32,
    norm1: f32,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    tile: &mut SharedMemory<f32>,
) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;
    let topleftx = (bx * TILE_SIZE) as i32 - HALO as i32;
    let toplefty = (by * TILE_SIZE) as i32 - HALO as i32;
    let serial_idx = tx + ty * TILE_SIZE;
    let serial_stride = TILE_SIZE * TILE_SIZE;

    let mut i = serial_idx;
    while i < SHARED_TOTAL {
        let work_x = topleftx + (i % SHARED_SIZE) as i32;
        let work_y = toplefty + (i / SHARED_SIZE) as i32;
        let in_bounds =
            work_x >= 0 && work_x < width as i32 && work_y >= 0 && work_y < height as i32;
        let i_us = i as usize;
        if in_bounds {
            let global_idx = (work_y as u32 * width + work_x as u32) as usize;
            let l0 = lum0[global_idx];
            let l1 = lum1[global_idx];
            tile[i_us] = compute_diff(l0, l1, norm1, norm2_0gt1, norm2_0lt1);
        } else {
            tile[i_us] = f32::new(0.0);
        }
        i += serial_stride;
    }
}

/// Malta high-frequency diff map. Launch with cube-dim (16, 16, 1) and
/// cube-count (ceil(width/16), ceil(height/16), 1).
///
/// `norm2_0gt1` / `norm2_0lt1` are pre-computed on host with f64
/// precision (see `butteraugli-cuda` for the formula); we take them as
/// scalars so this kernel doesn't need an f64 sqrt.
#[cube(launch_unchecked)]
pub fn malta_diff_map_hf_kernel(
    lum0: &Array<f32>,
    lum1: &Array<f32>,
    block_diff_ac: &mut Array<f32>,
    width: u32,
    height: u32,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1: f32,
) {
    let mut tile = SharedMemory::<f32>::new(SHARED_TOTAL_USIZE);
    load_tile(
        lum0, lum1, width, height, norm1, norm2_0gt1, norm2_0lt1, &mut tile,
    );
    sync_cube();

    let x = CUBE_POS_X * TILE_SIZE + UNIT_POS_X;
    let y = CUBE_POS_Y * TILE_SIZE + UNIT_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let pos = ((UNIT_POS_Y + HALO) * SHARED_SIZE + (UNIT_POS_X + HALO)) as i32;
    let result = malta_unit_hf(&tile, pos, SHARED_SIZE as i32);
    let out_idx = (y * width + x) as usize;
    block_diff_ac[out_idx] = block_diff_ac[out_idx] + result;
}

/// Malta low-frequency diff map.
#[cube(launch_unchecked)]
pub fn malta_diff_map_lf_kernel(
    lum0: &Array<f32>,
    lum1: &Array<f32>,
    block_diff_ac: &mut Array<f32>,
    width: u32,
    height: u32,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1: f32,
) {
    let mut tile = SharedMemory::<f32>::new(SHARED_TOTAL_USIZE);
    load_tile(
        lum0, lum1, width, height, norm1, norm2_0gt1, norm2_0lt1, &mut tile,
    );
    sync_cube();

    let x = CUBE_POS_X * TILE_SIZE + UNIT_POS_X;
    let y = CUBE_POS_Y * TILE_SIZE + UNIT_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let pos = ((UNIT_POS_Y + HALO) * SHARED_SIZE + (UNIT_POS_X + HALO)) as i32;
    let result = malta_unit_lf(&tile, pos, SHARED_SIZE as i32);
    let out_idx = (y * width + x) as usize;
    block_diff_ac[out_idx] = block_diff_ac[out_idx] + result;
}

/// Cooperative tile load with broadcast reference and per-batch
/// distorted slot. `lum0_off` lets the reference share across slots
/// (set to 0); `lum1_off = batch_idx * plane_stride` selects the
/// distorted slot. The accumulator output is per-slot.
#[cube]
fn load_tile_split(
    lum0: &Array<f32>,
    lum0_off: u32,
    lum1: &Array<f32>,
    lum1_off: u32,
    width: u32,
    height: u32,
    norm1: f32,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    tile: &mut SharedMemory<f32>,
) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;
    let topleftx = (bx * TILE_SIZE) as i32 - HALO as i32;
    let toplefty = (by * TILE_SIZE) as i32 - HALO as i32;
    let serial_idx = tx + ty * TILE_SIZE;
    let serial_stride = TILE_SIZE * TILE_SIZE;

    let mut i = serial_idx;
    while i < SHARED_TOTAL {
        let work_x = topleftx + (i % SHARED_SIZE) as i32;
        let work_y = toplefty + (i / SHARED_SIZE) as i32;
        let in_bounds =
            work_x >= 0 && work_x < width as i32 && work_y >= 0 && work_y < height as i32;
        let i_us = i as usize;
        if in_bounds {
            let global_idx = (work_y as u32 * width + work_x as u32) as usize;
            let l0 = lum0[(lum0_off as usize) + global_idx];
            let l1 = lum1[(lum1_off as usize) + global_idx];
            tile[i_us] = compute_diff(l0, l1, norm1, norm2_0gt1, norm2_0lt1);
        } else {
            tile[i_us] = f32::new(0.0);
        }
        i += serial_stride;
    }
}

/// Batched Malta HF kernel: one cached reference (broadcast) vs
/// `batch_size` distorted planes packed contiguously into `lum1`.
/// Launch with `cube_count = (ceil(w/16), ceil(h/16), batch_size)`
/// and `cube_dim = (16, 16, 1)`. `block_diff_ac` is the per-slot AC
/// accumulator (also `batch_size` planes packed contiguously).
#[cube(launch_unchecked)]
pub fn malta_diff_map_hf_batched_kernel(
    lum0: &Array<f32>,
    lum1: &Array<f32>,
    block_diff_ac: &mut Array<f32>,
    width: u32,
    height: u32,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1: f32,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Z;
    let lum1_off = batch_idx * plane_stride;
    let mut tile = SharedMemory::<f32>::new(SHARED_TOTAL_USIZE);
    load_tile_split(
        lum0, 0, lum1, lum1_off, width, height, norm1, norm2_0gt1, norm2_0lt1, &mut tile,
    );
    sync_cube();

    let x = CUBE_POS_X * TILE_SIZE + UNIT_POS_X;
    let y = CUBE_POS_Y * TILE_SIZE + UNIT_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let pos = ((UNIT_POS_Y + HALO) * SHARED_SIZE + (UNIT_POS_X + HALO)) as i32;
    let result = malta_unit_hf(&tile, pos, SHARED_SIZE as i32);
    let out_idx = (lum1_off + y * width + x) as usize;
    block_diff_ac[out_idx] = block_diff_ac[out_idx] + result;
}

/// Batched Malta LF kernel — see `malta_diff_map_hf_batched_kernel`.
#[cube(launch_unchecked)]
pub fn malta_diff_map_lf_batched_kernel(
    lum0: &Array<f32>,
    lum1: &Array<f32>,
    block_diff_ac: &mut Array<f32>,
    width: u32,
    height: u32,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1: f32,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Z;
    let lum1_off = batch_idx * plane_stride;
    let mut tile = SharedMemory::<f32>::new(SHARED_TOTAL_USIZE);
    load_tile_split(
        lum0, 0, lum1, lum1_off, width, height, norm1, norm2_0gt1, norm2_0lt1, &mut tile,
    );
    sync_cube();

    let x = CUBE_POS_X * TILE_SIZE + UNIT_POS_X;
    let y = CUBE_POS_Y * TILE_SIZE + UNIT_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let pos = ((UNIT_POS_Y + HALO) * SHARED_SIZE + (UNIT_POS_X + HALO)) as i32;
    let result = malta_unit_lf(&tile, pos, SHARED_SIZE as i32);
    let out_idx = (lum1_off + y * width + x) as usize;
    block_diff_ac[out_idx] = block_diff_ac[out_idx] + result;
}
