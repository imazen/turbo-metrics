//! Color conversion kernels for Butteraugli
//!
//! Converts sRGB to linear RGB, then to Butteraugli's XYB color space.

use nvptx_std::math::StdMathExt;

/// Constants for Butteraugli's opsin absorbance matrix
/// Different from SSIMULACRA2/jpegli's XYB!
mod consts {
    // Bias values added after opsin absorbance
    pub const OPSIN_BIAS_X: f32 = 1.7557483643287353;
    pub const OPSIN_BIAS_Y: f32 = 1.7557483643287353;
    pub const OPSIN_BIAS_B: f32 = 12.226454707163354;

    // Gamma function coefficients
    pub const GAMMA_MUL: f32 = 19.245013259874995;
    pub const GAMMA_ADD: f32 = 9.9710635769299145;
    pub const GAMMA_SUB: f32 = 23.16046239805755;
}

/// Convert sRGB value to linear RGB (gamma 2.4)
#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v < 0.0 {
        -(-v).powf(2.4)
    } else {
        v.powf(2.4)
    }
}

/// Butteraugli gamma function
#[inline]
fn gamma(v: f32) -> f32 {
    consts::GAMMA_MUL * (v + consts::GAMMA_ADD).log() - consts::GAMMA_SUB
}

/// Apply Butteraugli's opsin absorbance matrix
/// Transforms linear RGB to pre-XYB color space
#[inline]
fn opsin_absorbance(r: f32, g: f32, b: f32, clamp: bool) -> (f32, f32, f32) {
    let mut x = 0.29956550340058319 * r
        + 0.63373087833825936 * g
        + 0.077705617820981968 * b
        + consts::OPSIN_BIAS_X;

    let mut y = 0.22158691104574774 * r
        + 0.69391388044116142 * g
        + 0.0987313588422 * b
        + consts::OPSIN_BIAS_Y;

    let mut z = 0.02 * r + 0.02 * g + 0.20480129041026129 * b + consts::OPSIN_BIAS_B;

    if clamp {
        x = x.max(consts::OPSIN_BIAS_X);
        y = y.max(consts::OPSIN_BIAS_Y);
        z = z.max(consts::OPSIN_BIAS_B);
    }

    (x, y, z)
}

/// Convert sRGB (u8) to linear RGB (f32)
/// Processes packed RGB data (3 channels interleaved)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn srgb_to_linear_kernel(
    src: *const u8,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let x = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);
    let y = (core::arch::nvptx::_block_idx_y() as usize
        * core::arch::nvptx::_block_dim_y() as usize
        + core::arch::nvptx::_thread_idx_y() as usize);

    if x >= width * 3 || y >= height {
        return;
    }

    let src_row = src.byte_add(y * src_pitch);
    let dst_row = dst.byte_add(y * dst_pitch) as *mut f32;

    let val = *src_row.add(x) as f32 / 255.0;
    *dst_row.add(x) = srgb_to_linear(val);
}

/// Opsin dynamics image transformation
/// Takes linear RGB and blurred linear RGB, produces XYB output
///
/// This is the core Butteraugli color space transformation that uses
/// spatially-varying adaptation based on local brightness.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn opsin_dynamics_kernel(
    // Linear RGB planes (input, modified in-place to XYB)
    src_r: *mut f32,
    src_g: *mut f32,
    src_b: *mut f32,
    // Blurred linear RGB planes
    blur_r: *const f32,
    blur_g: *const f32,
    blur_b: *const f32,
    width: usize,
    height: usize,
    intensity_multiplier: f32,
) {
    let idx = (core::arch::nvptx::_block_idx_x() as usize
        * core::arch::nvptx::_block_dim_x() as usize
        + core::arch::nvptx::_thread_idx_x() as usize);

    if idx >= width * height {
        return;
    }

    // Read source and blurred values
    let r = *src_r.add(idx) * intensity_multiplier;
    let g = *src_g.add(idx) * intensity_multiplier;
    let b = *src_b.add(idx) * intensity_multiplier;

    let blur_r_val = *blur_r.add(idx) * intensity_multiplier;
    let blur_g_val = *blur_g.add(idx) * intensity_multiplier;
    let blur_b_val = *blur_b.add(idx) * intensity_multiplier;

    // Compute sensitivity from blurred values
    let (bx, by, bz) = opsin_absorbance(blur_r_val, blur_g_val, blur_b_val, true);
    let bx = bx.max(1e-4);
    let by = by.max(1e-4);
    let bz = bz.max(1e-4);

    let sens_x = (gamma(bx) / bx).max(1e-4);
    let sens_y = (gamma(by) / by).max(1e-4);
    let sens_z = (gamma(bz) / bz).max(1e-4);

    // Apply opsin absorbance to source
    let (mut sx, mut sy, mut sz) = opsin_absorbance(r, g, b, false);

    // Multiply by sensitivity
    sx *= sens_x;
    sy *= sens_y;
    sz *= sens_z;

    // Clamp
    sx = sx.max(consts::OPSIN_BIAS_X);
    sy = sy.max(consts::OPSIN_BIAS_Y);
    sz = sz.max(consts::OPSIN_BIAS_B);

    // Convert to final XYB form
    // X = opsin_x - opsin_y (opponent channel)
    // Y = opsin_x + opsin_y (luminance-like)
    // B = opsin_z (blue channel)
    *src_r.add(idx) = sx - sy;
    *src_g.add(idx) = sx + sy;
    *src_b.add(idx) = sz;
}

/// Deinterleave 3-channel image: RGB packed to R, G, B planes
/// Input: RGBRGBRGB... Output: RRR..., GGG..., BBB...
#[no_mangle]
pub unsafe extern "ptx-kernel" fn deinterleave_3ch_kernel(
    src: *const f32,
    src_pitch: usize,
    dst0: *mut f32, // First channel plane
    dst1: *mut f32, // Second channel plane
    dst2: *mut f32, // Third channel plane
    width: usize,
    height: usize,
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

    let src_row = src.byte_add(y * src_pitch) as *const f32;
    let idx = y * width + x;

    *dst0.add(idx) = *src_row.add(x * 3);
    *dst1.add(idx) = *src_row.add(x * 3 + 1);
    *dst2.add(idx) = *src_row.add(x * 3 + 2);
}

/// Convert interleaved linear RGB to planar XYB (without opsin dynamics)
/// More efficient version that does conversion and deinterleave in one pass
#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_xyb_planar_kernel(
    src: *const f32,
    src_pitch: usize,
    dst_x: *mut f32, // X plane
    dst_y: *mut f32, // Y plane
    dst_b: *mut f32, // B plane
    width: usize,
    height: usize,
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

    let src_row = src.byte_add(y * src_pitch) as *const f32;
    let idx = y * width + x;

    // Read linear RGB
    let r = *src_row.add(x * 3);
    let g = *src_row.add(x * 3 + 1);
    let b = *src_row.add(x * 3 + 2);

    // Apply opsin absorbance
    let (ox, oy, oz) = opsin_absorbance(r, g, b, false);

    // Apply gamma
    let gx = gamma(ox.max(consts::OPSIN_BIAS_X));
    let gy = gamma(oy.max(consts::OPSIN_BIAS_Y));
    let gz = gamma(oz.max(consts::OPSIN_BIAS_B));

    // Convert to XYB and store in planar format
    *dst_x.add(idx) = gx - gy;
    *dst_y.add(idx) = gx + gy;
    *dst_b.add(idx) = gz;
}

/// Convert linear RGB to Butteraugli XYB (without opsin dynamics)
/// Simpler version for direct conversion without adaptation
#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_xyb_kernel(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
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

    let src_row = src.byte_add(y * src_pitch) as *const f32;
    let dst_row = dst.byte_add(y * dst_pitch) as *mut f32;

    // Read RGB (packed)
    let r = *src_row.add(x * 3);
    let g = *src_row.add(x * 3 + 1);
    let b = *src_row.add(x * 3 + 2);

    // Apply opsin absorbance
    let (ox, oy, oz) = opsin_absorbance(r, g, b, false);

    // Apply gamma
    let gx = gamma(ox.max(consts::OPSIN_BIAS_X));
    let gy = gamma(oy.max(consts::OPSIN_BIAS_Y));
    let gz = gamma(oz.max(consts::OPSIN_BIAS_B));

    // Convert to XYB
    *dst_row.add(x * 3) = gx - gy;
    *dst_row.add(x * 3 + 1) = gx + gy;
    *dst_row.add(x * 3 + 2) = gz;
}
