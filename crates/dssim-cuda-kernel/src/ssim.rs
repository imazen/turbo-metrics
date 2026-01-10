//! SSIM computation kernel
//!
//! Computes the Structural Similarity Index between two images.
//!
//! SSIM formula:
//! SSIM(x,y) = (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
//!
//! Where:
//! - μx = local mean of x (blur(x))
//! - μy = local mean of y (blur(y))
//! - σx² = local variance of x (blur(x²) - μx²)
//! - σy² = local variance of y (blur(y²) - μy²)
//! - σxy = local covariance (blur(x*y) - μx*μy)

use nvptx_std::prelude::*;

// SSIM constants (from dssim-core)
const C1: f32 = 0.0001; // (0.01)²
const C2: f32 = 0.0009; // (0.03)²

/// Compute per-pixel SSIM from precomputed statistics.
///
/// Inputs:
/// - mu1: blur(img1) - local mean of reference
/// - mu2: blur(img2) - local mean of distorted
/// - sigma1_sq: blur(img1²) - μ1² will be computed here, OR precomputed variance
/// - sigma2_sq: blur(img2²) - μ2² will be computed here, OR precomputed variance
/// - sigma12: blur(img1*img2) - μ1*μ2 will be computed here, OR precomputed covariance
///
/// The kernel computes the full SSIM formula including subtracting μ² terms.
///
/// Output: SSIM map (values typically 0-1, 1 = identical)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn compute_ssim(
    mu1: *const f32,
    mu1_pitch: usize,
    mu2: *const f32,
    mu2_pitch: usize,
    img1_sq_blur: *const f32, // blur(img1²)
    img1_sq_blur_pitch: usize,
    img2_sq_blur: *const f32, // blur(img2²)
    img2_sq_blur_pitch: usize,
    img12_blur: *const f32, // blur(img1*img2)
    img12_blur_pitch: usize,
    ssim_out: *mut f32,
    ssim_out_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let m1 = *mu1.byte_add(y * mu1_pitch).add(x);
        let m2 = *mu2.byte_add(y * mu2_pitch).add(x);
        let i1sq_blur = *img1_sq_blur.byte_add(y * img1_sq_blur_pitch).add(x);
        let i2sq_blur = *img2_sq_blur.byte_add(y * img2_sq_blur_pitch).add(x);
        let i12_blur = *img12_blur.byte_add(y * img12_blur_pitch).add(x);

        // Compute statistics
        let mu1_sq = m1 * m1;
        let mu2_sq = m2 * m2;
        let mu1_mu2 = m1 * m2;

        let sigma1_sq = i1sq_blur - mu1_sq;
        let sigma2_sq = i2sq_blur - mu2_sq;
        let sigma12 = i12_blur - mu1_mu2;

        // SSIM formula
        // Numerator: (2*μ1*μ2 + C1) * (2*σ12 + C2)
        let num1 = 2.0f32.mul_add(mu1_mu2, C1);
        let num2 = 2.0f32.mul_add(sigma12, C2);
        let numerator = num1 * num2;

        // Denominator: (μ1² + μ2² + C1) * (σ1² + σ2² + C2)
        let den1 = mu1_sq + mu2_sq + C1;
        let den2 = sigma1_sq + sigma2_sq + C2;
        let denominator = den1 * den2;

        let ssim = numerator / denominator;
        *ssim_out.byte_add(y * ssim_out_pitch).add(x) = ssim;
    }
}

/// Compute per-pixel SSIM using precomputed reference statistics.
///
/// This variant is optimized for the precompute API where reference
/// statistics (mu1, sigma1_sq) are cached and reused.
///
/// Inputs for reference (precomputed):
/// - mu1: blur(ref)
/// - sigma1_sq: variance of reference (blur(ref²) - μ1²), already computed
/// - ref_img: original reference pixels (needed for covariance)
///
/// Inputs for distorted (computed fresh):
/// - mu2: blur(dis)
/// - img2_sq_blur: blur(dis²)
/// - img12_blur: blur(ref * dis)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn compute_ssim_precomputed_ref(
    // Reference (precomputed)
    mu1: *const f32,
    mu1_pitch: usize,
    sigma1_sq: *const f32, // Already computed: blur(ref²) - μ1²
    sigma1_sq_pitch: usize,
    // Distorted
    mu2: *const f32,
    mu2_pitch: usize,
    img2_sq_blur: *const f32,
    img2_sq_blur_pitch: usize,
    // Cross term
    img12_blur: *const f32,
    img12_blur_pitch: usize,
    // Output
    ssim_out: *mut f32,
    ssim_out_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let m1 = *mu1.byte_add(y * mu1_pitch).add(x);
        let m2 = *mu2.byte_add(y * mu2_pitch).add(x);
        let s1_sq = *sigma1_sq.byte_add(y * sigma1_sq_pitch).add(x);
        let i2sq_blur = *img2_sq_blur.byte_add(y * img2_sq_blur_pitch).add(x);
        let i12_blur = *img12_blur.byte_add(y * img12_blur_pitch).add(x);

        let mu1_sq = m1 * m1;
        let mu2_sq = m2 * m2;
        let mu1_mu2 = m1 * m2;

        // sigma1_sq is already computed for reference
        let sigma2_sq = i2sq_blur - mu2_sq;
        let sigma12 = i12_blur - mu1_mu2;

        // SSIM formula
        let num1 = 2.0f32.mul_add(mu1_mu2, C1);
        let num2 = 2.0f32.mul_add(sigma12, C2);
        let numerator = num1 * num2;

        let den1 = mu1_sq + mu2_sq + C1;
        let den2 = s1_sq + sigma2_sq + C2;
        let denominator = den1 * den2;

        let ssim = numerator / denominator;
        *ssim_out.byte_add(y * ssim_out_pitch).add(x) = ssim;
    }
}

/// Compute SSIM from LAB channel statistics, averaging across channels.
///
/// This matches dssim-core's approach where L, a, b statistics are averaged
/// before computing the SSIM formula.
///
/// Takes 15 input buffers (5 statistics × 3 channels) and produces single SSIM map.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn compute_ssim_lab(
    // Reference mu (L, a, b)
    mu1_l: *const f32,
    mu1_a: *const f32,
    mu1_b: *const f32,
    mu1_pitch: usize,
    // Distorted mu (L, a, b)
    mu2_l: *const f32,
    mu2_a: *const f32,
    mu2_b: *const f32,
    mu2_pitch: usize,
    // Reference blur(img²) (L, a, b)
    sq1_l: *const f32,
    sq1_a: *const f32,
    sq1_b: *const f32,
    sq1_pitch: usize,
    // Distorted blur(img²) (L, a, b)
    sq2_l: *const f32,
    sq2_a: *const f32,
    sq2_b: *const f32,
    sq2_pitch: usize,
    // Cross blur(img1*img2) (L, a, b)
    cross_l: *const f32,
    cross_a: *const f32,
    cross_b: *const f32,
    cross_pitch: usize,
    // Output
    ssim_out: *mut f32,
    ssim_out_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        // Load all values
        let m1_l = *mu1_l.byte_add(y * mu1_pitch).add(x);
        let m1_a = *mu1_a.byte_add(y * mu1_pitch).add(x);
        let m1_b = *mu1_b.byte_add(y * mu1_pitch).add(x);

        let m2_l = *mu2_l.byte_add(y * mu2_pitch).add(x);
        let m2_a = *mu2_a.byte_add(y * mu2_pitch).add(x);
        let m2_b = *mu2_b.byte_add(y * mu2_pitch).add(x);

        let s1_l = *sq1_l.byte_add(y * sq1_pitch).add(x);
        let s1_a = *sq1_a.byte_add(y * sq1_pitch).add(x);
        let s1_b = *sq1_b.byte_add(y * sq1_pitch).add(x);

        let s2_l = *sq2_l.byte_add(y * sq2_pitch).add(x);
        let s2_a = *sq2_a.byte_add(y * sq2_pitch).add(x);
        let s2_b = *sq2_b.byte_add(y * sq2_pitch).add(x);

        let c_l = *cross_l.byte_add(y * cross_pitch).add(x);
        let c_a = *cross_a.byte_add(y * cross_pitch).add(x);
        let c_b = *cross_b.byte_add(y * cross_pitch).add(x);

        // Average mu products across channels (like dssim-core)
        let mu1_sq = (m1_l * m1_l + m1_a * m1_a + m1_b * m1_b) * (1.0 / 3.0);
        let mu2_sq = (m2_l * m2_l + m2_a * m2_a + m2_b * m2_b) * (1.0 / 3.0);
        let mu1_mu2 = (m1_l * m2_l + m1_a * m2_a + m1_b * m2_b) * (1.0 / 3.0);

        // Average blur(img²) across channels
        let img1_sq_blur = (s1_l + s1_a + s1_b) * (1.0 / 3.0);
        let img2_sq_blur = (s2_l + s2_a + s2_b) * (1.0 / 3.0);
        let img12_blur = (c_l + c_a + c_b) * (1.0 / 3.0);

        // Compute sigma values
        let sigma1_sq = img1_sq_blur - mu1_sq;
        let sigma2_sq = img2_sq_blur - mu2_sq;
        let sigma12 = img12_blur - mu1_mu2;

        // SSIM formula
        let num1 = 2.0f32.mul_add(mu1_mu2, C1);
        let num2 = 2.0f32.mul_add(sigma12, C2);
        let numerator = num1 * num2;

        let den1 = mu1_sq + mu2_sq + C1;
        let den2 = sigma1_sq + sigma2_sq + C2;
        let denominator = den1 * den2;

        let ssim = numerator / denominator;
        *ssim_out.byte_add(y * ssim_out_pitch).add(x) = ssim;
    }
}

/// Compute |scalar - value| for each pixel (for MAD computation).
///
/// Used to compute the Mean Absolute Deviation from the average SSIM.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn compute_abs_diff_scalar(
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    scalar: f32, // The value to subtract (avg)
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();

    if x < width && y < height {
        let val = *src.byte_add(y * src_pitch).add(x);
        let abs_diff = (scalar - val).abs();
        *dst.byte_add(y * dst_pitch).add(x) = abs_diff;
    }
}
