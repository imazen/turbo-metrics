//! DSSIM-CUDA: GPU-accelerated DSSIM image quality metric
//!
//! This crate provides a CUDA implementation of the DSSIM (Structural Dissimilarity)
//! image quality metric, following the algorithm from the dssim-core crate.
//!
//! # Features
//!
//! - Multi-scale SSIM computation (5 scales)
//! - LAB color space for perceptual accuracy
//! - Precompute API for efficient batch comparisons
//!
//! # Example
//!
//! ```ignore
//! use dssim_cuda::Dssim;
//!
//! let mut dssim = Dssim::new(1920, 1080, &stream)?;
//! let score = dssim.compute_sync(&ref_image, &dis_image, &stream)?;
//! ```

mod kernel;

use cudarse_driver::sys::CuError;
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, C};

pub use kernel::Kernel;

/// Error type for DSSIM operations
#[derive(Debug)]
pub enum Error {
    Cuda(CuError),
    Npp(cudarse_npp::sys::Error),
}

impl From<CuError> for Error {
    fn from(e: CuError) -> Self {
        Error::Cuda(e)
    }
}

impl From<cudarse_npp::sys::Error> for Error {
    fn from(e: cudarse_npp::sys::Error) -> Self {
        Error::Npp(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cuda(e) => write!(f, "CUDA error: {:?}", e),
            Error::Npp(e) => write!(f, "NPP error: {:?}", e),
        }
    }
}

impl std::error::Error for Error {}

/// Result type for DSSIM operations
pub type Result<T> = std::result::Result<T, Error>;

/// Number of scales in the multi-scale SSIM computation
const NUM_SCALES: usize = 5;

/// Scale weights from dssim-core
const SCALE_WEIGHTS: [f64; NUM_SCALES] = [0.028, 0.197, 0.322, 0.298, 0.155];

/// Per-scale buffers for DSSIM computation
struct ScaleBuffers {
    /// Width at this scale
    width: u32,
    /// Height at this scale
    height: u32,

    // Linear RGB at this scale (for RGB-based downsampling like dssim-core)
    linear_rgb_ref: Image<f32, C<3>>,
    linear_rgb_dis: Image<f32, C<3>>,

    // LAB channel planes for reference image
    ref_l: Image<f32, C<1>>,
    ref_a: Image<f32, C<1>>,
    ref_b: Image<f32, C<1>>,

    // LAB channel planes for distorted image
    dis_l: Image<f32, C<1>>,
    dis_a: Image<f32, C<1>>,
    dis_b: Image<f32, C<1>>,

    // Blur statistics for reference: mu, blur(x^2)
    ref_mu_l: Image<f32, C<1>>,
    ref_mu_a: Image<f32, C<1>>,
    ref_mu_b: Image<f32, C<1>>,
    ref_sq_blur_l: Image<f32, C<1>>,
    ref_sq_blur_a: Image<f32, C<1>>,
    ref_sq_blur_b: Image<f32, C<1>>,

    // Blur statistics for distorted: mu, blur(x^2)
    dis_mu_l: Image<f32, C<1>>,
    dis_mu_a: Image<f32, C<1>>,
    dis_mu_b: Image<f32, C<1>>,
    dis_sq_blur_l: Image<f32, C<1>>,
    dis_sq_blur_a: Image<f32, C<1>>,
    dis_sq_blur_b: Image<f32, C<1>>,

    // Cross-correlation: blur(ref * dis)
    cross_blur_l: Image<f32, C<1>>,
    cross_blur_a: Image<f32, C<1>>,
    cross_blur_b: Image<f32, C<1>>,

    // Temporary buffer for blur operations
    temp1: Image<f32, C<1>>,
}

impl ScaleBuffers {
    fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            width,
            height,
            // Linear RGB
            linear_rgb_ref: Image::malloc(width, height)?,
            linear_rgb_dis: Image::malloc(width, height)?,
            // Reference LAB
            ref_l: Image::malloc(width, height)?,
            ref_a: Image::malloc(width, height)?,
            ref_b: Image::malloc(width, height)?,
            // Distorted LAB
            dis_l: Image::malloc(width, height)?,
            dis_a: Image::malloc(width, height)?,
            dis_b: Image::malloc(width, height)?,
            // Reference blur stats
            ref_mu_l: Image::malloc(width, height)?,
            ref_mu_a: Image::malloc(width, height)?,
            ref_mu_b: Image::malloc(width, height)?,
            ref_sq_blur_l: Image::malloc(width, height)?,
            ref_sq_blur_a: Image::malloc(width, height)?,
            ref_sq_blur_b: Image::malloc(width, height)?,
            // Distorted blur stats
            dis_mu_l: Image::malloc(width, height)?,
            dis_mu_a: Image::malloc(width, height)?,
            dis_mu_b: Image::malloc(width, height)?,
            dis_sq_blur_l: Image::malloc(width, height)?,
            dis_sq_blur_a: Image::malloc(width, height)?,
            dis_sq_blur_b: Image::malloc(width, height)?,
            // Cross-correlation
            cross_blur_l: Image::malloc(width, height)?,
            cross_blur_a: Image::malloc(width, height)?,
            cross_blur_b: Image::malloc(width, height)?,
            // Temporary buffer
            temp1: Image::malloc(width, height)?,
        })
    }

    fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// DSSIM GPU computation context
pub struct Dssim {
    kernel: Kernel,
    /// Buffers for each scale (includes RGB at each scale)
    scales: [ScaleBuffers; NUM_SCALES],
}

impl Dssim {
    /// Create a new DSSIM computation context for the given image dimensions.
    ///
    /// Allocates all necessary GPU buffers for the multi-scale pipeline.
    pub fn new(width: u32, height: u32, _stream: &CuStream) -> Result<Self> {
        let kernel = Kernel::load();

        // Allocate per-scale buffers
        let mut w = width;
        let mut h = height;
        let scales = array_init::try_array_init(|_| {
            let buffers = ScaleBuffers::new(w, h);
            // Halve dimensions for next scale (with minimum)
            w = (w + 1) / 2;
            h = (h + 1) / 2;
            if w < 8 {
                w = 8;
            }
            if h < 8 {
                h = 8;
            }
            buffers
        })?;

        Ok(Self { kernel, scales })
    }

    /// Compute DSSIM score between two sRGB images (synchronous).
    ///
    /// Both images must be the same dimensions as specified in `new()`.
    pub fn compute_sync(
        &mut self,
        ref_srgb: &Image<u8, C<3>>,
        dis_srgb: &Image<u8, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        // Stage 1: sRGB -> Linear RGB (into scale 0's RGB buffers)
        self.kernel
            .srgb_to_linear(stream, ref_srgb, &mut self.scales[0].linear_rgb_ref);
        self.kernel
            .srgb_to_linear(stream, dis_srgb, &mut self.scales[0].linear_rgb_dis);

        // Stage 2: Multi-scale processing
        // dssim-core: downsample RGB, convert to LAB at each scale, then compute SSIM
        let mut weighted_ssim_sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for scale in 0..NUM_SCALES {
            // For scales > 0: downsample RGB from previous scale
            if scale > 0 {
                let (left, right) = self.scales.split_at_mut(scale);
                let prev_scale = &left[scale - 1];
                let curr_scale = &mut right[0];

                // Downsample RGB (not LAB!) like dssim-core
                self.kernel.downscale_rgb_by_2(
                    stream,
                    &prev_scale.linear_rgb_ref,
                    &mut curr_scale.linear_rgb_ref,
                );
                self.kernel.downscale_rgb_by_2(
                    stream,
                    &prev_scale.linear_rgb_dis,
                    &mut curr_scale.linear_rgb_dis,
                );
            }

            // Convert RGB to LAB at this scale
            {
                let s = &mut self.scales[scale];
                self.kernel.linear_to_lab_planar(
                    stream,
                    &s.linear_rgb_ref,
                    &mut s.ref_l,
                    &mut s.ref_a,
                    &mut s.ref_b,
                );
                self.kernel.linear_to_lab_planar(
                    stream,
                    &s.linear_rgb_dis,
                    &mut s.dis_l,
                    &mut s.dis_a,
                    &mut s.dis_b,
                );
            }

            // Apply chroma pre-blur (dssim-core blurs a/b channels in place before SSIM)
            // This is a two-pass blur on the image data itself
            {
                let s = &mut self.scales[scale];
                // Pre-blur ref_a (two-pass)
                self.kernel.blur_3x3(stream, &s.ref_a, &mut s.temp1);
                self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_a);
                // Pre-blur ref_b (two-pass)
                self.kernel.blur_3x3(stream, &s.ref_b, &mut s.temp1);
                self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_b);
                // Pre-blur dis_a (two-pass)
                self.kernel.blur_3x3(stream, &s.dis_a, &mut s.temp1);
                self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_a);
                // Pre-blur dis_b (two-pass)
                self.kernel.blur_3x3(stream, &s.dis_b, &mut s.temp1);
                self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_b);
            }

            // Process this scale
            let scale_score = self.process_scale(scale, stream)?;
            weighted_ssim_sum += scale_score * SCALE_WEIGHTS[scale];
            weight_sum += SCALE_WEIGHTS[scale];
        }

        // Compute final SSIM and convert to DSSIM
        let ssim = weighted_ssim_sum / weight_sum;
        let dssim = ssim_to_dssim(ssim);

        Ok(dssim)
    }

    /// Process a single scale, computing SSIM for all 3 LAB channels.
    /// Returns the score for this scale using dssim-core's formula.
    fn process_scale(&mut self, scale: usize, stream: &CuStream) -> Result<f64> {
        let s = &mut self.scales[scale];
        let pixel_count = s.pixel_count();

        // dssim-core applies blur TWICE (two-pass Gaussian)
        // Note: chroma channels are already pre-blurred above

        // Process L channel - TWO-PASS blur for mu
        self.kernel.blur_3x3(stream, &s.ref_l, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_mu_l);
        self.kernel.blur_3x3(stream, &s.dis_l, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_mu_l);

        // blur_squared and blur_product do single blur, so add second pass
        self.kernel.blur_squared(stream, &s.ref_l, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_sq_blur_l);
        self.kernel.blur_squared(stream, &s.dis_l, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_sq_blur_l);
        self.kernel
            .blur_product(stream, &s.ref_l, &s.dis_l, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.cross_blur_l);

        // Process a channel - TWO-PASS blur (already pre-blurred, so 4 total passes)
        self.kernel.blur_3x3(stream, &s.ref_a, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_mu_a);
        self.kernel.blur_3x3(stream, &s.dis_a, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_mu_a);
        self.kernel.blur_squared(stream, &s.ref_a, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_sq_blur_a);
        self.kernel.blur_squared(stream, &s.dis_a, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_sq_blur_a);
        self.kernel
            .blur_product(stream, &s.ref_a, &s.dis_a, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.cross_blur_a);

        // Process b channel - TWO-PASS blur (already pre-blurred, so 4 total passes)
        self.kernel.blur_3x3(stream, &s.ref_b, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_mu_b);
        self.kernel.blur_3x3(stream, &s.dis_b, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_mu_b);
        self.kernel.blur_squared(stream, &s.ref_b, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.ref_sq_blur_b);
        self.kernel.blur_squared(stream, &s.dis_b, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.dis_sq_blur_b);
        self.kernel
            .blur_product(stream, &s.ref_b, &s.dis_b, &mut s.temp1);
        self.kernel.blur_3x3(stream, &s.temp1, &mut s.cross_blur_b);

        // Sync and download statistics
        stream.sync().map_err(Error::Cuda)?;

        // Download all statistics to CPU for proper SSIM computation
        let mu1_l = s.ref_mu_l.copy_to_cpu(stream.inner() as _)?;
        let mu1_a = s.ref_mu_a.copy_to_cpu(stream.inner() as _)?;
        let mu1_b = s.ref_mu_b.copy_to_cpu(stream.inner() as _)?;
        let mu2_l = s.dis_mu_l.copy_to_cpu(stream.inner() as _)?;
        let mu2_a = s.dis_mu_a.copy_to_cpu(stream.inner() as _)?;
        let mu2_b = s.dis_mu_b.copy_to_cpu(stream.inner() as _)?;

        let sq1_l = s.ref_sq_blur_l.copy_to_cpu(stream.inner() as _)?;
        let sq1_a = s.ref_sq_blur_a.copy_to_cpu(stream.inner() as _)?;
        let sq1_b = s.ref_sq_blur_b.copy_to_cpu(stream.inner() as _)?;
        let sq2_l = s.dis_sq_blur_l.copy_to_cpu(stream.inner() as _)?;
        let sq2_a = s.dis_sq_blur_a.copy_to_cpu(stream.inner() as _)?;
        let sq2_b = s.dis_sq_blur_b.copy_to_cpu(stream.inner() as _)?;

        let cross_l = s.cross_blur_l.copy_to_cpu(stream.inner() as _)?;
        let cross_a = s.cross_blur_a.copy_to_cpu(stream.inner() as _)?;
        let cross_b = s.cross_blur_b.copy_to_cpu(stream.inner() as _)?;

        stream.sync().map_err(Error::Cuda)?;

        // SSIM constants
        const C1: f32 = 0.0001; // 0.01^2
        const C2: f32 = 0.0009; // 0.03^2

        // Compute per-pixel SSIM using dssim-core's approach:
        // Average the L,a,b statistics before computing SSIM
        let mut ssim_map = Vec::with_capacity(pixel_count);

        for i in 0..pixel_count {
            // Compute mu products (averaging component-wise products like dssim-core)
            let mu1_sq = (mu1_l[i] * mu1_l[i] + mu1_a[i] * mu1_a[i] + mu1_b[i] * mu1_b[i]) / 3.0;
            let mu2_sq = (mu2_l[i] * mu2_l[i] + mu2_a[i] * mu2_a[i] + mu2_b[i] * mu2_b[i]) / 3.0;
            let mu1_mu2 = (mu1_l[i] * mu2_l[i] + mu1_a[i] * mu2_a[i] + mu1_b[i] * mu2_b[i]) / 3.0;

            // Average blur(img^2) and blur(img1*img2) across channels
            let img1_sq_blur = (sq1_l[i] + sq1_a[i] + sq1_b[i]) / 3.0;
            let img2_sq_blur = (sq2_l[i] + sq2_a[i] + sq2_b[i]) / 3.0;
            let img12_blur = (cross_l[i] + cross_a[i] + cross_b[i]) / 3.0;

            // Compute sigma values
            let sigma1_sq = img1_sq_blur - mu1_sq;
            let sigma2_sq = img2_sq_blur - mu2_sq;
            let sigma12 = img12_blur - mu1_mu2;

            // SSIM formula
            let ssim = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
                / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

            ssim_map.push(ssim);
        }

        // Compute score using dssim-core's formula:
        // avg = mean(ssim)^(0.5^scale_index)
        // score = 1 - mean(|avg - ssim_i|)
        let sum: f64 = ssim_map.iter().map(|&x| x as f64).sum();
        let len = pixel_count as f64;
        let mean_ssim = sum / len;
        let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(scale as i32));

        let mad: f64 = ssim_map
            .iter()
            .map(|&x| (avg - x as f64).abs())
            .sum::<f64>()
            / len;

        let score = 1.0 - mad;

        Ok(score)
    }

    /// Get estimated GPU memory usage in bytes.
    pub fn mem_usage(&self) -> usize {
        let scale_size: usize = self
            .scales
            .iter()
            .map(|s| {
                let pixels = s.width as usize * s.height as usize;
                // 2 RGB buffers (3 channels each) + 24 f32 planar buffers + 1 temp buffer
                pixels * 4 * (2 * 3 + 24 + 1)
            })
            .sum();

        scale_size
    }
}

/// Convert SSIM (0-1, higher is better) to DSSIM (0+, lower is better)
fn ssim_to_dssim(ssim: f64) -> f64 {
    1.0 / ssim.max(f64::EPSILON) - 1.0
}
