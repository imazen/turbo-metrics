//! Butteraugli CUDA implementation
//!
//! GPU-accelerated implementation of the Butteraugli perceptual image quality metric.
//!
//! Based on the Vship GPU implementation (https://github.com/Line-fr/Vship).
//!
//! # Example
//!
//! ```ignore
//! use butteraugli_cuda::Butteraugli;
//! use cudarse_npp::image::{Image, C};
//! use cudarse_npp::image::isu::Malloc;
//!
//! // Initialize CUDA context first
//! cudarse_driver::init_cuda_and_primary_ctx().unwrap();
//!
//! let mut butteraugli = Butteraugli::new(1920, 1080).unwrap();
//! let reference: Image<u8, C<3>> = Image::malloc(1920, 1080).unwrap();
//! let distorted: Image<u8, C<3>> = Image::malloc(1920, 1080).unwrap();
//! let score = butteraugli.compute(reference.full_view(), distorted.full_view()).unwrap();
//! println!("Butteraugli distance: {}", score);
//! ```

mod kernel;

use cudarse_driver::{CuBox, CuStream};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, C};
use cudarse_npp::ScratchBuffer;

pub use kernel::Kernel;

/// Error type for Butteraugli operations
#[derive(Debug)]
pub enum Error {
    /// CUDA driver error
    Cuda(String),
    /// NPP error
    Npp(String),
    /// Invalid dimensions
    InvalidDimensions(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cuda(s) => write!(f, "CUDA error: {}", s),
            Error::Npp(s) => write!(f, "NPP error: {}", s),
            Error::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
        }
    }
}

impl std::error::Error for Error {}

/// Butteraugli algorithm constants
mod consts {
    // Blur sigmas for different stages
    pub const SIGMA_OPSIN: f32 = 1.2; // Opsin dynamics blur (local adaptation)
    pub const SIGMA_LF: f32 = 7.15593339443; // LF separation

    // Pre-computed normalized weights for sigma=1.2 mirrored blur (5x5 kernel)
    // Computed using: scaler = -1/(2*sigma²), weight_i = exp(scaler*i²), normalized
    // sigma=1.2, scaler=-0.3472222, diff=2
    // kernel=[0.24935222, 0.7066483, 1.0, 0.7066483, 0.24935222], sum=2.9120011
    pub const OPSIN_BLUR_W0: f32 = 0.3434064686298370; // center weight
    pub const OPSIN_BLUR_W1: f32 = 0.2426676005125046; // ±1 offset weight
    pub const OPSIN_BLUR_W2: f32 = 0.0856291651725769; // ±2 offset weight
    pub const SIGMA_MF: f32 = 3.22489901262; // MF separation
    pub const SIGMA_HF: f32 = 1.56416327805; // HF separation
    pub const SIGMA_UHF: f32 = 1.2; // UHF blur
    pub const SIGMA_MASK: f32 = 2.7; // Mask blur

    // Range processing constants (from CPU butteraugli consts)
    pub const REMOVE_MF_RANGE: f32 = 0.29; // Remove range for MF X channel
    pub const ADD_MF_RANGE: f32 = 0.1; // Amplify range for MF Y channel
    pub const REMOVE_HF_RANGE: f32 = 1.5; // Remove range for HF X channel
    pub const REMOVE_UHF_RANGE: f32 = 0.04; // Remove range for UHF X channel

    // Cross-channel suppression
    pub const SUPPRESS_XY: f32 = 46.0; // suppress_x_by_y weight

    // Malta filter weights for UHF band (Y channel)
    // w_0gt1 = W * hf_asymmetry, w_0lt1 = W / hf_asymmetry
    pub const W_UHF_MALTA: f32 = 1.10039032555;
    pub const W_UHF_MALTA_0GT1: f32 = W_UHF_MALTA * HF_ASYMMETRY; // = 0.88
    pub const W_UHF_MALTA_0LT1: f32 = W_UHF_MALTA / HF_ASYMMETRY; // = 1.375
    pub const NORM1_UHF: f32 = 71.7800275169;

    // Malta filter weights for UHF X channel
    pub const W_UHF_MALTA_X: f32 = 173.5;
    pub const W_UHF_MALTA_X_0GT1: f32 = W_UHF_MALTA_X * HF_ASYMMETRY; // = 138.8
    pub const W_UHF_MALTA_X_0LT1: f32 = W_UHF_MALTA_X / HF_ASYMMETRY; // = 216.875
    pub const NORM1_UHF_X: f32 = 5.0;

    // Malta filter weights for HF band (Y channel)
    // HF uses sqrt(hf_asymmetry) for asymmetry
    pub const W_HF_MALTA: f32 = 18.7237414387;
    pub const W_HF_MALTA_0GT1: f32 = W_HF_MALTA * 0.894427191; // sqrt(0.8) ≈ 0.894
    pub const W_HF_MALTA_0LT1: f32 = W_HF_MALTA / 0.894427191;
    pub const NORM1_HF: f32 = 4498534.45232;

    // Malta filter weights for HF X channel
    pub const W_HF_MALTA_X: f32 = 6923.99476109;
    pub const W_HF_MALTA_X_0GT1: f32 = W_HF_MALTA_X * 0.894427191;
    pub const W_HF_MALTA_X_0LT1: f32 = W_HF_MALTA_X / 0.894427191;
    pub const NORM1_HF_X: f32 = 8051.15833247;

    // Malta LF weights for MF band (Y channel)
    // MF uses sqrt(hf_asymmetry) for asymmetry
    pub const W_MF_MALTA: f32 = 37.0819870399;
    pub const W_MF_MALTA_0GT1: f32 = W_MF_MALTA * 0.894427191;
    pub const W_MF_MALTA_0LT1: f32 = W_MF_MALTA / 0.894427191;
    pub const NORM1_MF: f32 = 130262059.556;

    // Malta LF weights for MF X channel
    pub const W_MF_MALTA_X: f32 = 8246.75321353;
    pub const W_MF_MALTA_X_0GT1: f32 = W_MF_MALTA_X * 0.894427191;
    pub const W_MF_MALTA_X_0LT1: f32 = W_MF_MALTA_X / 0.894427191;
    pub const NORM1_MF_X: f32 = 1009002.70582;

    // L2 difference weights [DC_X, DC_Y, DC_B, MF_X, MF_Y, MF_B, LF_X, LF_Y, LF_B]
    pub const WMUL: [f32; 9] = [
        400.0,          // DC X
        1.50815703118,  // DC Y
        0.0,            // DC B (unused)
        2150.0,         // MF X
        10.6195433239,  // MF Y
        16.2176043152,  // MF B
        29.2353797994,  // LF X
        0.844626970982, // LF Y
        0.703646627719, // LF B
    ];

    // Intensity target for opsin dynamics (nits for 1.0 input)
    // Default is 80.0 for standard sRGB content
    pub const INTENSITY_TARGET: f32 = 80.0;

    // HF asymmetry factor
    pub const HF_ASYMMETRY: f32 = 0.8;

    // Norm exponent for Butteraugli distance
    pub const NORM_Q: f32 = 4.0;
}

/// GPU-accelerated Butteraugli quality metric
///
/// Computes the perceptual distance between two images using the Butteraugli
/// algorithm on the GPU.
pub struct Butteraugli {
    kernel: Kernel,
    stream: CuStream,
    width: usize,
    height: usize,
    size: usize,

    // GPU buffers for linear RGB (interleaved, for sRGB conversion)
    linear1: Image<f32, C<3>>,
    linear2: Image<f32, C<3>>,

    // Planar linear RGB buffers for opsin dynamics [R, G, B]
    linear_planar1: [CuBox<[f32]>; 3],
    linear_planar2: [CuBox<[f32]>; 3],
    // Blurred planar linear RGB for opsin dynamics adaptation
    linear_blur1: [CuBox<[f32]>; 3],
    linear_blur2: [CuBox<[f32]>; 3],

    // Planar XYB buffers for each image [X, Y, B]
    xyb1: [CuBox<[f32]>; 3],
    xyb2: [CuBox<[f32]>; 3],

    // Frequency bands for image 1: [UHF, HF, MF, LF] x [X, Y, B]
    // We reuse some buffers, so we just need enough temp space
    freq1: [[CuBox<[f32]>; 3]; 4], // [band][channel]
    freq2: [[CuBox<[f32]>; 3]; 4],

    // Block difference accumulators [X, Y, B]
    block_diff_dc: [CuBox<[f32]>; 3],
    block_diff_ac: [CuBox<[f32]>; 3],

    // Masking buffers
    mask: CuBox<[f32]>,
    mask_temp: CuBox<[f32]>,

    // Final diffmap
    diffmap: CuBox<[f32]>,

    // Temp buffers for blur operations
    temp1: CuBox<[f32]>,
    temp2: CuBox<[f32]>,

    // Half-resolution buffers for multi-scale
    half_width: usize,
    half_height: usize,
    half_size: usize,
    diffmap_half: CuBox<[f32]>,

    // Reduction scratch buffer
    sum_scratch: ScratchBuffer,
    sum_result: CuBox<[f64]>,
}

impl Butteraugli {
    /// Create a new Butteraugli instance for images of the given dimensions
    pub fn new(width: u32, height: u32) -> Result<Self, Error> {
        let stream = CuStream::new().map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        let kernel = Kernel::load();

        let width = width as usize;
        let height = height as usize;
        let size = width * height;

        // Allocate main image buffers (interleaved for NPP compatibility)
        let linear1 = Image::<f32, C<3>>::malloc(width as u32, height as u32)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;
        let linear2 = Image::<f32, C<3>>::malloc(width as u32, height as u32)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;

        // Helper to allocate planar buffers
        let alloc_plane = || -> Result<CuBox<[f32]>, Error> {
            CuBox::<[f32]>::new_zeroed(size, &stream).map_err(|e| Error::Cuda(format!("{:?}", e)))
        };

        let alloc_3_planes = || -> Result<[CuBox<[f32]>; 3], Error> {
            Ok([alloc_plane()?, alloc_plane()?, alloc_plane()?])
        };

        // Planar linear RGB for opsin dynamics
        let linear_planar1 = alloc_3_planes()?;
        let linear_planar2 = alloc_3_planes()?;
        let linear_blur1 = alloc_3_planes()?;
        let linear_blur2 = alloc_3_planes()?;

        // XYB planes
        let xyb1 = alloc_3_planes()?;
        let xyb2 = alloc_3_planes()?;

        // Frequency bands: [UHF, HF, MF, LF] x [X, Y, B]
        let freq1 = [
            alloc_3_planes()?,
            alloc_3_planes()?,
            alloc_3_planes()?,
            alloc_3_planes()?,
        ];
        let freq2 = [
            alloc_3_planes()?,
            alloc_3_planes()?,
            alloc_3_planes()?,
            alloc_3_planes()?,
        ];

        // Block differences
        let block_diff_dc = alloc_3_planes()?;
        let block_diff_ac = alloc_3_planes()?;

        // Masking
        let mask = alloc_plane()?;
        let mask_temp = alloc_plane()?;

        // Diffmap
        let diffmap = alloc_plane()?;

        // Temp buffers for blur
        let temp1 = alloc_plane()?;
        let temp2 = alloc_plane()?;

        // Half-resolution for multi-scale
        let half_width = (width + 1) / 2;
        let half_height = (height + 1) / 2;
        let half_size = half_width * half_height;
        let diffmap_half = CuBox::<[f32]>::new_zeroed(half_size, &stream)
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        // Reduction buffers
        // We need to sum the diffmap^q, so allocate scratch for NPP Sum
        let sum_scratch = ScratchBuffer::alloc_len(size * 4 + 1024, stream.inner() as _)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;
        let sum_result =
            CuBox::<[f64]>::new_zeroed(1, &stream).map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        Ok(Self {
            kernel,
            stream,
            width,
            height,
            size,
            linear1,
            linear2,
            linear_planar1,
            linear_planar2,
            linear_blur1,
            linear_blur2,
            xyb1,
            xyb2,
            freq1,
            freq2,
            block_diff_dc,
            block_diff_ac,
            mask,
            mask_temp,
            diffmap,
            temp1,
            temp2,
            half_width,
            half_height,
            half_size,
            diffmap_half,
            sum_scratch,
            sum_result,
        })
    }

    /// Get the image dimensions this instance was created for
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }

    /// Helper to get raw pointer from CuBox
    fn ptr(buf: &CuBox<[f32]>) -> *const f32 {
        buf.ptr() as *const f32
    }

    /// Helper to get mutable raw pointer from CuBox
    fn ptr_mut(buf: &mut CuBox<[f32]>) -> *mut f32 {
        buf.ptr() as *mut f32
    }

    /// Compute Butteraugli distance between two images
    ///
    /// Both images must be sRGB, 8-bit per channel, with dimensions matching
    /// what this instance was created with.
    ///
    /// Returns the Butteraugli distance score (lower is better, 0 = identical).
    pub fn compute(
        &mut self,
        reference: impl Img<u8, C<3>>,
        distorted: impl Img<u8, C<3>>,
    ) -> Result<f32, Error> {
        // Validate dimensions
        if reference.width() as usize != self.width
            || reference.height() as usize != self.height
            || distorted.width() as usize != self.width
            || distorted.height() as usize != self.height
        {
            return Err(Error::InvalidDimensions(format!(
                "Expected {}x{}, got ref={}x{}, dist={}x{}",
                self.width,
                self.height,
                reference.width(),
                reference.height(),
                distorted.width(),
                distorted.height()
            )));
        }

        // === Stage 1: Color Space Conversion ===
        // sRGB -> linear RGB -> XYB with opsin dynamics

        // Convert sRGB to linear RGB
        self.kernel
            .srgb_to_linear(&self.stream, reference, self.linear1.full_view_mut());
        self.kernel
            .srgb_to_linear(&self.stream, distorted, self.linear2.full_view_mut());

        // Deinterleave linear RGB to planar format (stored in xyb buffers temporarily)
        // We do this BEFORE opsin dynamics so we can downsample for multi-scale
        self.kernel.deinterleave_3ch(
            &self.stream,
            self.linear1.full_view(),
            Self::ptr_mut(&mut self.xyb1[0]),
            Self::ptr_mut(&mut self.xyb1[1]),
            Self::ptr_mut(&mut self.xyb1[2]),
        );
        self.kernel.deinterleave_3ch(
            &self.stream,
            self.linear2.full_view(),
            Self::ptr_mut(&mut self.xyb2[0]),
            Self::ptr_mut(&mut self.xyb2[1]),
            Self::ptr_mut(&mut self.xyb2[2]),
        );

        // Downsample planar linear RGB to half-resolution for multi-scale processing
        // Store in linear_planar1/2 (using first half_size elements only)
        for ch in 0..3 {
            self.kernel.downsample_2x(
                &self.stream,
                Self::ptr(&self.xyb1[ch]),
                Self::ptr_mut(&mut self.linear_planar1[ch]),
                self.width,
                self.height,
                self.half_width,
                self.half_height,
            );
            self.kernel.downsample_2x(
                &self.stream,
                Self::ptr(&self.xyb2[ch]),
                Self::ptr_mut(&mut self.linear_planar2[ch]),
                self.width,
                self.height,
                self.half_width,
                self.half_height,
            );
        }

        // Now convert planar linear RGB to XYB with opsin dynamics
        self.convert_planar_to_xyb()?;

        // === Stage 2: Frequency Separation ===
        self.separate_frequencies()?;

        // === Stage 3: Compute Differences ===
        self.compute_differences()?;

        // === Stage 4: Psychovisual Masking ===
        self.compute_masking()?;

        // === Stage 5: Combine into Diffmap ===
        self.combine_diffmap()?;

        // === Stage 6: Multi-scale Processing ===
        // Run the pipeline at half resolution and combine with full-res result
        // Only do multi-scale if image is large enough (matches CPU threshold of 15x15)
        const MIN_SIZE_FOR_MULTISCALE: usize = 15;
        if self.width >= MIN_SIZE_FOR_MULTISCALE && self.height >= MIN_SIZE_FOR_MULTISCALE {
            self.run_half_scale_and_combine()?;
        }

        // === Stage 7: Reduce to Score ===
        let score = self.reduce_to_score()?;

        Ok(score)
    }

    /// Convert planar linear RGB to XYB color space using full opsin dynamics
    ///
    /// Assumes xyb1/xyb2 already contain planar linear RGB (from deinterleave step).
    /// This implements the proper Butteraugli color space conversion with
    /// blur-based local adaptation, matching libjxl's OpsinDynamicsImage.
    fn convert_planar_to_xyb(&mut self) -> Result<(), Error> {
        let w = self.width;
        let h = self.height;

        // === Image 1 ===
        // xyb1 currently contains planar linear RGB

        // Blur each channel with sigma=1.2 for local adaptation
        // Uses MIRRORED boundaries to match CPU butteraugli exactly
        for ch in 0..3 {
            self.kernel.blur_mirrored_5x5(
                &self.stream,
                Self::ptr(&self.xyb1[ch]), // source: planar linear RGB
                Self::ptr_mut(&mut self.linear_blur1[ch]), // dest: blurred for adaptation
                Self::ptr_mut(&mut self.temp1), // scratch buffer
                w,
                h,
                consts::OPSIN_BLUR_W0,
                consts::OPSIN_BLUR_W1,
                consts::OPSIN_BLUR_W2,
            );
        }

        // Apply opsin dynamics transformation
        // This transforms xyb1 from linear RGB to XYB in place
        self.kernel.opsin_dynamics(
            &self.stream,
            Self::ptr_mut(&mut self.xyb1[0]), // R -> X (modified in place)
            Self::ptr_mut(&mut self.xyb1[1]), // G -> Y (modified in place)
            Self::ptr_mut(&mut self.xyb1[2]), // B -> B (modified in place)
            Self::ptr(&self.linear_blur1[0]), // blurred R for adaptation
            Self::ptr(&self.linear_blur1[1]), // blurred G for adaptation
            Self::ptr(&self.linear_blur1[2]), // blurred B for adaptation
            w,
            h,
            consts::INTENSITY_TARGET,
        );

        // === Image 2 ===
        // Also use mirrored blur for consistent boundary handling
        for ch in 0..3 {
            self.kernel.blur_mirrored_5x5(
                &self.stream,
                Self::ptr(&self.xyb2[ch]),
                Self::ptr_mut(&mut self.linear_blur2[ch]),
                Self::ptr_mut(&mut self.temp1),
                w,
                h,
                consts::OPSIN_BLUR_W0,
                consts::OPSIN_BLUR_W1,
                consts::OPSIN_BLUR_W2,
            );
        }

        self.kernel.opsin_dynamics(
            &self.stream,
            Self::ptr_mut(&mut self.xyb2[0]),
            Self::ptr_mut(&mut self.xyb2[1]),
            Self::ptr_mut(&mut self.xyb2[2]),
            Self::ptr(&self.linear_blur2[0]),
            Self::ptr(&self.linear_blur2[1]),
            Self::ptr(&self.linear_blur2[2]),
            w,
            h,
            consts::INTENSITY_TARGET,
        );

        Ok(())
    }

    /// Separate XYB into frequency bands (UHF, HF, MF, LF)
    ///
    /// Matches CPU butteraugli's cascaded frequency separation:
    /// 1. LF = blur(src, sigma_lf=7.156)
    /// 2. MF_raw = src - LF
    /// 3. MF = blur(MF_raw, sigma=3.225)
    /// 4. HF_raw = MF_raw - MF (for X,Y channels only)
    /// 5. HF = blur(HF_raw, sigma=1.564)
    /// 6. UHF = HF_raw - HF (for X,Y channels only)
    ///
    /// Note: The sigmas are:
    /// - SIGMA_LF = 7.15593339443 (for LF extraction)
    /// - CPU SIGMA_HF = 3.22489901262 (for MF→HF separation, our SIGMA_MF)
    /// - CPU SIGMA_UHF = 1.56416327805 (for HF→UHF separation, our SIGMA_HF)
    fn separate_frequencies(&mut self) -> Result<(), Error> {
        let w = self.width;
        let h = self.height;
        let size = self.size;

        for img_idx in 0..2 {
            let (xyb, freq) = if img_idx == 0 {
                (&self.xyb1, &mut self.freq1)
            } else {
                (&self.xyb2, &mut self.freq2)
            };

            // ====== Step 1: Separate LF and MF ======
            // For all 3 channels: LF = blur(src, SIGMA_LF), MF_raw = src - LF
            for ch in 0..3 {
                let src = Self::ptr(&xyb[ch]);
                let temp1 = Self::ptr_mut(&mut self.temp1);

                // LF = blur(src, SIGMA_LF) → store in freq[3]
                let lf = Self::ptr_mut(&mut freq[3][ch]);
                self.kernel
                    .blur(&self.stream, src, lf, temp1, w, h, consts::SIGMA_LF);

                // MF_raw = src - LF → store in freq[2] temporarily
                let mf = Self::ptr_mut(&mut freq[2][ch]);
                self.kernel.subtract_arrays(&self.stream, src, lf, mf, size);
            }

            // ====== Step 2: Separate MF and HF ======
            // For X and Y channels: blur MF_raw, then HF = MF_raw - blur(MF_raw)
            // For B channel: just blur MF_raw (no HF for B)
            for ch in 0..3 {
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);

                // freq[2][ch] currently contains MF_raw
                let mf_raw = Self::ptr(&freq[2][ch]);

                // Blur MF_raw with sigma=3.225 → store in temp1
                self.kernel
                    .blur(&self.stream, mf_raw, temp1, temp2, w, h, consts::SIGMA_MF);

                if ch < 2 {
                    // For X and Y channels: HF = MF_raw - blur(MF_raw)
                    // Store HF_raw in freq[1][ch]
                    let hf = Self::ptr_mut(&mut freq[1][ch]);
                    self.kernel
                        .subtract_arrays(&self.stream, mf_raw, temp1, hf, size);
                }

                // Copy blurred MF back to freq[2][ch] (MF = blur(MF_raw))
                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[2][ch].ptr(),
                        self.temp1.ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }
            }

            // Post-process MF bands (matches CPU separate_mf_and_hf)
            // MF[X] = remove_range_around_zero(MF[X], REMOVE_MF_RANGE)
            // MF[Y] = amplify_range_around_zero(MF[Y], ADD_MF_RANGE)
            self.kernel.remove_range(
                &self.stream,
                Self::ptr_mut(&mut freq[2][0]),
                size,
                consts::REMOVE_MF_RANGE,
            );
            self.kernel.amplify_range(
                &self.stream,
                Self::ptr_mut(&mut freq[2][1]),
                size,
                consts::ADD_MF_RANGE,
            );

            // Apply suppress_x_by_y to HF[X] using HF[Y]
            self.kernel.suppress_x_by_y(
                &self.stream,
                Self::ptr_mut(&mut freq[1][0]), // HF X (modified)
                Self::ptr(&freq[1][1]),         // HF Y (read only)
                size,
                consts::SUPPRESS_XY,
            );

            // ====== Step 3: Separate HF and UHF ======
            // For X channel: standard subtraction + remove_range
            {
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);
                let hf_raw = Self::ptr(&freq[1][0]);

                // Blur HF_raw with sigma=1.564 → store in temp1
                self.kernel
                    .blur(&self.stream, hf_raw, temp1, temp2, w, h, consts::SIGMA_HF);

                // UHF = HF_raw - blur(HF_raw)
                let uhf = Self::ptr_mut(&mut freq[0][0]);
                self.kernel
                    .subtract_arrays(&self.stream, hf_raw, temp1, uhf, size);

                // HF = blur(HF_raw)
                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[1][0].ptr(),
                        self.temp1.ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }

                // Post-process X channel
                self.kernel.remove_range(
                    &self.stream,
                    Self::ptr_mut(&mut freq[1][0]),
                    size,
                    consts::REMOVE_HF_RANGE,
                );
                self.kernel.remove_range(
                    &self.stream,
                    Self::ptr_mut(&mut freq[0][0]),
                    size,
                    consts::REMOVE_UHF_RANGE,
                );
            }

            // For Y channel: use separate_hf_uhf_kernel which does maximum_clamp, scaling, and amplify
            {
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);

                // Copy HF_raw (current HF) to UHF before blurring
                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[0][1].ptr(), // UHF = HF_raw
                        freq[1][1].ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }

                // Blur HF_raw with sigma=1.564 → store in HF
                let hf_raw = Self::ptr(&freq[0][1]); // UHF now holds HF_raw
                self.kernel
                    .blur(&self.stream, hf_raw, temp1, temp2, w, h, consts::SIGMA_HF);

                // Copy blur result to HF
                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[1][1].ptr(), // HF = blur(HF_raw)
                        self.temp1.ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }

                // Apply Y channel post-processing using separate_hf_uhf_kernel
                // This does: HF = amplify(max_clamp(HF) * HF_MUL), UHF = max_clamp(UHF - max_clamp(HF)) * UHF_MUL
                self.kernel.separate_hf_uhf(
                    &self.stream,
                    Self::ptr_mut(&mut freq[1][1]), // HF Y
                    Self::ptr_mut(&mut freq[0][1]), // UHF Y (currently holds HF_raw)
                    size,
                );
            }

            // Apply XybLowFreqToVals to LF bands (multiply by 14-50x)
            // This is critical for proper DC/LF difference scaling
            self.kernel.xyb_low_freq_to_vals(
                &self.stream,
                Self::ptr_mut(&mut freq[3][0]), // LF X
                Self::ptr_mut(&mut freq[3][1]), // LF Y
                Self::ptr_mut(&mut freq[3][2]), // LF B
                size,
            );
        }

        Ok(())
    }

    /// Compute differences between frequency bands of the two images
    fn compute_differences(&mut self) -> Result<(), Error> {
        let w = self.width;
        let h = self.height;
        let size = self.size;

        // Clear accumulators
        for i in 0..3 {
            self.kernel.clear_buffer(
                &self.stream,
                Self::ptr_mut(&mut self.block_diff_dc[i]),
                size,
            );
            self.kernel.clear_buffer(
                &self.stream,
                Self::ptr_mut(&mut self.block_diff_ac[i]),
                size,
            );
        }

        // === UHF differences (Malta HF kernel) ===
        // Y channel - w_0gt1 = W * hf_asym, w_0lt1 = W / hf_asym
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[0][1]), // UHF Y image 1
            Self::ptr(&self.freq2[0][1]), // UHF Y image 2
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w,
            h,
            consts::W_UHF_MALTA_0GT1,
            consts::W_UHF_MALTA_0LT1,
            consts::NORM1_UHF,
        );

        // X channel
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[0][0]),
            Self::ptr(&self.freq2[0][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w,
            h,
            consts::W_UHF_MALTA_X_0GT1,
            consts::W_UHF_MALTA_X_0LT1,
            consts::NORM1_UHF_X,
        );

        // === HF differences (Malta LF kernel - 5 samples, not 9) ===
        // Y channel - HF uses sqrt(hf_asymmetry) for asymmetry
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[1][1]),
            Self::ptr(&self.freq2[1][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w,
            h,
            consts::W_HF_MALTA_0GT1,
            consts::W_HF_MALTA_0LT1,
            consts::NORM1_HF,
        );

        // X channel
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[1][0]),
            Self::ptr(&self.freq2[1][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w,
            h,
            consts::W_HF_MALTA_X_0GT1,
            consts::W_HF_MALTA_X_0LT1,
            consts::NORM1_HF_X,
        );

        // === MF differences (Malta LF kernel) ===
        // Y channel - MF uses SYMMETRIC weights (unlike HF which uses sqrt(hf_asymmetry))
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[2][1]),
            Self::ptr(&self.freq2[2][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w,
            h,
            consts::W_MF_MALTA, // Symmetric - same for both
            consts::W_MF_MALTA,
            consts::NORM1_MF,
        );

        // X channel - also symmetric
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[2][0]),
            Self::ptr(&self.freq2[2][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w,
            h,
            consts::W_MF_MALTA_X, // Symmetric - same for both
            consts::W_MF_MALTA_X,
            consts::NORM1_MF_X,
        );

        // === L2 differences for HF bands (asymmetric) ===
        // Adds to AC differences - CPU uses L2DiffAsymmetric
        // HF X: WMUL[0] * hf_asymmetry (0gt1), WMUL[0] / hf_asymmetry (0lt1)
        self.kernel.l2_asym_diff(
            &self.stream,
            Self::ptr(&self.freq1[1][0]), // HF X image 1
            Self::ptr(&self.freq2[1][0]), // HF X image 2
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            size,
            consts::WMUL[0] * consts::HF_ASYMMETRY,
            consts::WMUL[0] / consts::HF_ASYMMETRY,
        );
        // HF Y: WMUL[1] * hf_asymmetry, WMUL[1] / hf_asymmetry
        self.kernel.l2_asym_diff(
            &self.stream,
            Self::ptr(&self.freq1[1][1]), // HF Y image 1
            Self::ptr(&self.freq2[1][1]), // HF Y image 2
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            size,
            consts::WMUL[1] * consts::HF_ASYMMETRY,
            consts::WMUL[1] / consts::HF_ASYMMETRY,
        );

        // === L2 differences for MF bands ===
        // MF X (WMUL[3])
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][0]),
            Self::ptr(&self.freq2[2][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            size,
            consts::WMUL[3], // MF_X weight
        );
        // MF Y (WMUL[4])
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][1]),
            Self::ptr(&self.freq2[2][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            size,
            consts::WMUL[4], // MF_Y weight
        );
        // MF B (WMUL[5])
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][2]),
            Self::ptr(&self.freq2[2][2]),
            Self::ptr_mut(&mut self.block_diff_ac[2]),
            size,
            consts::WMUL[5], // MF_B weight
        );

        // === LF/DC differences (L2) ===
        // CPU uses WMUL[6+c] for DC/LF differences
        // LF X (WMUL[6])
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][0]),
            Self::ptr(&self.freq2[3][0]),
            Self::ptr_mut(&mut self.block_diff_dc[0]),
            size,
            consts::WMUL[6], // LF_X weight
        );
        // LF Y (WMUL[7])
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][1]),
            Self::ptr(&self.freq2[3][1]),
            Self::ptr_mut(&mut self.block_diff_dc[1]),
            size,
            consts::WMUL[7], // LF_Y weight
        );
        // LF B (WMUL[8])
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][2]),
            Self::ptr(&self.freq2[3][2]),
            Self::ptr_mut(&mut self.block_diff_dc[2]),
            size,
            consts::WMUL[8], // LF_B weight
        );

        Ok(())
    }

    /// Compute psychovisual masking
    ///
    /// This computes the visibility mask that determines how perceptible differences
    /// are based on local image content. Matches libjxl's MaskPsychoImage.
    ///
    /// Pipeline (matches CPU):
    /// 1. CombineChannelsForMasking(img1) → mask0
    /// 2. CombineChannelsForMasking(img2) → mask1
    /// 3. DiffPrecompute(mask0) → diff0
    /// 4. DiffPrecompute(mask1) → diff1
    /// 5. Blur(diff0) → blurred0
    /// 6. Blur(diff1) → blurred1
    /// 7. FuzzyErosion(blurred0) → final mask
    /// 8. MaskToErrorMul: add (blurred0 - blurred1)² * 10 to block_diff_ac[Y]
    fn compute_masking(&mut self) -> Result<(), Error> {
        let w = self.width;
        let h = self.height;
        let size = self.size;

        // Step 1: CombineChannelsForMasking for image 1 → mask
        // Formula: sqrt((uhf_x + hf_x)² * 2.5² + (uhf_y * 0.4 + hf_y * 0.4)²)
        self.kernel.combine_channels_for_masking(
            &self.stream,
            Self::ptr(&self.freq1[1][0]), // HF X image 1
            Self::ptr(&self.freq1[0][0]), // UHF X image 1
            Self::ptr(&self.freq1[1][1]), // HF Y image 1
            Self::ptr(&self.freq1[0][1]), // UHF Y image 1
            Self::ptr_mut(&mut self.mask),
            size,
        );

        // Step 2: CombineChannelsForMasking for image 2 → mask_temp
        self.kernel.combine_channels_for_masking(
            &self.stream,
            Self::ptr(&self.freq2[1][0]), // HF X image 2
            Self::ptr(&self.freq2[0][0]), // UHF X image 2
            Self::ptr(&self.freq2[1][1]), // HF Y image 2
            Self::ptr(&self.freq2[0][1]), // UHF Y image 2
            Self::ptr_mut(&mut self.mask_temp),
            size,
        );

        // Step 3: DiffPrecompute on mask0 → temp1
        self.kernel.diff_precompute(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr_mut(&mut self.temp1),
            size,
        );

        // Step 4: DiffPrecompute on mask1 → temp2
        self.kernel.diff_precompute(
            &self.stream,
            Self::ptr(&self.mask_temp),
            Self::ptr_mut(&mut self.temp2),
            size,
        );

        // Step 5: Blur diff0 → mask (blurred0)
        // Using mask_temp as scratch (we already consumed mask1)
        self.kernel.blur(
            &self.stream,
            Self::ptr(&self.temp1),
            Self::ptr_mut(&mut self.mask),
            Self::ptr_mut(&mut self.mask_temp), // scratch
            w,
            h,
            consts::SIGMA_MASK,
        );

        // Step 6: Blur diff1 → mask_temp (blurred1)
        // Using temp1 as scratch (we already consumed diff0)
        self.kernel.blur(
            &self.stream,
            Self::ptr(&self.temp2),
            Self::ptr_mut(&mut self.mask_temp),
            Self::ptr_mut(&mut self.temp1), // scratch
            w,
            h,
            consts::SIGMA_MASK,
        );

        // Step 7: FuzzyErosion on blurred0 → temp1 (final mask)
        self.kernel.fuzzy_erosion(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr_mut(&mut self.temp1),
            w,
            h,
        );

        // Step 8: MaskToErrorMul: add (blurred0 - blurred1)² * 10 to block_diff_ac[Y]
        // Uses mask (blurred0) and mask_temp (blurred1)
        self.kernel.mask_to_error_mul(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr(&self.mask_temp),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            size,
        );

        // Copy final mask (temp1) back to mask
        unsafe {
            cudarse_driver::sys::cuMemcpyAsync(
                self.mask.ptr(),
                self.temp1.ptr(),
                size * 4,
                self.stream.raw(),
            )
            .result()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Combine masked differences into final diffmap
    fn combine_diffmap(&mut self) -> Result<(), Error> {
        self.kernel.compute_diffmap(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr(&self.block_diff_dc[0]),
            Self::ptr(&self.block_diff_dc[1]),
            Self::ptr(&self.block_diff_dc[2]),
            Self::ptr(&self.block_diff_ac[0]),
            Self::ptr(&self.block_diff_ac[1]),
            Self::ptr(&self.block_diff_ac[2]),
            Self::ptr_mut(&mut self.diffmap),
            self.size,
        );

        Ok(())
    }

    /// Run the full pipeline at half resolution and combine with full-res diffmap
    ///
    /// This implements multi-scale processing: the half-res diffmap is upsampled
    /// and added to the full-res diffmap with weight 0.5.
    fn run_half_scale_and_combine(&mut self) -> Result<(), Error> {
        let w = self.half_width;
        let h = self.half_height;
        let size = self.half_size;

        // Half-res linear RGB is already in linear_planar1/2 (from downsample in compute())
        // We need to run the full pipeline at half resolution, reusing buffers.

        // === Step 1: Copy half-res linear to xyb buffers (reusing for half-res) ===
        // We copy to the first half_size elements of xyb1/2
        for ch in 0..3 {
            unsafe {
                cudarse_driver::sys::cuMemcpyAsync(
                    self.xyb1[ch].ptr(),
                    self.linear_planar1[ch].ptr(),
                    size * 4,
                    self.stream.raw(),
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                cudarse_driver::sys::cuMemcpyAsync(
                    self.xyb2[ch].ptr(),
                    self.linear_planar2[ch].ptr(),
                    size * 4,
                    self.stream.raw(),
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
        }

        // === Step 2: Opsin dynamics at half resolution ===
        // Blur for adaptation using mirrored boundaries
        for ch in 0..3 {
            self.kernel.blur_mirrored_5x5(
                &self.stream,
                Self::ptr(&self.xyb1[ch]),
                Self::ptr_mut(&mut self.linear_blur1[ch]),
                Self::ptr_mut(&mut self.temp1),
                w,
                h,
                consts::OPSIN_BLUR_W0,
                consts::OPSIN_BLUR_W1,
                consts::OPSIN_BLUR_W2,
            );
        }
        self.kernel.opsin_dynamics(
            &self.stream,
            Self::ptr_mut(&mut self.xyb1[0]),
            Self::ptr_mut(&mut self.xyb1[1]),
            Self::ptr_mut(&mut self.xyb1[2]),
            Self::ptr(&self.linear_blur1[0]),
            Self::ptr(&self.linear_blur1[1]),
            Self::ptr(&self.linear_blur1[2]),
            w,
            h,
            consts::INTENSITY_TARGET,
        );

        for ch in 0..3 {
            self.kernel.blur_mirrored_5x5(
                &self.stream,
                Self::ptr(&self.xyb2[ch]),
                Self::ptr_mut(&mut self.linear_blur2[ch]),
                Self::ptr_mut(&mut self.temp1),
                w,
                h,
                consts::OPSIN_BLUR_W0,
                consts::OPSIN_BLUR_W1,
                consts::OPSIN_BLUR_W2,
            );
        }
        self.kernel.opsin_dynamics(
            &self.stream,
            Self::ptr_mut(&mut self.xyb2[0]),
            Self::ptr_mut(&mut self.xyb2[1]),
            Self::ptr_mut(&mut self.xyb2[2]),
            Self::ptr(&self.linear_blur2[0]),
            Self::ptr(&self.linear_blur2[1]),
            Self::ptr(&self.linear_blur2[2]),
            w,
            h,
            consts::INTENSITY_TARGET,
        );

        // === Step 3: Frequency separation at half resolution (cascaded) ===
        for img_idx in 0..2 {
            let (xyb, freq) = if img_idx == 0 {
                (&self.xyb1, &mut self.freq1)
            } else {
                (&self.xyb2, &mut self.freq2)
            };

            // Step 3a: Separate LF and MF
            for ch in 0..3 {
                let src = Self::ptr(&xyb[ch]);
                let temp1 = Self::ptr_mut(&mut self.temp1);

                // LF = blur(src, SIGMA_LF)
                let lf = Self::ptr_mut(&mut freq[3][ch]);
                self.kernel
                    .blur(&self.stream, src, lf, temp1, w, h, consts::SIGMA_LF);

                // MF_raw = src - LF
                let mf = Self::ptr_mut(&mut freq[2][ch]);
                self.kernel.subtract_arrays(&self.stream, src, lf, mf, size);
            }

            // Step 3b: Separate MF and HF
            for ch in 0..3 {
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);
                let mf_raw = Self::ptr(&freq[2][ch]);

                // Blur MF_raw with sigma=3.225
                self.kernel
                    .blur(&self.stream, mf_raw, temp1, temp2, w, h, consts::SIGMA_MF);

                if ch < 2 {
                    // HF = MF_raw - blur(MF_raw)
                    let hf = Self::ptr_mut(&mut freq[1][ch]);
                    self.kernel
                        .subtract_arrays(&self.stream, mf_raw, temp1, hf, size);
                }

                // MF = blur(MF_raw)
                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[2][ch].ptr(),
                        self.temp1.ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }
            }

            // Post-process MF bands
            self.kernel.remove_range(
                &self.stream,
                Self::ptr_mut(&mut freq[2][0]),
                size,
                consts::REMOVE_MF_RANGE,
            );
            self.kernel.amplify_range(
                &self.stream,
                Self::ptr_mut(&mut freq[2][1]),
                size,
                consts::ADD_MF_RANGE,
            );

            // Apply suppress_x_by_y to HF[X] using HF[Y]
            self.kernel.suppress_x_by_y(
                &self.stream,
                Self::ptr_mut(&mut freq[1][0]),
                Self::ptr(&freq[1][1]),
                size,
                consts::SUPPRESS_XY,
            );

            // Step 3c: Separate HF and UHF (X channel)
            {
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);
                let hf_raw = Self::ptr(&freq[1][0]);

                self.kernel
                    .blur(&self.stream, hf_raw, temp1, temp2, w, h, consts::SIGMA_HF);

                let uhf = Self::ptr_mut(&mut freq[0][0]);
                self.kernel
                    .subtract_arrays(&self.stream, hf_raw, temp1, uhf, size);

                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[1][0].ptr(),
                        self.temp1.ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }

                self.kernel.remove_range(
                    &self.stream,
                    Self::ptr_mut(&mut freq[1][0]),
                    size,
                    consts::REMOVE_HF_RANGE,
                );
                self.kernel.remove_range(
                    &self.stream,
                    Self::ptr_mut(&mut freq[0][0]),
                    size,
                    consts::REMOVE_UHF_RANGE,
                );
            }

            // Step 3c: Separate HF and UHF (Y channel with special processing)
            {
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);

                // Copy HF_raw to UHF before blurring
                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[0][1].ptr(),
                        freq[1][1].ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }

                let hf_raw = Self::ptr(&freq[0][1]);
                self.kernel
                    .blur(&self.stream, hf_raw, temp1, temp2, w, h, consts::SIGMA_HF);

                unsafe {
                    cudarse_driver::sys::cuMemcpyAsync(
                        freq[1][1].ptr(),
                        self.temp1.ptr(),
                        size * 4,
                        self.stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }

                self.kernel.separate_hf_uhf(
                    &self.stream,
                    Self::ptr_mut(&mut freq[1][1]),
                    Self::ptr_mut(&mut freq[0][1]),
                    size,
                );
            }

            // Apply XybLowFreqToVals to LF bands
            self.kernel.xyb_low_freq_to_vals(
                &self.stream,
                Self::ptr_mut(&mut freq[3][0]),
                Self::ptr_mut(&mut freq[3][1]),
                Self::ptr_mut(&mut freq[3][2]),
                size,
            );
        }

        // === Step 4: Compute differences at half resolution ===
        // Clear accumulators
        for i in 0..3 {
            self.kernel.clear_buffer(
                &self.stream,
                Self::ptr_mut(&mut self.block_diff_dc[i]),
                size,
            );
            self.kernel.clear_buffer(
                &self.stream,
                Self::ptr_mut(&mut self.block_diff_ac[i]),
                size,
            );
        }

        // UHF Malta Y
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[0][1]),
            Self::ptr(&self.freq2[0][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w,
            h,
            consts::W_UHF_MALTA_0GT1,
            consts::W_UHF_MALTA_0LT1,
            consts::NORM1_UHF,
        );
        // UHF Malta X
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[0][0]),
            Self::ptr(&self.freq2[0][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w,
            h,
            consts::W_UHF_MALTA_X_0GT1,
            consts::W_UHF_MALTA_X_0LT1,
            consts::NORM1_UHF_X,
        );
        // HF Malta Y (LF kernel - 5 samples)
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[1][1]),
            Self::ptr(&self.freq2[1][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w,
            h,
            consts::W_HF_MALTA_0GT1,
            consts::W_HF_MALTA_0LT1,
            consts::NORM1_HF,
        );
        // HF Malta X (LF kernel - 5 samples)
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[1][0]),
            Self::ptr(&self.freq2[1][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w,
            h,
            consts::W_HF_MALTA_X_0GT1,
            consts::W_HF_MALTA_X_0LT1,
            consts::NORM1_HF_X,
        );
        // MF Malta Y - symmetric weights
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[2][1]),
            Self::ptr(&self.freq2[2][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w,
            h,
            consts::W_MF_MALTA,
            consts::W_MF_MALTA,
            consts::NORM1_MF,
        );
        // MF Malta X - symmetric weights
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[2][0]),
            Self::ptr(&self.freq2[2][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w,
            h,
            consts::W_MF_MALTA_X,
            consts::W_MF_MALTA_X,
            consts::NORM1_MF_X,
        );

        // L2 differences
        // HF X, Y
        self.kernel.l2_asym_diff(
            &self.stream,
            Self::ptr(&self.freq1[1][0]),
            Self::ptr(&self.freq2[1][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            size,
            consts::WMUL[0] * consts::HF_ASYMMETRY,
            consts::WMUL[0] / consts::HF_ASYMMETRY,
        );
        self.kernel.l2_asym_diff(
            &self.stream,
            Self::ptr(&self.freq1[1][1]),
            Self::ptr(&self.freq2[1][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            size,
            consts::WMUL[1] * consts::HF_ASYMMETRY,
            consts::WMUL[1] / consts::HF_ASYMMETRY,
        );
        // MF X, Y, B
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][0]),
            Self::ptr(&self.freq2[2][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            size,
            consts::WMUL[3],
        );
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][1]),
            Self::ptr(&self.freq2[2][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            size,
            consts::WMUL[4],
        );
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][2]),
            Self::ptr(&self.freq2[2][2]),
            Self::ptr_mut(&mut self.block_diff_ac[2]),
            size,
            consts::WMUL[5],
        );
        // LF X, Y, B
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][0]),
            Self::ptr(&self.freq2[3][0]),
            Self::ptr_mut(&mut self.block_diff_dc[0]),
            size,
            consts::WMUL[6],
        );
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][1]),
            Self::ptr(&self.freq2[3][1]),
            Self::ptr_mut(&mut self.block_diff_dc[1]),
            size,
            consts::WMUL[7],
        );
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][2]),
            Self::ptr(&self.freq2[3][2]),
            Self::ptr_mut(&mut self.block_diff_dc[2]),
            size,
            consts::WMUL[8],
        );

        // === Step 5: Masking at half resolution ===
        // Same pipeline as compute_masking but at half resolution
        // Step 1: CombineChannelsForMasking for image 1 → mask
        self.kernel.combine_channels_for_masking(
            &self.stream,
            Self::ptr(&self.freq1[1][0]),
            Self::ptr(&self.freq1[0][0]),
            Self::ptr(&self.freq1[1][1]),
            Self::ptr(&self.freq1[0][1]),
            Self::ptr_mut(&mut self.mask),
            size,
        );
        // Step 2: CombineChannelsForMasking for image 2 → mask_temp
        self.kernel.combine_channels_for_masking(
            &self.stream,
            Self::ptr(&self.freq2[1][0]),
            Self::ptr(&self.freq2[0][0]),
            Self::ptr(&self.freq2[1][1]),
            Self::ptr(&self.freq2[0][1]),
            Self::ptr_mut(&mut self.mask_temp),
            size,
        );
        // Step 3: DiffPrecompute on mask0 → temp1
        self.kernel.diff_precompute(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr_mut(&mut self.temp1),
            size,
        );
        // Step 4: DiffPrecompute on mask1 → temp2
        self.kernel.diff_precompute(
            &self.stream,
            Self::ptr(&self.mask_temp),
            Self::ptr_mut(&mut self.temp2),
            size,
        );
        // Step 5: Blur diff0 → mask (blurred0)
        self.kernel.blur(
            &self.stream,
            Self::ptr(&self.temp1),
            Self::ptr_mut(&mut self.mask),
            Self::ptr_mut(&mut self.mask_temp),
            w,
            h,
            consts::SIGMA_MASK,
        );
        // Step 6: Blur diff1 → mask_temp (blurred1)
        self.kernel.blur(
            &self.stream,
            Self::ptr(&self.temp2),
            Self::ptr_mut(&mut self.mask_temp),
            Self::ptr_mut(&mut self.temp1),
            w,
            h,
            consts::SIGMA_MASK,
        );
        // Step 7: FuzzyErosion on blurred0 → temp1 (final mask)
        self.kernel.fuzzy_erosion(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr_mut(&mut self.temp1),
            w,
            h,
        );
        // Step 8: MaskToErrorMul using blurred0 - blurred1
        self.kernel.mask_to_error_mul(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr(&self.mask_temp),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            size,
        );
        // Copy final mask (temp1) back to mask
        unsafe {
            cudarse_driver::sys::cuMemcpyAsync(
                self.mask.ptr(),
                self.temp1.ptr(),
                size * 4,
                self.stream.raw(),
            )
            .result()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        }

        // === Step 6: Combine into half-res diffmap ===
        self.kernel.compute_diffmap(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr(&self.block_diff_dc[0]),
            Self::ptr(&self.block_diff_dc[1]),
            Self::ptr(&self.block_diff_dc[2]),
            Self::ptr(&self.block_diff_ac[0]),
            Self::ptr(&self.block_diff_ac[1]),
            Self::ptr(&self.block_diff_ac[2]),
            Self::ptr_mut(&mut self.diffmap_half),
            size,
        );

        // === Step 7: Upsample and add to full-res diffmap with weight 0.5 ===
        self.kernel.add_upsample_2x(
            &self.stream,
            Self::ptr(&self.diffmap_half),
            Self::ptr_mut(&mut self.diffmap),
            self.half_width,
            self.half_height,
            self.width,
            self.height,
            0.5, // Weight for half-res contribution
        );

        Ok(())
    }

    /// Compute Butteraugli distance with control over multi-scale processing
    ///
    /// This is useful for debugging and testing individual stages.
    /// Set `enable_multiscale` to false to skip multi-scale processing.
    pub fn compute_with_options(
        &mut self,
        reference: impl Img<u8, C<3>>,
        distorted: impl Img<u8, C<3>>,
        enable_multiscale: bool,
    ) -> Result<f32, Error> {
        // Validate dimensions
        if reference.width() as usize != self.width
            || reference.height() as usize != self.height
            || distorted.width() as usize != self.width
            || distorted.height() as usize != self.height
        {
            return Err(Error::InvalidDimensions(format!(
                "Expected {}x{}, got ref={}x{}, dist={}x{}",
                self.width,
                self.height,
                reference.width(),
                reference.height(),
                distorted.width(),
                distorted.height()
            )));
        }

        // === Stage 1: Color Space Conversion ===
        self.kernel
            .srgb_to_linear(&self.stream, reference, self.linear1.full_view_mut());
        self.kernel
            .srgb_to_linear(&self.stream, distorted, self.linear2.full_view_mut());

        self.kernel.deinterleave_3ch(
            &self.stream,
            self.linear1.full_view(),
            Self::ptr_mut(&mut self.xyb1[0]),
            Self::ptr_mut(&mut self.xyb1[1]),
            Self::ptr_mut(&mut self.xyb1[2]),
        );
        self.kernel.deinterleave_3ch(
            &self.stream,
            self.linear2.full_view(),
            Self::ptr_mut(&mut self.xyb2[0]),
            Self::ptr_mut(&mut self.xyb2[1]),
            Self::ptr_mut(&mut self.xyb2[2]),
        );

        // Downsample for multi-scale (still needed even if multiscale disabled)
        for ch in 0..3 {
            self.kernel.downsample_2x(
                &self.stream,
                Self::ptr(&self.xyb1[ch]),
                Self::ptr_mut(&mut self.linear_planar1[ch]),
                self.width,
                self.height,
                self.half_width,
                self.half_height,
            );
            self.kernel.downsample_2x(
                &self.stream,
                Self::ptr(&self.xyb2[ch]),
                Self::ptr_mut(&mut self.linear_planar2[ch]),
                self.width,
                self.height,
                self.half_width,
                self.half_height,
            );
        }

        // Convert planar linear RGB to XYB with opsin dynamics
        self.convert_planar_to_xyb()?;

        // === Stage 2: Frequency Separation ===
        self.separate_frequencies()?;

        // === Stage 3: Compute Differences ===
        self.compute_differences()?;

        // === Stage 4: Psychovisual Masking ===
        self.compute_masking()?;

        // === Stage 5: Combine into Diffmap ===
        self.combine_diffmap()?;

        // === Stage 6: Multi-scale Processing (optional) ===
        const MIN_SIZE_FOR_MULTISCALE: usize = 15;
        if enable_multiscale
            && self.width >= MIN_SIZE_FOR_MULTISCALE
            && self.height >= MIN_SIZE_FOR_MULTISCALE
        {
            self.run_half_scale_and_combine()?;
        }

        // === Stage 7: Reduce to Score ===
        let score = self.reduce_to_score()?;

        Ok(score)
    }

    /// Reduce diffmap to a single score by finding the maximum value
    ///
    /// The Butteraugli score is the MAXIMUM value in the diffmap, not a Q-norm.
    /// This matches the C++ libjxl implementation (butteraugli.cc compute_score_from_diffmap).
    fn reduce_to_score(&mut self) -> Result<f32, Error> {
        let size = self.size;

        // Sync and copy diffmap to CPU
        self.stream
            .sync()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        let mut cpu_buf = vec![0.0f32; size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                cpu_buf.as_mut_ptr() as *mut _,
                self.diffmap.ptr(),
                size * 4,
            )
            .result()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        }

        // Find maximum on CPU
        // TODO: Implement GPU-side max reduction for better performance
        let max_val = cpu_buf.iter().cloned().fold(0.0f32, f32::max);

        Ok(max_val)
    }
}

/// Constants used in Butteraugli computation (re-exported for testing)
pub mod constants {
    pub use super::consts::*;
}

// Debug methods for testing
impl Butteraugli {
    /// Get diffmap as CPU buffer (for debugging)
    pub fn get_diffmap(&mut self) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.diffmap.ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get block_diff_ac[channel] as CPU buffer (for debugging)
    pub fn get_block_diff_ac(&mut self, channel: usize) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.block_diff_ac[channel].ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get block_diff_dc[channel] as CPU buffer (for debugging)
    pub fn get_block_diff_dc(&mut self, channel: usize) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.block_diff_dc[channel].ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get mask as CPU buffer (for debugging)
    pub fn get_mask(&mut self) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.mask.ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get frequency band for image 1: band=0=UHF, 1=HF, 2=MF, 3=LF; channel=0=X, 1=Y, 2=B
    pub fn get_freq1(&mut self, band: usize, channel: usize) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.freq1[band][channel].ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get frequency band for image 2: band=0=UHF, 1=HF, 2=MF, 3=LF; channel=0=X, 1=Y, 2=B
    pub fn get_freq2(&mut self, band: usize, channel: usize) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.freq2[band][channel].ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get XYB values for image 1: channel=0=X, 1=Y, 2=B
    pub fn get_xyb1(&mut self, channel: usize) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.xyb1[channel].ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }

    /// Get XYB values for image 2: channel=0=X, 1=Y, 2=B
    pub fn get_xyb2(&mut self, channel: usize) -> Vec<f32> {
        self.stream.sync().expect("sync failed");
        let mut buf = vec![0.0f32; self.size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut _,
                self.xyb2[channel].ptr(),
                self.size * 4,
            )
            .result()
            .expect("memcpy failed");
        }
        buf
    }
}
