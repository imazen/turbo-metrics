//! Butteraugli CUDA implementation
//!
//! GPU-accelerated implementation of the Butteraugli perceptual image quality metric.
//!
//! Based on the Vship GPU implementation (https://github.com/Line-fr/Vship).
//!
//! # Example
//!
//! ```no_run
//! use butteraugli_cuda::Butteraugli;
//!
//! // Initialize CUDA context first
//! cudarse_driver::init_cuda().unwrap();
//!
//! let mut butteraugli = Butteraugli::new(1920, 1080).unwrap();
//! let score = butteraugli.compute(&reference_rgb, &distorted_rgb).unwrap();
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
    pub const SIGMA_LF: f32 = 7.15593339443;    // LF separation (opsin dynamics)
    pub const SIGMA_MF: f32 = 3.22489901262;    // MF separation
    pub const SIGMA_HF: f32 = 1.56416327805;    // HF separation
    pub const SIGMA_UHF: f32 = 1.2;             // UHF blur
    pub const SIGMA_MASK: f32 = 2.7;            // Mask blur

    // Malta filter weights for UHF band (Y channel)
    pub const W_UHF_MALTA: f32 = 1.10039032555;
    pub const W_UHF_MALTA_ASYM: f32 = 0.8 * 1.10039032555;  // with HF_ASYMMETRY
    pub const NORM1_UHF: f32 = 71.7800275169;

    // Malta filter weights for UHF X channel
    pub const W_UHF_MALTA_X: f32 = 173.5;
    pub const W_UHF_MALTA_X_ASYM: f32 = 0.8 * 173.5;
    pub const NORM1_UHF_X: f32 = 5.0;

    // Malta filter weights for HF band (Y channel)
    pub const W_HF_MALTA: f32 = 18.7237414387;
    pub const W_HF_MALTA_ASYM: f32 = 0.8 * 18.7237414387;
    pub const NORM1_HF: f32 = 4498534.45232;

    // Malta filter weights for HF X channel
    pub const W_HF_MALTA_X: f32 = 6923.99476109;
    pub const W_HF_MALTA_X_ASYM: f32 = 0.8 * 6923.99476109;
    pub const NORM1_HF_X: f32 = 8051.15833247;

    // Malta LF weights for MF band (Y channel)
    pub const W_MF_MALTA: f32 = 37.0819870399;
    pub const W_MF_MALTA_ASYM: f32 = 0.8 * 37.0819870399;
    pub const NORM1_MF: f32 = 130262059.556;

    // Malta LF weights for MF X channel
    pub const W_MF_MALTA_X: f32 = 8246.75321353;
    pub const W_MF_MALTA_X_ASYM: f32 = 0.8 * 8246.75321353;
    pub const NORM1_MF_X: f32 = 1009002.70582;

    // L2 difference weights [DC_X, DC_Y, DC_B, MF_X, MF_Y, MF_B, LF_X, LF_Y, LF_B]
    pub const WMUL: [f32; 9] = [
        400.0,           // DC X
        1.50815703118,   // DC Y
        0.0,             // DC B (unused)
        2150.0,          // MF X
        10.6195433239,   // MF Y
        16.2176043152,   // MF B
        29.2353797994,   // LF X
        0.844626970982,  // LF Y
        0.703646627719,  // LF B
    ];

    // Intensity multiplier for opsin dynamics
    pub const INTENSITY_MULT: f32 = 255.0;

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
        let sum_result = CuBox::<[f64]>::new_zeroed(1, &stream)
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        Ok(Self {
            kernel,
            stream,
            width,
            height,
            size,
            linear1,
            linear2,
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
        self.kernel.srgb_to_linear(&self.stream, reference, self.linear1.full_view_mut());
        self.kernel.srgb_to_linear(&self.stream, distorted, self.linear2.full_view_mut());

        // For now, use simple linear->XYB conversion
        // TODO: Implement full opsin dynamics with blur-based adaptation
        // This requires: blur linear RGB, use blurred as adaptation, then convert

        // Convert to XYB (stores in freq1[3] and freq2[3] as temporary LF storage)
        // We'll use the xyb buffers to store the result
        self.convert_to_xyb()?;

        // === Stage 2: Frequency Separation ===
        self.separate_frequencies()?;

        // === Stage 3: Compute Differences ===
        self.compute_differences()?;

        // === Stage 4: Psychovisual Masking ===
        self.compute_masking()?;

        // === Stage 5: Combine into Diffmap ===
        self.combine_diffmap()?;

        // === Stage 6: Multi-scale Processing ===
        // TODO: Implement half-resolution processing and merging
        // For now, skip multi-scale

        // === Stage 7: Reduce to Score ===
        let score = self.reduce_to_score()?;

        Ok(score)
    }

    /// Convert linear RGB to XYB color space
    fn convert_to_xyb(&mut self) -> Result<(), Error> {
        // Convert interleaved linear RGB directly to planar XYB
        // This is a simplified version without full opsin dynamics (blur-based adaptation)

        // Image 1
        self.kernel.linear_to_xyb_planar(
            &self.stream,
            self.linear1.full_view(),
            Self::ptr_mut(&mut self.xyb1[0]),
            Self::ptr_mut(&mut self.xyb1[1]),
            Self::ptr_mut(&mut self.xyb1[2]),
        );

        // Image 2
        self.kernel.linear_to_xyb_planar(
            &self.stream,
            self.linear2.full_view(),
            Self::ptr_mut(&mut self.xyb2[0]),
            Self::ptr_mut(&mut self.xyb2[1]),
            Self::ptr_mut(&mut self.xyb2[2]),
        );

        Ok(())
    }

    /// Separate XYB into frequency bands (UHF, HF, MF, LF)
    fn separate_frequencies(&mut self) -> Result<(), Error> {
        let w = self.width;
        let h = self.height;
        let size = self.size;

        // For each image and channel, compute frequency bands:
        // LF = blur(src, sigma_lf)
        // MF = blur(src - LF, sigma_mf)
        // HF = blur(residual, sigma_hf)
        // UHF = residual - HF

        for img_idx in 0..2 {
            let (xyb, freq) = if img_idx == 0 {
                (&self.xyb1, &mut self.freq1)
            } else {
                (&self.xyb2, &mut self.freq2)
            };

            for ch in 0..3 {
                let src = Self::ptr(&xyb[ch]);
                let temp1 = Self::ptr_mut(&mut self.temp1);
                let temp2 = Self::ptr_mut(&mut self.temp2);

                // LF = blur(src, sigma_lf)
                let lf = Self::ptr_mut(&mut freq[3][ch]);
                self.kernel.blur(&self.stream, src, lf, temp1, w, h, consts::SIGMA_LF);

                // residual = src - LF (store in temp2)
                self.kernel.subtract_arrays(&self.stream, src, lf, temp2, size);

                // MF = blur(residual, sigma_mf)
                let mf = Self::ptr_mut(&mut freq[2][ch]);
                self.kernel.blur(&self.stream, temp2, mf, temp1, w, h, consts::SIGMA_MF);

                // residual2 = residual - MF (store in temp1)
                self.kernel.subtract_arrays(&self.stream, temp2, mf, temp1, size);

                // HF = blur(residual2, sigma_hf)
                let hf = Self::ptr_mut(&mut freq[1][ch]);
                self.kernel.blur(&self.stream, temp1, hf, temp2, w, h, consts::SIGMA_HF);

                // UHF = residual2 - HF
                let uhf = Self::ptr_mut(&mut freq[0][ch]);
                self.kernel.subtract_arrays(&self.stream, temp1, hf, uhf, size);
            }
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
            self.kernel.clear_buffer(&self.stream, Self::ptr_mut(&mut self.block_diff_dc[i]), size);
            self.kernel.clear_buffer(&self.stream, Self::ptr_mut(&mut self.block_diff_ac[i]), size);
        }

        // === UHF differences (Malta HF kernel) ===
        // Y channel
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[0][1]), // UHF Y image 1
            Self::ptr(&self.freq2[0][1]), // UHF Y image 2
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w, h,
            consts::W_UHF_MALTA,
            consts::W_UHF_MALTA_ASYM,
            consts::NORM1_UHF,
        );

        // X channel
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[0][0]),
            Self::ptr(&self.freq2[0][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w, h,
            consts::W_UHF_MALTA_X,
            consts::W_UHF_MALTA_X_ASYM,
            consts::NORM1_UHF_X,
        );

        // === HF differences (Malta HF kernel) ===
        // Y channel
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[1][1]),
            Self::ptr(&self.freq2[1][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w, h,
            consts::W_HF_MALTA,
            consts::W_HF_MALTA_ASYM,
            consts::NORM1_HF,
        );

        // X channel
        self.kernel.malta_diff_map(
            &self.stream,
            Self::ptr(&self.freq1[1][0]),
            Self::ptr(&self.freq2[1][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w, h,
            consts::W_HF_MALTA_X,
            consts::W_HF_MALTA_X_ASYM,
            consts::NORM1_HF_X,
        );

        // === MF differences (Malta LF kernel) ===
        // Y channel
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[2][1]),
            Self::ptr(&self.freq2[2][1]),
            Self::ptr_mut(&mut self.block_diff_ac[1]),
            w, h,
            consts::W_MF_MALTA,
            consts::W_MF_MALTA_ASYM,
            consts::NORM1_MF,
        );

        // X channel
        self.kernel.malta_diff_map_lf(
            &self.stream,
            Self::ptr(&self.freq1[2][0]),
            Self::ptr(&self.freq2[2][0]),
            Self::ptr_mut(&mut self.block_diff_ac[0]),
            w, h,
            consts::W_MF_MALTA_X,
            consts::W_MF_MALTA_X_ASYM,
            consts::NORM1_MF_X,
        );

        // === LF/DC differences (L2) ===
        // DC X
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][0]),
            Self::ptr(&self.freq2[3][0]),
            Self::ptr_mut(&mut self.block_diff_dc[0]),
            size,
            consts::WMUL[0], // DC_X weight
        );

        // DC Y
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[3][1]),
            Self::ptr(&self.freq2[3][1]),
            Self::ptr_mut(&mut self.block_diff_dc[1]),
            size,
            consts::WMUL[1], // DC_Y weight
        );

        // MF B
        self.kernel.l2_diff(
            &self.stream,
            Self::ptr(&self.freq1[2][2]),
            Self::ptr(&self.freq2[2][2]),
            Self::ptr_mut(&mut self.block_diff_ac[2]),
            size,
            consts::WMUL[5], // MF_B weight
        );

        // LF channels
        for (ch, &weight) in [6, 7, 8].iter().enumerate() {
            self.kernel.l2_diff(
                &self.stream,
                Self::ptr(&self.freq1[3][ch]),
                Self::ptr(&self.freq2[3][ch]),
                Self::ptr_mut(&mut self.block_diff_dc[ch]),
                size,
                consts::WMUL[weight],
            );
        }

        Ok(())
    }

    /// Compute psychovisual masking
    fn compute_masking(&mut self) -> Result<(), Error> {
        let w = self.width;
        let h = self.height;
        let size = self.size;

        // Initialize mask from HF and UHF bands (X and Y channels)
        self.kernel.mask_init(
            &self.stream,
            Self::ptr(&self.freq1[1][0]), // HF X
            Self::ptr(&self.freq1[0][0]), // UHF X
            Self::ptr(&self.freq1[1][1]), // HF Y
            Self::ptr(&self.freq1[0][1]), // UHF Y
            Self::ptr_mut(&mut self.mask),
            size,
        );

        // Precompute diff values
        self.kernel.diff_precompute(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr_mut(&mut self.mask_temp),
            size,
        );

        // Blur the mask
        self.kernel.blur(
            &self.stream,
            Self::ptr(&self.mask_temp),
            Self::ptr_mut(&mut self.mask),
            Self::ptr_mut(&mut self.temp1),
            w, h,
            consts::SIGMA_MASK,
        );

        // Fuzzy erosion
        self.kernel.fuzzy_erosion(
            &self.stream,
            Self::ptr(&self.mask),
            Self::ptr_mut(&mut self.mask_temp),
            w, h,
        );

        // Copy back
        unsafe {
            cudarse_driver::sys::cuMemcpyAsync(
                self.mask.ptr(),
                self.mask_temp.ptr(),
                size * 4,
                self.stream.raw(),
            ).result().map_err(|e| Error::Cuda(format!("{:?}", e)))?;
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

    /// Reduce diffmap to a single score using the Q-norm
    fn reduce_to_score(&mut self) -> Result<f32, Error> {
        let size = self.size;

        // Compute diffmap^q
        self.kernel.power_elements(
            &self.stream,
            Self::ptr(&self.diffmap),
            Self::ptr_mut(&mut self.temp1),
            size,
            consts::NORM_Q,
        );

        // Sum the powered elements using a simple reduction
        // For production, we'd use NPP's Sum function, but for now use a simple approach

        // Sync and copy to CPU for reduction
        self.stream.sync().map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        // Copy powered diffmap to CPU
        let mut cpu_buf = vec![0.0f32; size];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                cpu_buf.as_mut_ptr() as *mut _,
                self.temp1.ptr(),
                size * 4,
            ).result().map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        }

        // Sum on CPU
        let sum: f64 = cpu_buf.iter().map(|&x| x as f64).sum();

        // Compute Q-norm: (sum / n)^(1/q)
        let norm = (sum / size as f64).powf(1.0 / consts::NORM_Q as f64);

        Ok(norm as f32)
    }
}

/// Constants used in Butteraugli computation (re-exported for testing)
pub mod constants {
    pub use super::consts::*;
}
