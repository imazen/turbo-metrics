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

use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, C};

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

/// GPU-accelerated Butteraugli quality metric
///
/// Computes the perceptual distance between two images using the Butteraugli
/// algorithm on the GPU.
pub struct Butteraugli {
    kernel: Kernel,
    stream: CuStream,
    width: u32,
    height: u32,

    // GPU buffers for linear RGB
    linear1: Image<f32, C<3>>,
    linear2: Image<f32, C<3>>,
}

impl Butteraugli {
    /// Create a new Butteraugli instance for images of the given dimensions
    pub fn new(width: u32, height: u32) -> Result<Self, Error> {
        let stream = CuStream::new().map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        let kernel = Kernel::load();

        // Allocate main image buffers
        let linear1 =
            Image::<f32, C<3>>::malloc(width, height).map_err(|e| Error::Npp(format!("{:?}", e)))?;
        let linear2 =
            Image::<f32, C<3>>::malloc(width, height).map_err(|e| Error::Npp(format!("{:?}", e)))?;

        Ok(Self {
            kernel,
            stream,
            width,
            height,
            linear1,
            linear2,
        })
    }

    /// Get the image dimensions this instance was created for
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
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
        if reference.width() != self.width
            || reference.height() != self.height
            || distorted.width() != self.width
            || distorted.height() != self.height
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

        // Step 1: Convert sRGB to linear RGB
        self.kernel
            .srgb_to_linear(&self.stream, reference, self.linear1.full_view_mut());
        self.kernel
            .srgb_to_linear(&self.stream, distorted, self.linear2.full_view_mut());

        // TODO: Implement full pipeline:
        // Step 2: Apply opsin dynamics (linear RGB -> XYB)
        // Step 3: Separate frequencies (LF, MF, HF, UHF)
        // Step 4: Compute Malta diff maps
        // Step 5: Compute psychovisual masking
        // Step 6: Combine into final diffmap
        // Step 7: Reduce to single score

        // For now, return a placeholder
        // Full implementation requires more buffer management
        self.stream.sync().map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        // Placeholder score
        Ok(0.0)
    }
}

/// Constants used in Butteraugli computation
pub mod constants {
    /// Asymmetry factor for high-frequency differences
    pub const HF_ASYMMETRY: f32 = 0.8;

    /// Malta filter weights for UHF band
    pub const W_UHF_MALTA: f32 = 1.10039032555;
    pub const NORM1_UHF: f32 = 71.7800275169;

    /// Malta filter weights for UHF X channel
    pub const W_UHF_MALTA_X: f32 = 173.5;
    pub const NORM1_UHF_X: f32 = 5.0;

    /// Malta filter weights for HF band
    pub const W_HF_MALTA: f32 = 18.7237414387;
    pub const NORM1_HF: f32 = 4498534.45232;

    /// Malta filter weights for HF X channel
    pub const W_HF_MALTA_X: f32 = 6923.99476109;
    pub const NORM1_HF_X: f32 = 8051.15833247;

    /// Malta filter weights for MF band
    pub const W_MF_MALTA: f32 = 37.0819870399;
    pub const NORM1_MF: f32 = 130262059.556;

    /// Malta filter weights for MF X channel
    pub const W_MF_MALTA_X: f32 = 8246.75321353;
    pub const NORM1_MF_X: f32 = 1009002.70582;

    /// L2 difference weights [DC_X, DC_Y, DC_B, MF_X, MF_Y, MF_B, LF_X, LF_Y, LF_B]
    pub const WMUL: [f32; 9] = [
        400.0,
        1.50815703118,
        0.0,
        2150.0,
        10.6195433239,
        16.2176043152,
        29.2353797994,
        0.844626970982,
        0.703646627719,
    ];

    /// Gaussian blur sigmas used at different stages
    pub const BLUR_SIGMAS: [f64; 5] = [
        1.2,            // Index 0
        1.56416327805,  // Index 1 - HF separation
        2.7,            // Index 2 - mask blur
        3.22489901262,  // Index 3 - MF separation
        7.15593339443,  // Index 4 - LF separation (opsin dynamics)
    ];
}
