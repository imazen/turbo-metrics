//! zensim CUDA — GPU port of the zensim perceptual similarity metric.
//!
//! Numerically equivalent (within ~ULP of cross-arch FMA drift) to
//! `zen/zensim/zensim/src` on CPU. Kernels live in the companion
//! `zensim-cuda-kernel` crate. This host-side wrapper:
//!
//!   1. Uploads reference + distorted sRGB u8 images, converts to
//!      positive-XYB planar f32.
//!   2. For each of 4 scales (1×, 2×, 4×, 8×):
//!        a. Fused H-blur → 4 planes (mu1, mu2, sigma_sq, sigma12)
//!        b. Fused V-blur + feature extraction → 17 f64 sums + 3 peaks
//!        c. Downscale src/dst planes (for next scale)
//!   3. Returns the per-scale feature vector. Score mapping / trained
//!      weights live in the coefficient wrapper (matches CPU layering).
//!
//! This module is scaffolding; the pipeline implementation is
//! in-progress. See `zensim-cuda-kernel/src/{color,blur,downscale,features}.rs`
//! for the device code.

mod kernel;
pub use kernel::Kernel;

/// Placeholder host-side struct. The full pipeline (ref cache + graph
/// capture + batched scorer) is being ported incrementally; see
/// coefficient#13 follow-ups for status.
pub struct Zensim {
    _kernel: Kernel,
}

impl Zensim {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            _kernel: Kernel::load(),
        })
    }
}

#[derive(Debug)]
pub enum Error {
    Cuda(String),
    Npp(String),
    InvalidDimensions(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Cuda(s) => write!(f, "CUDA error: {}", s),
            Error::Npp(s) => write!(f, "NPP error: {}", s),
            Error::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
        }
    }
}

impl std::error::Error for Error {}
