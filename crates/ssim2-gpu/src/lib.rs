//! Multi-vendor GPU implementation of the SSIMULACRA2 perceptual image
//! quality metric.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl) — single Rust
//! kernel source, dispatchable across:
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** (SIMD) reference path when `cpu` is enabled
//!
//! Algorithmic parity target is the published `ssimulacra2` v0.5.1 crate
//! (the canonical Rust port of `cloudinary/ssimulacra2`). At the
//! resolution-pyramid level the implementation also matches
//! `crates/ssimulacra2-cuda/`, which uses the same Charalampidis
//! recursive Gaussian and the same 6-octave reduction.
//!
//! ## Status
//!
//! Initial port from `ssimulacra2-cuda`. See `PORT_STATUS.md` and
//! `HANDOFF.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod kernels;
pub mod pipeline;
pub mod pipeline_batch;

pub use pipeline::Ssim2;
pub use pipeline_batch::Ssim2Batch;

/// Number of pyramid scales — matches both the CPU and CUDA references.
pub const NUM_SCALES: usize = 6;

/// Result of an SSIMULACRA2 comparison.
///
/// `score` is in roughly the 0–100 range — higher = better quality, 100 =
/// identical, 0 = visually broken. Mirrors the scalar returned by
/// `ssimulacra2::compute_frame_ssimulacra2`.
#[derive(Debug, Clone, Copy)]
pub struct GpuSsim2Result {
    pub score: f64,
}

/// Errors that the GPU SSIMULACRA2 pipeline can return.
#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match the configured `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference*` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image is smaller than 8×8 — SSIMULACRA2 is undefined there.
    InvalidImageSize,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::InvalidImageSize => write!(f, "image must be at least 8×8 pixels"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
