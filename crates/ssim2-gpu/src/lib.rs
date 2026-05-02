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
//! ## Single-image usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use ssim2_gpu::Ssim2;
//!
//! let client = WgpuRuntime::client(&Default::default());
//! let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! let dist_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//!
//! let result = s.compute(&ref_srgb, &dist_srgb)?;
//! println!("score = {:.3}", result.score);
//! # Ok::<(), ssim2_gpu::Error>(())
//! ```
//!
//! ## Cached-reference usage (encoder rate-distortion)
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use ssim2_gpu::Ssim2;
//!
//! # fn candidates() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! s.set_reference(&ref_srgb)?;
//!
//! for distorted_candidate in candidates() {
//!     let r = s.compute_with_reference(&distorted_candidate)?;
//!     // ... use r.score in the rate-distortion search ...
//! }
//! # Ok::<(), ssim2_gpu::Error>(())
//! ```
//!
//! ## Batched usage (N images vs one cached reference)
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use ssim2_gpu::Ssim2Batch;
//!
//! # fn collect_distorted() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut batch = Ssim2Batch::<WgpuRuntime>::new(client, 256, 256, 8)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! batch.set_reference(&ref_srgb)?;
//!
//! let dis_images: Vec<Vec<u8>> = collect_distorted();
//! let results = batch.compute_batch(&dis_images)?;
//! for r in &results {
//!     println!("score = {:.3}", r.score);
//! }
//! # Ok::<(), ssim2_gpu::Error>(())
//! ```
//!
//! ## Score interpretation
//!
//! Output is in roughly the 0–100 range:
//! - **100** = identical (or near-identical)
//! - **90+** = visually indistinguishable for most observers
//! - **70+** = high quality
//! - **30–60** = noticeable distortion
//! - **<0** = the SSIMULACRA2 polynomial overshoot region for severely
//!   distorted images; the CPU `ssimulacra2` produces the same
//!   negative values there — not a GPU-side bug.
//!
//! ## Status
//!
//! Initial port from `ssimulacra2-cuda`. See `PORT_STATUS.md` and
//! `HANDOFF.md`. Validated against CPU `ssimulacra2` v0.5.1 to
//! ≤ 0.06 % relative on JPEG q5..q90; cached and batched paths agree
//! with the direct path to ≤ 1.3e-5 absolute.

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
    /// `Ssim2Batch::new` was called with `batch_size == 0`, or
    /// `compute_batch` got more inputs than the instance's batch_size.
    InvalidBatchSize { got: usize, max: usize },
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
            Error::InvalidBatchSize { got, max } => write!(
                f,
                "invalid batch size: got {got} images for batch_size = {max}"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
