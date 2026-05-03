//! GPU kernels for DSSIM.
//!
//! Pipeline order (per pyramid scale; 5 scales total):
//!
//! 1. `srgb` — sRGB u8 → linear f32 RGB (planar) via 256-entry-equivalent
//!    inline transfer function.
//! 2. `downscale` — 2×2 box-filter linear-RGB downscale (between scales,
//!    operating on planar buffers).
//! 3. `lab` — linear RGB → custom-scaled Lab (planar).
//! 4. `blur` — fixed 3×3 Gaussian convolution with edge clamping. Two-pass
//!    on every plane (matches dssim-core's `BLUR_KERNEL` × 2 application).
//! 5. `ssim` — 15-input fused per-pixel SSIM map (averages mu/cov terms
//!    across L/a/b before applying the SSIM formula). Plus a scalar
//!    `abs_diff` kernel for the per-scale MAD step.
//! 6. `reduction` — per-scale Σ, then Σ|x - avg|. Two scalars per scale.

pub mod blur;
pub mod downscale;
pub mod lab;
pub mod reduction;
pub mod srgb;
pub mod ssim;
