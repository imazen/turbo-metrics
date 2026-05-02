//! GPU kernels for SSIMULACRA2.
//!
//! Pipeline order (per pyramid scale):
//!
//! 1. `srgb` — sRGB u8 → linear f32 RGB (planar) via 256-entry LUT.
//! 2. `downscale` — 2× linear-RGB downscale to build the 6-octave pyramid.
//! 3. `xyb` — linear RGB → positive XYB (planar).
//! 4. `blur` — Charalampidis recursive Gaussian.
//! 5. `transpose` — square the IIR pass over the other axis.
//! 6. `error_maps` — fused SSIM + ringing + blurring per-pixel error.
//! 7. `reduction` — sum and sum-of-fourth-powers per (scale, channel,
//!    error-map). 18 scalars per scale × 6 scales = 108 raw stats; the
//!    pipeline folds them into the final score host-side.

pub mod blur;
pub mod downscale;
pub mod error_maps;
pub mod reduction;
pub mod srgb;
pub mod transpose;
pub mod xyb;
