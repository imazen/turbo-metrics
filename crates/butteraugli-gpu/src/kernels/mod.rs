//! GPU kernels for butteraugli.
//!
//! Each module covers one algorithmic stage. The pipeline (in execution order):
//!
//! 1. `colors` — sRGB → linear, opsin dynamics, linear → XYB
//! 2. `blur` — separable 1D convolutions for low/mid/high frequency split
//! 3. `frequency` — split XYB into UHF / HF / MF / LF bands
//! 4. `malta` — Malta filter (perceptual contrast at multiple frequencies)
//! 5. `masking` — visual masking and fuzzy erosion
//! 6. `diffmap` — combine channels into the final per-pixel diffmap
//! 7. `reduction` — fused max-norm + libjxl 3-norm aggregation (this stage
//!    is fully ported and validated against the CPU crate)
//!
//! Stages other than `reduction` are stubs at this commit; see PORT_STATUS.md.

pub mod blur;
pub mod colors;
pub mod diffmap;
pub mod downscale;
pub mod frequency;
pub mod malta;
pub mod masking;
pub mod reduction;
