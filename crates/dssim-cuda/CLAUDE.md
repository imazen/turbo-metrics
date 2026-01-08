# DSSIM-CUDA Development Log

## Project Overview

GPU implementation of the DSSIM image quality metric, following the ssimulacra2-cuda architecture pattern.

## Current Status

**Phase 1: Kernel Crate Setup** - Complete (13 kernels compiled)
**Phase 2: Host Crate Pipeline** - Complete (builds and runs)
**Phase 3: Accuracy Validation** - Complete (all tests passing with <0.1% error)

### Test Results (2026-01-08)

| Test | CPU | GPU | Error | Status |
|------|-----|-----|-------|--------|
| Identical images | 0.000000 | 0.000000 | 0% | PASS |
| Very different | 0.541865 | 0.541830 | 0.006% | PASS |
| Slightly different | 0.014610 | 0.014604 | 0.04% | PASS |

All tests pass with strict tolerances (< 1% for subtle differences, < 0.1% for large differences).

## Architecture

The implementation now matches dssim-core's architecture exactly:

1. **sRGB to linear RGB conversion** (LUT-based, like dssim-core)
2. **Multi-scale processing** (5 scales):
   - Downsample RGB (not LAB) between scales
   - Convert RGB to LAB at each scale
   - Apply chroma pre-blur (two-pass) to a/b channels
   - Compute SSIM statistics with two-pass Gaussian blur
3. **Score computation**: MAD formula `score = 1 - mean(|avg - ssim_i|)` where `avg = mean(ssim)^(0.5^scale)`
4. **Final conversion**: `dssim = 1/ssim - 1`

### Architecture Decisions

- **Color space**: LAB (D65 illuminant, custom 0-1 scaling like dssim-core)
- **Blur kernel**: Two-pass 3x3 Gaussian to match dssim-core
- **Multi-scale**: 5 scales with weights `[0.028, 0.197, 0.322, 0.298, 0.155]`
- **Downsampling**: RGB space (not LAB), matching dssim-core
- **Chroma pre-blur**: Two-pass blur on a/b channels before SSIM computation

## Key Algorithm Details (from dssim-core 3.2.11)

### LAB Conversion Constants
```rust
// D65 illuminant
const D65X: f32 = 0.9505;
const D65Y: f32 = 1.0;
const D65Z: f32 = 1.089;

// Nonlinear transform threshold
const EPSILON: f32 = 216.0 / 24389.0;  // ~0.008856
const K: f32 = 24389.0 / (27.0 * 116.0);  // ~7.787
```

### 3x3 Gaussian Kernel (applied twice = two-pass blur)
```rust
const KERNEL: [[f32; 3]; 3] = [
    [0.095332, 0.118095, 0.095332],
    [0.118095, 0.146293, 0.118095],
    [0.095332, 0.118095, 0.095332],
];
```

### SSIM Constants
```rust
const C1: f32 = 0.0001;  // 0.01^2
const C2: f32 = 0.0009;  // 0.03^2
```

## Files Created

### Kernel Crate (dssim-cuda-kernel)
- `src/lib.rs` - Module exports with nvptx features
- `src/srgb.rs` - LUT-based sRGB to linear (copied from ssimulacra2)
- `src/downscale.rs` - 2x downsampling for planes and RGB
- `src/lab.rs` - Linear RGB to LAB conversion
- `src/blur.rs` - 3x3 Gaussian blur + fused operations (blur_squared, blur_product)
- `src/ssim.rs` - Per-pixel SSIM computation

### Host Crate (dssim-cuda)
- `src/lib.rs` - Dssim struct with multi-scale pipeline
- `src/kernel.rs` - Kernel launch wrappers
- `tests/accuracy.rs` - CPU parity tests with strict tolerances

## Discoveries & Notes

### Key Implementation Details
1. dssim-core downsamples RGB, then converts to LAB at each scale
2. Chroma pre-blur: a/b channels get two-pass blur BEFORE SSIM statistics
3. SSIM stats are averaged across L/a/b channels BEFORE computing the SSIM formula
4. Score uses MAD (mean absolute deviation) from a power-scaled average

### Key Differences from ssimulacra2
1. LAB color space instead of XYB
2. Simple 3x3 blur (two-pass) instead of recursive Gaussian (IIR filter)
3. Standard SSIM formula with MAD scoring instead of custom error maps
4. No transpose-based optimization needed (blur is small enough)

## Red Herrings / Things That Weren't Issues

- Initial kernel crate build failed without CUDA_PATH env var
- Panic handler issue: needed `features = ["minimal-panic"]` in nvptx-std dependency
- Borrow checker in downscale loop: fixed with split_at_mut
- Score formula: dssim-core uses mean absolute deviation, not simple average
- Stats averaging: dssim-core averages LAB stats BEFORE computing SSIM, not after
- **Test bug (2026-01-08)**: Original test passed gamma-encoded sRGB values to dssim-core instead of using `ToRGBAPLU::to_rgblu()` for proper linearization. This caused apparent 40%+ error that was actually a test bug, not an implementation bug.

## Build Notes

```bash
# Build requires CUDA_PATH
CUDA_PATH=/usr/local/cuda-12.6 cargo build -p dssim-cuda

# Run tests
CUDA_PATH=/usr/local/cuda-12.6 cargo test -p dssim-cuda --test accuracy -- --nocapture
```

## TODO

- [x] Kernel crate setup
- [x] srgb.rs (copy)
- [x] downscale.rs (adapted for RGB)
- [x] lab.rs (new)
- [x] blur.rs (new)
- [x] ssim.rs (new)
- [x] Host crate
- [x] Pipeline integration
- [x] CPU parity tests
- [x] Fix score computation (MAD formula)
- [x] Fix two-pass blur
- [x] Restructure to RGB downsampling (matching dssim-core)
- [x] Add per-scale LAB conversion
- [x] Implement chroma pre-blur
- [x] Achieve <1% accuracy on all tests
- [x] Remove unused temp2 buffer
- [ ] Precompute API (future optimization)
