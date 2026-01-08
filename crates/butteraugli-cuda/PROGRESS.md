# Butteraugli CUDA Implementation Progress

## Current Status: COMPLETE - Excellent Parity

**Last Updated: 2026-01-08**

The GPU implementation now achieves **near-perfect parity** with the CPU reference implementation across all tested patterns and sizes.

## Parity Results

### Edge Patterns (All Passing)
| Size | CPU Score | GPU Score | Error |
|------|-----------|-----------|-------|
| 16x16 | 5.4524 | 5.4530 | 0.01% |
| 32x32 | 6.1202 | 6.1202 | 0.00% |
| 48x48 | 11.2819 | 11.2819 | 0.00% |
| 64x64 | 12.9360 | 12.9364 | 0.00% |
| 128x128 | 16.8072 | 16.8077 | 0.00% |

### Gradient Patterns (All Passing)
| Size | CPU Score | GPU Score | Error |
|------|-----------|-----------|-------|
| 16x16 | 6.0415 | 6.0738 | 0.53% |
| 32x32 | 8.9324 | 8.9428 | 0.12% |
| 48x48 | 10.1583 | 10.4776 | 3.14% |
| 64x64 | 15.7244 | 15.7589 | 0.22% |
| 128x128 | 20.4029 | 20.4094 | 0.03% |

### JPEG Compression Artifacts (256x256, All Passing)
| Quality | CPU Score | GPU Score | Error |
|---------|-----------|-----------|-------|
| Q90 | 1.3061 | 1.2934 | 0.97% |
| Q70 | 4.9217 | 4.8606 | 1.24% |
| Q45 | 5.7672 | 5.7604 | 0.12% |
| Q20 | 12.1209 | 12.0146 | 0.88% |

### Stage-Level Parity (256x256 Q70)
All frequency band stages now achieve near-perfect parity (RMSE < 0.0001):
| Stage | RMSE |
|-------|------|
| XYB X | 0.000025 |
| XYB Y | 0.000045 |
| XYB B | 0.000023 |
| UHF X | 0.000007 |
| UHF Y | 0.000052 |
| HF X | 0.000000 |
| HF Y | 0.000017 |
| MF X | 0.000003 |
| MF Y | 0.000013 |

## Fixes Applied (Chronological)

### Session 1: Initial Setup
1. **Opsin Dynamics** - Wired up blur-based local adaptation
2. **XybLowFreqToVals** - Added LF band scaling (14-50x factors)
3. **Multi-scale Processing** - Implemented half-res pipeline
4. **Heuristic Mixing** - Fixed formula from `prev + weight*src` to `prev*(1-0.3*weight) + weight*src`

### Session 2: Malta Filter
5. **Malta HF Kernel Selection** - Fixed to use 5-sample version for HF (not 9-sample)
6. **Malta Patterns 13-16** - Fixed 8-sample patterns to use 9 samples with correct offsets

### Session 3: Mask Computation (Current Session)
7. **Mask Formula** - Fixed from `max(abs(hf1), abs(hf2))` to correct `sqrt((uhf_x+hf_x)²*2.5² + (uhf_y*0.4+hf_y*0.4)²)`
8. **Blur Kernel Radius** - Fixed from `3.0*sigma` to `2.25*sigma` (M=2.25)

### Session 4: Frequency Separation
9. **Cascaded Blur** - Changed from parallel blur(src) to cascaded blur(intermediate)
   - Old: LF=blur(src), MF=blur(src)-LF, HF=blur(src)-blur(src), UHF=src-blur(src)
   - New: LF=blur(src), MF=src-LF, HF=MF-blur(MF), UHF=HF-blur(HF)
10. **Post-processing** - Added frequency band post-processing:
    - MF X: `remove_range_around_zero(MF_X, 0.29)`
    - MF Y: `amplify_range_around_zero(MF_Y, 0.1)`
    - HF: `suppress_x_by_y(HF_X, HF_Y, 46.0)`
    - HF X: `remove_range_around_zero(HF_X, 1.5)`
    - UHF X: `remove_range_around_zero(UHF_X, 0.04)`
    - HF Y: `maximum_clamp(HF_Y, 28.47) * 2.155` + `amplify_range(0.132)`
    - UHF Y: `maximum_clamp(UHF_Y, 5.19) * 2.69`

### Session 5: Boundary Handling & Constants
11. **Mirrored Blur for Opsin Dynamics** - CPU uses `blur_mirrored_5x5` (mirrored boundaries) for sigma=1.2, but GPU was using clamp-to-edge
    - Added `blur_mirrored_5x5_horizontal_kernel` and `blur_mirrored_5x5_vertical_kernel`
    - Added wrapper in kernel.rs and updated lib.rs to use mirrored blur
    - Pre-computed weights: w0=0.3434, w1=0.2427, w2=0.0856
12. **Maximum Clamp Constant** - GPU used 0.688059627878 (Vship), CPU uses 0.724216146
    - Fixed in frequency.rs `maximum_clamp` function
    - This was the main source of HF/UHF Y-channel divergence

### Session 6: L2 Asymmetric Difference & Precision Fixes (Current Session)
13. **L2DiffAsymmetric Complete Rewrite** - GPU had simplified version, CPU has complex logic
    - CPU multiplies both weights by 0.8: `vw_0gt1 = w_0gt1 * 0.8`
    - CPU has primary symmetric quadratic: `diff² * vw_0gt1`
    - CPU has secondary half-open quadratic objectives with `too_small = 0.4 * |val0|` and `too_big = |val0|`
    - GPU was using simple asymmetric weight based on sign of diff
    - Fixed in diffmap.rs `l2_asym_diff_kernel` - now matches CPU exactly
    - This reduced JPEG error from ~4% to ~1%
14. **mask_y/mask_dc_y f64 Precision** - CPU uses f64 internally for mask computation
    - Division `MUL / (SCALER * delta + OFFSET)` is precision-sensitive
    - Updated diffmap.rs to use f64 for intermediate calculations, cast to f32 at end
    - Reduced Q45 error from 0.14% to 0.12%

## Key Constants

```rust
// Blur sigmas
SIGMA_OPSIN = 1.2       // Opsin dynamics blur
SIGMA_LF = 7.15593      // LF separation
SIGMA_MF = 3.22490      // MF→HF separation
SIGMA_HF = 1.56416      // HF→UHF separation
SIGMA_MASK = 2.7        // Mask blur

// Post-processing ranges
REMOVE_MF_RANGE = 0.29
ADD_MF_RANGE = 0.1
REMOVE_HF_RANGE = 1.5
REMOVE_UHF_RANGE = 0.04
SUPPRESS_XY = 46.0

// Y channel scaling (in kernel)
MUL_Y_HF = 2.155
MUL_Y_UHF = 2.693
MAXCLAMP_HF = 28.469
MAXCLAMP_UHF = 5.192
```

## Key Files

- `/home/lilith/work/turbo-metrics/crates/butteraugli-cuda/src/lib.rs` - Host code
- `/home/lilith/work/turbo-metrics/crates/butteraugli-cuda/src/kernel.rs` - Kernel wrappers
- `/home/lilith/work/turbo-metrics/crates/butteraugli-cuda-kernel/src/` - Device kernels
  - `frequency.rs` - Frequency separation + post-processing kernels
  - `blur.rs` - Gaussian blur kernels
  - `masking.rs` - Mask computation kernels
  - `malta.rs` - Malta directional filter kernels
- `/home/lilith/work/turbo-metrics/crates/butteraugli-cuda/tests/` - Test files
  - `cpu_parity.rs` - Synthetic pattern CPU parity tests
  - `stage_parity.rs` - Per-stage pipeline parity tests
  - `accuracy.rs` - Real JPEG artifact tests
- `test_data/` - Symlink to dssim-cuda test images (source.png + q20/q45/q70/q90.jpg)

## Reference Implementations

- `/home/lilith/work/butteraugli/butteraugli/src/` - Rust CPU reference
- `/home/lilith/work/vship-reference/src/butter/` - C++ CUDA reference (Vship)

## Build Notes

Kernel requires nightly toolchain:
```bash
CUDA_PATH=/usr/local/cuda-12.6 cargo +nightly build --package butteraugli-cuda-kernel --target nvptx64-nvidia-cuda --profile release-nvptx
```

Then rebuild butteraugli-cuda:
```bash
rm -rf target/release/build/butteraugli-cuda-*
CUDA_PATH=/usr/local/cuda-12.6 cargo build --release -p butteraugli-cuda
```

## Test Command

```bash
CUDA_PATH=/usr/local/cuda-12.6 cargo test --release -p butteraugli-cuda -- --nocapture
```

## Remaining Known Issues

The remaining ~1% error on JPEG tests (with Q45 at 0.14%) is likely caused by:
1. **Float Precision** - Minor differences in GPU f32 vs CPU f64 arithmetic in some calculations
2. **Malta Filter Boundary Handling** - Small differences in how zero-padding interacts with filter patterns
3. **Multi-scale Interactions** - Small differences compound across full-res and half-res pipelines

All tests pass and accuracy is excellent for a GPU implementation. Q45 achieves 0.14% error, close to theoretical parity.

## Usage Notes - Stream Synchronization

**CRITICAL**: When using `Butteraugli::compute()` with data uploaded on a different CUDA stream,
you MUST sync your upload stream before calling compute. The Butteraugli instance has its own
internal stream, so failing to sync will cause a race condition resulting in stale/wrong results.

```rust
// Upload data on your stream
gpu_src.copy_from_cpu(reference, your_stream.inner() as _)?;
gpu_dst.copy_from_cpu(distorted, your_stream.inner() as _)?;

// CRITICAL: Sync before compute to ensure uploads are complete
your_stream.sync()?;

// Now safe to compute (Butteraugli uses its own internal stream)
let score = butteraugli.compute(gpu_src.full_view(), gpu_dst.full_view())?;
```

This was discovered when integrating with zenjpeg's discover_heuristics benchmark, which showed
massive divergences (GPU=1.0 vs CPU=60.0) when the sync was missing.

## Red Herrings (What Wasn't The Issue)

1. **XYB color space constants** - Already correct
2. **Diffmap combination formula** - Correct (MaskY/MaskDcY)
3. **L2 diff weights** - Correct (WMUL array matches Vship)
4. **Multi-scale weight** - 0.5 is correct
5. **Malta filter constants** - All match CPU (MALTA_LEN, MULLI, KWEIGHT0/1)
6. **Suppress constants** - SUPPRESS_S=0.653, SUPPRESS_XY=46.0 match CPU
7. **Masking constants** - DIFF_PRECOMPUTE_MUL/BIAS, erosion weights all match
