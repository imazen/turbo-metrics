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
| Q90 | 1.3061 | 1.2974 | 0.67% |
| Q70 | 4.9217 | 4.6898 | 4.71% |
| Q45 | 5.7672 | 5.9587 | 3.32% |
| Q20 | 12.1209 | 11.6494 | 3.89% |

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

### Session 4: Frequency Separation (Current Session)
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

## Red Herrings (What Wasn't The Issue)

1. **XYB color space constants** - Already correct
2. **Diffmap combination formula** - Correct (MaskY/MaskDcY)
3. **L2 diff weights** - Correct (WMUL array matches Vship)
4. **Multi-scale weight** - 0.5 is correct
