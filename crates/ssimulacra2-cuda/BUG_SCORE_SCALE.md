# Bug: GPU SSIMULACRA2 returns wrong score scale

## Summary

The ssimulacra2-cuda library returns raw/internal scores instead of properly transformed 0-100 SSIMULACRA2 scores.

## Expected Behavior

SSIMULACRA2 scores should be in the 0-100 range:
- 100 = identical images
- 90+ = imperceptible differences
- 70+ = high quality
- <50 = significant degradation

The CPU implementation (fast-ssim2) returns correct scores in this range.

## Actual Behavior

The GPU implementation returns large negative values like -860 to -1182 for typical JPEG comparisons.

## Evidence

Comparing the same image (CID22/1583339) encoded with jpegli at various quality levels:

**CPU baseline (fast-ssim2):**
| BPP | Quality |
|-----|---------|
| 0.39 | 46.9 |
| 0.49 | 56.3 |
| 1.12 | 74.7 |
| 2.55 | 85.8 |

**GPU baseline (ssimulacra2-cuda):**
| BPP | Quality |
|-----|---------|
| 0.39 | -864.8 |
| 0.49 | -862.8 |
| 1.12 | -861.5 |
| 2.55 | -860.8 |

## Analysis

The transformation code in `post_process_scores()` looks correct:
```rust
if score > 0.0f64 {
    score = score.powf(0.627633...).mul_add(-10.0f64, 100.0f64);
} else {
    score = 100.0f64;
}
```

However, the raw score before this transformation appears to be ~2500-3000 instead of the expected ~1-10 range. This suggests the issue is in the earlier computation stages - possibly:
1. Missing normalization by image dimensions
2. Incorrect aggregation of per-scale/per-channel scores
3. Different weighting constants than the CPU implementation

## Impact

- Absolute quality values are meaningless
- Relative comparisons still work (Pareto distance calculations are valid)
- Cannot compare GPU scores to CPU scores or published SSIMULACRA2 benchmarks

## Discovered

2026-01-19 during coefficient SA optimization testing.
