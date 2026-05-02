# ssim2-gpu — Handoff

Status as of 2026-05-02 (overnight run). **Single-image pipeline +
cached-reference path + thin `Ssim2Batch` wrapper landed and validated
end-to-end against the published `ssimulacra2` v0.5.1 CPU crate.**
Score parity within 0.06 % relative on the JPEG quality corpus
(q ∈ {5, 20, 45, 70, 90}); identical-image → 99.99; cached vs direct
path drift ≤ 8e-6.

## TL;DR

- **All 7 kernel modules ported, validated, and wired end-to-end.**
- **`Ssim2::compute(ref, dis)` matches CPU `ssimulacra2` to ≤ 0.5 % rel
  (or ≤ 0.1 absolute) across the full JPEG-q1..q90 corpus.**
- **`Ssim2::set_reference` + `compute_with_reference`** populate cached
  pyramid + ref-XYB + ref blur and skip the reference side on every
  subsequent call. Bit-exact-modulo-atomic-reordering vs the direct path
  (≤ 8e-6 drift).
- **`Ssim2Batch`** is currently a thin sequential wrapper around the
  cached-reference path — correct, validated against the single-image
  path, but **does not yet use kernel-level batching**. That's the
  follow-up (see "What's left").
- **3 lock tests** (`tests/parity_lock.rs`) cover JPEG-corpus parity,
  cached-vs-direct equivalence, and identical-image → 100. All green.

Repo: https://github.com/imazen/turbo-metrics (commit at handoff: see
`git log -1` for the most recent ssim2-gpu commit).

## Validated parity (RTX 5070 + CUDA 13.2, 256×256)

JPEG corpus from `crates/dssim-cuda/test_data/`:

| q | CPU score | GPU score | Δ | rel |
|---|---|---|---|---|
|  1 |   1.2391 |   1.2104 | 0.029 | 2.31 % |
|  5 | -10.4452 | -10.4510 | 0.006 | 0.06 % |
| 20 |  57.0726 |  57.0581 | 0.015 | 0.03 % |
| 45 |  68.6823 |  68.6470 | 0.035 | 0.05 % |
| 70 |  79.5139 |  79.4766 | 0.037 | 0.05 % |
| 90 |  90.8900 |  90.8447 | 0.045 | 0.05 % |

q=1 sits at 2.3 % rel only because the absolute score is small (1.2);
absolute Δ ≈ 0.029 is the f32-vs-f64 reduction noise floor. Every
realistic-quality target (q ≥ 5) is within 0.06 % relative.

## What's done

| Module | LOC | Validation | Status |
|---|---|---|---|
| `kernels::srgb` | 56 | < 3e-7 abs vs CPU `srgb_gamma_to_lin` over 256×3 inputs | ✅ |
| `kernels::xyb` | 86 | < 1e-5 abs vs `yuvxyb::linear_rgb_to_xyb` + `make_positive_xyb` over 1024 random samples | ✅ |
| `kernels::downscale` | 47 | Matches CPU `downscale_by_2` per-pixel | ✅ |
| `kernels::blur` | 137 | < 1e-4 abs vs CPU `Blur::blur` over 6 cases up to 1024×768 | ✅ |
| `kernels::transpose` | 27 | trivial; verified through blur parity | ✅ |
| `kernels::error_maps` | 67 | exercised through end-to-end parity | ✅ |
| `kernels::reduction` | 80 | exercised through end-to-end parity (cached vs direct ≤ 8e-6) | ✅ |
| `pipeline::Ssim2` (single + cached) | ~600 | < 0.06 % rel vs CPU `compute_frame_ssimulacra2` | ✅ |
| `pipeline_batch::Ssim2Batch` | ~50 | matches `Ssim2::compute_with_reference` per-call | 🟡 wrapper |

In-tree examples (all run on RTX 5070 + CUDA 13.2):
- `srgb_parity`, `xyb_parity`, `blur_parity` — per-kernel validation.
- `parity_real_image` — synthetic 256×256, full pipeline vs CPU.
- `parity_jpeg_corpus` — `source.png` + JPEG q1..q90, full pipeline vs CPU.
- `cached_reference` — direct vs cached path drift ≤ 8e-6.
- `batch_smoke` — `Ssim2Batch` per-call equivalence.
- `end_to_end` — synthetic stripe pattern, score sanity print.

In-tree tests (`tests/parity_lock.rs`): 3 cases — `parity_jpeg_corpus`,
`cached_reference_matches_direct`, `identical_image_scores_100`.

## What's left

### 1. Kernel-level `Ssim2Batch` (the optional day-4+ work)

Right now `Ssim2Batch` loops over `compute_with_reference` once per
distorted slot, so per-image launch overhead is the same as the cached
single-image path. The `butteraugli-gpu::ButteraugliBatch` shape —
broadcast reference inputs + 3D launch grids with `CUBE_POS_Z` =
batch_idx — is the next step. Estimated ~300–500 LOC mostly in
batched variants of:
- `error_maps_batched_kernel` (broadcast `source`/`mu1`/`sigma11`
  vs per-slot `distorted`/`mu2`/`sigma22`/`sigma12`)
- `transpose_batched_kernel` (per-image plane stride)
- `blur_pass_batched_kernel` (per-image height clamp; same trap as
  butteraugli's `vertical_blur_batched_kernel` — see G4.8 in
  `docs/CUBECL_GOTCHAS.md`)
- `pointwise_mul_batched_kernel`
- `reduction::launch_sum_p4_batched`

Throughput win at small images is large (~6× at 256² per the
butteraugli numbers); at 1 MP+ it's marginal. Defer until a consumer
needs it.

### 2. Multi-vendor validation

- `--no-default-features --features wgpu` on a host with a working
  Vulkan or Metal ICD. WSL2 doesn't expose one for NVIDIA, so this
  needs native Linux/Mac/Windows.
- HIP backend on AMD silicon when available.

### 3. Cross-arch lock parity

The `parity_lock.rs` tests assert *current* behaviour against current
CPU `ssimulacra2`. A cross-arch lock that hashes a fixed score table
across a small image corpus would catch silent precision regressions
across cubecl versions / CUDA versions / GPUs. Adapt the
`butteraugli-cuda/cross_arch_parity.rs` shape.

### 4. Reduction precision tightening

The q=1 case's 0.029 absolute drift comes from f32 atomic-add reordering
in the reductions. Two options:
- **CUDA-only fast path**: feature-detect `Atomic<f64>::fetch_add` and
  switch the sums kernel. Adds < 50 LOC; CUDA-only.
- **Two-pass deterministic reduction**: reduce per-thread to local
  partials, then a single tree reduction in shared memory + one
  atomic per block. ~80 LOC, portable across all backends. Probably
  the right answer.

Not blocking — the existing precision matches every quality target a
production consumer cares about — but worth doing when extending
parity.

## How to continue

### Environment

You need:
- **CUDA 13.2** (cubecl 0.10's CUDA backend wants `nvrtcGetTileIR`).
  `/usr/local/cuda` should symlink to `cuda-13.2`.
- For multi-vendor validation: native Linux/Mac/Windows where the
  wgpu/Vulkan/Metal ICD is reachable. WSL2 won't do it for NVIDIA GPUs.

### Build commands

```bash
CUDA_PATH=/usr/local/cuda cargo build -p ssim2-gpu
CUDA_PATH=/usr/local/cuda cargo test --release -p ssim2-gpu

# Per-stage parity examples:
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example srgb_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example xyb_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example blur_parity

# Full-pipeline parity:
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example parity_real_image
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example parity_jpeg_corpus

# Cached-reference + batch:
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example cached_reference
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example batch_smoke
```

Cold compile: 5–9 min the first time. Incremental: ~60 s for an
example rebuild after a kernel edit.

### Iteration tips when porting more kernels

- **Always validate against the actual CPU crate**, not a hand-written
  reference. The published `ssimulacra2`'s exported `Blur::blur` and
  `compute_frame_ssimulacra2` are the source of truth — use them
  directly in parity examples.
- **Per-stage parity first, end-to-end second.** Build sub-1e-5
  confidence on each kernel before wiring the whole pipeline. The
  blur-parity test caught a couple of off-by-one IIR-loop-bound bugs
  in five minutes; finding those at the score-parity level would have
  been hours.
- **Identical-image-scores-100 is the cheapest sanity check in the
  world.** Add it to your lock test suite from day one.

### File map

```
crates/ssim2-gpu/
├── Cargo.toml
├── README.md
├── PORT_STATUS.md
├── HANDOFF.md
├── build.rs                 # ✅ Charalampidis IIR coefficients
├── src/
│   ├── lib.rs               # ✅ GpuSsim2Result + re-exports
│   ├── pipeline.rs          # ✅ Ssim2 + score-from-stats (108-weight WEIGHT)
│   ├── pipeline_batch.rs    # 🟡 Ssim2Batch (wrapper; kernel-batched TBD)
│   └── kernels/
│       ├── mod.rs
│       ├── srgb.rs          # ✅ inline sRGB→linear
│       ├── xyb.rs           # ✅ linear→positive XYB
│       ├── downscale.rs     # ✅ 2× planar average
│       ├── blur.rs          # ✅ recursive Gaussian (column walker + ring)
│       ├── transpose.rs     # ✅ used between blur passes
│       ├── error_maps.rs    # ✅ SSIM + ringing + blurring
│       └── reduction.rs     # ✅ fused (Σ, Σ⁴) per plane
├── examples/                # ✅ all 8 examples build and run
└── tests/
    └── parity_lock.rs       # ✅ 3 CI-friendly tests, all passing
```

### Suggested next-session order

1. **Kernel-batched `Ssim2Batch`** — see "What's left §1". Mirror
   `butteraugli-gpu::pipeline_batch.rs` shape; copy the broadcast
   pattern from `l2_diff_broadcast_batched_kernel` for the
   sigma12-style cross terms.
2. **WGPU validation on a native host** — confirms portability.
3. **Multi-vendor lock test** — port a small known-good score table.
4. **Reduction precision** — switch to deterministic two-pass tree
   reduction, then re-run the parity corpus and tighten the threshold.

## Cross-references

- General porting patterns: `docs/CUBECL_PORTING_GUIDE.md`
- Comprehensive cubecl gotchas: `docs/CUBECL_GOTCHAS.md`
- Per-kernel plan that drove this port: `docs/SSIMULACRA2_PORTING_PLAN.md`
- Prior worked example: `crates/butteraugli-gpu/`
- Original CUDA implementation: `crates/ssimulacra2-cuda/` +
  `crates/ssimulacra2-cuda-kernel/`
- Published CPU reference (validation target):
  https://crates.io/crates/ssimulacra2 v0.5.1
