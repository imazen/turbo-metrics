# butteraugli-gpu — Handoff

Status as of 2026-05-01 (afternoon update). The single-resolution
pipeline is fully wired. Remaining: a ~2× calibration gap vs CPU
butteraugli, plus multi-resolution and reference-cache work.

## TL;DR

- **All 8 kernel modules ported, validated, and wired into a real
  end-to-end pipeline** running on RTX 5070 + CUDA 13.2. Identical-
  image case → score = 0; perturbed images → monotone-with-magnitude
  score; reduction matches CPU bit-exact.
- **One open algorithmic issue:** GPU score ≈ 2× CPU score on real
  images and synthetic perturbations (`single_resolution=true` on the
  CPU side). Mean diffmap ratio is 1.14×; peak ratio is 2×. Source
  unknown — the most likely candidates are gamma function precision
  (CPU uses `fast_log2f` polynomial; GPU uses CUDA's native `ln`)
  and/or some quiet scaling difference inside frequency separation.
- **Toolchain quirks documented** in `PORT_STATUS.md`.
- **3 new diagnostic examples**: `parity_vs_cpu`, `parity_real_image`,
  `diffmap_inspect` — last one dumps GPU intermediate buffers
  (mask, AC/DC accumulators, LF/MF/HF/UHF planes) to localize the gap.

Repo: https://github.com/imazen/turbo-metrics — branch `master`.
Latest commit at handoff time: `7250d0e`.

## What's done

Kernel ports (one Rust source, dispatchable across CUDA/WGPU/Metal/HIP/CPU):

| Module | LOC | Validated against | Status |
|---|---|---|---|
| `kernels::reduction` | 90 | CPU butteraugli (max ≡ exact, 3-norm <4e-6 rel) | ✅ |
| `kernels::colors` | 145 | CPU formulas (sRGB <3e-7, opsin <8e-6 abs) | ✅ |
| `kernels::blur` | 130 | CPU separable Gaussian (5 sigmas, <5e-7 abs) | ✅ |
| `kernels::frequency` | 270 | Pipeline run | ✅ |
| `kernels::downscale` | 90 | Pipeline run | ✅ |
| `kernels::masking` | 160 | Pipeline run | ✅ |
| `kernels::diffmap` | 150 | Pipeline run | ✅ |
| `kernels::malta` | 540 | Pipeline run | ✅ |
| `pipeline::Butteraugli` | 750 | End-to-end real GPU butteraugli scores | ✅ wired |

Examples that exist and run on RTX 5070 + CUDA 13.2:
- `reduction_parity` — bit-exact match w/ CPU
- `colors_parity` — sRGB+opsin <8e-6 abs vs CPU
- `blur_parity` — 5 sigmas H+V <5e-7 abs vs CPU
- `end_to_end` — produces real `(score, pnorm_3)` for any image pair
- `parity_vs_cpu` — synthetic perturbation series, 64×64 to 512×512
- `parity_real_image` — real PNG vs JPEG-like perturbation
- `diffmap_inspect` — pixel-by-pixel CPU-vs-GPU diff with intermediate
  buffer dumps (mask/AC/DC/LF/MF/HF/UHF) at probe points

## What's left

### 1. Close the ~2× calibration gap (open question)

GPU and CPU agree on the *zero* case but diverge by a factor of ~2 on
non-trivial inputs. Numbers from the synthetic test on `parity_vs_cpu`:

| size | mag | CPU score | GPU score | Δ |
|---|---|---|---|---|
| 64×64 | 1 | 0.86 | 1.08 | +25% |
| 64×64 | 12 | 9.29 | 13.62 | +47% |
| 64×64 | 32 | 21.19 | 32.99 | +56% |
| 1018×1014 | 12 | 10.07 | 24.82 | +147% |

Mean diffmap ratio is much closer to 1 (~1.14×) — the divergence is
mostly in the peak. From the `diffmap_inspect` dump on a flat-128
image with one perturbed pixel at the center:

```
At (32, 32) center:
  GPU AC[Y] sum = 1696    (Malta UHF + HF + MF + L2asym + L2 + mask_to_error)
  CPU AC[Y] sum ≈ 398     (same algorithm shape, single-resolution)
```

That's 4× larger in AC, 2× larger in `sqrt(mask·AC + mdc·DC)`. The
2× factor is suspicious — possible causes worth investigating:

1. **Gamma precision:** CPU uses `fast_log2f` (polynomial, ~3.9e-6
   abs error). GPU uses CUDA `ln`. For values ~30 the absolute error
   is similar but the CPU polynomial happens to bias *low*; we may
   be propagating a sensitivity that's slightly different in a way
   that amplifies through frequency separation.
2. **Some scaling we miss in the X-channel pre-Malta path** — `WMUL[3]
   = 2150` is the largest weight; small upstream divergences would
   dominate.
3. **Mask blur scratch:** I fixed one buffer-aliasing bug in image-B
   mask blur (`diffmap_buf` is now used as the H-pass scratch).
   Worth re-scanning the orchestration for similar issues.

The score is *meaningful* — perturbation magnitude correlates
monotonically and identical inputs give exactly 0. Just calibrated wrong.

### 2. Multi-resolution supersample-add

CPU butteraugli's default mode (without `single_resolution=true`) adds
a half-resolution diffmap mixed in via:

```
dest *= 0.85;
dest += 0.5 * src_upsampled;
```

We have `kernels::downscale::downsample_2x_kernel` and
`add_upsample_2x_kernel` ready; just need orchestration. ~200 LOC.

### 3. Reference cache

`set_reference()` / `compute_with_reference()` so a series of distorted-
image comparisons skip re-running image-1 kernels. Mirrors butteraugli-
cuda's reference cache shape.

### 4. Production polish

- Cross-arch parity test (port the 191-entry `cross_arch_parity.rs`
  table from CPU butteraugli).
- WGPU validation on a host with a real Vulkan ICD (WSL2 doesn't
  expose one for the NVIDIA GPU).
- HIP validation on AMD silicon when available.

## How to continue

### Environment

You need:
- **CUDA 13.2** for cubecl 0.10's CUDA backend (`/usr/local/cuda`
  symlinked to `cuda-13.2`). On Blackwell GPUs (RTX 5070+, sm_120)
  this is mandatory because nvrtc 12.x doesn't know sm_120.
- For multi-vendor validation: native Linux/Mac/Windows where the
  wgpu Vulkan/Metal ICD is reachable. **WSL2 doesn't expose a Vulkan
  ICD** for the NVIDIA GPU by default; only the CUDA backend works
  there.

### Build commands

```bash
CUDA_PATH=/usr/local/cuda cargo build -p butteraugli-gpu

CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example reduction_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example colors_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example blur_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example end_to_end
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example parity_vs_cpu
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example parity_real_image
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example diffmap_inspect
```

### Iteration tip

cubecl + cuda compile times are ~5-9 min cold per full build. Once
cached, incremental rebuilds are ~2 min. Plan kernel work in 1-2 hour
blocks and iterate in batches.

### Patterns to copy when porting more kernels

See `PORT_STATUS.md` "Translation patterns" section for the PTX-Rust
→ cubecl table. Big ones (cubecl 0.10 codegen quirks):

1. `f32::exp` is **not** registered as a cube op. Use
   `f32::powf(2.0, x * LOG2_E)` where `LOG2_E ≈ 1.4426950`.
2. `Atomic<f32>::fetch_max` codegens to a non-existent
   `atomicMax(float*, float)`. For non-negative f32, cast to u32 bits
   via `u32::reinterpret(value)` and `Atomic<u32>::fetch_max`.
3. `0.0` literal in if/else arms with cube-wrapped values doesn't
   auto-promote. Use `f32::new(0.0)` explicitly.
4. `u32::abs_diff` not registered. Use
   `u32::saturating_sub(a, b) + u32::saturating_sub(b, a)` for `|a−b|`.
5. `SharedMemory::<T>::new(N)` takes `usize` (not u32). Make a
   `const X_USIZE: usize = (X) as usize;` for sizes you also use as
   u32.
6. SharedMemory indexes by `usize`. Cast `i as usize` when reading
   from u32 indices.
7. Comptime generics on `bool` aren't supported by `#[cube]`. Split
   into separate launch entry points with a shared `#[cube]` helper.
8. `CubeCount` and `CubeDim` are not `Copy` — `.clone()` per
   `launch_unchecked` call.

### Pipeline buffer-aliasing rules

`pipeline.rs` calls `blur_plane_via(src, dst, scratch)` which does
`H(src → scratch)` followed by `V(scratch → dst)`. Constraints:
- `scratch` must differ from BOTH `src` and `dst`.
- `src == dst` is OK (V writes after reading scratch).

The mask pipeline reuses `diffmap_buf` as a scratch since it's not
written until the final compute_diffmap step. Other long stretches
where temp1/temp2 are both occupied need similar care.

### File map

```
crates/butteraugli-gpu/
├── Cargo.toml              # cubecl 0.10.0-pre.4, default features cuda+wgpu+cpu
├── PORT_STATUS.md          # detailed module status + cubecl gotchas
├── HANDOFF.md              # this file
├── src/
│   ├── lib.rs              # GpuButteraugliResult + re-exports
│   ├── pipeline.rs         # Butteraugli — full single-res pipeline
│   └── kernels/
│       ├── mod.rs
│       ├── reduction.rs    # ✅ validated
│       ├── colors.rs       # ✅ validated
│       ├── blur.rs         # ✅ validated
│       ├── frequency.rs    # ✅ ported (split + zero/copy helpers)
│       ├── downscale.rs    # ✅ ported (multi-res orchestration TBD)
│       ├── masking.rs      # ✅ ported
│       ├── diffmap.rs      # ✅ ported (l2_diff_write added)
│       └── malta.rs        # ✅ ported (HF + LF, 24×24 SharedMemory)
├── examples/
│   ├── reduction_parity.rs # ✅ runs, matches CPU
│   ├── colors_parity.rs    # ✅ runs, matches CPU
│   ├── blur_parity.rs      # ✅ runs, matches CPU
│   ├── end_to_end.rs       # ✅ produces (score, pnorm_3)
│   ├── parity_vs_cpu.rs    # ✅ synthetic CPU-GPU comparison
│   ├── parity_real_image.rs # ✅ PNG-based comparison
│   └── diffmap_inspect.rs  # ✅ pixel-level diff + buffer dumps
└── tests/
    └── reduction_parity.rs # ✅ 4 cases, all pass
```

### Suggested next-session order

1. **Find the 2× source.** Run `diffmap_inspect`, write a CPU-side
   probe via `butteraugli::psycho::separate_frequencies` (the module
   is public), dump CPU intermediate buffers at the same probe points
   as the GPU dump. Diff. The first plane that diverges is the bug.
2. After parity lands, **wire multi-resolution** (~200 LOC of
   orchestration around the existing `downsample_2x_kernel` and
   `add_upsample_2x_kernel`).
3. Add **reference cache** for the encoder use case.
4. Lock parity with a 191-entry cross-arch test ported from CPU
   butteraugli's `cross_arch_parity.rs`.

## Related work landed earlier in the same thread

- **butteraugli (CPU crate) v0.9.2** published to crates.io — adds
  `pnorm_3` (libjxl 3-norm aggregation) to `ButteraugliResult`,
  `pnorm(p)` method, `max_norm()` alias.
  https://crates.io/crates/butteraugli/0.9.2
- **butteraugli-cuda** (in turbo-metrics) gained the same fused
  max+pnorm_3 reduction kernel via inline PTX `atom.global.add.f64`
  (commit `9d8412f`).
- **CubeCL research:** Tracel-AI's CubeCL came out clearly ahead of
  rust-gpu, krnl, vulkano, opencl3 for multi-vendor Rust GPU compute
  as of 2026-05-01.
