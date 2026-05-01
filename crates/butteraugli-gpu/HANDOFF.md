# butteraugli-gpu — Handoff

Status as of 2026-05-01 (evening). **Single-resolution pipeline now
matches CPU butteraugli to <1 % on synthetic and 0.0 % on real images.**
Remaining: multi-resolution supersample-add and the reference-cache
optimisation.

## TL;DR

- **All 8 kernel modules ported, validated, and wired end-to-end.**
- **Score parity with CPU butteraugli** (`single_resolution=true`):
  GPU within 0.0–0.8 % on synthetic and effectively 0 % on a real
  1018×1014 PNG comparison. Earlier ~2× gap was caused by feeding the
  LF blur (sigma=7.156) into opsin's sensitivity input where CPU uses a
  separate sigma=1.2 5-tap blur. Fixed in commit `a80b6bd`.
- **Identical-image case → exactly 0.** Reduction is bit-exact vs CPU.
- **Toolchain quirks documented** in `PORT_STATUS.md` (8 cubecl 0.10
  codegen issues + workarounds).
- **2 in-tree parity examples**: `parity_vs_cpu` (synthetic perturbations
  64×64 → 512×512) and `parity_real_image` (PNG-based).

Repo: https://github.com/imazen/turbo-metrics — branch `master`.
Latest commit at handoff time: `a80b6bd`.

## Validated parity (post-fix)

`parity_vs_cpu` (synthetic gradient + diagonal stripes):

| size | mag | CPU score | GPU score | Δ |
|---|---|---|---|---|
| 64×64   |  1 |  0.8632 |  0.8634 | +0.0 % |
| 64×64   | 12 |  9.2891 |  9.2556 | +0.4 % |
| 64×64   | 32 | 21.1890 | 21.1232 | +0.3 % |
| 256×256 | 12 |  9.7129 |  9.7149 | +0.0 % |
| 512×512 | 32 | 22.7274 | 22.5454 | +0.8 % |

`parity_real_image` (1018×1014 PNG, JPEG-like 8×8 block perturbation):

| mag | CPU score | GPU score | Δ |
|---|---|---|---|
|  1 |  1.0312 |  1.0314 | +0.0 % |
|  4 |  3.3640 |  3.3646 | +0.0 % |
| 12 | 10.0659 | 10.0678 | +0.0 % |
| 32 | 25.9080 | 25.9132 | +0.0 % |

`pnorm_3` deltas are similarly within 0.1 %. Sub-percent residual is
consistent with f32 round-off across the 30+ kernel pipeline.

## What's done

Kernel ports (one Rust source, dispatchable across CUDA/WGPU/Metal/HIP/CPU):

| Module | LOC | Validation | Status |
|---|---|---|---|
| `kernels::reduction` | 90 | bit-exact max-norm vs CPU; 3-norm <4e-6 rel | ✅ |
| `kernels::colors` | 145 | sRGB <3e-7, opsin <8e-6 abs vs CPU | ✅ |
| `kernels::blur` | 130 | <3e-5 abs vs CPU butteraugli's actual blur over 9 cases | ✅ |
| `kernels::frequency` | 270 | Pipeline parity at <1 % | ✅ |
| `kernels::downscale` | 90 | Ready for multi-res orchestration | ✅ |
| `kernels::masking` | 160 | Pipeline parity at <1 % | ✅ |
| `kernels::diffmap` | 150 | Pipeline parity at <1 % | ✅ |
| `kernels::malta` | 540 | Pipeline parity at <1 % | ✅ |
| `pipeline::Butteraugli` | 750 | <0.0–0.8 % vs CPU on every test | ✅ |

In-tree examples that run on RTX 5070 + CUDA 13.2:
- `reduction_parity` — bit-exact match w/ CPU butteraugli's reduction
- `colors_parity` — sRGB+opsin <8e-6 abs vs CPU
- `blur_parity` — 5 sigmas H+V <5e-7 abs vs CPU
- `end_to_end` — produces real `(score, pnorm_3)` for any image pair
- `parity_vs_cpu` — synthetic perturbation series, all sizes
- `parity_real_image` — real PNG vs JPEG-like perturbation
- `diffmap_inspect` — pixel-by-pixel CPU-vs-GPU diff with intermediate
  buffer dumps (mask/AC/DC/LF/MF/HF/UHF) at probe points

## What's left

### 1. Multi-resolution supersample-add

CPU butteraugli's default mode (without `single_resolution=true`) adds
a half-resolution diffmap mixed in via:

```
dest *= 0.85;
dest += 0.5 * src_upsampled;
```

We have `kernels::downscale::downsample_2x_kernel` and
`add_upsample_2x_kernel` ready; just need orchestration. ~200 LOC.
Adds ~5–15 % to the score on natural images.

### 2. Reference cache

`set_reference()` / `compute_with_reference()` so a series of distorted-
image comparisons skip re-running image-1 kernels. Mirrors butteraugli-
cuda's reference cache shape.

### 3. Production polish

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
│   ├── parity_vs_cpu.rs    # ✅ synthetic CPU-GPU comparison <1 %
│   ├── parity_real_image.rs # ✅ PNG-based comparison <0.1 %
│   └── diffmap_inspect.rs  # ✅ pixel-level diff + buffer dumps
└── tests/
    └── reduction_parity.rs # ✅ 4 cases, all pass
```

### Suggested next-session order

1. **Wire multi-resolution.** Subsample inputs to half-res, run a second
   pipeline pass, supersample-add the result with weight=0.5. ~200 LOC
   of orchestration around the existing `downsample_2x_kernel` and
   `add_upsample_2x_kernel`.
2. **Add reference cache** for the encoder use case.
3. **Lock parity** with a 191-entry cross-arch test ported from CPU
   butteraugli's `cross_arch_parity.rs`.

## Diagnostic notes (kept for future debugging)

The 2× score gap that motivated the SIGMA_OPSIN fix was found by:
1. `parity_vs_cpu` showed scores ≈ 2× CPU on every test.
2. `diffmap_inspect` showed GPU's max-norm pixel value was 2× CPU's at
   the perturbation centre. Wing values (far from peak) were
   wrong-signed on GPU vs zero on CPU.
3. A throwaway diagnostic (since deleted) used a path-dep on
   `butteraugli` with `internals` feature, dumped XYB / LF / MF / HF /
   UHF planes side-by-side. The first divergent stage was XYB:
   GPU produced Y_pert ≈ 210 at the perturbed pixel, CPU produced
   Y_pert ≈ 188.
4. Reading CPU's `opsin_dynamics_image` revealed the internal
   sigma=1.2 `blur_mirrored_5x5` for sensitivity — different from
   the SIGMA_LF=7.156 LF blur that the GPU pipeline was reusing.

If a future bug needs similar localisation, the path-dep diagnostic
pattern is:

```toml
butteraugli = { version = "0.9.2", path = "/home/lilith/work/butteraugli/butteraugli", features = ["internals"] }
```

This exposes `butteraugli::opsin`, `butteraugli::psycho`,
`butteraugli::blur`, `butteraugli::image` for direct comparison
against the GPU's debug accessors (`debug_lf`, `debug_freq`,
`debug_block_diff_ac/dc`, `debug_mask`).

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
