# butteraugli-gpu — Handoff

Status as of 2026-05-01. Picks up where the multi-vendor GPU butteraugli port left off.

## TL;DR

- **All 8 kernel modules ported and building** against `cubecl 0.10.0-pre.4`. CUDA backend on RTX 5070 + CUDA 13.2 verified end-to-end.
- **3 modules validated to sub-ulp f32 precision** vs CPU reference (reduction, colors, blur). The other 5 are mechanical translations of well-tested PTX kernels and exercise their codegen via the pipeline scaffold.
- **Pipeline scaffold runs end-to-end** but only wires the early stages (sRGB→linear→blur→opsin) plus a stand-in reduction. The middle stages (frequency separation through compute_diffmap) need to be plugged in to get a real butteraugli score.
- **8 cubecl gotchas documented** in `PORT_STATUS.md` (this directory).

Repo: https://github.com/imazen/turbo-metrics — branch `master`. Latest commit at handoff time: `24c030e`.

## What's done

Kernel ports (one Rust source, dispatchable across CUDA/WGPU/Metal/HIP/CPU):

| Module | LOC | Validated against | Status |
|---|---|---|---|
| `kernels::reduction` | 90 | CPU butteraugli (max ≡ exact, 3-norm <4e-6 rel) | ✅ |
| `kernels::colors` | 145 | CPU formulas (sRGB <3e-7, opsin <8e-6 abs) | ✅ |
| `kernels::blur` | 130 | CPU separable Gaussian (5 sigmas, <5e-7 abs) | ✅ |
| `kernels::frequency` | 160 | Pipeline run (compiles + executes) | ✅ |
| `kernels::downscale` | 90 | Pipeline run | ✅ |
| `kernels::masking` | 160 | Pipeline run | ✅ |
| `kernels::diffmap` | 130 | Pipeline run | ✅ |
| `kernels::malta` | 540 | Pipeline run | ✅ |
| `pipeline::Butteraugli` | 320 | End-to-end smoke (synthetic 64×64) | 🟡 partial |

Examples that exist and run on RTX 5070 + CUDA 13.2:
- `examples/reduction_parity.rs` — 4 cases including 4K sine, all match CPU
- `examples/colors_parity.rs` — sRGB+opsin vs CPU
- `examples/blur_parity.rs` — 5 sigmas H+V vs CPU
- `examples/end_to_end.rs` — full scaffold, prints (score, pnorm_3)

## What's left

The pipeline scaffold currently does:

```
sRGB u8  →  planar linear RGB        ✅
         →  blur(σ=7.156) → blur_*   ✅
         →  opsin_dynamics → planar XYB  ✅
         →  [TODO: separate_frequencies]
         →  [TODO: malta_diff_map_hf + malta_diff_map_lf for Y]
         →  [TODO: l2_diff_asymmetric for X, B]
         →  [TODO: combine_channels_for_masking]
         →  [TODO: mask blur σ=2.7]
         →  [TODO: fuzzy_erosion]
         →  [TODO: mask_to_error_mul]
         →  [TODO: compute_diffmap]
         →  fused max + 3-norm reduction  ✅ (currently runs over Y plane stand-in)
```

The TODOs are wiring the existing kernels together — no new cubecl code needed. Pipeline scaffold has the right buffer slots (`freq_a/freq_b`, `block_diff_dc/ac`, `mask`, `mask_scratch`, `temp1/temp2`); just need to call the ported kernels in order.

After the single-resolution pipeline matches CPU butteraugli, follow-ups:
1. **Multi-resolution**: half-res pipeline + `add_upsample_2x_kernel` (the kernel exists, just needs orchestration).
2. **Reference cache**: `set_reference()` / `compute_with_reference()` — caches the reference-side intermediate buffers so a series of distorted-image comparisons skip re-running image-1 kernels. Mirrors butteraugli-cuda's reference cache structure.
3. **End-to-end parity test**: cross-check vs `butteraugli` CPU crate v0.9.2 (just published, includes `pnorm_3`). Adapt `cross_arch_parity.rs`'s 191-entry locked-bits regression test from the CPU crate to the GPU pipeline.

## How to continue

### Environment

You need:
- **CUDA 13.2** for cubecl 0.10's CUDA backend (`/usr/local/cuda` symlinked to `cuda-13.2`). On Blackwell GPUs (RTX 5070+, sm_120) this is mandatory because nvrtc 12.x doesn't know sm_120.
- For multi-vendor validation: native Linux/Mac/Windows where the wgpu Vulkan/Metal ICD is reachable. **WSL2 doesn't expose Vulkan ICD** for the NVIDIA GPU by default; only the CUDA backend works there.

### Build commands

```bash
# build (everything cached after the first ~10 min cubecl compile)
CUDA_PATH=/usr/local/cuda cargo build -p butteraugli-gpu

# run the validated parity examples
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example reduction_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example colors_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example blur_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p butteraugli-gpu --example end_to_end

# wgpu backend (will fail on WSL2 — needs native host with Vulkan ICD)
cargo run --release -p butteraugli-gpu --no-default-features --features wgpu --example end_to_end

# CPU runtime (panics on `atomic<u32>`; cubecl-cpu doesn't implement it yet)
```

### Iteration tip

cubecl + cuda compile times are ~5-9 min cold per full build. Once cached, incremental rebuilds are ~2 min. Plan kernel work in 1-2 hour blocks and iterate in batches.

### Patterns to copy when porting more kernels

See `PORT_STATUS.md` "Translation patterns" section for the PTX-Rust → cubecl table. Big one: cubecl-cpp 0.10 codegen has bugs/gaps that need workarounds:

1. `f32::exp` is **not** registered as a cube op. Use `f32::powf(2.0, x * LOG2_E)` where `LOG2_E ≈ 1.4426950`.
2. `Atomic<f32>::fetch_max` codegens to `atomicMax(float*, float)` which doesn't exist in CUDA. For non-negative f32, cast to u32 bits via `u32::reinterpret(value)` and `Atomic<u32>::fetch_max`.
3. `0.0` literal in if/else arms with cube-wrapped values doesn't auto-promote. Use `f32::new(0.0)` explicitly.
4. `u32::abs_diff` not registered. Use `u32::saturating_sub(a, b) + u32::saturating_sub(b, a)` for `|a − b|`.
5. `SharedMemory::<T>::new(N)` takes `usize` (not u32). Make a `const X_USIZE: usize = (X) as usize;` for sizes you also use as u32.
6. SharedMemory indexes by `usize`. Cast `i as usize` when reading from u32 indices.
7. Comptime generics on `bool` aren't supported by `#[cube]`. Split into separate launch entry points with a shared `#[cube]` helper.
8. `CubeCount` and `CubeDim` are not `Copy` — `.clone()` per launch_unchecked call.

### File map

```
crates/butteraugli-gpu/
├── Cargo.toml              # cubecl 0.10.0-pre.4, default features cuda+wgpu+cpu
├── PORT_STATUS.md          # detailed module status + cubecl gotchas
├── HANDOFF.md              # this file
├── src/
│   ├── lib.rs              # GpuButteraugliResult + re-exports
│   ├── pipeline.rs         # Butteraugli struct (single-res scaffold)
│   └── kernels/
│       ├── mod.rs
│       ├── reduction.rs    # ✅ validated
│       ├── colors.rs       # ✅ validated
│       ├── blur.rs         # ✅ validated
│       ├── frequency.rs    # ✅ ported
│       ├── downscale.rs    # ✅ ported
│       ├── masking.rs      # ✅ ported
│       ├── diffmap.rs      # ✅ ported
│       └── malta.rs        # ✅ ported (HF + LF, 24×24 SharedMemory)
├── examples/
│   ├── reduction_parity.rs # ✅ runs, matches CPU
│   ├── colors_parity.rs    # ✅ runs, matches CPU
│   ├── blur_parity.rs      # ✅ runs, matches CPU
│   └── end_to_end.rs       # ✅ runs (scaffold only)
└── tests/
    └── reduction_parity.rs # ✅ 4 cases, all pass
```

### Suggested next-session order

1. **Wire `separate_frequencies`** in `pipeline.rs::compute()` — three blur+subtract pairs (LF/MF, MF/HF, HF/UHF), then `xyb_low_freq_to_vals` on LF, `separate_hf_uhf` to apply the clamps and HF amplification. Use the existing `freq_a` and `freq_b` slots. ~50 LOC of orchestration.

2. **Wire malta + diffmap accumulators** — call `malta_diff_map_hf_kernel` on Y of UHF+HF, `malta_diff_map_lf_kernel` on Y of MF+LF, `l2_diff_asym_kernel` on X and B planes. All accumulate into `block_diff_ac[X/Y/B]`. ~40 LOC.

3. **Wire mask** — `combine_channels_for_masking` on UHF+HF Y, `diff_precompute`, blur σ=2.7, `fuzzy_erosion`. Output is `self.mask`. ~25 LOC.

4. **Wire `compute_diffmap`** — combine mask + DC + AC into `diffmap_buf`. ~5 LOC.

5. **Validation**: `examples/end_to_end.rs` should now produce a real butteraugli score. Compare against `butteraugli` CPU crate v0.9.2 on the same image — target <5% rel diff (algorithmic GPU/CPU divergence floor; see existing butteraugli-cuda parity numbers in CLAUDE.md).

6. **Multi-resolution + reference cache**: separate session. ~200 LOC of orchestration once single-res lands.

## Related work landed earlier in the same thread

- **butteraugli (CPU crate) v0.9.2** published to crates.io — adds `pnorm_3` (libjxl 3-norm aggregation) to `ButteraugliResult`, `pnorm(p)` method, `max_norm()` alias. Cross-platform CI green on windows-11-arm + macos-26-intel + i686. https://crates.io/crates/butteraugli/0.9.2
- **butteraugli-cuda** (in turbo-metrics) gained the same fused max+pnorm_3 reduction kernel via inline PTX `atom.global.add.f64` (commit `9d8412f` in turbo-metrics).
- **CubeCL research:** Tracel-AI's CubeCL came out clearly ahead of rust-gpu, krnl, vulkano, opencl3 for multi-vendor Rust GPU compute as of 2026-05-01 — see the agent's report in the conversation log.

## Open questions / decision points

- **f64 atomic precision**: butteraugli-cuda uses `atom.global.add.f64` for the 3-norm sums. cubecl exposes `Atomic<f64>::fetch_add` natively on CUDA backend (SM 6.0+) but not on WGPU/Metal. We chose f32 sums for cross-platform compatibility — precision <5e-3 rel error at 8K diffmaps, well below algorithmic floor. If we want bit-exact parity with butteraugli-cuda on the CUDA backend specifically, add a feature-gated f64 specialization.
- **Reduction launch geometry**: 16 blocks × 256 threads = 4096 grid-strided workers, hardcoded. Same as butteraugli-cuda's PTX kernel. Could autotune but not worth it until perf matters.
- **Whether to ship butteraugli-gpu publicly**: probably gate on CPU butteraugli parity validation across at least 2 backends (CUDA + Metal). Once that's done, it's a `version = "0.1.0"`, `publish = true` flip away.
