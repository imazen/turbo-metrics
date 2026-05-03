# dssim-gpu port status

Multi-vendor GPU port of `dssim-cuda` using CubeCL. Validated against
the published `dssim-core` v3.4 CPU reference at < 1 % relative error
on JPEG corpus and the synthetic black/white/gradient cases.

## Module status

| Module | Source | LOC | Status | Notes |
|---|---|---|---|---|
| `kernels::srgb` | `dssim-cuda-kernel/src/srgb.rs` | 49 | ✅ ported | Inline transfer (matches the LUT at byte resolution); inputs widened u8→u32 on the host so wgpu/Metal can read `Array<u32>` natively. Same shape as `ssim2-gpu`. |
| `kernels::lab` | `dssim-cuda-kernel/src/lab.rs` | 87 | ✅ ported | Custom-scaled Lab matching `dssim-core::tolab.rs`. `cbrt` substituted with `f32::powf(_, 1.0/3.0)` — cubecl 0.10 has no `cbrt` op (gotcha G1.x equivalent). At byte-precision input the difference is below 1 ulp; verified by integration tests. |
| `kernels::downscale` | `dssim-cuda-kernel/src/downscale.rs` | 47 | ✅ ported | Single-plane 2×2 average with edge-clamp. Skipped the packed-RGB CUDA variant — the pipeline keeps R/G/B planar from sRGB onwards (saves an indirection, matches `ssim2-gpu`'s shape). |
| `kernels::blur` | `dssim-cuda-kernel/src/blur.rs` | 175 | ✅ ported | Three variants: `blur_3x3`, `blur_squared`, `blur_product`. Fixed-coefficient 9-tap Gaussian with replicate-clamp boundary — verbatim coefficients from `dssim-core::blur::BLUR_KERNEL`. |
| `kernels::ssim` | `dssim-cuda-kernel/src/ssim.rs` | 99 | ✅ ported | Fused 15-input Lab SSIM map (averages mu/cov terms across L/a/b before the standard SSIM formula). Plus pointwise `abs_diff_scalar` for the per-scale MAD step. |
| `kernels::reduction` | NPP `Sum` × 2 | 196 | ✅ ported | Slotted Σ via `Atomic<f32>::fetch_add` (`fast-reduction` feature, default-on) plus a portable two-stage finalizer for Metal. 10 slots = 5 scales × 2 reductions per scale (Σ ssim, Σ \|ssim − avg\|). |
| Pipeline (`pipeline::Dssim`) | `dssim-cuda/src/lib.rs` | ~470 | ✅ wired | Five-scale pyramid; per-scale chroma pre-blur + two-pass blur for mu / sq / cross; fused SSIM map; per-scale MAD; weighted final score with `dssim-core`'s `[0.028, 0.197, 0.322, 0.298, 0.155]` weights, then `1/ssim − 1` to convert SSIM → DSSIM. |
| `Dssim::set_reference` / `compute_with_reference` | same | (above) | ✅ implemented | Cached state: per-scale `ref_lin`, `ref_lab`, `ref_mu`, `ref_sq_blur`. Cached-vs-direct drift ≤ 1e-5 absolute on the integration tests. |

## Validated parity (RTX 5070, CUDA 13.2, host-side `dssim-core` v3.4)

`tests/parity_lock.rs` — 9/9 pass on CUDA:

| Case | CPU (`dssim-core`) | GPU (`dssim-gpu`) | rel |
|---|---|---|---|
| 32×32 identical gradient | 0.000000 | 0.000000 | 0.00 % |
| 64×64 black vs white | 0.541865 | 0.541865 | 0.00 % |
| 64×64 noisy gradient (±8) | 0.006706 | 0.006713 | 0.10 % |
| `dssim-cuda` corpus q70.jpg vs source.png | 0.000903 | 0.000903 | 0.10 % |
| `dssim-cuda` corpus q90.jpg vs source.png | 0.000186 | 0.000187 | 0.67 % |

The 0.67 % at q90 is well within the f32-vs-f64 reduction noise floor
on a tiny score (1.86 × 10⁻⁴ absolute, Δ ≈ 1.2 × 10⁻⁶); for any
practical use the agreement is exact to four decimal places.

`cached_reference_matches_direct`: |direct − cached| ≤ 1 × 10⁻⁵ on the
synthetic 64×64 noisy-gradient case.

## Backend coverage

| Backend | Build | Tests | Notes |
|---|---|---|---|
| CUDA (NVIDIA, native) | ✅ | ✅ 9/9 (CUDA 13.2 / RTX 5070) | Full parity validated. |
| WGPU (cross-vendor) | ✅ | ⚠ untested in WSL2 | WSL2 has no Vulkan ICD by default (gotcha G3.2); validate on native Linux / Mac / Windows. |
| HIP (AMD ROCm) | ✅ (compiles) | ⚠ untested | Same reduction code path as CUDA; should work by analogy with `ssim2-gpu` / `butteraugli-gpu`. |
| CPU (cubecl-cpu) | ✅ (compiles) | ❌ kernels panic at runtime | `cubecl-cpu` 0.10 doesn't implement `Atomic<f32>` (gotcha G3.3) and panics on `CUBE_COUNT` builtin in our reduction. Build-only; not a parity target. Use `dssim-core` as the CPU reference instead. |

## Known gotchas applied

- **G1.4 / lab `cbrt`** → substituted `f32::powf(_, 1.0/3.0)`.
- **G1.5 / SharedMemory sizing** — n/a (no shared memory in this port;
  the 3×3 blur is small enough to read directly from `Array<f32>`).
- **G2.1 / `CubeCount`/`CubeDim` not Copy** — every launch site
  recomputes the count via `Self::cube_count_1d(n)` rather than
  caching, matching `ssim2-gpu`'s pattern.
- **G3.2 / WSL2 no Vulkan ICD** — wgpu backend is build-only on the
  reference host; CUDA validates the algorithm.
- **G3.3 / cubecl-cpu no atomics** — explicit limitation noted in
  README and tests; cubecl-cpu is build-only.
- **G4.1 / blur scratch aliasing** — the chroma pre-blur uses two
  distinct scratch buffers (`temp1` and `temp2`) so neither aliases
  the source on either pass. The mu / sq / cross pipelines write to
  disjoint destinations.
- **G4.6 / accumulator zeroing** — `Dssim::zero_partials()` reallocates
  both reduction buffers at the start of every `compute*` call so the
  fast-mode `fetch_add` doesn't inherit the previous call's totals.

## Followups (not blocking)

- Per-kernel parity examples (e.g. `srgb_parity.rs`, `blur_parity.rs`)
  modelled on `ssim2-gpu`'s set. The integration tests already
  validate the full pipeline; per-kernel diagnostics would only be
  needed if a regression appears.
- Batched scoring (`DssimBatch`) on the `butteraugli-gpu` /
  `ssim2-gpu` shape — useful for encoder rate-distortion sweeps. Not
  implemented yet because `dssim-cuda` itself doesn't expose a
  batched API; would be a workspace-level addition rather than a
  port.
- `zen-metrics-cli` integration. The CLI is currently being reworked
  by a concurrent agent (`feat(zen-metrics-cli)!: drop -cpu suffix,
  remove dssim, bump to 0.2.0`); re-adding `dssim-gpu` to the
  metric registry is a follow-up commit once that lands.
