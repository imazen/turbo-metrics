# ssim2-gpu — starter HANDOFF

**Status: not started.** The plan, gotchas, and worked example are
all written; this file is the resume point for whoever picks up the
implementation. Move it to `crates/ssim2-gpu/HANDOFF.md` once the
crate skeleton lands.

## Read first (in order)

1. **[`CUBECL_GOTCHAS.md`](CUBECL_GOTCHAS.md)** — every cubecl-0.10-era
   trap with symptoms / fixes / examples. Skim end-to-end before
   writing the first line of cubecl code; bookmark for debugging.
2. **[`SSIMULACRA2_PORTING_PLAN.md`](SSIMULACRA2_PORTING_PLAN.md)** —
   per-kernel breakdown for ssimulacra2 specifically, with sizing
   table and 3-day plan. Drives the work.
3. **[`CUBECL_PORTING_GUIDE.md`](CUBECL_PORTING_GUIDE.md)** — general
   patterns (skeleton, validation strategy, batching). Reference back
   when needed.
4. **`crates/butteraugli-gpu/`** — worked example. Mirror the layout,
   the API shape, and the `pipeline.rs` / `pipeline_batch.rs` / `kernels/`
   structure 1:1. Copy `PORT_STATUS.md` and adapt.

## Resume here

The very first session should:

1. **Verify toolchain.** `CUDA_PATH=/usr/local/cuda cargo build -p
   butteraugli-gpu` should succeed end-to-end without errors. If
   cudarse fails, see G7.2 in CUBECL_GOTCHAS.md (the patches are in
   commit `f745da9`, should already be on master). If cubecl fails,
   verify CUDA 13 is installed (G3.1).
2. **Scaffold the crate** per
   [`SSIMULACRA2_PORTING_PLAN.md` §1](SSIMULACRA2_PORTING_PLAN.md):
   `crates/ssim2-gpu/{Cargo.toml, src/{lib.rs, pipeline.rs, kernels/{mod.rs,
   srgb.rs, xyb.rs, downscale.rs, blur.rs, error_maps.rs, reduction.rs}},
   examples/, PORT_STATUS.md, HANDOFF.md}`. Copy the Cargo.toml
   feature layout from butteraugli-gpu.
3. **Port `srgb` and write `srgb_parity.rs` first** (sub-day,
   bit-exact target). Get an end-to-end "I have something building
   and validating" win in the first hour.

## What's expected to be hard

The recursive Charalampidis blur (§2 of the porting plan).
Everything else is direct translation; that one is a stateful per-
column IIR with shared-memory ring buffer, six FMAs per iteration,
and a sigma response that won't match a hand-written Gaussian. **Do
not proceed past blur parity until it matches the published
ssimulacra2 crate's actual `blur` to ≤ 1e-5 abs on a real image.**
This was the single biggest correctness lesson from butteraugli-gpu
(see G5.1, G4.5 in CUBECL_GOTCHAS.md).

Budget a full day for blur parity. If the second day is bleeding into
the third, slip day-3 work (`error_maps` + final orchestration) — do
not compromise on blur parity.

## Toolchain reality (lifted from butteraugli-gpu HANDOFF)

- **CUDA 13.2** for cubecl 0.10's CUDA backend (`/usr/local/cuda`
  symlinked to `cuda-13.2`). On Blackwell GPUs (RTX 5070+, sm_120)
  this is mandatory because nvrtc 12.x doesn't know sm_120.
- For multi-vendor validation: native Linux/Mac/Windows where the
  wgpu Vulkan/Metal ICD is reachable. **WSL2 doesn't expose a Vulkan
  ICD** for the NVIDIA GPU by default; only the CUDA backend works
  there.
- cubecl-cpu doesn't implement `atomic<u32>` — not useful as a CPU
  validator for kernels that use atomics (i.e., the reductions).

## Build commands (once the crate exists)

```bash
CUDA_PATH=/usr/local/cuda cargo build -p ssim2-gpu

CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example srgb_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example xyb_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example blur_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example error_maps_parity
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example end_to_end
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example parity_real_image
```

Cold compile time will be 5–9 minutes the first time (G6.1).
Incremental rebuilds ~2 min.

## Suggested next-session order

Day 1 (≈ 6–8 hours of focused work):

1. Scaffold + Cargo.toml + lib.rs + kernels/mod.rs (~30 min).
2. Port `srgb_to_linear` kernel + `srgb_parity.rs` (~1 hour). Validate
   bit-exact against `ssimulacra2::xyb::srgb_to_linear`. Don't
   forget the LUT (256 entries, copy verbatim).
3. Port `xyb` (linear→XYB) kernel + `xyb_parity.rs` (~2 hours). The
   `cbrt` op may need the `powf(1/3)` substitution; check cubecl
   ops list.
4. Port `downscale_2x_packed` kernel + run it from end_to_end.rs (~1
   hour). Skip the warp-shuffle plane variant — no portable WGPU
   equivalent.
5. Port `reduction` (per-octave fused reduce) — start with the simple
   3-launch version, fuse later (~2 hours).
6. Stub out `pipeline.rs::compute()` with sRGB→linear→XYB→downscale
   pyramid + reduction (skip blur and error_maps for now). Should run
   end-to-end and produce *some* scalar that's wildly wrong — that's
   fine for day 1. Goal: nothing panics.

Day 2 (the hard day, ≈ 6–8 hours):

7. Port the recursive Charalampidis blur. Start by copying
   `build.rs` from ssimulacra2-cuda-kernel verbatim (it just
   generates Charalampidis coefficients into
   `recursive_gaussian.rs`). Translate the 137-LOC IIR kernel from
   the CUDA Rust to `#[cube]` Rust (see SSIMULACRA2_PORTING_PLAN.md
   §2 for the kernel signature sketch). Drop the 5-plane fanout —
   one launch per plane.
8. Write `blur_parity.rs` validating against the *actual*
   `ssimulacra2` CPU crate's blur (path-dep with `internals` feature
   if needed — see G5.1). Iterate until ≤ 1e-5 abs on a 256×256
   gradient image.

Day 3 (≈ 4–6 hours):

9. Port `error_maps` fused 3-output kernel (~1 hour, direct
   translation).
10. Wire the per-octave loop in `pipeline.rs::compute()`: blur →
    pointwise multiplies for sigma11/22/12 → blur of those → run
    `error_maps` → reduce.
11. `end_to_end.rs` should now produce a real ssimulacra2 score.
    Validate against published `ssimulacra2` CPU crate on a real PNG;
    target Δ ≤ 0.1 % score deviation.
12. Add `set_reference_linear` / `compute_with_reference_linear`
    mirroring the butteraugli-gpu pattern. Cache `ref_linear[*]`,
    `ref_xyb[*]`, `mu1[*]`, `sigma11[*]`. Validate cache drift = 0.0
    vs full-compute.

Day 4+ (optional):

13. `Ssim2Batch<R>` for encoder use case. Same pattern as
    `ButteraugliBatch` — pack N distorted images, batched kernels,
    broadcast cached reference. ~300–500 LOC depending on how many
    kernels need batched variants.
14. Cross-arch lock test (port the CPU's known-good score table).
15. Multi-vendor validation: build with `--no-default-features
    --features wgpu` on a Linux/Mac host with a Vulkan/Metal ICD.

## Open questions / decision points

- **6-octave pyramid: separate buffers per level or shared
  pool?** ssimulacra2-cuda allocates per-level (~800 MB at 1440×1080
  per the CUDA crate's docstring). For the first cut, mirror that.
  If memory pressure becomes an issue, look at reusing the smaller
  octaves' buffers — but the CUDA version doesn't bother and ships
  fine. Don't pre-optimize.

- **Single-image vs always-multi-octave API?** ssimulacra2 IS the
  6-octave pyramid; there's no meaningful "single resolution" mode
  the way butteraugli has. So no `new` vs `new_multires` split —
  just one constructor.

- **Score struct shape.** `GpuSsim2Result { score: f64 }` is the
  obvious shape. There's no `pnorm_3` analogue (ssimulacra2 has its
  own pnorm-like aggregation, but it's part of the per-octave
  reduction, not a separate exported metric). Mirror the CPU
  crate's return type.

- **Reduction kernel: fused vs split?** Each octave needs three
  aggregations (sum, max-norm, libssimulacra2-pnorm) per channel.
  Three launches per octave is simpler, ~2× slower than one fused
  launch. Start with three; fuse only if profiling shows it
  matters. (At 6 octaves × 3 channels × 3 metrics = 54 small
  reductions per call, the fused version probably matters more than
  it does for butteraugli's single reduction.)

- **Whether to ship `Ssim2Batch`.** Encoder-side rate-distortion
  loops are the big winner. The batch implementation is ~1 day of
  extra work. If your near-term consumer is just video metric
  reporting (one comparison at a time), skip it — `compute_with_
  reference` already cuts per-call cost by ~2×.

## Score interpretation

ssimulacra2 outputs a scalar in roughly the 0–100 range (higher =
better; 100 = identical, 0 = visually broken). This is different
from butteraugli's 0–30 max-norm range. The score-from-features
weights are hard-coded constants from libssimulacra2; pull them from
the CPU crate's source as a `[f64; 54]` and dot-product host-side.

## Files you'll touch in turbo-metrics master

New (under `crates/ssim2-gpu/`):
- `Cargo.toml`
- `src/lib.rs`
- `src/pipeline.rs`
- `src/pipeline_batch.rs` (day 4+)
- `src/kernels/{mod, srgb, xyb, downscale, blur, error_maps, reduction}.rs`
- `examples/{srgb,xyb,blur,error_maps}_parity.rs`
- `examples/end_to_end.rs`
- `examples/parity_real_image.rs`
- `examples/batch_parity.rs` (day 4+)
- `tests/lock.rs` (day 3 or later)
- `PORT_STATUS.md`, `HANDOFF.md`, `build.rs`

Optionally modified (only if you find new gotchas worth recording):
- `docs/CUBECL_GOTCHAS.md`
- `docs/SSIMULACRA2_PORTING_PLAN.md`

Existing crates that should NOT be touched:
- `crates/ssimulacra2-cuda*` — keep the CUDA path running for
  parallel validation. Port to a *new* crate `ssim2-gpu`, don't
  in-place rewrite.
- `crates/butteraugli-gpu/` — reference only.
- `crates/cudarse/*` — already CUDA-13-patched in commit `f745da9`,
  no further changes needed for ssim2.

## Risks

| risk | mitigation |
|---|---|
| Blur IIR parity slips into day 3 | The plan assumes this; slip day-3 work, don't compromise on parity |
| cubecl-cuda's nvrtc breaks on a future CUDA point release | Pin a working CUDA version in PORT_STATUS.md; don't aggressively upgrade mid-port |
| Score weight constants drift between libssimulacra2 versions | Pin to the same CPU crate version you're validating against |
| Memory pressure at 4K/8K (per-octave allocations sum to GB) | First-cut: don't worry. If it becomes real: reuse buffers across octaves. CUDA version's docstring claims 800 MB at 1440×1080 — call it the working budget |

## Where to ask for help

- The general guide and gotchas docs cover ~95 % of likely failure
  modes. If your bug isn't there, walk through the
  CUBECL_GOTCHAS.md "quick-reference checklist" at the bottom.
- For cubecl-specific questions (graph capture, stream priority,
  bool comptime generics), file an issue at
  https://github.com/tracel-ai/cubecl/issues — see
  [#1319](https://github.com/tracel-ai/cubecl/issues/1319) as an
  example of the format.
- For ssimulacra2 algorithm questions, the upstream is
  https://github.com/cloudinary/ssimulacra2.

## Cross-references

- General patterns: [`CUBECL_PORTING_GUIDE.md`](CUBECL_PORTING_GUIDE.md)
- Comprehensive gotchas: [`CUBECL_GOTCHAS.md`](CUBECL_GOTCHAS.md)
- Detailed per-kernel plan: [`SSIMULACRA2_PORTING_PLAN.md`](SSIMULACRA2_PORTING_PLAN.md)
- Worked example: [`crates/butteraugli-gpu/`](../crates/butteraugli-gpu/)
- Original CUDA implementation: [`crates/ssimulacra2-cuda/`](../crates/ssimulacra2-cuda/) +
  [`crates/ssimulacra2-cuda-kernel/`](../crates/ssimulacra2-cuda-kernel/)
- Open upstream blocker: [tracel-ai/cubecl#1319](https://github.com/tracel-ai/cubecl/issues/1319)
  (CUDA stream priority — not a porting blocker, just a follow-up
  perf knob)
