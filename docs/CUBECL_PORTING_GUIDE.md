# CUDA → CubeCL porting guide

Distilled from porting `butteraugli-cuda` → `butteraugli-gpu` over the
2026-05-01 sessions. Targets the next wave: `ssimulacra2-cuda`,
`dssim-cuda`, `zensim-cuda`. Following this guide should get a
CUDA-only metric crate to a multi-vendor CubeCL one in 1–3 days each
depending on kernel count.

## Why port

| | CUDA-only (PTX) | CubeCL (this guide) |
|---|---|---|
| NVIDIA | ✓ | ✓ (same hardware path via cubecl-cuda) |
| AMD ROCm | ✗ | ✓ via `cubecl/hip` |
| Intel / Apple / cross-platform | ✗ | ✓ via `cubecl/wgpu` (Vulkan / Metal / DX12) |
| WASM compute | ✗ | ✓ via `cubecl/wgpu` (WebGPU) |
| CPU validator | ✗ (NPP-bound) | partially (cubecl-cpu, see Gotchas) |
| Kernel source | `.cu` files + nvcc + nvptx-builder build.rs | `#[cube]`-annotated Rust, no external compile |
| NPP dependency | yes (= NVIDIA-only) | no |
| Build complexity | nvcc, nvrtc, ptx, NPP linkage | one cargo build, no system tools beyond CUDA SDK for the cuda backend |
| Dispatch flexibility | one kernel binary per arch, baked at build | runtime kernel codegen per backend |

Portability is the headline; the secondary win is not having to
maintain build.rs / `.cu` files.

The cost: cubecl 0.10 has rough edges (the codegen quirks below) and
~5–9 min cold compile time on first build. Once cached, incremental
rebuilds are ~2 min.

## When to skip the port

- The crate already serves a CUDA-only consumer with no plans to broaden
  reach (e.g., a video decoder pipeline that's CUDA all the way down).
- Hot kernels that depend on Tensor Cores, warp shuffle, or
  warp-cooperative primitives that cubecl 0.10 doesn't expose. Check
  cubecl's `prelude::*` for what's wrapped before committing.

## The 8 cubecl 0.10 gotchas to internalise first

These bit me at least once each. Document them in the new crate's
`PORT_STATUS.md` so the next person doesn't relearn.

1. **`f32::exp` is not registered as a cube op.** Use
   `f32::powf(2.0, x * LOG2_E)` where `LOG2_E ≈ 1.4426950`.
2. **`Atomic<f32>::fetch_max` codegens to `atomicMax(float*, float)`**
   which doesn't exist in CUDA. For non-negative f32, cast to u32 bits
   via `u32::reinterpret(value)` and use `Atomic<u32>::fetch_max`. f32
   IEEE-754 bit-pattern ordering matches f32 value ordering for
   non-negative values.
3. **`0.0` literal in if/else arms with cube-wrapped values doesn't
   auto-promote.** Use `f32::new(0.0)` explicitly.
4. **`u32::abs_diff` not registered.** Use
   `u32::saturating_sub(a, b) + u32::saturating_sub(b, a)` for `|a−b|`.
5. **`SharedMemory::<T>::new(N)` takes `usize`** (not u32). When the
   same constant is used both as a shared-memory size and as a u32 in
   a launch shape, declare it twice:
   ```rust
   const SHARED_SIZE: u32 = 24;
   const SHARED_TOTAL_USIZE: usize = (SHARED_SIZE * SHARED_SIZE) as usize;
   ```
6. **SharedMemory indexes by `usize`.** Cast `i as usize` when reading
   from u32 indices.
7. **Comptime generics on `bool` aren't supported by `#[cube]`.** Split
   into separate launch entry points with a shared `#[cube]` helper.
8. **`CubeCount` and `CubeDim` are not `Copy`** — `.clone()` per
   `launch_unchecked` call.

Other smaller surprises:

- `f32::ln` is base-e (correct natural log). `f32::log` is base-2.
- `cubecl-cpu` doesn't implement `atomic<u32>` — so it can't be used as
  a CPU validator for any kernel that uses atomics.
- WSL2 has no Vulkan ICD for NVIDIA GPUs by default → wgpu backend
  panics there. Test on native host.
- Cubecl 0.10.0-pre.4 needs CUDA 13's `nvrtcGetTileIR`. CUDA 12 won't
  link the cuda backend. RTX 5070 (sm_120) requires CUDA 13 anyway.

## Crate skeleton

For a metric crate `foo-cuda` you're porting to `foo-gpu`:

```
crates/foo-gpu/
├── Cargo.toml
├── PORT_STATUS.md          # gotchas + per-module status
├── HANDOFF.md              # session-handoff notes
├── src/
│   ├── lib.rs              # GpuFooResult + re-exports
│   ├── pipeline.rs         # Foo<R> single-image pipeline
│   ├── pipeline_batch.rs   # FooBatch<R> if applicable
│   └── kernels/
│       ├── mod.rs
│       ├── reduction.rs    # always start here
│       ├── colors.rs       # sRGB / Lab / XYB conversions
│       ├── blur.rs         # separable Gaussian
│       └── … domain-specific
├── examples/
│   ├── reduction_parity.rs # validate kernel-by-kernel
│   ├── colors_parity.rs
│   ├── blur_parity.rs
│   ├── end_to_end.rs       # full pipeline smoke
│   └── parity_real_image.rs # vs CPU reference, bit-exact target
└── tests/
    └── reduction_parity.rs # CI-friendly parity gate
```

`Cargo.toml`:

```toml
[features]
default = ["cuda", "wgpu", "cpu"]
cuda = ["cubecl/cuda"]
wgpu = ["cubecl/wgpu"]
hip  = ["cubecl/hip"]
cpu  = ["cubecl/cpu"]

[dependencies]
cubecl = { version = "0.10.0-pre.4", default-features = false, features = ["std"] }

[dev-dependencies]
foo = "X.Y"          # the CPU reference for parity tests
bytemuck = "1"
imgref = "1"
rgb = "0.8"
image = { version = "0.25", default-features = false, features = ["png"] }
```

Public API for `pipeline.rs`:

```rust
pub struct Foo<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    n: usize,
    /* per-instance buffers, all created via client.create_from_slice(zeros) */
    /* cached_* fields for set_reference/compute_with_reference */
    has_cached_reference: bool,
    half_res: Option<Box<Self>>,
}

impl<R: Runtime> Foo<R> {
    pub fn new(client: ComputeClient<R>, w: u32, h: u32) -> Self;
    pub fn new_multires(client: ComputeClient<R>, w: u32, h: u32) -> Self;
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> GpuFooResult;
    pub fn set_reference(&mut self, ref_srgb: &[u8]);
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> GpuFooResult;
    pub fn dimensions(&self) -> (u32, u32);
    pub fn copy_diffmap(&self) -> Vec<f32>; // or whatever the per-pixel output is
    pub fn has_cached_reference(&self) -> bool;
    pub fn half_res(&self) -> Option<&Self>;
    /* cached_* accessors so a batched scorer can broadcast them */
}
```

## Kernel-by-kernel translation

### Pointwise kernels (sRGB→linear, opsin, l2_diff, etc.)

The straight one-to-one translation. Each thread handles one element.

```rust
#[cube(launch_unchecked)]
pub fn srgb_u8_to_linear_planar_kernel(
    src: &Array<u8>,
    dst_r: &mut Array<f32>,
    dst_g: &mut Array<f32>,
    dst_b: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst_r.len() { terminate!(); }
    let i3 = idx * 3;
    dst_r[idx] = srgb_byte_to_linear(src[i3]);
    dst_g[idx] = srgb_byte_to_linear(src[i3 + 1]);
    dst_b[idx] = srgb_byte_to_linear(src[i3 + 2]);
}
```

Launch with `cube_count = ceil(n / 256)`, `cube_dim = (256, 1, 1)`.

For pointwise kernels, batching is free: extend buffer sizes by `N`,
extend launch by `N`, no kernel changes.

### Separable blur (H + V)

Translate per-pixel. Each thread reads a row (H pass) or column (V
pass) within the image, multiplies by Gaussian weights, normalises.

```rust
#[cube(launch_unchecked)]
pub fn vertical_blur_kernel(
    src: &Array<f32>, dst: &mut Array<f32>,
    width: u32, height: u32, sigma: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (width * height) as usize { terminate!(); }
    let w = width as usize;
    let y = idx / w;
    let x = idx % w;
    let radius_us = u32::max(u32::cast_from(M * sigma), 1u32) as usize;
    let begin = usize::saturating_sub(y, radius_us);
    let end = u32::min((y + radius_us) as u32, (height - 1) as u32) as usize;

    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    let mut i = begin;
    while i <= end {
        let dist = (u32::saturating_sub(i as u32, y as u32)
                  + u32::saturating_sub(y as u32, i as u32)) as f32;
        let weight = gauss(dist, sigma);
        sum += src[i * w + x] * weight;
        wsum += weight;
        i += 1;
    }
    dst[y * w + x] = sum / wsum;
}
```

Use `f32::powf(2.0, x * LOG2_E)` for `exp(-0.5 * z * z)` because of
gotcha #1.

### Tile-based kernels (e.g. SSIM 11×11 window, Malta 24×24)

Use `SharedMemory<f32>` with cooperative load via thread serial-id
within the cube. Launch with cube_dim = tile-interior size, cube_count
= ceil(image / tile_interior).

See `crates/butteraugli-gpu/src/kernels/malta.rs` for a worked example
with a 16×16 work tile + 4-pixel halo (24×24 shared array).

Watch out:

- `sync_cube()` must come AFTER all cooperative loads and BEFORE any
  thread reads from shared memory.
- `terminate!()` for out-of-bounds threads must be AFTER `sync_cube()`,
  not before — otherwise the surviving threads deadlock waiting for the
  terminated ones.

### Reductions

The fused-reduction pattern from `butteraugli-gpu/src/kernels/reduction.rs`
generalises:

```rust
#[cube(launch_unchecked)]
fn fused_reduce_kernel(
    diffmap: &Array<f32>,
    output_max_bits: &mut Array<Atomic<u32>>,    // u32 because of gotcha #2
    output_sums: &mut Array<Atomic<f32>>,
) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let n = diffmap.len();
    let mut local_max = 0.0f32;
    let mut local_sums = 0.0f32;
    let mut i = tid;
    while i < n {
        let v = diffmap[i];
        if v > local_max { local_max = v; }
        local_sums += v * v;
        i += stride;
    }
    output_max_bits[0].fetch_max(u32::reinterpret(local_max));
    output_sums[0].fetch_add(local_sums);
}
```

Launch with a small fixed grid (16 cubes × 256 threads has worked well
across Malta-class workloads), grid-stride within. Caller zeroes
output buffers, then folds.

For batched reductions, make `cube_count = (CUBES_PER_IMAGE,
batch_size, 1)` and use `CUBE_POS_Y` as the batch index. See
`reduce_batched_with_pnorm` in butteraugli-gpu for the per-image
fused max + Σpⁿ pattern.

## Pipeline orchestration patterns

### Allocate buffers up front

`Foo::new(client, w, h)` allocates every per-pixel intermediate as a
`cubecl::server::Handle`:

```rust
fn alloc_plane<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}
```

These are zero-initialised — important because some kernels read
the previous-call value of an accumulator (e.g., `l2_diff` adds to
`dst[idx]`).

### Launch helpers

```rust
fn cube_count_1d(&self) -> CubeCount { /* ceil(n/256) */ }
fn cube_dim_1d(&self) -> CubeDim { CubeDim::new_1d(256) }
fn cube_count_2d(&self) -> CubeCount { /* ceil(w/16), ceil(h/16) */ }
fn cube_dim_2d(&self) -> CubeDim { CubeDim::new_2d(16, 16) }
```

Wrap each launch as a method on `Foo<R>` that names the inputs/outputs
clearly (e.g., `self.malta_hf(a, b, acc, ...)`). The launch site
becomes a one-liner.

### Buffer aliasing

`blur_plane_via(src, dst, scratch)` does
`H(src → scratch)` followed by `V(scratch → dst)`. Constraints:

- `scratch` must differ from BOTH `src` and `dst`
- `src == dst` is OK (V writes after reading scratch)

When you run out of scratch buffers in a long pipeline, reuse buffers
that are written-only later (e.g., `diffmap_buf` is fine as a scratch
for the mask blur because `compute_diffmap` overwrites it as the very
last step).

### Reference cache + multi-resolution

The two big API features. Layered correctly:

```rust
pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> GpuFooResult {
    self.populate_linear_from_srgb(true, ref_srgb);
    self.populate_linear_from_srgb(false, dist_srgb);
    self.run_pipeline_from_linear(true, true);
    reduction::reduce(...)
}

pub fn set_reference(&mut self, ref_srgb: &[u8]) {
    self.populate_linear_from_srgb(true, ref_srgb);
    if let Some(half) = self.half_res.as_deref() {
        populate_half_res_linear(self, half, true);
    }
    self.apply_opsin(true);
    self.separate_frequencies(true);
    self.compute_mask_pipeline_reference_only();
    self.has_cached_reference = true;
    if let Some(half) = self.half_res.as_mut() { /* recurse for half */ }
}

pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> GpuFooResult {
    assert!(self.has_cached_reference);
    self.populate_linear_from_srgb(false, dist_srgb);
    self.run_pipeline_from_linear(false, true);
    reduction::reduce(...)
}
```

Half-res sibling lives as `Option<Box<Foo<R>>>`. Recursion terminates
because the inner half has `half_res = None`.

The single critical rule: **populate the half-res linear BEFORE
opsin overwrites the full-res linear with opsin output**. Get this
backwards and the half-res sibling runs opsin on already-XYB inputs
(my batch implementation hit this exact bug). The fix is splitting
`compute` into "to linear" + "downsample" + "rest of pipeline" phases.

### Batching pattern

Stack `N` images contiguously per buffer:
`[image_0_pixel_0, image_0_pixel_1, …, image_0_pixel_{P-1}, image_1_pixel_0, …]`.

Most pointwise kernels work as-is — just launch with `N×P` threads. The
ones that need batched variants:

- **Vertical blur** — y is per-image-clamped, so the kernel needs
  `plane_stride` and `batch_size` parameters and computes
  `batch_idx = idx / plane_stride; local_y = …` internally.
- **Tile-based kernels (Malta, SSIM windows)** — launch with 3D cube
  grid `(bx, by, batch_idx)`, use `CUBE_POS_Z` for the batch index
  inside.
- **Per-image reduction** — `(CUBES_PER_IMAGE, batch_size, 1)` grid,
  one cube grouping per image, atomics into batch_idx-indexed slots.
- **Broadcast variants** — when one input is the cached reference and
  the other is batched, write a `*_broadcast_batched_kernel` variant
  that reads the reference at `idx % plane_stride` and the batched
  side at `idx`. See `l2_diff_broadcast_batched_kernel` in butteraugli-gpu.

The throughput win is biggest at small images:

| size | speedup vs unbatched (butteraugli measured) |
|---|---|
| 256² batch=8 | 6.6× |
| 512² batch=8 | 2.6× |
| 1024² batch=4 | 1.2× |

Below 512² launch overhead dominates total time; batching collapses it.

## Validation strategy

**Layer 1 — kernel-level parity.** For each kernel, write
`examples/<kernel>_parity.rs` that runs the GPU kernel and the CPU
reference on the same input and reports `max_abs_diff` and
`max_rel_diff`. Target sub-ulp precision (< 1e-6 abs for unit-range
data, < 1e-5 abs for opsin-scale data).

The CPU reference must be the **actual published crate's**
implementation, not a hand-written re-derivation. For
`butteraugli-gpu` we initially compared against a hand-coded clamp-to-
edge blur, which silently masked a real divergence vs the published
butteraugli's actual blur. Catch this by adding a path-dep diagnostic
example that uses `features = ["internals"]` to expose the CPU crate's
internals:

```toml
[dev-dependencies.foo]
version = "X.Y"
path = "/path/to/foo-source"
features = ["internals"]
```

Such examples don't ship in the committed code (the path-dep is
host-specific), but they're invaluable while debugging.

**Layer 2 — pipeline parity.** `parity_real_image.rs` runs the full
GPU pipeline against the CPU on a real PNG and reports score
deviation. Target `+0.0%` to `+0.1%` (the floor is f32 round-off).

**Layer 3 — bit-exact lock test.** Once parity is achieved, freeze it
with a regression test that hashes the output diffmap (or hashes the
sequence of scores across a small corpus). Adapt the 191-entry
`cross_arch_parity.rs` from butteraugli for the same shape.

Unit tests should claim small images (32×32) for speed; the parity
example uses a real PNG (≥ 512²) to exercise the actual interesting
range.

## Per-crate notes

### ssimulacra2-gpu (porting from `crates/ssimulacra2-cuda`)

ssimulacra2 is structurally similar to butteraugli — it has:

- sRGB → linear → XYB conversion (`xyb.rs`)
- Multi-resolution downscaling (6 octaves, `downscale.rs`)
- Per-octave per-pixel error maps (`error_maps.rs`)
- Per-octave reduction
- Final score = weighted sum of per-octave reductions

What ports cleanly from butteraugli's playbook:

- `kernels::blur` lifts almost verbatim — same separable Gaussian.
- `kernels::colors` (sRGB → linear) lifts verbatim.
- The XYB conversion is different from butteraugli's (no opsin
  sensitivity loop) but simpler: pure pointwise matrix multiply.

Things to add new for SSIM2:

- **6-octave pyramid orchestration** — N+1 instances of `Ssim2<R>`
  daisy-chained via downsample. Or: single instance with embedded
  octave loop, allocating per-octave buffers up front.
- **Per-pixel SSIM1 / SSIM3 error maps** — the 11×11 windowed mean +
  variance + covariance kernels. These are TILE-BASED with shared
  memory (see Malta gotchas) — 16×16 work tile + 5-pixel halo on each
  side gives a 26×26 shared array (676 f32s, fine for any GPU).
- **Per-octave reduction (mean of error map)** — fused with the SSIM
  computation if you want a single launch per octave.

`shared.cu` in the existing crate is hand-written CUDA C for some
performance-sensitive bits. Translate to `#[cube]` Rust and validate
parity with the existing PTX output.

Estimated effort: 2–3 days. Higher than butteraugli because of the
6-octave control flow.

### dssim-gpu (porting from `crates/dssim-cuda`)

DSSIM uses Lab colour space (not XYB) and a single 8×8 (or 11×11
depending on impl) windowed SSIM. Simpler than SSIMULACRA2.

Reusable from butteraugli playbook:

- `kernels::blur` and `kernels::colors::srgb_to_linear` lift directly.
- The Lab conversion (`lab.rs`) is pointwise and trivially translatable.

New work:

- **Tile-based SSIM kernel** — same shape as the SSIM2 one above
  (cooperative tile load, per-pixel mean/variance/covariance). DSSIM's
  window may be smaller than SSIM2's so the shared array is smaller.
- **Per-pixel scaling** — DSSIM does a `1 / (1 + SSIM)` style transform
  before reducing.

Estimated effort: 1–2 days. Smallest of the three.

### zensim-gpu (porting from `crates/zensim-cuda`)

zensim is the feature-extractor variant — produces a `Vec<f64>` of
features (per-octave pooled statistics) rather than a single score.

Notes:

- `set_reference` / `compute_with_reference` shape is already there
  in the CUDA crate; mirror the API.
- The output is a feature vector, not a scalar — adjust the reduction
  to write `Vec<f32>` (or `Vec<f64>` if precision matters) for all the
  per-octave statistics rather than one scalar.
- `padding.rs` handles non-multiple-of-N image dimensions by padding.
  In CubeCL you can mostly avoid explicit padding by making your
  kernels respect the actual `width × height` and clamping at the
  edges (same as the blur example above) — saves you a memory pass.
- The zensim `score_from_features` function stays on the host (a
  weighted dot-product); doesn't need to move.

Estimated effort: 2 days. Mostly because feature accounting is fiddly,
not because the kernels are hard.

### General order to port

1. Add `<crate>-gpu` skeleton with feature flags and an empty `lib.rs`.
2. Port `srgb_to_linear` first → `kernels::colors`. Add a
   `colors_parity.rs` example that compares to the CPU lookup table.
3. Port `blur` next → `kernels::blur`. Add `blur_parity.rs` against the
   actual published CPU `gaussian_blur` (not a hand-written reference,
   per the validation gotcha).
4. Port the reduction → `kernels::reduction`. Add
   `reduction_parity.rs`. This is the easiest correctness check and
   gives an early "I have something working" milestone.
5. Port the metric-specific kernels (XYB, frequency separation, SSIM
   window, …).
6. Wire `pipeline.rs::compute` end-to-end, validate against a real PNG.
7. Add `set_reference` / `compute_with_reference`. Run the cache-vs-
   full-compute drift test (target = 0.0).
8. Add `new_multires` if the metric has a multi-resolution component.
9. Add `<crate>Batch<R>` for per-image batched scoring. Watch the
   downsample-before-opsin ordering bug.
10. Lock parity with a small test corpus.

## Performance expectations

Based on the butteraugli numbers:

- **Steady-state** at 1 MP+, CubeCL hits ~150–250 MP/s on RTX 5070 +
  CUDA 13.2. Within 5–15 % of equivalent hand-written PTX.
- **Cached reference** halves per-call cost on a typical encoder loop.
- **Batched** at 256² gives 5–7× per-image speedup; at 1024² it's
  marginal (1.2×).
- **Cold compile** is the worst pain point — first build of a new
  CubeCL kernel module is 5–9 minutes on a Ryzen 9. Plan for that.

## Cross-references

- Worked example: `crates/butteraugli-gpu/src/{pipeline.rs, pipeline_batch.rs, kernels/}`
- Original CUDA implementations: `crates/{ssimulacra2,dssim,zensim}-cuda/`
- cubecl docs: https://github.com/tracel-ai/cubecl
- The `PORT_STATUS.md` and `HANDOFF.md` patterns from butteraugli-gpu
  are worth replicating in each new gpu crate.
