# ssimulacra2-cuda → ssim2-gpu — concrete porting plan

Companion to [`CUBECL_PORTING_GUIDE.md`](CUBECL_PORTING_GUIDE.md). That
doc covers the general patterns; this one is the kernel-by-kernel
playbook for **ssimulacra2 specifically**.

Source: `crates/ssimulacra2-cuda{,-kernel}` — 2 178 LOC of Rust + 305
kernel launches per scored pair. Target: `crates/ssim2-gpu` — multi-vendor
via cubecl 0.10, same algorithm, parity within f32 precision.

Estimated effort: **2–3 days** end-to-end. Single biggest unknown is the
recursive Gaussian blur — see §2.

## At-a-glance sizing

| component | CUDA-PTX LOC | est. CubeCL LOC | notes |
|---|---|---|---|
| `srgb` kernel + LUT | 70 | 30 | LUT translates to a `const [f32; 256]` exactly; kernel is one pointwise read+lookup |
| `xyb` kernels (linear→XYB) | 129 | 80 | pure pointwise, no shared mem |
| `downscale` (2×2 average) | 67 | 50 | 1 kernel each for planar and packed |
| `blur` (recursive 5-pass) | 137 | 200 | stateful per-column IIR — the hardest port |
| `error_maps` (SSIM + artifact + detail_loss) | 60 | 80 | fused 3-output pointwise; needs precomputed mu/sigma |
| Pipeline orchestration (lib.rs) | 1 217 | ~700 | 6-octave pyramid + ref cache + score reduction |
| **Total** | **~1 680** | **~1 140** | smaller because no NPP wrapping, no CUDA graph capture, no `Image<f32>` plumbing |

## 1. Crate skeleton — same as butteraugli-gpu

```
crates/ssim2-gpu/
├── Cargo.toml          # cubecl 0.10.0-pre.4, default features cuda+wgpu+cpu
├── PORT_STATUS.md      # the 8 cubecl gotchas (copy from butteraugli-gpu)
├── HANDOFF.md
├── src/
│   ├── lib.rs          # GpuSsim2Result + re-exports
│   ├── pipeline.rs     # Ssim2<R> single-image pipeline
│   ├── pipeline_batch.rs  # Ssim2Batch<R> if needed (encoder use case)
│   └── kernels/
│       ├── mod.rs
│       ├── srgb.rs
│       ├── xyb.rs
│       ├── downscale.rs
│       ├── blur.rs        # recursive Charalampidis IIR — see §2
│       ├── error_maps.rs  # SSIM + artifact + detail_loss fused
│       └── reduction.rs   # per-octave + final weighted sum
└── examples/
    ├── srgb_parity.rs
    ├── xyb_parity.rs
    ├── blur_parity.rs    # ← parity vs the actual ssimulacra2 CPU crate, not a hand reference
    ├── error_maps_parity.rs
    ├── end_to_end.rs
    ├── parity_real_image.rs
    └── batch_parity.rs
```

## 2. The hard one: recursive Gaussian blur

ssimulacra2's blur is *not* a separable Gaussian convolution — it's
Charalampidis 2016 truncated-cosine **IIR**. The CUDA version
(`ssimulacra2-cuda-kernel/src/blur.rs`) processes each column top-to-
bottom maintaining 6 floats of state per thread, plus a shared-memory
ring buffer for the `y - N - 1` lookback term.

### What the CUDA kernel does

For each (column `x`, plane index `index ∈ 0..5`):

1. Initialise a 33-slot ring buffer in shared memory (per-thread).
2. Walk `y` from `-N + 1` to `height - 1`:
   - Load `right_val = src[y+N-1, x]` (zero past the edge).
   - Load `left_val = ring[(y - N - 1) % RING_SIZE]` (zero before y-N-1 ≥ 0).
   - Compute three IIR taps in parallel (`prev_1`, `prev_3`, `prev_5`)
     using six FMAs:
     ```
     out_k = sum * MUL_IN_k + prev2_k * MUL_PREV2_k + prev_k * MUL_PREV_k
     ```
   - Sum the three outputs and store to `dst[y, x]`.
   - Push `right_val` into the ring buffer.

Coefficients (`MUL_IN_*`, `MUL_PREV_*`, `RADIUS=N`) come from
`build.rs`: a small Rust program that solves Charalampidis's recurrence
once at build time and writes a generated `recursive_gaussian.rs` with
the constants.

The 5-plane fanout (`src0..src4` / `dst0..dst4`) lets one grid launch
blur all 5 same-shaped planes concurrently — that's a CUDA occupancy
trick, not algorithmic.

### CubeCL translation strategy

cubecl supports `SharedMemory<f32>::new(N)` and `sync_cube()` (gotchas
#5, #6 in the general guide). The IIR loop body is straight Rust math
— `f32::mul_add` should map to FMA. The state machine (six `prev*`
registers carried across iterations) is a `while` loop with locals.

What changes vs CUDA:

- **Drop the 5-plane fanout.** CubeCL doesn't have an obvious
  equivalent of "pick `(src, dst)` based on `block_idx_y`". Easiest
  port: launch the kernel five times, once per plane. You lose ~1.5×
  occupancy at small image sizes; gain it back at the pipeline level
  if you stream the launches concurrently (cubecl handles dispatch
  queueing).
- **Move the constants table into Rust.** Keep `build.rs` (it just
  solves and emits the constants); `include!` the generated file
  exactly the same way. Delete the `.cu` and `.ll` files entirely.
- **Ring buffer:** `SharedMemory::<f32>::new((BLOCK_WIDTH * RING_SIZE) as usize)`,
  index as `tx * RING_SIZE + (y_offset % RING_SIZE)`. cubecl
  `SharedMemory<T>` is flat 1D; you'd have done a 2D `[BLOCK_WIDTH][33]`
  in CUDA, but flatten in cubecl.
- **The `if y >= 0 { *dst[...] = ... }` guard** stays as-is; cubecl
  threads can write conditionally without re-launching.

Recommended kernel signature (single-plane variant):

```rust
#[cube(launch_unchecked)]
pub fn blur_plane_pass_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    src_pitch_f32: u32,   // already in f32 stride; CUDA used byte stride, cubecl is cleaner with element stride
    dst_pitch_f32: u32,
    /* coefficients packed as 9 scalars or read from a const */
) {
    let mut ring = SharedMemory::<f32>::new(BLOCK_WIDTH_USIZE * RING_SIZE_USIZE);
    let tx = UNIT_POS_X;
    let bx = CUBE_POS_X;
    let x = bx * (CUBE_DIM_X as u32) + tx;
    if x >= width { terminate!(); }
    // … initialise ring, then the y-loop matching the CUDA reference verbatim
}
```

Coefficients: pass as 9 scalar f32 args (cubecl can take many scalars).
Or store them in a single `Array<f32, 9>` constant; cubecl 0.10 has
`#[cube(launch)] const` for compile-time constants.

### Validation gate for the blur

This is the only kernel you should expect to spend a full day on.
Validation must compare against **the published `ssimulacra2`** CPU
crate's actual blur output, not against a hand-rolled separable
Gaussian. The IIR has phase lag and a different impulse response than
direct convolution — comparing to the wrong reference will mask real
bugs (this is the same lesson as butteraugli's σ=1.2 mirrored-blur
gotcha — see CUBECL_PORTING_GUIDE.md §"Validation strategy").

## 3. The easy ones

### `srgb` → `kernels::srgb.rs`

**Direct translation.** The 256-entry LUT is the same constant array.
Pointwise kernel; one ABSOLUTE_POS-indexed read+lookup per thread.

```rust
const SRGB8_TO_LINEARF32_LUT: [f32; 256] = [/* paste from CUDA kernel */];

#[cube(launch_unchecked)]
pub fn srgb_to_linear_kernel(
    src: &Array<u8>,
    dst: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() { terminate!(); }
    dst[idx] = SRGB8_TO_LINEARF32_LUT[src[idx] as usize];
}
```

The CUDA version uses byte-stride pitch. CubeCL Arrays are flat — for
non-padded planar buffers (the common case) you don't need stride;
for padded buffers, take a `pitch_f32` scalar param and compute
`dst[y * pitch_f32 + x]`. ssimulacra2's images aren't padded by
default, so the simple version is fine.

Validation: compare against `ssimulacra2::xyb::srgb_to_linear` on a
random u8 buffer — should be **bit-exact** (same LUT, same indexing).

### `xyb` → `kernels::xyb.rs`

**Direct translation.** `px_linear_rgb_to_positive_xyb` is pointwise,
two passes:

1. opsin_absorbance (3×3 matrix multiply + 3 biases) → 3 mul_add chains
2. cube root + per-channel bias subtract
3. final mix to (X, Y, B)

All pointwise. Translates to `#[cube]` Rust verbatim. The constants
(`K_M00..K_M22`, `K_B0`, `K_B0_ROOT`) are bit-identical scalar
constants.

The only CubeCL gotcha: `f32::cbrt` may not be a registered cube op.
Check `kernels::xyb::cbrt`-equivalent in cubecl prelude; if missing,
substitute `x.powf(1.0 / 3.0)` (gotcha #1's exp/powf trick).

Validation: bit-exact for sane inputs; up to ~3e-7 abs noise for `cbrt`
substitution if you go the powf route.

### `downscale` → `kernels::downscale.rs`

**Direct translation.** 2×2 average with edge clamping. The CUDA
version has both a packed 3-channel variant (`downscale_by_2`) and a
single-plane shuffle-warp variant (`downscale_plane_by_2`). Skip the
warp-shuffle one — it's a CUDA-specific micro-optimization that
cubecl can't express portably (no warp shuffle in WGPU). The plain
nested-loop version maps to `#[cube]` cleanly:

```rust
#[cube(launch_unchecked)]
pub fn downscale_2x_packed_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32, src_h: u32,
    dst_w: u32, dst_h: u32,
) {
    // (essentially the existing CUDA function body, indexed via ABSOLUTE_POS)
}
```

You already have a similar kernel in `butteraugli-gpu/src/kernels/downscale.rs`
— port it side-by-side with packed-3-channel handling.

## 4. `error_maps` → `kernels::error_maps.rs`

**Direct translation.** This is the SSIM-plus-artifact-plus-detail-loss
fused pointwise kernel. Inputs: source, distorted, mu1/2, sigma11/12/22
(all 7 same-size planes). Outputs: 3 planes (out, artifact, detail_loss).

The kernel logic is straight scalar math — translate the CUDA function
body verbatim into a `#[cube(launch_unchecked)]` with 10 `&Array<f32>`
parameters. No shared memory, no atomics, no warp tricks.

Where do mu1/sigma11/etc. come from? Each is a blurred image (mu = blur
of the plane, sigma11 = blur of plane², sigma12 = blur of plane1·plane2,
etc.). The pipeline computes them via the IIR blur kernel from §2 plus
some pointwise multiplies. Allocate them as additional buffers in the
`Ssim2<R>` struct.

## 5. Pipeline orchestration — the bulk of the work

`ssimulacra2-cuda/src/lib.rs` is 1 217 LOC, but most of it is buffer
allocation (NPP `Image<f32, C<3>>` with NPP pitch handling) and CUDA
graph capture. CubeCL drops both: allocations are flat handles via
`client.create_from_slice(zeros)`, and there's no graph capture (yet
— see [tracel-ai/cubecl#1319](https://github.com/tracel-ai/cubecl/issues/1319)).

Conceptual pipeline (matches the CUDA implementation, just trimmed):

1. **sRGB → linear** for both images (full-res planes).
2. **Build pyramid:** `linear[0]` = full-res; `linear[i+1]` = downscale_2x(`linear[i]`). 6 levels.
3. **Convert each linear-RGB level to XYB** in place.
4. **For each level** (6 octaves):
   a. Blur both XYB planes with the recursive Gaussian → mu1, mu2.
   b. Compute `xyb1²`, `xyb2²`, `xyb1·xyb2` pointwise → 3 plane triples.
   c. Blur each → sigma11, sigma22, sigma12.
   d. Run `error_maps` kernel → 3 planes (out/artifact/detail_loss) per channel.
   e. Reduce each → 3 scalars (sum / max-norm / pnorm) per channel.
5. **Final score:** weighted dot-product of the 6×3×3 = 54 reduction outputs
   with libssimulacra2's score weights. Host-side; trivial.

### Pyramid + per-octave buffer layout

```rust
const SCALES: usize = 6;

pub struct Ssim2<R: Runtime> {
    client: ComputeClient<R>,
    width: u32, height: u32,
    /// Per-octave dimensions. `dims[0] = (width, height)`, `dims[i+1] = (dims[i]/2)`.
    dims: [(u32, u32); SCALES],
    /// Reference linear RGB pyramid (6 levels, planar 3-channel each).
    ref_linear: [[Handle; 3]; SCALES],
    dis_linear: [[Handle; 3]; SCALES],
    /// XYB after linear→XYB conversion (overwrites linear in place — same size).
    /// (Actually allocate separately if you also want to cache linear for set_reference; trade-off.)
    /// Per-octave intermediates: mu, sigma11, sigma22, sigma12 (each a 3-channel triple).
    mu1: [[Handle; 3]; SCALES],
    mu2: [[Handle; 3]; SCALES],
    sigma11: [[Handle; 3]; SCALES],
    sigma22: [[Handle; 3]; SCALES],
    sigma12: [[Handle; 3]; SCALES],
    /// error_maps outputs.
    err_ssim: [[Handle; 3]; SCALES],
    err_artifact: [[Handle; 3]; SCALES],
    err_detail: [[Handle; 3]; SCALES],
    /// Per-octave per-channel reduced scalars (54 floats total).
    scores_dev: Handle,
    /// Cached reference state for set_reference / compute_with_reference.
    has_cached_reference: bool,
}
```

This is a lot of buffers (≈ 200 MB at 1 MP, scaling down each octave),
but matches the CUDA crate's footprint (which says "~800 MiB at 1440×1080").
You can reduce by reusing buffers across octaves — but the CUDA version
doesn't bother and ships fine, so don't pre-optimise.

### Reference cache

The CUDA crate already has `set_reference_linear` /
`compute_with_reference_linear` — mirror them. The cacheable state is:
- `ref_linear[*]` (the pyramid)
- `ref_xyb[*]` (post-XYB conversion)
- `mu1[*]`, `sigma11[*]` (the reference-only blur outputs)

Then `compute_with_reference_linear` only runs the distorted side and
the cross terms (`sigma12 = blur(xyb1·xyb2)`).

This pattern is identical to what's in `butteraugli-gpu/src/pipeline.rs`
(`compute_mask_pipeline_reference_only` etc.).

## 6. Reduction

For each octave, for each channel, you reduce three error maps with
three different aggregations (sum, max-norm, libssimulacra2-pnorm).
That's 3 × 3 × 6 = 54 scalars per call.

Two implementation choices:

a. **One fused per-octave reduction kernel** that computes all 9
   per-channel scalars in a single grid-strided pass. Mirror the
   `fused_max_pnorm_sums_kernel` pattern from `butteraugli-gpu`.

b. **Three reduction launches per octave**, each producing 3 scalars.
   Simpler to implement, ~2× slower than (a) at small octaves; not
   visible at full-res.

Option (b) for the first cut; switch to (a) if profiling shows reduction
dominating after pipeline parity is achieved.

## 7. Validation order

Same as the general guide:

1. **`srgb_parity.rs`** — bit-exact LUT match, day 1.
2. **`xyb_parity.rs`** — sub-ulp f32 vs CPU `ssimulacra2::xyb`, day 1.
3. **`blur_parity.rs`** — IIR vs published CPU `ssimulacra2::blur`. This
   is the gate; do **not** proceed past this until parity is < 1e-5
   abs on a real image.
4. **`error_maps_parity.rs`** — assemble synthetic mu/sigma inputs,
   compare scalar SSIM/artifact/detail outputs to the CPU formula.
5. **`end_to_end.rs`** — full pipeline, real PNG, target ≤ 0.1 % score
   deviation from CPU `ssimulacra2`.
6. **Cross-arch lock test** — port CPU's known-good score table.

## 8. Score interpretation

ssimulacra2 outputs a final scalar in roughly the 0–100 range (not
`butteraugli`'s 0–30 range). The score-from-features weights are
hard-coded constants from libssimulacra2; pull them from the CPU
crate's source as a `[f64; 54]` array, dot-product host-side.

The output struct should match shape with the CPU crate:

```rust
pub struct GpuSsim2Result {
    /// Ssimulacra2 score in the 0–100 range (higher = better quality;
    /// 100 = identical, 0 = visually broken).
    pub score: f64,
}
```

(Just one scalar — there's no `pnorm_3` analogue here.)

## 9. Per-day breakdown

**Day 1:** scaffold + srgb + xyb + downscale + reduction. End-of-day:
`end_to_end.rs` runs and produces *some* scalar (probably wildly wrong
without blur, but no panics).

**Day 2:** blur. The whole day on parity vs CPU `ssimulacra2::blur`,
because the IIR is the only nontrivial kernel.

**Day 3:** wire `error_maps`, full pipeline parity, ≤ 0.1 % vs CPU on
a real image. Add `set_reference_linear` / `compute_with_reference_linear`.
Optional: `Ssim2Batch<R>` for encoder use (same pattern as
butteraugli-gpu's `ButteraugliBatch`).

If day 2 takes longer than expected on the IIR (likely), slip day 3
out. Don't compromise on blur parity.

## 10. Cross-references

- **General patterns:** [`CUBECL_PORTING_GUIDE.md`](CUBECL_PORTING_GUIDE.md)
- **Worked example:** `crates/butteraugli-gpu/src/pipeline.rs` for the
  reference-cache + multi-resolution patterns
- **Original CUDA:** `crates/ssimulacra2-cuda{,-kernel}/src/`
- **CPU crate (parity reference):** `ssimulacra2` on crates.io (use
  `features = ["internals"]` if it has them, or check what it exposes
  publicly)
- **libssimulacra2 reference:** https://github.com/cloudinary/ssimulacra2
  for cross-checking the score weights
- **Open upstream blocker:** [tracel-ai/cubecl#1319](https://github.com/tracel-ai/cubecl/issues/1319)
  — once landed, add CUDA stream priority for shared-GPU workstation use
