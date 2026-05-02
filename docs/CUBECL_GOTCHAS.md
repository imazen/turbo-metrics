# CubeCL gotchas — long-form reference

Everything that bit me during the butteraugli-cuda → butteraugli-gpu
port. cubecl 0.10.0-pre.4 era; some entries will dissolve in later
versions.

Each entry has the same shape:
- **Symptom** — what you'll see (compile error, silent wrong output, runtime panic, perf cliff)
- **Cause** — why it happens
- **Detect** — fastest way to confirm it's this gotcha and not something else
- **Fix** — the workaround / proper fix
- **Example** — broken vs working code

Categories:
1. Codegen / `#[cube]` body
2. Launch API
3. Backend-specific
4. Pipeline orchestration
5. Validation traps
6. Performance surprises
7. Toolchain & build

---

## 1. Codegen & `#[cube]` body

### G1.1 — `f32::exp` is not a registered cube op

**Symptom.** The `#[cube]` macro accepts the code, but launch fails with
"unknown intrinsic" or the codegen emits a non-existent symbol.
Sometimes errors show only at runtime when the runtime tries to JIT.

**Cause.** cubecl 0.10's op table for `f32` doesn't include `exp`. It
includes `ln`, `log` (base-2!), `powf`, `sqrt`, `sin`, `cos`, etc. but
not `exp`.

**Detect.** Search your `#[cube]` functions for `f32::exp(`. If any
match, you have the bug.

**Fix.** Use the identity `exp(x) = 2^(x · log₂(e))` and call
`f32::powf(2.0, x * LOG2_E)`. CUDA lowers `powf` to its native `__powf`
intrinsic — same hardware path as a direct `exp` call would have been.

```rust
// ❌ BROKEN
#[cube]
fn gauss(d: f32, s: f32) -> f32 {
    let z = d / s;
    f32::exp(-0.5 * z * z)
}

// ✅ FIXED
const LOG2_E: f32 = 1.442_695_040_888_963_4;

#[cube]
fn exp_f32(x: f32) -> f32 {
    f32::powf(2.0, x * LOG2_E)
}

#[cube]
fn gauss(d: f32, s: f32) -> f32 {
    let z = d / s;
    exp_f32(-0.5 * z * z)
}
```

---

### G1.2 — `Atomic<f32>::fetch_max` codegens to a non-existent CUDA function

**Symptom.** Compile-time "atomicMax(float\*, float) is undefined" or
runtime kernel-load failure. Sometimes silent wrong output if the
backend tolerates it.

**Cause.** CUDA's hardware atomicMax is integer-only (signed and
unsigned 32/64). There's no `atomicMax(float*, float)`. cubecl's IR
allows you to write `Atomic<f32>::fetch_max` but the CUDA backend has
no path to lower it.

**Detect.** Grep your kernels for `Atomic<f32>` paired with
`fetch_max`. Either pattern alone is fine; the combination is the bug.

**Fix.** For non-negative f32 (which butteraugli diffmaps always are —
they're `sqrt` of sums of squares), bit-cast to u32 and atomic-max on
the bit pattern. f32 IEEE-754 ordering matches u32 ordering for
non-negative values.

```rust
// ❌ BROKEN
output_max[0].fetch_max(local_max);  // compiles, then fails on launch

// ✅ FIXED
let max_bits = u32::reinterpret(local_max);
output_max_bits[0].fetch_max(max_bits);  // Atomic<u32> works fine

// Host-side bit-cast back:
let max = f32::from_bits(max_bits_readback);
```

---

### G1.3 — `0.0` literal in if/else arms with cube-wrapped values doesn't auto-promote

**Symptom.** Cryptic compile error like "type mismatch: expected
`Expand<f32>`, found `f64`".

**Cause.** When a value flowing into a `#[cube]` if/else is a
cube-wrapped `f32` (which the macro represents as `Expand<f32>`),
literal `0.0` (which Rust types as `f64` or `{f64,f32}` ambiguously)
doesn't get coerced.

**Detect.** Compile error mentions `Expand<f32>` and a literal float in
an if/else.

**Fix.** Use `f32::new(0.0)` explicitly.

```rust
// ❌ BROKEN
let v = if x > 1.0 { x } else { 0.0 };

// ✅ FIXED
let v = if x > 1.0 { x } else { f32::new(0.0) };
```

This affects any literal in an if/else arm where the other arm is a
runtime f32. Default to `f32::new(...)` for any constant in conditional
arms.

---

### G1.4 — `u32::abs_diff` not registered

**Symptom.** Compile error "no method named `abs_diff` for type `u32`"
inside a `#[cube]` body, even though stable Rust has it.

**Cause.** cubecl's u32 op table doesn't include `abs_diff`.

**Detect.** Look for `u32::abs_diff` or `.abs_diff(` on a u32 inside a
cube body.

**Fix.** Two saturating subs and a sum.

```rust
// ❌ BROKEN
let dist = u32::abs_diff(i, j);

// ✅ FIXED
let dist = u32::saturating_sub(i, j) + u32::saturating_sub(j, i);
```

---

### G1.5 — `SharedMemory::<T>::new(N)` takes `usize`, indexes by `usize`

**Symptom.** `expected usize, got u32` errors when the same constant is
used both for shared-memory sizing and for u32 launch shapes.

**Cause.** cubecl's `SharedMemory<T>` is sized at `usize` and indexed
at `usize`. Many launch-shape values (CubeDim, image dimensions) want
`u32`. Mixing produces type mismatches.

**Detect.** Compile errors at SharedMemory allocation or indexing
involving the same constant used elsewhere as a u32.

**Fix.** Define both forms of the constant.

```rust
// ✅ pattern
const SHARED_SIZE: u32 = 24;
const SHARED_TOTAL_USIZE: usize = (SHARED_SIZE * SHARED_SIZE) as usize;

#[cube]
fn load_tile(tile: &mut SharedMemory<f32>) {
    // alloc:
    let mut tile = SharedMemory::<f32>::new(SHARED_TOTAL_USIZE);
    // index:
    let i: u32 = ...;
    let v = tile[i as usize];
}
```

Same for any shared array dimensioned by a `u32` you also need at
`usize`.

---

### G1.6 — `f32::log` is base-2, not natural

**Symptom.** Numerically wrong output — values systematically off by a
factor of `ln(2) ≈ 0.6931`.

**Cause.** cubecl's `f32::log(x)` is `log₂(x)`, not `ln(x)`. The Rust
standard library has `f32::log(self, base)` (with explicit base) and
`f32::log2`/`f32::log10`/`f32::ln` — cubecl picked log2 as the
unspecified `log`, which is the opposite of std.

**Detect.** Compare a kernel's output for a known input against the
CPU; if it's off by exactly `ln(2)` or `log₂(e)`, it's this.

**Fix.** Use `f32::ln` for natural log explicitly.

```rust
// ❌ BROKEN — silently gives log₂, not ln
#[cube]
fn gamma(v: f32) -> f32 {
    GAMMA_MUL * f32::log(v + GAMMA_ADD) - GAMMA_SUB
}

// ✅ FIXED
#[cube]
fn gamma(v: f32) -> f32 {
    GAMMA_MUL * f32::ln(v + GAMMA_ADD) - GAMMA_SUB
}
```

This bit me on butteraugli's gamma function. The diff was small enough
that early tests "passed" but downstream propagation produced 12 %
score errors.

---

### G1.7 — Comptime generics on `bool` aren't supported by `#[cube]`

**Symptom.** Compile error "cannot find type parameter ... in this
scope" when a `#[cube(launch_unchecked)] fn foo<USE_LF: bool>(...)`
tries to switch behaviour on the comptime bool.

**Cause.** cubecl 0.10's `#[cube]` macro can monomorphize over numeric
generics but bool comptime generics aren't fully wired.

**Detect.** Compile fails on a `<X: bool>` parameter inside `#[cube]`.

**Fix.** Split into two kernel entry points sharing a non-launch-able
`#[cube]` helper.

```rust
// ❌ BROKEN
#[cube(launch_unchecked)]
fn malta_diff<USE_LF: bool>(...) { ... }

// ✅ FIXED
#[cube]
fn malta_diff_inner(use_lf_branch: u32, ...) {
    // body, branch on use_lf_branch
}

#[cube(launch_unchecked)]
fn malta_diff_hf(...) { malta_diff_inner(0, ...) }

#[cube(launch_unchecked)]
fn malta_diff_lf(...) { malta_diff_inner(1, ...) }
```

Or, if the two variants share little code, just write two complete
kernels — that's what butteraugli-gpu's `malta_diff_map_hf_kernel` /
`malta_diff_map_lf_kernel` do.

---

## 2. Launch API

### G2.1 — `CubeCount` and `CubeDim` are not `Copy`

**Symptom.** "use of moved value" errors when launching multiple
kernels in a row using the same precomputed shapes.

**Cause.** `CubeCount` and `CubeDim` in cubecl 0.10 are not `Copy`.
Each `launch_unchecked` call moves them.

**Detect.** Move-error on the second launch in a sequence.

**Fix.** `.clone()` per launch, or recompute per launch.

```rust
// ❌ BROKEN
let dim = self.cube_count_1d();
let block = self.cube_dim_1d();
unsafe {
    kernel_a::launch_unchecked::<R>(client, dim, block, ...);
    kernel_b::launch_unchecked::<R>(client, dim, block, ...);  // moved!
}

// ✅ FIXED
let dim = self.cube_count_1d();
let block = self.cube_dim_1d();
unsafe {
    kernel_a::launch_unchecked::<R>(client, dim.clone(), block.clone(), ...);
    kernel_b::launch_unchecked::<R>(client, dim, block, ...);
}
```

`.clone()` on these is cheap (small structs) — don't worry about the
overhead.

---

### G2.2 — `ArrayArg::from_raw_parts` API change between 0.9 and 0.10

**Symptom.** Compile error after a cubecl version bump: wrong number
of arguments, wrong type.

**Cause.** The signature changed:
- **0.9**: `from_raw_parts<T>(handle: &Handle, len: usize, vec_factor: u8)` — handle by ref, type generic, vec_factor.
- **0.10**: `from_raw_parts(handle: Handle, len: usize)` — handle by value, no type generic, no vec_factor.

**Detect.** Compile fails on `ArrayArg::from_raw_parts` after a cubecl
upgrade.

**Fix.** Drop the type generic and the vec_factor; pass the handle by
value (clone if you need to use it again later).

```rust
// 0.9 (broken in 0.10)
ArrayArg::from_raw_parts::<f32>(&handle, n, 1)

// 0.10
ArrayArg::from_raw_parts(handle.clone(), n)
```

---

### G2.3 — `launch_unchecked` returns `()` in 0.10, not `Result`

**Symptom.** Compile error "expected `()`, found `Result`" when
`?`-propagating a launch.

**Cause.** Same version-bump hazard. cubecl 0.9's `launch_unchecked`
returned `Result<(), ServerError>`; 0.10 returns `()`.

**Detect.** Compile fails on `?` after a launch call.

**Fix.** Drop the `?`. Errors are now logged via the runtime's logger;
panics still surface for catastrophic failures.

```rust
// 0.9 (broken in 0.10)
unsafe {
    kernel::launch_unchecked::<R>(client, dim, block, ...)?;
}

// 0.10
unsafe {
    kernel::launch_unchecked::<R>(client, dim, block, ...);
}
```

---

### G2.4 — Aliasing `&mut Array<f32>` arguments may panic on some backends

**Symptom.** Backend-specific panic (CUDA usually fine; WGPU may
reject) when the same handle is passed as two different `ArrayArg`
parameters where one is `&mut`.

**Cause.** WGPU's binding model forbids aliasing read+write on the
same buffer; CUDA tolerates it.

**Detect.** Test on WGPU specifically. If it works on CUDA but panics
on WGPU, this is it.

**Fix.** Use a scratch buffer to break the aliasing. The pattern shows
up most often in in-place transforms.

```rust
// ❌ FRAGILE (panics on WGPU)
unsafe {
    in_place_kernel::launch_unchecked::<R>(
        client, dim, block,
        ArrayArg::from_raw_parts(buf.clone(), n),  // read
        ArrayArg::from_raw_parts(buf.clone(), n),  // write — alias!
    );
}

// ✅ ROBUST
unsafe {
    out_of_place_kernel::launch_unchecked::<R>(
        client, dim, block,
        ArrayArg::from_raw_parts(buf.clone(), n),
        ArrayArg::from_raw_parts(scratch.clone(), n),  // distinct
    );
}
copy_plane(scratch, buf);
```

butteraugli-gpu's `split_uhf_hf_x_kernel` does this on purpose: writes
to a temp buffer, then copies back via `copy_plane`. Same pattern
works for any in-place transform.

---

## 3. Backend-specific

### G3.1 — cubecl 0.10 needs CUDA 13's nvrtc

**Symptom.** Link error mentioning `nvrtcGetTileIR` or build script
failing on `nvrtc` headers.

**Cause.** cubecl 0.10 calls `nvrtcGetTileIR` which was added in CUDA
13.x. CUDA 12's `nvrtc.h` doesn't expose it.

**Detect.** Build script error for `cubecl-cuda` or its sys deps
mentioning `nvrtc`. `nvcc --version` reports 12.x.

**Fix.** Install CUDA 13. On Ubuntu/WSL:
```bash
sudo apt install -y cuda-toolkit-13-2
# update-alternatives auto-symlinks /usr/local/cuda
```

For Blackwell (RTX 5070+, sm_120) you need CUDA 13 anyway because
nvrtc 12.x can't target sm_120.

---

### G3.2 — WSL2 has no Vulkan ICD for NVIDIA GPUs by default

**Symptom.** Running a `--no-default-features --features wgpu` example
panics with "no adapter available" or similar.

**Cause.** WSL2's default NVIDIA driver exposes only CUDA, not Vulkan.
The wgpu backend can't enumerate any adapter.

**Detect.** `vulkaninfo` shows no devices. `nvidia-smi` shows the GPU.

**Fix.** Install the WSL2 Vulkan ICD package (varies by distro), or
test the wgpu path on a native Linux/Mac/Windows host. For routine
development, just stick with the CUDA backend on WSL2 and validate
WGPU on a separate machine before shipping.

---

### G3.3 — `cubecl-cpu` doesn't implement `atomic<u32>`

**Symptom.** Panic "not yet implemented: This type is not implemented
yet. atomic<u32>" when running with the CPU backend.

**Cause.** cubecl-cpu's IR lowering for atomics is incomplete in 0.10.

**Detect.** Panic only on CPU backend; CUDA/WGPU work.

**Fix.** Don't use cubecl-cpu as a parity validator for any kernel
that uses atomics. Validate on CUDA against a CPU reference instead.
The CPU backend is useful for non-atomic kernels (sub-ulp parity is a
nice property to assert) but not for fused reductions.

---

### G3.4 — CUDA stream priority is not exposed

**Symptom.** No knob to make cubecl-cuda use a low-priority stream
(equivalent to `cuStreamCreateWithPriority(LEAST_PRIORITY)`). A
long-running batch starves the desktop compositor.

**Cause.** `cubecl-cuda::compute::stream::create_stream` hardcodes
`StreamKind::NonBlocking` with no priority arg. cudarc has the
`cuStreamCreateWithPriority` FFI binding but cubecl doesn't plumb it
through.

**Detect.** Display compositor stutters during long batches on a
shared-GPU dev machine.

**Fix.** Tracked upstream as
[tracel-ai/cubecl#1319](https://github.com/tracel-ai/cubecl/issues/1319).
Workarounds: vendor cubecl-cuda with a 5-line patch (Cargo
`[patch.crates-io]`), or wait for the upstream fix.

---

### G3.5 — CUDA graph capture is not exposed

**Symptom.** Per-call kernel-launch overhead can't be amortised by
capturing the post-upload pipeline once and replaying it.
butteraugli-cuda does this for batch dispatches; cubecl-cuda can't.

**Cause.** cubecl-runtime has no public API for graph
begin_capture/end_capture/replay.

**Detect.** Profile shows ~1 ms of fixed launch overhead per
compute() at small image sizes. Equivalent CUDA-graph-using code
(e.g., butteraugli-cuda's `compute_batch_with_reference`) shows
~0.1 ms.

**Fix.** Upstream-blocked. The kernel-batching workaround
(per-image-stack buffers + 3D launch grids) gets most of the same
speedup without graph capture — see butteraugli-gpu's
`pipeline_batch.rs`.

---

## 4. Pipeline orchestration

### G4.1 — `blur_plane_via(src, dst, scratch)` aliasing constraints

**Symptom.** Silent wrong output (often non-deterministic — depends on
warp scheduling). Hard to reproduce in unit tests; usually shows up
only in pipeline parity vs CPU.

**Cause.** A separable blur does `H(src → scratch)` followed by
`V(scratch → dst)`. If `scratch == src`, the H pass overwrites pixels
that other threads in the same wavefront still need to read across the
row. If `scratch == dst`, the V pass reads from `scratch` after `dst`
has been partially written.

**Detect.** Pipeline parity vs CPU is wrong by some structured pattern
(stripes, row-pairs, etc.). Single-pixel sanity tests pass.

**Fix.**
- `scratch` must differ from BOTH `src` and `dst`.
- `src == dst` is OK (V writes to dst after reading scratch — no
  in-flight aliasing).

When you run out of scratch buffers, reuse a buffer that's
written-only later (e.g., `diffmap_buf` is fine as scratch for the
mask blur because `compute_diffmap` overwrites it as the very last
step).

---

### G4.2 — Downsample-before-opsin ordering on multi-resolution

**Symptom.** Multi-resolution scores are wildly wrong (e.g., 100× too
large) but single-resolution scores are correct.

**Cause.** The half-resolution pipeline needs *linear-RGB* input
(it'll run its own opsin), but if you downsample after opsin has
overwritten the full-res linear-RGB buffer with XYB, the half-res
pipeline ends up running opsin on already-XYB data.

**Detect.** Single-res mode passes parity; multi-res mode is broken.
Half-res sibling's intermediate buffers contain values 10× too large.

**Fix.** Split the pipeline phases:
1. sRGB → linear (full-res only).
2. Downsample full-res linear → half-res linear.
3. Opsin on full-res (in place: linear → XYB).
4. Run rest of full-res pipeline.
5. Opsin on half-res (in place).
6. Run rest of half-res pipeline.
7. Supersample-add half-res diffmap into full-res.
8. Reduce.

The wrong order: `(1) → (3) → (2)`. Opsin at step 3 destroys the
linear values step 2 needs.

```rust
// ❌ BROKEN
self.populate_linear_from_srgb(true, ref_srgb);
self.apply_opsin(true);                          // overwrites lin_a with XYB
populate_half_res_linear(self, half, true);      // downsamples XYB, wrong!

// ✅ FIXED
self.populate_linear_from_srgb(true, ref_srgb);
populate_half_res_linear(self, half, true);      // downsamples linear ✓
self.apply_opsin(true);                          // OK: lin_a → XYB
half.apply_opsin(true);                          // OK: half.lin_a → XYB
```

---

### G4.3 — Borrow split when calling helpers across struct fields

**Symptom.** `cannot borrow *self as immutable because it is also
borrowed as mutable` when iterating a recursive sibling struct
(typical for half_res support).

**Cause.** `if let Some(half) = self.half_res.as_mut() { call(self,
half) }` simultaneously borrows `self.half_res` mutably and `self`
immutably (because `call`'s signature is `(&Self, &mut Other)` or
similar).

**Detect.** Borrow-checker error specifically on a sibling-recursion
helper.

**Fix.** Two patterns:

a) **Use `as_deref()` for read-only sibling access** when the helper
   only needs `&` to both:

```rust
// ❌ BROKEN
if let Some(half) = self.half_res.as_mut() {
    populate_half_res_linear(self, half, true);
}

// ✅ FIXED — populate_half_res_linear takes (&Foo, &Foo)
if let Some(half) = self.half_res.as_deref() {
    populate_half_res_linear(self, half, true);
}
```

b) **Take the box out, mutate, put back** when both `self` and `half`
   need `&mut`:

```rust
if let Some(mut half) = self.half_res.take() {
    half.run_pipeline_from_linear(do_a, do_b);
    let src = half.diffmap_buf.clone();
    let (sw, sh) = (half.width, half.height);
    self.launch_add_supersampled_2x_from(&src, sw, sh);
    self.half_res = Some(half);
}
```

Pattern (b) is what butteraugli-gpu's `run_pipeline_from_linear`
uses.

---

### G4.4 — `intensity_multiplier` confusion (off-by-255 trap)

**Symptom.** Diffmap scores are 1/255 of the expected magnitude. Or
2-3× too large, depending on direction.

**Cause.** Some CPU implementations (e.g., butteraugli) take an
`intensity_target` value (e.g., 80.0 nits) and multiply it into the
*linear* RGB inside opsin — where the linear inputs are already on
[0, 1] after the sRGB transfer. The mistake is to additionally
divide by 255, treating the multiplier as if it were applied to the u8
side.

**Detect.** Score differs from CPU by a factor of 80, 80/255, or
255/80 across all images.

**Fix.** Pass the raw intensity target (e.g., 80.0). Never combine
with a `/255` factor that's already absorbed into the sRGB→linear
transfer.

```rust
// ❌ BROKEN — double-divide by 255
const DEFAULT_INTENSITY_MULTIPLIER: f32 = 80.0 / 255.0;

// ✅ FIXED
const DEFAULT_INTENSITY_MULTIPLIER: f32 = 80.0;
```

---

### G4.5 — Sigma-of-the-wrong-blur (algorithmic invariant inside a kernel that takes σ as a param)

**Symptom.** Single-pixel-perturbation test gives a diffmap with
wrong-sign wings far from the perturbation. Score is ~2× expected.

**Cause.** Some metrics (butteraugli, dssim) have multiple internal
blurs at different sigmas: a small one for opsin sensitivity (e.g.,
σ=1.2) and a large one for low-frequency separation (e.g., σ=7.16).
If you reuse the LF blur's sigma where the opsin blur is needed, the
sensitivity gets smoothed over a 16-pixel radius instead of a 2-pixel
radius — destroys the per-pixel contrast information.

**Detect.** `diffmap_inspect`-style example shows non-zero wings far
from a single perturbation, with sign that doesn't make physical
sense (perturbation should brighten neighbors; you see darkening).

**Fix.** Read the CPU implementation carefully — every internal blur
has a specific σ. Don't share constants across stages.

```rust
// ❌ BROKEN — reusing SIGMA_LF for opsin
self.blur_plane(&lin[ch], &blur_a[ch], SIGMA_LF);   // wrong σ
self.launch_opsin(true);

// ✅ FIXED — separate constant
const SIGMA_OPSIN: f32 = 1.2;
self.blur_plane(&lin[ch], &blur_a[ch], SIGMA_OPSIN);
self.launch_opsin(true);
```

---

### G4.6 — Forgetting to zero accumulators between calls

**Symptom.** First call gives correct result; second call gives
2× / 3× / N× the value. Identical-image test gives non-zero.

**Cause.** Buffers like `block_diff_ac[ch]` are accumulators (kernels
do `dst[idx] += weight * (a-b)²`). They need to be zero before the
first contribution per call. Allocation initialises to zero (good for
the first call), but subsequent calls inherit the previous call's sum.

**Detect.** Identical-image score is non-zero on the second call but
zero on the first.

**Fix.** Either:
- Explicit `zero_plane(buf)` launch before the first accumulating
  kernel.
- Use a *write-only* variant for the first contribution (e.g.,
  `l2_diff_write_kernel` instead of `l2_diff_kernel`) — overwrites
  rather than accumulates, no zero needed.

butteraugli-gpu uses both: `zero_plane` for `block_diff_ac[0..2]`
because they receive multiple contributions; `l2_diff_write` for
`block_diff_ac[2]` because the only contribution there is the MF B
L2 diff.

---

### G4.7 — Slot-aware kernels for batched per-pair operations

**Symptom.** Batched scores all collapse to the same value (slot 0's
result), or are wrong by `2×`-`Nx` based on broadcast pattern.

**Cause.** Kernels that read two inputs of *different* shapes
(broadcast reference vs per-slot distorted) need slot-aware indexing.
Plain pointwise kernels work for two same-shape batched inputs but
not for broadcast.

**Detect.** Batched output equals 2× single output across all slots.
Single-image output is correct.

**Fix.** Add `*_broadcast_batched_kernel` variants that take
`plane_stride` and use `idx % plane_stride` for the broadcast input,
`idx` for the batched input.

```rust
// ✅ Pattern
#[cube(launch_unchecked)]
pub fn l2_diff_broadcast_batched_kernel(
    src_ref: &Array<f32>,    // single plane, broadcast
    src_dis: &Array<f32>,    // batched, N planes packed
    dst: &mut Array<f32>,    // batched accumulator
    plane_stride: u32,
    weight: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() { terminate!(); }
    let local = idx - (idx / (plane_stride as usize)) * (plane_stride as usize);
    let diff = src_ref[local] - src_dis[idx];
    dst[idx] = dst[idx] + weight * diff * diff;
}
```

butteraugli-gpu has the full set:
`l2_diff_broadcast_batched_kernel`,
`l2_asym_diff_broadcast_batched_kernel`,
`l2_diff_write_broadcast_batched_kernel`,
`mask_to_error_mul_batched_kernel`.

---

### G4.8 — Vertical blur in batched layout crosses image boundaries

**Symptom.** Batched output is correct in the centre of each image
but wrong near the y boundaries. Horizontal blur is fine.

**Cause.** Stacking batched images vertically as one tall buffer
(`height = h * batch_size`) makes horizontal blur work as-is (rows
are independent). But vertical blur reads `src[i*w + x]` for `i` in
`[y - radius, y + radius]` clamped to `[0, height - 1]`. Near image-N
boundaries, the kernel reads pixels from image-(N+1).

**Detect.** Per-image error map has a band of corruption in the bottom
~radius rows of every image except the last.

**Fix.** Write a batch-aware vertical-blur kernel that takes
`plane_stride` and clamps within each image's local height.

```rust
#[cube(launch_unchecked)]
pub fn vertical_blur_batched_kernel(
    src: &Array<f32>, dst: &mut Array<f32>,
    width: u32, height: u32, sigma: f32,
    plane_stride: u32, batch_size: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (plane_stride * batch_size) as usize;
    if idx >= total { terminate!(); }
    let plane_us = plane_stride as usize;
    let batch_idx = idx / plane_us;
    let local_idx = idx - batch_idx * plane_us;
    if local_idx >= (width * height) as usize { terminate!(); }
    let y = local_idx / (width as usize);
    let x = local_idx - y * (width as usize);
    let plane_off = batch_idx * plane_us;
    // …blur, clamping y to local image's [0, height - 1]
}
```

---

## 5. Validation traps

### G5.1 — Validate against the published CPU crate, not a hand-rolled re-derivation

**Symptom.** Pipeline parity test passes, but real-world scores
diverge from CPU butteraugli/ssimulacra2 by 5–100 %.

**Cause.** When writing a kernel-level parity test (e.g., for blur),
it's tempting to write the CPU reference yourself in the test file
("a Gaussian blur is just a separable convolution, easy"). But the
*actual* CPU crate may use a non-standard variant — different
boundary handling, recursive IIR vs FIR, mirrored vs clamped, custom
sigma derivation, etc. Your hand reference matches your GPU code, so
the test passes, but neither matches the truth.

**Detect.** Kernel parity tests pass at sub-ulp, but real-image
pipeline parity is off by structured amounts.

**Fix.** Always compare against the actual CPU crate's exported
functions. If the crate doesn't expose internals publicly, add a
dev-dependency path-dep with `features = ["internals"]`:

```toml
[dev-dependencies]
butteraugli = { version = "0.9.2", path = "/path/to/source", features = ["internals"] }
```

The path-dep won't ship in committed code (it's host-specific) but
it's invaluable while debugging. Once parity is established, the lock
test can hash the expected-good outputs and run without the path-dep.

This is exactly how the σ=1.2 vs σ=7.156 opsin-blur bug (G4.5) was
found — the hand-written reference matched the wrong σ.

---

### G5.2 — Three-layer validation strategy

**Symptom.** "I have a parity gap, where do I start looking?"

**Cause.** Without layered checks, a divergence at any of 30+ kernels
shows up only at the final score, with no localisation.

**Detect.** When you can't tell whether the bug is in stage 3 or
stage 13 of the pipeline.

**Fix.** Build the validation in layers, top-to-bottom:

1. **Per-kernel parity** — `examples/<kernel>_parity.rs` compares
   one kernel against the CPU's actual implementation, sub-ulp target.
2. **Pipeline parity** — `examples/parity_real_image.rs` runs the
   full GPU pipeline against the CPU on a real PNG, target Δ ≤ 0.1 %.
3. **Bit-exact lock test** — once parity holds, hash the diffmap (or
   a sequence of scores across a small corpus) and assert the hash in
   a unit test. Catches regressions quickly.

When the pipeline test starts failing but the kernel tests still pass,
you know to look at the orchestration — buffer aliasing, ordering,
caching state.

---

### G5.3 — Identical-image and flat-input sanity tests

**Symptom.** Hard-to-localise divergence — symptoms are quantitative
(score off by some amount) rather than structural.

**Cause.** The hardest bugs are the ones that *almost* work. Without
trivial-input baselines, you're trying to debug "score is 2.04× too
large" rather than "score is non-zero when it should be zero".

**Detect.** N/A — preventive.

**Fix.** Add two zeroth checks to every pipeline:

a) **Identical-image:** ref == dist → score must be exactly 0.
   Catches uninitialized accumulators (G4.6), wrong cached state, etc.

b) **Flat-128 input:** uniform sRGB-128 image (constant after
   linearisation) → score must be exactly 0 (all frequency bands
   should produce 0 differences).

Both ran first at every refactor in butteraugli-gpu. They caught the
stale-cached-state bug in `compute_with_reference` and the wrong-
direction-of-flags-on-recursion bug in `set_reference` immediately.

---

### G5.4 — Diff-dumping intermediate buffers when stage-level localisation matters

**Symptom.** Pipeline output is wrong, no idea where divergence
starts.

**Cause.** With 30+ stages, narrowing down by re-running the test
with prints in every kernel takes too long.

**Detect.** When per-kernel tests pass but full-pipeline doesn't.

**Fix.** Add `debug_*` accessors that read individual GPU buffers
back to host. Then write a one-off example that:

1. Runs both the GPU pipeline and a CPU reference (path-dep
   `internals` feature) on the same input.
2. Dumps each intermediate buffer side-by-side at a few probe pixels.
3. Reports the first stage where they diverge significantly.

butteraugli-gpu's `diffmap_inspect.rs` is the worked example. The
throwaway `cpu_vs_gpu_intermediates.rs` (deleted from the committed
code; it needed the path-dep) was what found the σ=1.2 opsin-blur
bug in 30 minutes — without it, blind grepping would have taken hours.

---

## 6. Performance surprises

### G6.1 — Cold-build compile time is 5–9 minutes

**Symptom.** First `cargo build` of a cubecl-cuda crate takes
forever; you wonder if it's hung.

**Cause.** cubecl-cuda + the cubecl macro infrastructure pulls in a
lot of dependencies that take a while to build the first time:
tracel-mlir-rs, cubecl-cpp, cubecl-runtime, etc. ~5 minutes on a
Ryzen 9 with cargo's default codegen settings.

**Detect.** Cold `cargo build -p your-gpu-crate` runs for 5+ minutes
with the cubecl crates appearing in the build output.

**Fix.** Nothing to fix — just plan around it. After the first
build, incremental rebuilds are ~2 minutes. Plan kernel work in
1–2 hour batches and iterate within the cached build.

---

### G6.2 — Per-instance allocation is expensive at large dimensions

**Symptom.** `Foo::new(client, w, h)` takes hundreds of milliseconds
at 1 MP+. Surprising for "just allocating buffers".

**Cause.** Each per-instance buffer is allocated AND zero-filled by
uploading zeros from host over PCIe. With ~30 buffers per instance,
4 MB each at 1 MP, that's ~120 MB of zero-fill upload.

Measured (RTX 5070, CUDA 13.2, butteraugli-gpu):

| size | `new()` | first `compute()` | steady |
|---|---|---|---|
| 128² | 2.3 ms | 1.14 ms | 1.05 ms |
| 256² | 9.7 ms | 2.00 ms | 1.13 ms |
| 512² | 53 ms | 5.02 ms | 1.82 ms |
| 1024² | 231 ms | 15.6 ms | 6.02 ms |
| 2048² | 625 ms | 59.9 ms | 23.7 ms |

**Detect.** Profile shows construction dominating call time on small
images. Or just run a `bench_dim_switch`-style test.

**Fix.** Pre-allocate one instance per distinct dimension you'll
encounter. Don't construct per-call. Round-robin across pre-allocated
instances has near-zero switching overhead beyond the per-size
baseline.

---

### G6.3 — Kernel launch overhead dominates at small sizes

**Symptom.** ~1 ms floor on per-call latency regardless of image
size. At 128² the wall-clock is 1 ms but the actual GPU work is
~0.02 ms.

**Cause.** Each kernel launch has a fixed CUDA driver overhead
(~30–40 µs). A typical butteraugli compute() does 30+ launches, so
~1 ms floor.

**Detect.** Steady-state ms doesn't drop linearly as image size shrinks
below 256².

**Fix.** Two paths:

a) **Batch multiple images per kernel launch** —
   `*Batch<R>::compute_batch_with_reference` packs N images per
   buffer and launches each kernel once for the whole batch. At
   256², batching N=8 gives 6.6× per-image speedup.

b) **CUDA graph capture** — capture the post-upload pipeline once,
   replay as a single graph launch. Currently blocked on cubecl
   exposing graph capture (G3.5).

For sizes ≥ 1 MP, kernel launch overhead is < 20 % and these
optimisations don't move the needle.

---

## 7. Toolchain & build

### G7.1 — Cargo feature flags must be forwarded explicitly to cubecl

**Symptom.** Your crate has `default = ["cuda", "wgpu", "cpu"]` but
running with `--no-default-features --features cuda` panics with
"no available adapter".

**Cause.** Your features need to forward to cubecl's:
```toml
default = ["cuda", "wgpu", "cpu"]
cuda = ["cubecl/cuda"]
wgpu = ["cubecl/wgpu"]
cpu  = ["cubecl/cpu"]
```

If you forget the forwarding line (e.g., `cuda = []`), cubecl-cuda
isn't actually pulled in and there's no CUDA runtime to construct.

**Detect.** Build succeeds but runtime adapter selection fails.

**Fix.** Always forward features explicitly to cubecl. This burned
butteraugli-gpu once with avx512: the `Cargo.toml` had
`avx512 = ["archmage/avx512"]` but missed `magetypes/avx512`, and the
v4 kernels never got compiled in.

---

### G7.2 — `cudarse-driver` and `cudarse-npp` need CUDA-13 patches

**Symptom.** Building turbo-metrics' butteraugli-cuda or any crate
depending on cudarse against CUDA 13 fails with:
- `cuEventElapsedTime` not found in scope (renamed `_v2`)
- `nppGetGpuName`, `nppSetStream`, etc. not found (removed in NPP 13)

**Cause.** CUDA 13 dropped the legacy non-`_v2` event API and
removed NPP's old global-state helpers. cudarse-driver's
`event.rs` and cudarse-npp's `lib.rs` were written against CUDA 12.

**Detect.** Compile failure on either of those crates after a CUDA
13 install.

**Fix.** Already merged in turbo-metrics commit `f745da9`:
- `cudarse-driver`: `cuEventElapsedTime` → `cuEventElapsedTime_v2`
- `cudarse-npp-sys/build.rs`: allowlist `cudaGetDevice`,
  `cudaDeviceGetAttribute`, `cudaGetDeviceProperties*`
- `cudarse-npp`: emulate the removed NPP globals via a thread-local
  `NppStreamContext` populated from `cudaDeviceGetAttribute`
- `cudarse-npp/src/image/ist.rs`: switch the two
  `nppiSumGetBufferHostSize_32f_C{1,3}R` calls to the `_Ctx` variants

---

### G7.3 — Don't hold onto stale bindings.rs files

**Symptom.** After `cargo clean -p` of a sys crate that uses
bindgen, the next build still uses the old bindings (sometimes).

**Cause.** `cargo clean -p` doesn't always remove the bindgen output
in `target/release/build/<hash>/out/bindings.rs`. Bindgen runs again
but the consuming crate may still link against an older copy from a
different feature-hash directory.

**Detect.** Build error references symbols that should/shouldn't
exist depending on which bindings file is used.

**Fix.** Remove the build directory directly:
```bash
rm -rf target/release/build/<sys-crate-name>-* target/debug/build/<sys-crate-name>-*
cargo build -p <consumer>
```

Sledgehammer alternative: `cargo clean` (deletes everything, costs a
full rebuild — see G6.1).

---

## Quick-reference checklist before opening a bug report

When something breaks in your cubecl port, walk through this:

1. **Identical-image test passes?** (G5.3)
   - If no: probably stale buffer or wrong cached state.
2. **Flat-128 test passes?** (G5.3)
   - If no: probably wrong σ, wrong constant, or freq-separation bug.
3. **Per-kernel parity tests pass against the *actual* CPU crate?** (G5.1)
   - If no: kernel-level bug (G1.x).
   - If yes: orchestration bug (G4.x).
4. **WGPU and CUDA agree?** (G3.x)
   - If only one fails: backend-specific (G2.4 or G3.x).
5. **Score off by `255×`, `80×`, or some clean integer multiple?** (G4.4)
   - That's an intensity_target / intensity_multiplier confusion.
6. **Score off by `2×`?** (G4.5, G4.6, G4.7)
   - σ-mismatch, double-accumulation, or batch-broadcast bug.

---

## See also

- **General porting guide:** [`CUBECL_PORTING_GUIDE.md`](CUBECL_PORTING_GUIDE.md)
- **Worked example:** [`crates/butteraugli-gpu/`](../crates/butteraugli-gpu/)
- **Per-crate ssim2 plan:** [`SSIMULACRA2_PORTING_PLAN.md`](SSIMULACRA2_PORTING_PLAN.md)
- **Open upstream issues:**
  - [tracel-ai/cubecl#1319](https://github.com/tracel-ai/cubecl/issues/1319) — CUDA stream priority
