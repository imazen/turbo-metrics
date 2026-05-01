# butteraugli-gpu port status

Multi-vendor GPU port of `butteraugli-cuda` using CubeCL (NVIDIA + AMD + Intel + Apple from one Rust source).

## Module status

| Module | LOC (cuda) | Status | Notes |
|---|---:|---|---|
| `reduction` (max + 3-norm sums) | 90 | ✅ ported + validated | Bit-exact match on max-norm, <4e-6 rel diff on 3-norm vs CPU reference. RTX 5070 + CUDA 13.2. |
| `colors` (sRGB / opsin / XYB / deinterleave) | ~250 | ✅ ported + validated (sRGB+opsin) | sRGB→linear within 3e-7, opsin within 8e-6 vs CPU. `linear_to_xyb` and `deinterleave_3ch` not yet needed. |
| `blur` (separable 1D + 5×5 mirrored) | ~420 | ✅ ported + validated (sep. 1D) | H + V Gaussian both within 5e-7 abs diff vs CPU on all 5 butteraugli sigmas (1.2, 1.56, 2.7, 3.22, 7.16). 5×5 mirrored variant not yet ported. |
| `frequency` (UHF/HF/MF/LF split) | ~320 | ✅ ported | 8 pointwise kernels: xyb_low_freq, subtract, sub_remove/amplify_range, suppress_x_by_y, separate_hf_uhf, remove/amplify_range. |
| `downscale` (2× subsample) | ~110 | ✅ ported | downsample_2x + add_upsample_2x with 0.3 heuristic mix. |
| `malta` (perceptual contrast) | ~700 | ✅ ported | HF (9-tap × 16) + LF (5-tap × 16) variants, 24×24 SharedMemory tile, sync_cube(). |
| `masking` (mask_to_error_mul, fuzzy_erosion) | ~280 | ✅ ported | combine_channels_for_masking, mask_to_error_mul, diff_precompute, fuzzy_erosion. |
| `diffmap` (combine_channels_to_diffmap_fused) | ~150 | ✅ ported | compute_diffmap (sqrt(maskY·ΣAC + maskDcY·ΣDC)), l2_diff, l2_asym_diff. |
| Pipeline orchestration (`Butteraugli` struct, multi-res, ref cache) | ~2400 (lib.rs) | 🟡 partial | Single-res scaffold built (sRGB→linear, opsin, blur). Frequency separation + malta + masking + compute_diffmap wiring is the remaining 30% TODO. Multi-res supersample-add and reference cache pending. |

## Translation patterns (reference for porting)

### PTX-Rust → CubeCL `#[cube]`

| PTX-Rust (`butteraugli-cuda-kernel`) | CubeCL (`butteraugli-gpu`) |
|---|---|
| `pub unsafe extern "ptx-kernel" fn foo(...)` | `#[cube(launch_unchecked)] fn foo(...)` |
| `core::arch::nvptx::_thread_idx_x()` | `UNIT_POS_X` (or `UNIT_POS` for absolute-in-cube) |
| `core::arch::nvptx::_block_idx_x()` | `CUBE_POS_X` |
| `core::arch::nvptx::_block_dim_x()` | `CUBE_DIM_X` |
| `core::arch::nvptx::_grid_dim_x()` | `CUBE_COUNT_X` |
| `tid + bid * bdim` (linearized) | `ABSOLUTE_POS` |
| `*src.add(i)` | `array[i]` |
| `core::arch::asm!("atom.global.max.u32 ...")` | `atomic.fetch_max(value)` |
| `core::arch::asm!("atom.global.add.f64 ...")` | `atomic.fetch_add(value)` (use `Atomic<f32>` cross-platform; `Atomic<f64>` CUDA-only via runtime check) |
| `__shared__ float buf[N]` | `let mut buf = SharedMemory::<f32>::new(N);` |
| `__syncthreads()` | `sync_units()` |
| `f.powf(q)` | same — `cubecl_core::frontend::Float::powf` is in scope |

### Host-side: `cudarse-driver` → CubeCL

| `cudarse-driver` / `cudarse-npp` | CubeCL |
|---|---|
| `CuStream` | `ComputeClient<R>` (handles streaming internally) |
| `CuBox::<[T]>::new_zeroed(n, &stream)` | `client.empty(n * size_of::<T>())` then a zero-fill kernel, or `client.create_from_slice(&[T::default(); N])` |
| `cuMemcpyHtoD_v2` | `client.create(bytes)` |
| `cuMemcpyDtoH_v2` | `client.read_one(handle)` |
| `cuStreamSynchronize` | implicit on `read_one`; explicit `client.sync()` |
| Kernel launch `kernel.launch(...)` | `kernel_name::launch_unchecked::<R>(client, count, dim, args)` |
| `LaunchConfig { grid_dim, block_dim, ... }` | `CubeCount::Static(x, y, z)` + `CubeDim::new_3d(x, y, z)` |

### Atomics — backend-specific feature gating

CubeCL exposes feature checks via `client.properties().type_usage(...)`. For
diffmap reduction, the relevant probe is:

```rust
use cubecl::ir::{StorageType, ElemType, FloatKind};
use cubecl::ir::features::TypeUsage;
let f64_atomic_ok = client
    .properties()
    .type_usage(StorageType::Atomic(ElemType::Float(FloatKind::F64)))
    .contains(TypeUsage::AtomicAdd);
```

Cuda backend supports `Atomic<f64>` AtomicAdd since SM 6.0 (always true on
Volta+). WGPU/Metal don't. Strategy: use `Atomic<f32>` unconditionally for
the cross-platform path. Add a CUDA-only specialization later if precision
parity with `butteraugli-cuda` becomes important.

### Buffer aliasing / scratch reuse

`butteraugli-cuda` aggressively reuses temp buffers (e.g. `temp1`, `temp2`,
`mask_temp`) across stages. CubeCL handles this via memory pools — calling
`client.empty(size)` repeatedly returns recycled handles after the previous
operation drains. So the manual buffer-pool dance can disappear.

### Multi-resolution + reference cache

Once all kernels are ported, the orchestration in `butteraugli-cuda/src/lib.rs`
maps to:

- `set_reference` → run all reference-side kernels into cached `Tensor` handles
- `compute_with_reference` → run distorted-side kernels using cached refs
- Graph capture (CUDA only) → CubeCL's `client.command_buffer()` API (if
  needed for perf — typical CubeCL workloads don't need it)

## Validation gates

1. **Reduction-only**: `cargo run --example reduction_parity` — confirms
   the toolchain compiles and the smallest kernel matches CPU within fp
   tolerance.
   - Kernel logic verified correct (CUDA C++ codegen inspected).
   - End-to-end run **blocked on this dev box** (see "Toolchain reality").
2. **End-to-end**: once full pipeline ports, compare against
   `butteraugli-cuda` on the same image pair. Target rel diff < 1% on
   max-norm and 3-norm.
3. **Cross-backend**: same test on CUDA + WGPU + (where available) HIP.
   Target rel diff < 1% across backends.
4. **Cross-arch lock-file**: 191-entry locked-bits regression test
   modelled on `butteraugli`'s `cross_arch_parity.rs`. Adapt to GPU by
   relaxing the `eq` check to a `< 1e-3` tolerance (GPU SIMD width drift
   is real).

## Toolchain reality

CubeCL 0.10 requires a CUDA 13 toolkit (the `nvrtcGetTileIR` symbol).
On WSL2 Ubuntu this is `cuda-toolkit-13-2` from the standard NVIDIA repo;
RTX 50-series (Blackwell, sm_120) needs CUDA 13 anyway since CUDA 12.x
nvrtc only knows up to sm_90.

WSL2 doesn't expose a Vulkan ICD to Linux processes by default; the
wgpu/Vulkan path is unreachable here. Use CUDA backend on WSL2; reserve
WGPU testing for native Linux/Mac/Windows.

`cubecl-cpu` 0.10.0-pre.4 doesn't implement `atomic<u32>` yet — not
useful as a CI validator until that lands upstream.

### Current state on this dev box (validated 2026-05-01)

CUDA 13.2 + RTX 5070 + cubecl 0.10.0-pre.4 — reduction kernel matches
CPU reference bit-exactly on max-norm and within 4e-6 relative error on
3-norm (f32 accumulator round-off only). End-to-end CubeCL pipeline is
verified working.

## CubeCL gotchas discovered during port

Items worth knowing if you continue this work or port other kernels:

0. **`f32::log` is base-2 log; natural log is `f32::ln`.** `f32::exp`
   is *not registered* in cubecl 0.10 (only available for f16/bf16/
   flex32/tf32/f64). Substitute `exp(x) ≡ 2^(x · log₂(e))` using
   `f32::powf(2.0, …)` which IS registered. The CUDA codegen lowers
   `powf` to its native `powf` intrinsic, same hardware path.



1. **`Atomic<f32>::fetch_max` codegens as `atomicMax(float*, float)` on
   the CUDA backend, which is invalid C++** (CUDA only has `atomicMax`
   for integer types). cubecl-cpp 0.9.0 registers f32 atomic-max as
   supported but the codegen is broken. Workaround: cast f32 → u32 bits
   via `u32::reinterpret(value)` and use `Atomic<u32>::fetch_max`. Safe
   for non-negative f32 values (their bit-pattern ordering matches value
   ordering). This is what the reduction kernel does.

2. **`CUBE_DIM_X` is `u32` but `CUBE_COUNT` is `usize`.** Compose them
   with `CUBE_COUNT * (CUBE_DIM_X as usize)` for grid stride.

3. **`Array::len()` returns `usize`** in cube context — match the type
   when comparing.

4. **`launch_unchecked` returns a `Result`** that's easy to silently
   ignore (no `#[must_use]` warning at the proc-macro layer in some
   shapes). Wrap with `if let Err(e) = ... { eprintln!(...) }` during
   bring-up; debugging "kernel did nothing" is much easier than guessing.

5. **`ArrayArg::from_raw_parts::<T>(handle, len, vec)`** in 0.9.0 takes
   3 args; 0.10.0-pre.4 dropped to 2 args. Pin the version explicitly.

6. **Build times are real:** ~5–9 min for an incremental example rebuild
   when cubecl is already cached. Plan iteration around this.
