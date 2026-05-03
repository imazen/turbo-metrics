# dssim-gpu

Multi-vendor GPU implementation of the DSSIM (Structural Dissimilarity)
perceptual image quality metric, built on
[CubeCL](https://github.com/tracel-ai/cubecl). One Rust kernel source,
runs on:

- **CUDA** (NVIDIA) via `cubecl-cuda`
- **WGPU** (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- **HIP** (AMD ROCm) via `cubecl-hip` (`hip` feature)
- **CPU** fallback via `cubecl-cpu` for build-checking only — the
  reduction kernels need atomics + `CUBE_COUNT`, which `cubecl-cpu` 0.10
  doesn't yet support, so the CPU backend is **not** a runtime parity
  target. Use the published `dssim-core` crate as the CPU reference
  instead (that's what the integration tests do).

Algorithmic parity target is `dssim-core` v3.4 (the canonical Rust
DSSIM crate). At the pyramid level this also matches the existing
`dssim-cuda` crate — same five scales, same custom-Lab conversion,
same fixed 3×3 Gaussian, same per-scale MAD score. The CUDA-specific
crate stays in the workspace; this one extends reach to AMD / Intel /
Apple / WebGPU without giving anything up on NVIDIA.

## Single-image usage

```rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use dssim_gpu::Dssim;

let client = WgpuRuntime::client(&Default::default());
let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;

let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
let dist_srgb: Vec<u8> = vec![0; 256 * 256 * 3];

let result = d.compute(&ref_srgb, &dist_srgb)?;
println!("DSSIM = {:.6}", result.score);
# Ok::<(), dssim_gpu::Error>(())
```

## Cached-reference usage

When scoring many distorted candidates against one reference (encoder
rate-distortion search), call `set_reference` once and then
`compute_with_reference` per candidate. Skips the reference-side
pyramid + Lab + reference-blur work each call.

```rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use dssim_gpu::Dssim;

let client = WgpuRuntime::client(&Default::default());
let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;

let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
d.set_reference(&ref_srgb)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let r = d.compute_with_reference(&candidate)?;
    // ... use r.score ...
}
# Ok::<(), dssim_gpu::Error>(())
```

## Score interpretation

Output is the standard DSSIM scalar: 0 = identical, larger = more
distortion. Mirrors the `f64` returned by `dssim_core::Dssim::compare`.

## Features

| Feature | Default | Effect |
|---|---|---|
| `cuda` | yes | Compile the cubecl-cuda backend. |
| `wgpu` | yes | Compile the cubecl-wgpu backend. |
| `cpu`  | yes | Compile the cubecl-cpu backend (build-only — see above). |
| `hip`  | no  | Compile the cubecl-hip backend. |
| `fast-reduction` | yes | Use `Atomic<f32>::fetch_add` for the per-scale Σ reduction. Works on CUDA / DX12 / HIP; silently no-ops on Metal — disable for Metal builds. |

For Metal targets:

```bash
cargo build -p dssim-gpu --no-default-features --features wgpu
```

## Status

Initial port from `dssim-cuda`. See `PORT_STATUS.md` for the
per-kernel breakdown, runtime coverage, and validated parity numbers.
