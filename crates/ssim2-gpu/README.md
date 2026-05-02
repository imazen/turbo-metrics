# ssim2-gpu

Multi-vendor GPU implementation of the SSIMULACRA2 perceptual image
quality metric, built on [CubeCL](https://github.com/tracel-ai/cubecl).

One Rust kernel source, dispatchable across:
- **CUDA** (NVIDIA) via the cubecl CUDA runtime
- **WGPU** (cross-platform) — Vulkan / Metal / DX12 / WebGPU
- **HIP** (AMD ROCm) when the `hip` feature is enabled
- **CPU** (SIMD) reference path when `cpu` is enabled

Algorithmic parity target is the published [`ssimulacra2`](https://crates.io/crates/ssimulacra2)
v0.5.1 CPU crate (the canonical Rust port of [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2)).

## Usage

```rust
use cubecl::cuda::CudaRuntime;
use cubecl::Runtime;
use ssim2_gpu::Ssim2;

let client = CudaRuntime::client(&Default::default());
let mut s = Ssim2::<CudaRuntime>::new(client, width, height)?;

// One-off comparison.
let result = s.compute(&ref_srgb, &dis_srgb)?;
println!("score = {:.3}", result.score);

// Encoder rate-distortion search: cache the reference once.
s.set_reference(&ref_srgb)?;
for dis in distorted_candidates {
    let r = s.compute_with_reference(&dis)?;
    // ...
}
```

The `Ssim2Batch<R>` wrapper exposes the same API for batched scoring.

## Score interpretation

Output is in roughly the 0–100 range:
- **100** = identical (or near-identical)
- **90+** = visually indistinguishable for most observers
- **70+** = high quality
- **30–60** = noticeable distortion
- **<0** = the SSIMULACRA2 polynomial overshoot region for severely
  distorted images; not a bug, just how the curve behaves.

## Status

See [`PORT_STATUS.md`](PORT_STATUS.md) and [`HANDOFF.md`](HANDOFF.md).
Validated against CPU `ssimulacra2` to ≤ 0.06 % relative error on the
JPEG q5..q90 corpus; cached vs direct path drift ≤ 8e-6.

## Build

CUDA 13.2 required for cubecl 0.10's CUDA backend. RTX 50-series
(Blackwell, sm_120) needs CUDA 13 anyway.

```bash
CUDA_PATH=/usr/local/cuda cargo build -p ssim2-gpu
CUDA_PATH=/usr/local/cuda cargo test --release -p ssim2-gpu
```

## License

MIT — same terms as the rest of `turbo-metrics`.
