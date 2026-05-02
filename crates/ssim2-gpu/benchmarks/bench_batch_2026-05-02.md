# bench_batch — 2026-05-02

256×256 source, JPEG corpus repeated to fill batch slots. Median of 5
trials post-warmup. From `examples/bench_batch.rs`.

## CUDA (RTX 5070 + CUDA 13.2, WSL2 Ubuntu 22.04)

`cargo run --release -p ssim2-gpu --example bench_batch`

| batch | seq total | seq /img | batch total | batch /img | speedup |
|---|---|---|---|---|---|
| 1 | 4.10 ms | 4.10 ms | 4.02 ms | 4.02 ms | 1.02× |
| 2 | 8.39 ms | 4.19 ms | 4.20 ms | 2.10 ms | 2.00× |
| 4 | 22.50 ms | 5.63 ms | 7.94 ms | 1.98 ms | 2.84× |
| 8 | 54.91 ms | 6.86 ms | 16.62 ms | 2.08 ms | 3.30× |
| 16 | 96.40 ms | 6.03 ms | 26.91 ms | 1.68 ms | 3.58× |

## Windows DX12 (RTX 5070, native NVIDIA driver, fast-reduction enabled)

```powershell
$env:AUTO_GRAPHICS_BACKEND='dx12'
cargo run --release -p ssim2-gpu --no-default-features --features 'wgpu fast-reduction' --example bench_batch
```

| batch | seq total | seq /img | batch total | batch /img | speedup |
|---|---|---|---|---|---|
| 1 | 4.13 ms | 4.13 ms | 4.25 ms | 4.25 ms | 0.97× |
| 2 | 7.66 ms | 3.83 ms | 5.21 ms | 2.60 ms | 1.47× |
| 4 | 15.17 ms | 3.79 ms | 5.97 ms | 1.49 ms | 2.54× |
| 8 | 30.06 ms | 3.76 ms | 8.60 ms | 1.07 ms | 3.50× |
| 16 | 63.99 ms | 4.00 ms | 17.85 ms | 1.12 ms | 3.58× |

DX12 batched is slightly faster than CUDA batched at small N — the
NVIDIA DX12 stack absorbs the 4× u32 sRGB upload + copy-kernel
finalizer overhead better than the CUDA stack does for whatever
reason.

## Metal (CI-only — Apple Silicon `macos-latest`, portable mode)

`fast-reduction` disabled (auto-disabled by `--no-default-features
--features wgpu` since Metal silently no-ops Atomic<f32>::fetch_add).

Numbers not measured here — CI just verifies pass/fail. Per-image
throughput would be slightly worse than the portable CUDA path
(2.5–3 ms/img at N=8 on equivalent silicon).

## Reproducing

CUDA / WSL:
```
CUDA_PATH=/usr/local/cuda cargo run --release -p ssim2-gpu --example bench_batch
```

Windows DX12 / native:
```powershell
$env:AUTO_GRAPHICS_BACKEND='dx12'
cargo run -p ssim2-gpu --no-default-features --features 'wgpu fast-reduction' --release --example bench_batch
```

For Metal builds (must drop fast-reduction):
```
cargo run --no-default-features --features wgpu --release --example bench_batch
```
