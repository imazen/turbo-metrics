//! Multi-vendor GPU implementation of the butteraugli perceptual image quality metric.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl). Single Rust kernel
//! source, dispatchable across:
//! - **CUDA** (NVIDIA) — native PTX/SASS via CubeCL CUDA runtime
//! - **WGPU** (cross-platform) — Vulkan/Metal/DX12/WebGPU via wgpu
//! - **HIP** (AMD ROCm) — when the `hip` feature is enabled
//! - **CPU** (SIMD) — when the `cpu` feature is enabled
//!
//! The CPU backend is intended only as a correctness reference; it's not
//! competitive with the dedicated [`butteraugli`](https://crates.io/crates/butteraugli)
//! crate's autoversioned SIMD path.
//!
//! ## Algorithmic parity with `butteraugli` v0.9.2
//!
//! Aggregations match the CPU crate exactly: `score` is the max-norm
//! distance, `pnorm_3` is the libjxl 3-norm aggregation
//! (`butteraugli_main --pnorm` default). Both are produced in a single
//! fused on-device reduction pass over the diffmap.
//!
//! ## Status
//!
//! Early port from `butteraugli-cuda`. The reduction is the first kernel
//! ported end-to-end; full pipeline (opsin / blur / Malta / masking / diffmap
//! combination) is in progress. See `PORT_STATUS.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod kernels;

use cubecl::prelude::*;

/// Result of a butteraugli comparison.
///
/// Mirrors `butteraugli::ButteraugliResult` from the CPU crate. `score` is
/// the max-norm; `pnorm_3` is the libjxl 3-norm aggregation, available
/// "for free" because the fused reduction kernel produces both in one pass.
#[derive(Debug, Clone, Copy)]
pub struct GpuButteraugliResult {
    /// Max-norm difference score. < 1.0 is "good", > 2.0 is "bad".
    pub score: f32,
    /// libjxl 3-norm aggregation — average of three p-norms at exponents
    /// 3, 6, 12. Matches `butteraugli_main --pnorm` and the CPU crate's
    /// `ButteraugliResult.pnorm_3`.
    pub pnorm_3: f32,
}

/// Aggregate a diffmap into (score, pnorm_3) on the GPU using a single
/// fused reduction pass — runs on whatever CubeCL runtime `R` you pick.
///
/// This is the smallest end-to-end CubeCL kernel in the crate; it serves
/// as both the score-extraction step of the full butteraugli pipeline
/// (when the rest is ported) and as a self-contained validation target.
///
/// Diffmap values must be non-negative finite f32 (the butteraugli pipeline
/// guarantees this — diffmap is `sqrt` of sums of squares).
pub fn reduce_diffmap_to_score<R: Runtime>(
    client: &ComputeClient<R>,
    diffmap_handle: cubecl::server::Handle,
    n_pixels: usize,
) -> GpuButteraugliResult {
    kernels::reduction::reduce(client, diffmap_handle, n_pixels)
}
