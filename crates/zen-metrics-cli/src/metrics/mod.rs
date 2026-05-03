#![forbid(unsafe_code)]

//! Metric backend dispatch.
//!
//! `MetricKind` is the user-facing enum (parsed from the CLI's `--metric`
//! flag). `run_metric` resolves it to the actual backend and runs the
//! comparison on the supplied RGB8 image pair.

use clap::ValueEnum;

use crate::decode::Rgb8Image;

#[cfg(feature = "cpu-metrics")]
mod butteraugli;
#[cfg(feature = "cpu-metrics")]
mod dssim;
#[cfg(feature = "cpu-metrics")]
mod ssim2;
#[cfg(feature = "cpu-metrics")]
mod zensim;

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim"
))]
mod gpu_runtime_dispatch;

#[cfg(feature = "gpu-butteraugli")]
mod butteraugli_gpu;
#[cfg(feature = "gpu-dssim")]
mod dssim_gpu;
#[cfg(feature = "gpu-ssim2")]
mod ssim2_gpu;

/// Metric identifier exposed on the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MetricKind {
    /// SSIMULACRA2 — CPU implementation via the `ssimulacra2` crate.
    #[value(name = "ssim2")]
    Ssim2,
    /// SSIMULACRA2 — GPU implementation via the `ssim2-gpu` crate.
    #[value(name = "ssim2-gpu")]
    Ssim2Gpu,
    /// Butteraugli (3-norm) — CPU implementation via the `butteraugli` crate.
    #[value(name = "butteraugli")]
    Butteraugli,
    /// Butteraugli (3-norm) — GPU implementation via the `butteraugli-gpu` crate.
    #[value(name = "butteraugli-gpu")]
    ButteraugliGpu,
    /// DSSIM — CPU implementation via the `dssim-core` crate. Distance metric: 0 = identical.
    #[value(name = "dssim")]
    Dssim,
    /// DSSIM — GPU implementation via the `dssim-gpu` crate. Distance metric: 0 = identical.
    #[value(name = "dssim-gpu")]
    DssimGpu,
    /// Zensim — CPU implementation via the `zensim` crate.
    #[value(name = "zensim")]
    Zensim,
}

impl MetricKind {
    pub fn all() -> &'static [MetricKind] {
        &[
            MetricKind::Ssim2,
            MetricKind::Ssim2Gpu,
            MetricKind::Butteraugli,
            MetricKind::ButteraugliGpu,
            MetricKind::Dssim,
            MetricKind::DssimGpu,
            MetricKind::Zensim,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            MetricKind::Ssim2 => "ssim2",
            MetricKind::Ssim2Gpu => "ssim2-gpu",
            MetricKind::Butteraugli => "butteraugli",
            MetricKind::ButteraugliGpu => "butteraugli-gpu",
            MetricKind::Dssim => "dssim",
            MetricKind::DssimGpu => "dssim-gpu",
            MetricKind::Zensim => "zensim",
        }
    }

    pub fn backend(self) -> &'static str {
        match self {
            MetricKind::Ssim2Gpu | MetricKind::ButteraugliGpu | MetricKind::DssimGpu => "GPU",
            _ => "CPU",
        }
    }

    pub fn requires_gpu(self) -> bool {
        matches!(
            self,
            MetricKind::Ssim2Gpu | MetricKind::ButteraugliGpu | MetricKind::DssimGpu
        )
    }

    pub fn column_name(self) -> &'static str {
        // Friendly column header used in `batch` output TSVs.
        match self {
            MetricKind::Ssim2 => "ssim2",
            MetricKind::Ssim2Gpu => "ssim2_gpu",
            MetricKind::Butteraugli => "butteraugli",
            MetricKind::ButteraugliGpu => "butteraugli_gpu",
            MetricKind::Dssim => "dssim",
            MetricKind::DssimGpu => "dssim_gpu",
            MetricKind::Zensim => "zensim",
        }
    }
}

/// CubeCL runtime selector for GPU metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GpuRuntime {
    /// First runtime that initialises successfully.
    Auto,
    /// CUDA. Requires the `gpu-cuda` cargo feature.
    Cuda,
    /// wgpu (Vulkan / Metal / DX12 / WebGPU). Requires `gpu-wgpu`.
    Wgpu,
    /// AMD HIP / ROCm. Requires `gpu-hip`.
    Hip,
    /// CPU-fallback runtime in CubeCL. Requires `gpu-cpu`.
    Cpu,
}

/// Run `kind` on a `(reference, distorted)` RGB8 pair. GPU metrics route
/// `gpu_runtime` through the CubeCL backend; CPU metrics ignore it.
pub fn run_metric(
    kind: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    #[cfg_attr(
        not(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim"
        )),
        allow(unused_variables)
    )]
    gpu_runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ssim2 => ssim2::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ssim2 => Err(disabled_msg("ssim2", "cpu-metrics")),

        #[cfg(feature = "gpu-ssim2")]
        MetricKind::Ssim2Gpu => ssim2_gpu::score(reference, distorted, gpu_runtime),
        #[cfg(not(feature = "gpu-ssim2"))]
        MetricKind::Ssim2Gpu => Err(disabled_msg("ssim2-gpu", "gpu-ssim2")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Butteraugli => butteraugli::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Butteraugli => Err(disabled_msg("butteraugli", "cpu-metrics")),

        #[cfg(feature = "gpu-butteraugli")]
        MetricKind::ButteraugliGpu => butteraugli_gpu::score(reference, distorted, gpu_runtime),
        #[cfg(not(feature = "gpu-butteraugli"))]
        MetricKind::ButteraugliGpu => Err(disabled_msg("butteraugli-gpu", "gpu-butteraugli")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Dssim => dssim::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Dssim => Err(disabled_msg("dssim", "cpu-metrics")),

        #[cfg(feature = "gpu-dssim")]
        MetricKind::DssimGpu => dssim_gpu::score(reference, distorted, gpu_runtime),
        #[cfg(not(feature = "gpu-dssim"))]
        MetricKind::DssimGpu => Err(disabled_msg("dssim-gpu", "gpu-dssim")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Zensim => zensim::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Zensim => Err(disabled_msg("zensim", "cpu-metrics")),
    }
}

#[allow(dead_code)]
fn disabled_msg(metric: &str, feature: &str) -> Box<dyn std::error::Error> {
    format!("metric '{metric}' is disabled (rebuild with `--features {feature}`)").into()
}
