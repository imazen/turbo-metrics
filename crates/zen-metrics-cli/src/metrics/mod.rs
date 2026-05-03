#![forbid(unsafe_code)]

//! Metric backend dispatch.
//!
//! `MetricKind` is the user-facing enum (parsed from the CLI's `--metric`
//! flag). `run_metric` resolves it to the actual backend and runs the
//! comparison on the supplied RGB8 image pair.

use clap::ValueEnum;

use crate::decode::Rgb8Image;

#[cfg(feature = "cpu-metrics")]
mod butteraugli_cpu;
#[cfg(feature = "cpu-metrics")]
mod dssim_cpu;
#[cfg(feature = "cpu-metrics")]
mod ssim2_cpu;
#[cfg(feature = "cpu-metrics")]
mod zensim_cpu;

#[cfg(any(feature = "gpu-butteraugli", feature = "gpu-ssim2"))]
mod gpu_runtime_dispatch;

#[cfg(feature = "gpu-butteraugli")]
mod butteraugli_gpu;
#[cfg(feature = "gpu-ssim2")]
mod ssim2_gpu;

/// Metric identifier exposed on the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MetricKind {
    /// SSIMULACRA2 — CPU implementation via the `ssimulacra2` crate.
    #[value(name = "ssim2-cpu")]
    Ssim2Cpu,
    /// SSIMULACRA2 — GPU implementation via the `ssim2-gpu` crate.
    #[value(name = "ssim2-gpu")]
    Ssim2Gpu,
    /// Butteraugli — CPU implementation via the `butteraugli` crate.
    #[value(name = "butteraugli-cpu")]
    ButteraugliCpu,
    /// Butteraugli — GPU implementation via the `butteraugli-gpu` crate.
    #[value(name = "butteraugli-gpu")]
    ButteraugliGpu,
    /// DSSIM — CPU implementation via the `dssim-core` crate.
    #[value(name = "dssim-cpu")]
    DssimCpu,
    /// Zensim — CPU implementation via the `zensim` crate.
    #[value(name = "zensim")]
    Zensim,
}

impl MetricKind {
    pub fn all() -> &'static [MetricKind] {
        &[
            MetricKind::Ssim2Cpu,
            MetricKind::Ssim2Gpu,
            MetricKind::ButteraugliCpu,
            MetricKind::ButteraugliGpu,
            MetricKind::DssimCpu,
            MetricKind::Zensim,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            MetricKind::Ssim2Cpu => "ssim2-cpu",
            MetricKind::Ssim2Gpu => "ssim2-gpu",
            MetricKind::ButteraugliCpu => "butteraugli-cpu",
            MetricKind::ButteraugliGpu => "butteraugli-gpu",
            MetricKind::DssimCpu => "dssim-cpu",
            MetricKind::Zensim => "zensim",
        }
    }

    pub fn backend(self) -> &'static str {
        match self {
            MetricKind::Ssim2Gpu | MetricKind::ButteraugliGpu => "GPU",
            _ => "CPU",
        }
    }

    pub fn requires_gpu(self) -> bool {
        matches!(self, MetricKind::Ssim2Gpu | MetricKind::ButteraugliGpu)
    }

    pub fn column_name(self) -> &'static str {
        // Friendly column header used in `batch` output TSVs.
        match self {
            MetricKind::Ssim2Cpu => "ssim2_cpu",
            MetricKind::Ssim2Gpu => "ssim2_gpu",
            MetricKind::ButteraugliCpu => "butteraugli_cpu",
            MetricKind::ButteraugliGpu => "butteraugli_gpu",
            MetricKind::DssimCpu => "dssim_cpu",
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
        not(any(feature = "gpu-butteraugli", feature = "gpu-ssim2")),
        allow(unused_variables)
    )]
    gpu_runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ssim2Cpu => ssim2_cpu::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ssim2Cpu => Err(disabled_msg("ssim2-cpu", "cpu-metrics")),

        #[cfg(feature = "gpu-ssim2")]
        MetricKind::Ssim2Gpu => ssim2_gpu::score(reference, distorted, gpu_runtime),
        #[cfg(not(feature = "gpu-ssim2"))]
        MetricKind::Ssim2Gpu => Err(disabled_msg("ssim2-gpu", "gpu-ssim2")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::ButteraugliCpu => butteraugli_cpu::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::ButteraugliCpu => Err(disabled_msg("butteraugli-cpu", "cpu-metrics")),

        #[cfg(feature = "gpu-butteraugli")]
        MetricKind::ButteraugliGpu => butteraugli_gpu::score(reference, distorted, gpu_runtime),
        #[cfg(not(feature = "gpu-butteraugli"))]
        MetricKind::ButteraugliGpu => Err(disabled_msg("butteraugli-gpu", "gpu-butteraugli")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::DssimCpu => dssim_cpu::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::DssimCpu => Err(disabled_msg("dssim-cpu", "cpu-metrics")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Zensim => zensim_cpu::score(reference, distorted),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Zensim => Err(disabled_msg("zensim", "cpu-metrics")),
    }
}

#[allow(dead_code)]
fn disabled_msg(metric: &str, feature: &str) -> Box<dyn std::error::Error> {
    format!("metric '{metric}' is disabled (rebuild with `--features {feature}`)").into()
}
