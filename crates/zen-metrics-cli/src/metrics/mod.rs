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
    /// Butteraugli — CPU implementation via the `butteraugli` crate.
    /// Emits **two** columns per cell: `butteraugli_max`
    /// (`ButteraugliResult::score`, the per-block maximum) and
    /// `butteraugli_pnorm3` (`ButteraugliResult::pnorm_3`, the libjxl-style
    /// 3-norm aggregation matching `butteraugli_main --pnorm` and the
    /// Cloudinary CID22 paper). One `compute()` call yields both numbers,
    /// so emitting both is free.
    #[value(name = "butteraugli")]
    Butteraugli,
    /// Butteraugli — GPU implementation via the `butteraugli-gpu` crate.
    /// Emits two columns: `butteraugli_max_gpu` and `butteraugli_pnorm3_gpu`.
    /// Same single-`compute()` rationale as the CPU variant.
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

    /// Static list of TSV / parquet column suffixes emitted by this metric,
    /// in the order [`run_metric`] produces them. For most metrics this is a
    /// single column matching [`MetricKind::name`] with `-` rewritten to
    /// `_`. Butteraugli (CPU and GPU) emits two columns — `_max` and
    /// `_pnorm3` — because one `compute()` call yields both aggregations.
    ///
    /// Headers in the sweep TSV are formatted as `score_<column>` for each
    /// entry returned here.
    pub fn column_names(self) -> &'static [&'static str] {
        match self {
            MetricKind::Ssim2 => &["ssim2"],
            MetricKind::Ssim2Gpu => &["ssim2_gpu"],
            MetricKind::Butteraugli => &["butteraugli_max", "butteraugli_pnorm3"],
            MetricKind::ButteraugliGpu => &["butteraugli_max_gpu", "butteraugli_pnorm3_gpu"],
            MetricKind::Dssim => &["dssim"],
            MetricKind::DssimGpu => &["dssim_gpu"],
            MetricKind::Zensim => &["zensim"],
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
///
/// Returns `(column_name, value)` pairs in the order declared by
/// [`MetricKind::column_names`]. For most metrics this is a single pair;
/// for butteraugli (CPU and GPU) it is two pairs (max-norm + 3-norm) yielded
/// from a single `compute()` call so callers don't pay twice.
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
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ssim2 => Ok(vec![("ssim2", ssim2::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ssim2 => Err(disabled_msg("ssim2", "cpu-metrics")),

        #[cfg(feature = "gpu-ssim2")]
        MetricKind::Ssim2Gpu => Ok(vec![(
            "ssim2_gpu",
            ssim2_gpu::score(reference, distorted, gpu_runtime)?,
        )]),
        #[cfg(not(feature = "gpu-ssim2"))]
        MetricKind::Ssim2Gpu => Err(disabled_msg("ssim2-gpu", "gpu-ssim2")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Butteraugli => {
            let (max, pnorm3) = butteraugli::score_both(reference, distorted)?;
            Ok(vec![
                ("butteraugli_max", max),
                ("butteraugli_pnorm3", pnorm3),
            ])
        }
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Butteraugli => Err(disabled_msg("butteraugli", "cpu-metrics")),

        #[cfg(feature = "gpu-butteraugli")]
        MetricKind::ButteraugliGpu => {
            let (max, pnorm3) = butteraugli_gpu::score_both(reference, distorted, gpu_runtime)?;
            Ok(vec![
                ("butteraugli_max_gpu", max),
                ("butteraugli_pnorm3_gpu", pnorm3),
            ])
        }
        #[cfg(not(feature = "gpu-butteraugli"))]
        MetricKind::ButteraugliGpu => Err(disabled_msg("butteraugli-gpu", "gpu-butteraugli")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Dssim => Ok(vec![("dssim", dssim::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Dssim => Err(disabled_msg("dssim", "cpu-metrics")),

        #[cfg(feature = "gpu-dssim")]
        MetricKind::DssimGpu => Ok(vec![(
            "dssim_gpu",
            dssim_gpu::score(reference, distorted, gpu_runtime)?,
        )]),
        #[cfg(not(feature = "gpu-dssim"))]
        MetricKind::DssimGpu => Err(disabled_msg("dssim-gpu", "gpu-dssim")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Zensim => Ok(vec![("zensim", zensim::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Zensim => Err(disabled_msg("zensim", "cpu-metrics")),
    }
}

#[allow(dead_code)]
fn disabled_msg(metric: &str, feature: &str) -> Box<dyn std::error::Error> {
    format!("metric '{metric}' is disabled (rebuild with `--features {feature}`)").into()
}

/// Run the zensim metric and additionally return the 300-feature extended
/// vector that the score is derived from. Only the zensim metric exposes a
/// feature vector — other metrics return `None` so callers can still use a
/// uniform call site. Callers that want feature output must therefore include
/// `MetricKind::Zensim` in their metric list.
///
/// Score values match what [`run_metric`] would return for `MetricKind::Zensim`,
/// so a sweep that scores zensim today and migrates to this entry point
/// later sees no shift in the TSV `score_zensim` column.
#[cfg(feature = "sweep")]
pub fn run_zensim_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    #[cfg(feature = "cpu-metrics")]
    {
        zensim::score_with_features(reference, distorted)
    }
    #[cfg(not(feature = "cpu-metrics"))]
    {
        Err(disabled_msg("zensim", "cpu-metrics"))
    }
}
