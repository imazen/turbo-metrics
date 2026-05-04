#![forbid(unsafe_code)]

//! GPU butteraugli score via the `butteraugli-gpu` crate. Returns the
//! libjxl-style **3-norm** aggregation (`GpuButteraugliResult::pnorm_3`),
//! matching the CPU `butteraugli` crate's `pnorm_3` output and
//! `butteraugli_main --pnorm`. The max-norm `score` field is dropped to
//! keep the CLI's metric API a single scalar.

use cubecl::Runtime;

use crate::decode::Rgb8Image;
use crate::metrics::GpuRuntime;
use crate::metrics::gpu_runtime_dispatch::{auto_order, runtime_label};

pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    score_kind(reference, distorted, runtime, ScoreKind::Pnorm3)
}

pub fn score_max(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    score_kind(reference, distorted, runtime, ScoreKind::Max)
}

#[derive(Clone, Copy)]
enum ScoreKind {
    Pnorm3,
    Max,
}

fn score_kind(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
    kind: ScoreKind,
) -> Result<f64, Box<dyn std::error::Error>> {
    let candidates: Vec<GpuRuntime> = match runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };

    let mut last_error: Option<String> = None;
    for rt in candidates {
        match score_with_runtime(reference, distorted, rt, kind) {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(format!("{}: {e}", runtime_label(rt)));
            }
        }
    }
    let label = match kind {
        ScoreKind::Pnorm3 => "butteraugli-gpu",
        ScoreKind::Max => "butteraugli-max-gpu",
    };
    Err(format!(
        "{}: no runtime succeeded; last error: {}",
        label,
        last_error.unwrap_or_else(|| "none".into())
    )
    .into())
}

fn score_with_runtime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
    kind: ScoreKind,
) -> Result<f64, Box<dyn std::error::Error>> {
    match runtime {
        GpuRuntime::Cuda => {
            #[cfg(feature = "gpu-cuda")]
            {
                run::<cubecl::cuda::CudaRuntime>(reference, distorted, kind)
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
        }
        GpuRuntime::Wgpu => {
            #[cfg(feature = "gpu-wgpu")]
            {
                run::<cubecl::wgpu::WgpuRuntime>(reference, distorted, kind)
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
        }
        GpuRuntime::Hip => {
            #[cfg(feature = "gpu-hip")]
            {
                run::<cubecl::hip::HipRuntime>(reference, distorted, kind)
            }
            #[cfg(not(feature = "gpu-hip"))]
            {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
        }
        GpuRuntime::Cpu => {
            #[cfg(feature = "gpu-cpu")]
            {
                run::<cubecl::cpu::CpuRuntime>(reference, distorted, kind)
            }
            #[cfg(not(feature = "gpu-cpu"))]
            {
                Err("cpu runtime not compiled in (rebuild with `--features gpu-cpu`)".into())
            }
        }
        GpuRuntime::Auto => unreachable!("Auto is expanded by score()"),
    }
}

fn run<R: Runtime>(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    kind: ScoreKind,
) -> Result<f64, Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut b =
        butteraugli_gpu::Butteraugli::<R>::new_multires(client, reference.width, reference.height);
    let result = b.compute(&reference.pixels, &distorted.pixels);
    let score = match kind {
        ScoreKind::Pnorm3 => result.pnorm_3 as f64,
        ScoreKind::Max => result.score as f64,
    };
    if !score.is_finite() {
        let label = match kind {
            ScoreKind::Pnorm3 => "pnorm_3",
            ScoreKind::Max => "score (max-norm)",
        };
        return Err(format!("butteraugli-gpu produced non-finite {label}: {score}").into());
    }
    Ok(score)
}
