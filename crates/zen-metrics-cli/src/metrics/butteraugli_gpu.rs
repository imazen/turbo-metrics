#![forbid(unsafe_code)]

//! GPU butteraugli score via the `butteraugli-gpu` crate. Returns the
//! max-norm score; `pnorm_3` is dropped to keep the CLI's metric API a
//! single scalar.

use cubecl::Runtime;

use crate::decode::Rgb8Image;
use crate::metrics::GpuRuntime;
use crate::metrics::gpu_runtime_dispatch::{auto_order, runtime_label};

pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    let candidates: Vec<GpuRuntime> = match runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };

    let mut last_error: Option<String> = None;
    for rt in candidates {
        match score_with_runtime(reference, distorted, rt) {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(format!("{}: {e}", runtime_label(rt)));
            }
        }
    }
    Err(format!(
        "butteraugli-gpu: no runtime succeeded; last error: {}",
        last_error.unwrap_or_else(|| "none".into())
    )
    .into())
}

fn score_with_runtime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    match runtime {
        GpuRuntime::Cuda => {
            #[cfg(feature = "gpu-cuda")]
            {
                run::<cubecl::cuda::CudaRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
        }
        GpuRuntime::Wgpu => {
            #[cfg(feature = "gpu-wgpu")]
            {
                run::<cubecl::wgpu::WgpuRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
        }
        GpuRuntime::Hip => {
            #[cfg(feature = "gpu-hip")]
            {
                run::<cubecl::hip::HipRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-hip"))]
            {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
        }
        GpuRuntime::Cpu => {
            #[cfg(feature = "gpu-cpu")]
            {
                run::<cubecl::cpu::CpuRuntime>(reference, distorted)
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
) -> Result<f64, Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut b =
        butteraugli_gpu::Butteraugli::<R>::new_multires(client, reference.width, reference.height);
    let result = b.compute(&reference.pixels, &distorted.pixels);
    let score = result.score as f64;
    if !score.is_finite() {
        return Err(format!("butteraugli-gpu produced non-finite score: {score}").into());
    }
    Ok(score)
}
