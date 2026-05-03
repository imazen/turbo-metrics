#![forbid(unsafe_code)]

//! Helpers to resolve a `GpuRuntime` selection into the requested CubeCL
//! runtime. Each runtime path is feature-gated so a host without (e.g.)
//! the CUDA toolkit can still build the wgpu / cpu paths.

use crate::metrics::GpuRuntime;

/// List of runtimes to try, in `auto` order, restricted to whatever the
/// crate was compiled with. Used by both metric backends.
pub fn auto_order() -> &'static [GpuRuntime] {
    &[
        GpuRuntime::Cuda,
        GpuRuntime::Wgpu,
        GpuRuntime::Hip,
        GpuRuntime::Cpu,
    ]
}

pub fn runtime_label(rt: GpuRuntime) -> &'static str {
    match rt {
        GpuRuntime::Auto => "auto",
        GpuRuntime::Cuda => "cuda",
        GpuRuntime::Wgpu => "wgpu",
        GpuRuntime::Hip => "hip",
        GpuRuntime::Cpu => "cpu",
    }
}
