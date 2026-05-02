//! Diagnostic: print whichever cubecl runtime is selected, the GPU
//! adapter info (when wgpu), and the cubecl atomic-usage table (which
//! atomic ops are registered for f32, u32, etc).
//!
//! cubecl-wgpu silently no-ops `Atomic<f32>::fetch_add` when the
//! device doesn't expose `SHADER_FLOAT32_ATOMIC`, which makes our
//! reductions return zero and the score collapse to ~100 regardless
//! of distortion. This example dumps the registered atomic usages so
//! we can see in CI exactly what each runner supports.

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

use cubecl::Runtime;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};

fn main() {
    let client = Backend::client(&Default::default());
    let props = client.properties();

    println!("== cubecl atomic usages ==");

    let probes: &[(&str, StorageType)] = &[
        (
            "Atomic<f32>",
            StorageType::Atomic(ElemType::Float(FloatKind::F32)),
        ),
        (
            "Atomic<f64>",
            StorageType::Atomic(ElemType::Float(FloatKind::F64)),
        ),
        (
            "Atomic<u32>",
            StorageType::Atomic(ElemType::UInt(UIntKind::U32)),
        ),
        (
            "Atomic<i32>",
            StorageType::Atomic(ElemType::Int(IntKind::I32)),
        ),
    ];
    for (name, ty) in probes {
        let usage = props.atomic_type_usage(Type::Scalar(*ty));
        let mut flags: Vec<&str> = Vec::new();
        if usage.contains(AtomicUsage::LoadStore) {
            flags.push("LoadStore");
        }
        if usage.contains(AtomicUsage::Add) {
            flags.push("Add");
        }
        if usage.contains(AtomicUsage::MinMax) {
            flags.push("MinMax");
        }
        println!(
            "  {:14} = {}",
            name,
            if flags.is_empty() {
                "<none>".to_string()
            } else {
                flags.join("|")
            },
        );
    }
}
