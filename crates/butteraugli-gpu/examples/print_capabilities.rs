//! Diagnostic: print which atomic ops the active cubecl backend
//! claims to support.
//!
//! butteraugli-gpu's reduction folds 4 atomics: one Atomic<u32> max
//! (works everywhere) and three Atomic<f32> adds (broken on Metal
//! despite reporting support). The portable two-stage path drops the
//! float atomic adds in favour of per-thread partials + a finalizer
//! kernel — selected at compile time via the `fast-reduction` feature
//! flag.
//!
//! Don't trust the report alone — Metal lies about Atomic<f32>=Add.
//! Use the empirical reduction parity tests to confirm.

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

    #[cfg(feature = "fast-reduction")]
    println!("\nfast-reduction: ENABLED (Atomic<f32>::fetch_add path)");
    #[cfg(not(feature = "fast-reduction"))]
    println!("\nfast-reduction: DISABLED (per-thread partials + finalizer)");
}
