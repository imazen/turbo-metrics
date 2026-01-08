fn main() {
    nvptx_builder::link_libdevice();

    // Link shared memory definition for Malta kernels
    let root = env!("CARGO_MANIFEST_DIR");
    nvptx_builder::link_llvm_ir_file(&format!("{root}/src/shared.ll"));
}
