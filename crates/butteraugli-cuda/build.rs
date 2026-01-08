fn main() {
    nvptx_builder::build_ptx_crate("butteraugli-cuda-kernel", "release-nvptx", true);
}
