fn main() {
    nvptx_builder::build_ptx_crate("dssim-cuda-kernel", "release-nvptx", true);
}
