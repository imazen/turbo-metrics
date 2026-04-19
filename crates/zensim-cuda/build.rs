fn main() {
    nvptx_builder::build_ptx_crate("zensim-cuda-kernel", "release-nvptx", true);
}
