fn main() {
    // Link libdevice for math functions (cbrt, exp, etc.)
    nvptx_builder::link_libdevice();
}
