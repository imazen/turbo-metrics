//! Kernel wrapper for Butteraugli CUDA implementation
//!
//! Loads the PTX and provides type-safe kernel launch wrappers.

use cudarse_driver::{kernel_params, CuFunction, CuModule, CuStream, LaunchConfig};
use cudarse_npp::debug_assert_same_size;
use cudarse_npp::image::{Img, ImgMut, C};

/// Container for all Butteraugli CUDA kernels
pub struct Kernel {
    _module: CuModule,
    // Color conversion
    srgb_to_linear: CuFunction,
    opsin_dynamics: CuFunction,
    linear_to_xyb: CuFunction,
    linear_to_xyb_planar: CuFunction,
    deinterleave_3ch: CuFunction,
    // Blur
    horizontal_blur: CuFunction,
    vertical_blur: CuFunction,
    tiled_blur: CuFunction,
    // Downscale
    downsample_2x: CuFunction,
    add_upsample_2x: CuFunction,
    // Frequency separation
    xyb_low_freq_to_vals: CuFunction,
    subtract_arrays: CuFunction,
    sub_remove_range: CuFunction,
    sub_amplify_range: CuFunction,
    suppress_x_by_y: CuFunction,
    separate_hf_uhf: CuFunction,
    remove_range: CuFunction,
    // Malta
    malta_diff_map: CuFunction,
    malta_diff_map_lf: CuFunction,
    // Masking
    mask_init: CuFunction,
    diff_precompute: CuFunction,
    fuzzy_erosion: CuFunction,
    // Diffmap
    compute_diffmap: CuFunction,
    l2_diff: CuFunction,
    l2_asym_diff: CuFunction,
    power_elements: CuFunction,
}

impl Kernel {
    /// Load kernels from embedded PTX
    pub fn load() -> Self {
        let module = CuModule::load_ptx(include_str!(concat!(
            env!("OUT_DIR"),
            "/butteraugli_cuda_kernel.ptx"
        )))
        .expect("Failed to load butteraugli PTX");

        Self {
            srgb_to_linear: module
                .function_by_name("srgb_to_linear_kernel")
                .expect("srgb_to_linear_kernel not found"),
            opsin_dynamics: module
                .function_by_name("opsin_dynamics_kernel")
                .expect("opsin_dynamics_kernel not found"),
            linear_to_xyb: module
                .function_by_name("linear_to_xyb_kernel")
                .expect("linear_to_xyb_kernel not found"),
            linear_to_xyb_planar: module
                .function_by_name("linear_to_xyb_planar_kernel")
                .expect("linear_to_xyb_planar_kernel not found"),
            deinterleave_3ch: module
                .function_by_name("deinterleave_3ch_kernel")
                .expect("deinterleave_3ch_kernel not found"),
            horizontal_blur: module
                .function_by_name("horizontal_blur_kernel")
                .expect("horizontal_blur_kernel not found"),
            vertical_blur: module
                .function_by_name("vertical_blur_kernel")
                .expect("vertical_blur_kernel not found"),
            tiled_blur: module
                .function_by_name("tiled_blur_kernel")
                .expect("tiled_blur_kernel not found"),
            downsample_2x: module
                .function_by_name("downsample_2x_kernel")
                .expect("downsample_2x_kernel not found"),
            add_upsample_2x: module
                .function_by_name("add_upsample_2x_kernel")
                .expect("add_upsample_2x_kernel not found"),
            xyb_low_freq_to_vals: module
                .function_by_name("xyb_low_freq_to_vals_kernel")
                .expect("xyb_low_freq_to_vals_kernel not found"),
            subtract_arrays: module
                .function_by_name("subtract_arrays_kernel")
                .expect("subtract_arrays_kernel not found"),
            sub_remove_range: module
                .function_by_name("sub_remove_range_kernel")
                .expect("sub_remove_range_kernel not found"),
            sub_amplify_range: module
                .function_by_name("sub_amplify_range_kernel")
                .expect("sub_amplify_range_kernel not found"),
            suppress_x_by_y: module
                .function_by_name("suppress_x_by_y_kernel")
                .expect("suppress_x_by_y_kernel not found"),
            separate_hf_uhf: module
                .function_by_name("separate_hf_uhf_kernel")
                .expect("separate_hf_uhf_kernel not found"),
            remove_range: module
                .function_by_name("remove_range_kernel")
                .expect("remove_range_kernel not found"),
            malta_diff_map: module
                .function_by_name("malta_diff_map_kernel")
                .expect("malta_diff_map_kernel not found"),
            malta_diff_map_lf: module
                .function_by_name("malta_diff_map_lf_kernel")
                .expect("malta_diff_map_lf_kernel not found"),
            mask_init: module
                .function_by_name("mask_init_kernel")
                .expect("mask_init_kernel not found"),
            diff_precompute: module
                .function_by_name("diff_precompute_kernel")
                .expect("diff_precompute_kernel not found"),
            fuzzy_erosion: module
                .function_by_name("fuzzy_erosion_kernel")
                .expect("fuzzy_erosion_kernel not found"),
            compute_diffmap: module
                .function_by_name("compute_diffmap_kernel")
                .expect("compute_diffmap_kernel not found"),
            l2_diff: module
                .function_by_name("l2_diff_kernel")
                .expect("l2_diff_kernel not found"),
            l2_asym_diff: module
                .function_by_name("l2_asym_diff_kernel")
                .expect("l2_asym_diff_kernel not found"),
            power_elements: module
                .function_by_name("power_elements_kernel")
                .expect("power_elements_kernel not found"),
            _module: module,
        }
    }

    /// Launch configuration for 1D kernels
    fn launch_config_1d(size: usize) -> LaunchConfig {
        const THREADS: u32 = 256;
        let blocks = ((size as u32) + THREADS - 1) / THREADS;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch configuration for 2D kernels
    fn launch_config_2d(width: u32, height: u32) -> LaunchConfig {
        const THREADS_X: u32 = 32;
        const THREADS_Y: u32 = 8;
        let blocks_x = (width + THREADS_X - 1) / THREADS_X;
        let blocks_y = (height + THREADS_Y - 1) / THREADS_Y;
        LaunchConfig {
            grid_dim: (blocks_x, blocks_y, 1),
            block_dim: (THREADS_X, THREADS_Y, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch configuration for Malta kernels (16x16 blocks)
    fn launch_config_malta(width: u32, height: u32) -> LaunchConfig {
        const TILE: u32 = 16;
        let blocks_x = (width + TILE - 1) / TILE;
        let blocks_y = (height + TILE - 1) / TILE;
        LaunchConfig {
            grid_dim: (blocks_x, blocks_y, 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 24 * 24 * 4, // 24x24 float shared memory
        }
    }

    /// Convert sRGB to linear RGB
    pub fn srgb_to_linear(
        &self,
        stream: &CuStream,
        src: impl Img<u8, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
    ) {
        debug_assert_same_size!(src, dst);
        unsafe {
            self.srgb_to_linear
                .launch(
                    &Self::launch_config_2d(src.width() * 3, src.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                        src.width() as usize,
                        src.height() as usize,
                    ),
                )
                .expect("srgb_to_linear launch failed");
        }
    }

    /// Apply opsin dynamics transformation (linear RGB -> XYB with adaptation)
    pub fn opsin_dynamics(
        &self,
        stream: &CuStream,
        src_r: *mut f32,
        src_g: *mut f32,
        src_b: *mut f32,
        blur_r: *const f32,
        blur_g: *const f32,
        blur_b: *const f32,
        width: usize,
        height: usize,
        intensity_multiplier: f32,
    ) {
        unsafe {
            self.opsin_dynamics
                .launch(
                    &Self::launch_config_1d(width * height),
                    stream,
                    kernel_params!(
                        src_r,
                        src_g,
                        src_b,
                        blur_r,
                        blur_g,
                        blur_b,
                        width,
                        height,
                        intensity_multiplier,
                    ),
                )
                .expect("opsin_dynamics launch failed");
        }
    }

    /// Simple linear RGB to XYB (without opsin dynamics)
    pub fn linear_to_xyb(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
    ) {
        debug_assert_same_size!(src, dst);
        unsafe {
            self.linear_to_xyb
                .launch(
                    &Self::launch_config_2d(src.width(), src.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                        src.width() as usize,
                        src.height() as usize,
                    ),
                )
                .expect("linear_to_xyb launch failed");
        }
    }

    /// Convert interleaved linear RGB to planar XYB
    pub fn linear_to_xyb_planar(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        dst_x: *mut f32,
        dst_y: *mut f32,
        dst_b: *mut f32,
    ) {
        unsafe {
            self.linear_to_xyb_planar
                .launch(
                    &Self::launch_config_2d(src.width(), src.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst_x,
                        dst_y,
                        dst_b,
                        src.width() as usize,
                        src.height() as usize,
                    ),
                )
                .expect("linear_to_xyb_planar launch failed");
        }
    }

    /// Deinterleave 3-channel image to planar format
    pub fn deinterleave_3ch(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        dst0: *mut f32,
        dst1: *mut f32,
        dst2: *mut f32,
    ) {
        unsafe {
            self.deinterleave_3ch
                .launch(
                    &Self::launch_config_2d(src.width(), src.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst0,
                        dst1,
                        dst2,
                        src.width() as usize,
                        src.height() as usize,
                    ),
                )
                .expect("deinterleave_3ch launch failed");
        }
    }

    /// Horizontal blur pass
    pub fn horizontal_blur(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        width: usize,
        height: usize,
        sigma: f32,
    ) {
        unsafe {
            self.horizontal_blur
                .launch(
                    &Self::launch_config_1d(width * height),
                    stream,
                    kernel_params!(src, dst, width, height, sigma,),
                )
                .expect("horizontal_blur launch failed");
        }
    }

    /// Vertical blur pass
    pub fn vertical_blur(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        width: usize,
        height: usize,
        sigma: f32,
    ) {
        unsafe {
            self.vertical_blur
                .launch(
                    &Self::launch_config_1d(width * height),
                    stream,
                    kernel_params!(src, dst, width, height, sigma,),
                )
                .expect("vertical_blur launch failed");
        }
    }

    /// Tiled blur (small kernel, in shared memory)
    pub fn tiled_blur(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        width: usize,
        height: usize,
    ) {
        unsafe {
            self.tiled_blur
                .launch(
                    &Self::launch_config_2d(width as u32, height as u32),
                    stream,
                    kernel_params!(src, dst, width, height,),
                )
                .expect("tiled_blur launch failed");
        }
    }

    /// Two-pass separable Gaussian blur
    pub fn blur(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        temp: *mut f32,
        width: usize,
        height: usize,
        sigma: f32,
    ) {
        // Horizontal pass: src -> temp
        self.horizontal_blur(stream, src, temp, width, height, sigma);
        // Vertical pass: temp -> dst
        self.vertical_blur(stream, temp, dst, width, height, sigma);
    }

    /// Malta difference map (high frequency)
    pub fn malta_diff_map(
        &self,
        stream: &CuStream,
        lum0: *const f32,
        lum1: *const f32,
        block_diff_ac: *mut f32,
        width: usize,
        height: usize,
        w_0gt1: f32,
        w_0lt1: f32,
        norm1: f32,
    ) {
        unsafe {
            self.malta_diff_map
                .launch(
                    &Self::launch_config_malta(width as u32, height as u32),
                    stream,
                    kernel_params!(lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1,),
                )
                .expect("malta_diff_map launch failed");
        }
    }

    /// Malta difference map (low frequency)
    pub fn malta_diff_map_lf(
        &self,
        stream: &CuStream,
        lum0: *const f32,
        lum1: *const f32,
        block_diff_ac: *mut f32,
        width: usize,
        height: usize,
        w_0gt1: f32,
        w_0lt1: f32,
        norm1: f32,
    ) {
        unsafe {
            self.malta_diff_map_lf
                .launch(
                    &Self::launch_config_malta(width as u32, height as u32),
                    stream,
                    kernel_params!(lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1,),
                )
                .expect("malta_diff_map_lf launch failed");
        }
    }

    /// Compute final diffmap from masked AC and DC differences
    pub fn compute_diffmap(
        &self,
        stream: &CuStream,
        mask: *const f32,
        dc0: *const f32,
        dc1: *const f32,
        dc2: *const f32,
        ac0: *const f32,
        ac1: *const f32,
        ac2: *const f32,
        dst: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.compute_diffmap
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(mask, dc0, dc1, dc2, ac0, ac1, ac2, dst, size,),
                )
                .expect("compute_diffmap launch failed");
        }
    }

    /// L2 difference (symmetric)
    pub fn l2_diff(
        &self,
        stream: &CuStream,
        src1: *const f32,
        src2: *const f32,
        dst: *mut f32,
        size: usize,
        weight: f32,
    ) {
        unsafe {
            self.l2_diff
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(src1, src2, dst, size, weight,),
                )
                .expect("l2_diff launch failed");
        }
    }

    /// L2 asymmetric difference
    pub fn l2_asym_diff(
        &self,
        stream: &CuStream,
        src1: *const f32,
        src2: *const f32,
        dst: *mut f32,
        size: usize,
        weight_gt: f32,
        weight_lt: f32,
    ) {
        unsafe {
            self.l2_asym_diff
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(src1, src2, dst, size, weight_gt, weight_lt,),
                )
                .expect("l2_asym_diff launch failed");
        }
    }

    /// Downsample by 2x
    pub fn downsample_2x(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) {
        unsafe {
            self.downsample_2x
                .launch(
                    &Self::launch_config_2d(dst_width as u32, dst_height as u32),
                    stream,
                    kernel_params!(src, dst, src_width, src_height, dst_width, dst_height,),
                )
                .expect("downsample_2x launch failed");
        }
    }

    /// Add upsampled image to destination
    pub fn add_upsample_2x(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) {
        unsafe {
            self.add_upsample_2x
                .launch(
                    &Self::launch_config_2d(dst_width as u32, dst_height as u32),
                    stream,
                    kernel_params!(src, dst, src_width, src_height, dst_width, dst_height,),
                )
                .expect("add_upsample_2x launch failed");
        }
    }

    /// Subtract arrays: dst = src1 - src2
    pub fn subtract_arrays(
        &self,
        stream: &CuStream,
        src1: *const f32,
        src2: *const f32,
        dst: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.subtract_arrays
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(src1, src2, dst, size,),
                )
                .expect("subtract_arrays launch failed");
        }
    }

    /// Initialize XYB low-frequency values (modifies in place)
    pub fn xyb_low_freq_to_vals(
        &self,
        stream: &CuStream,
        x: *mut f32,
        y: *mut f32,
        b: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.xyb_low_freq_to_vals
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(x, y, b, size,),
                )
                .expect("xyb_low_freq_to_vals launch failed");
        }
    }

    /// Suppress X channel by Y channel
    pub fn suppress_x_by_y(
        &self,
        stream: &CuStream,
        x: *mut f32,
        y: *const f32,
        size: usize,
        suppression: f32,
    ) {
        unsafe {
            self.suppress_x_by_y
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(x, y, size, suppression,),
                )
                .expect("suppress_x_by_y launch failed");
        }
    }

    /// Separate HF and UHF bands (modifies both in place)
    pub fn separate_hf_uhf(
        &self,
        stream: &CuStream,
        hf: *mut f32,
        uhf: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.separate_hf_uhf
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(hf, uhf, size,),
                )
                .expect("separate_hf_uhf launch failed");
        }
    }

    /// Initialize masking from HF and UHF bands
    pub fn mask_init(
        &self,
        stream: &CuStream,
        hf_x: *const f32,
        uhf_x: *const f32,
        hf_y: *const f32,
        uhf_y: *const f32,
        dst: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.mask_init
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(hf_x, uhf_x, hf_y, uhf_y, dst, size,),
                )
                .expect("mask_init launch failed");
        }
    }

    /// Precompute diff values for masking
    pub fn diff_precompute(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.diff_precompute
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(src, dst, size,),
                )
                .expect("diff_precompute launch failed");
        }
    }

    /// Fuzzy erosion for mask refinement
    pub fn fuzzy_erosion(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        width: usize,
        height: usize,
    ) {
        unsafe {
            self.fuzzy_erosion
                .launch(
                    &Self::launch_config_1d(width * height),
                    stream,
                    kernel_params!(src, dst, width, height,),
                )
                .expect("fuzzy_erosion launch failed");
        }
    }

    /// Power elements for norm computation
    pub fn power_elements(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        size: usize,
        q: f32,
    ) {
        unsafe {
            self.power_elements
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(src, dst, size, q,),
                )
                .expect("power_elements launch failed");
        }
    }

    /// Clear buffer to zero
    pub fn clear_buffer(&self, stream: &CuStream, dst: *mut f32, size: usize) {
        // Use cudaMemsetAsync via driver API
        unsafe {
            cudarse_driver::sys::cuMemsetD32Async(
                dst as u64,
                0,
                size,
                stream.raw(),
            )
            .result()
            .expect("clear_buffer failed");
        }
    }
}
