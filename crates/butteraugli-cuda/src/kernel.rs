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
    blur_mirrored_5x5_h: CuFunction,
    blur_mirrored_5x5_v: CuFunction,
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
    amplify_range: CuFunction,
    // Malta
    malta_diff_map: CuFunction,
    malta_diff_map_lf: CuFunction,
    // Masking
    combine_channels_for_masking: CuFunction,
    mask_to_error_mul: CuFunction,
    diff_precompute: CuFunction,
    fuzzy_erosion: CuFunction,
    // Diffmap
    compute_diffmap: CuFunction,
    l2_diff: CuFunction,
    l2_asym_diff: CuFunction,
    power_elements: CuFunction,
    max_reduce_f32_to_u32: CuFunction,
    // Batched variants (tracked in coefficient#13). Kernels whose
    // single-image version is already 1D size-indexed (e.g. opsin,
    // l2_diff, subtract_arrays) don't need a batch variant — callers
    // can feed them a concatenated N-image buffer and size = N * plane.
    srgb_to_linear_batch: CuFunction,
    deinterleave_3ch_batch: CuFunction,
    opsin_dynamics_batch: CuFunction,
    horizontal_blur_batch: CuFunction,
    vertical_blur_batch: CuFunction,
    blur_mirrored_5x5_h_batch: CuFunction,
    blur_mirrored_5x5_v_batch: CuFunction,
    downsample_2x_batch: CuFunction,
    add_upsample_2x_batch: CuFunction,
    malta_diff_map_batch: CuFunction,
    malta_diff_map_lf_batch: CuFunction,
    malta_diff_map_batch_split_stride: CuFunction,
    malta_diff_map_lf_batch_split_stride: CuFunction,
    mask_to_error_mul_batch_split_stride: CuFunction,
    broadcast_plane_batch: CuFunction,
    compute_diffmap_batch: CuFunction,
    fuzzy_erosion_batch: CuFunction,
    max_reduce_f32_to_u32_batch: CuFunction,
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
            blur_mirrored_5x5_h: module
                .function_by_name("blur_mirrored_5x5_horizontal_kernel")
                .expect("blur_mirrored_5x5_horizontal_kernel not found"),
            blur_mirrored_5x5_v: module
                .function_by_name("blur_mirrored_5x5_vertical_kernel")
                .expect("blur_mirrored_5x5_vertical_kernel not found"),
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
            amplify_range: module
                .function_by_name("amplify_range_kernel")
                .expect("amplify_range_kernel not found"),
            malta_diff_map: module
                .function_by_name("malta_diff_map_kernel")
                .expect("malta_diff_map_kernel not found"),
            malta_diff_map_lf: module
                .function_by_name("malta_diff_map_lf_kernel")
                .expect("malta_diff_map_lf_kernel not found"),
            combine_channels_for_masking: module
                .function_by_name("combine_channels_for_masking_kernel")
                .expect("combine_channels_for_masking_kernel not found"),
            mask_to_error_mul: module
                .function_by_name("mask_to_error_mul_kernel")
                .expect("mask_to_error_mul_kernel not found"),
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
            max_reduce_f32_to_u32: module
                .function_by_name("max_reduce_f32_to_u32_kernel")
                .expect("max_reduce_f32_to_u32_kernel not found"),
            srgb_to_linear_batch: module
                .function_by_name("srgb_to_linear_batch_kernel")
                .expect("srgb_to_linear_batch_kernel not found"),
            deinterleave_3ch_batch: module
                .function_by_name("deinterleave_3ch_batch_kernel")
                .expect("deinterleave_3ch_batch_kernel not found"),
            opsin_dynamics_batch: module
                .function_by_name("opsin_dynamics_batch_kernel")
                .expect("opsin_dynamics_batch_kernel not found"),
            horizontal_blur_batch: module
                .function_by_name("horizontal_blur_batch_kernel")
                .expect("horizontal_blur_batch_kernel not found"),
            vertical_blur_batch: module
                .function_by_name("vertical_blur_batch_kernel")
                .expect("vertical_blur_batch_kernel not found"),
            blur_mirrored_5x5_h_batch: module
                .function_by_name("blur_mirrored_5x5_horizontal_batch_kernel")
                .expect("blur_mirrored_5x5_horizontal_batch_kernel not found"),
            blur_mirrored_5x5_v_batch: module
                .function_by_name("blur_mirrored_5x5_vertical_batch_kernel")
                .expect("blur_mirrored_5x5_vertical_batch_kernel not found"),
            downsample_2x_batch: module
                .function_by_name("downsample_2x_batch_kernel")
                .expect("downsample_2x_batch_kernel not found"),
            add_upsample_2x_batch: module
                .function_by_name("add_upsample_2x_batch_kernel")
                .expect("add_upsample_2x_batch_kernel not found"),
            malta_diff_map_batch: module
                .function_by_name("malta_diff_map_batch_kernel")
                .expect("malta_diff_map_batch_kernel not found"),
            malta_diff_map_lf_batch: module
                .function_by_name("malta_diff_map_lf_batch_kernel")
                .expect("malta_diff_map_lf_batch_kernel not found"),
            malta_diff_map_batch_split_stride: module
                .function_by_name("malta_diff_map_batch_split_stride_kernel")
                .expect("malta_diff_map_batch_split_stride_kernel not found"),
            malta_diff_map_lf_batch_split_stride: module
                .function_by_name("malta_diff_map_lf_batch_split_stride_kernel")
                .expect("malta_diff_map_lf_batch_split_stride_kernel not found"),
            mask_to_error_mul_batch_split_stride: module
                .function_by_name("mask_to_error_mul_batch_split_stride_kernel")
                .expect("mask_to_error_mul_batch_split_stride_kernel not found"),
            broadcast_plane_batch: module
                .function_by_name("broadcast_plane_batch_kernel")
                .expect("broadcast_plane_batch_kernel not found"),
            compute_diffmap_batch: module
                .function_by_name("compute_diffmap_batch_kernel")
                .expect("compute_diffmap_batch_kernel not found"),
            fuzzy_erosion_batch: module
                .function_by_name("fuzzy_erosion_batch_kernel")
                .expect("fuzzy_erosion_batch_kernel not found"),
            max_reduce_f32_to_u32_batch: module
                .function_by_name("max_reduce_f32_to_u32_batch_kernel")
                .expect("max_reduce_f32_to_u32_batch_kernel not found"),
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

    /// Simple linear RGB to XYB (without opsin dynamics blur)
    /// Uses intensity_target = 80.0 (standard nits for sRGB content).
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
                        80.0f32, // intensity_target
                    ),
                )
                .expect("linear_to_xyb launch failed");
        }
    }

    /// Convert interleaved linear RGB to planar XYB
    ///
    /// Uses intensity_target = 80.0 (standard nits for sRGB content).
    /// This matches the CPU butteraugli's default behavior.
    pub fn linear_to_xyb_planar(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        dst_x: *mut f32,
        dst_y: *mut f32,
        dst_b: *mut f32,
    ) {
        self.linear_to_xyb_planar_with_intensity(stream, src, dst_x, dst_y, dst_b, 80.0);
    }

    /// Convert interleaved linear RGB to planar XYB with custom intensity target
    ///
    /// The intensity_target parameter specifies how many nits (cd/m²) correspond
    /// to 1.0 in the linear RGB input. Default is 80.0 for standard sRGB content.
    /// Use higher values (e.g., 250.0) for HDR content.
    pub fn linear_to_xyb_planar_with_intensity(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        dst_x: *mut f32,
        dst_y: *mut f32,
        dst_b: *mut f32,
        intensity_target: f32,
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
                        intensity_target,
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

    /// Two-pass separable Gaussian blur (clamp-to-edge boundaries)
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

    /// Two-pass separable Gaussian blur with mirrored boundaries.
    /// Uses pre-computed weights for sigma=1.2 (5x5 kernel).
    /// This matches CPU butteraugli's blur_mirrored_5x5 for opsin dynamics.
    pub fn blur_mirrored_5x5(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        temp: *mut f32,
        width: usize,
        height: usize,
        w0: f32, // center weight
        w1: f32, // 1-pixel offset weight
        w2: f32, // 2-pixel offset weight
    ) {
        // Horizontal pass: src -> temp (transposed)
        unsafe {
            self.blur_mirrored_5x5_h
                .launch(
                    &Self::launch_config_1d(width * height),
                    stream,
                    kernel_params!(src, temp, width, height, w0, w1, w2,),
                )
                .expect("blur_mirrored_5x5_h launch failed");
        }
        // Vertical pass: temp (transposed) -> dst
        unsafe {
            self.blur_mirrored_5x5_v
                .launch(
                    &Self::launch_config_1d(width * height),
                    stream,
                    kernel_params!(temp, dst, width, height, w0, w1, w2,),
                )
                .expect("blur_mirrored_5x5_v launch failed");
        }
    }

    /// Malta difference map (high frequency)
    ///
    /// Pre-computes Malta weights with f64 precision for better accuracy.
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
        // Pre-compute Malta weights with f64 precision (matches CPU)
        const MULLI_HF: f64 = 0.39905817637;
        const K_WEIGHT0: f64 = 0.5;
        const K_WEIGHT1: f64 = 0.33;
        const LEN2: f64 = 3.75 * 2.0 + 1.0;

        let w_0gt1_f64 = w_0gt1 as f64;
        let w_0lt1_f64 = w_0lt1 as f64;
        let norm1_f64 = norm1 as f64;

        let norm2_0gt1 = (MULLI_HF * (K_WEIGHT0 * w_0gt1_f64).sqrt() / LEN2 * norm1_f64) as f32;
        let norm2_0lt1 = (MULLI_HF * (K_WEIGHT1 * w_0lt1_f64).sqrt() / LEN2 * norm1_f64) as f32;

        unsafe {
            self.malta_diff_map
                .launch(
                    &Self::launch_config_malta(width as u32, height as u32),
                    stream,
                    kernel_params!(
                        lum0,
                        lum1,
                        block_diff_ac,
                        width,
                        height,
                        norm2_0gt1,
                        norm2_0lt1,
                        norm1,
                    ),
                )
                .expect("malta_diff_map launch failed");
        }
    }

    /// Malta difference map (low frequency)
    ///
    /// Pre-computes Malta weights with f64 precision for better accuracy.
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
        // Pre-compute Malta weights with f64 precision (matches CPU)
        const MULLI_LF: f64 = 0.611612573796;
        const K_WEIGHT0: f64 = 0.5;
        const K_WEIGHT1: f64 = 0.33;
        const LEN2: f64 = 3.75 * 2.0 + 1.0;

        let w_0gt1_f64 = w_0gt1 as f64;
        let w_0lt1_f64 = w_0lt1 as f64;
        let norm1_f64 = norm1 as f64;

        let norm2_0gt1 = (MULLI_LF * (K_WEIGHT0 * w_0gt1_f64).sqrt() / LEN2 * norm1_f64) as f32;
        let norm2_0lt1 = (MULLI_LF * (K_WEIGHT1 * w_0lt1_f64).sqrt() / LEN2 * norm1_f64) as f32;

        unsafe {
            self.malta_diff_map_lf
                .launch(
                    &Self::launch_config_malta(width as u32, height as u32),
                    stream,
                    kernel_params!(
                        lum0,
                        lum1,
                        block_diff_ac,
                        width,
                        height,
                        norm2_0gt1,
                        norm2_0lt1,
                        norm1,
                    ),
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

    /// Add upsampled image to destination with scale factor
    pub fn add_upsample_2x(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        scale: f32,
    ) {
        unsafe {
            self.add_upsample_2x
                .launch(
                    &Self::launch_config_2d(dst_width as u32, dst_height as u32),
                    stream,
                    kernel_params!(dst, src, dst_width, dst_height, src_width, src_height, scale,),
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

    /// Separate HF and UHF bands for Y channel (modifies both in place)
    /// Used after blur: hf = blur(HF_raw), uhf = HF_raw
    /// After: hf = amplify(max_clamp(hf) * HF_MUL), uhf = max_clamp(uhf - max_clamp(hf)) * UHF_MUL
    pub fn separate_hf_uhf(&self, stream: &CuStream, hf: *mut f32, uhf: *mut f32, size: usize) {
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

    /// Subtract and remove range around zero
    /// first = remove_range_around_zero(first, w)
    /// second = second - first (original first value)
    pub fn sub_remove_range(
        &self,
        stream: &CuStream,
        first: *mut f32,
        second: *mut f32,
        size: usize,
        w: f32,
    ) {
        unsafe {
            self.sub_remove_range
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(first, second, size, w,),
                )
                .expect("sub_remove_range launch failed");
        }
    }

    /// Subtract and amplify range around zero
    /// first = amplify_range_around_zero(first, w)
    /// second = second - first (original first value)
    pub fn sub_amplify_range(
        &self,
        stream: &CuStream,
        first: *mut f32,
        second: *mut f32,
        size: usize,
        w: f32,
    ) {
        unsafe {
            self.sub_amplify_range
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(first, second, size, w,),
                )
                .expect("sub_amplify_range launch failed");
        }
    }

    /// Remove range around zero (standalone)
    /// arr = remove_range_around_zero(arr, w)
    pub fn remove_range(&self, stream: &CuStream, arr: *mut f32, size: usize, w: f32) {
        unsafe {
            self.remove_range
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(arr, size, w,),
                )
                .expect("remove_range launch failed");
        }
    }

    /// Amplify range around zero (standalone)
    /// arr = amplify_range_around_zero(arr, w)
    pub fn amplify_range(&self, stream: &CuStream, arr: *mut f32, size: usize, w: f32) {
        unsafe {
            self.amplify_range
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(arr, size, w,),
                )
                .expect("amplify_range launch failed");
        }
    }

    /// Combine HF and UHF channels for masking (single image)
    /// Formula: sqrt((uhf_x + hf_x)² * 2.5² + (uhf_y * 0.4 + hf_y * 0.4)²)
    pub fn combine_channels_for_masking(
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
            self.combine_channels_for_masking
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(hf_x, uhf_x, hf_y, uhf_y, dst, size,),
                )
                .expect("combine_channels_for_masking launch failed");
        }
    }

    /// MaskToErrorMul: Add contribution from blurred UHF Y difference to block_diff_ac
    pub fn mask_to_error_mul(
        &self,
        stream: &CuStream,
        blurred1: *const f32,
        blurred2: *const f32,
        block_diff_ac: *mut f32,
        size: usize,
    ) {
        unsafe {
            self.mask_to_error_mul
                .launch(
                    &Self::launch_config_1d(size),
                    stream,
                    kernel_params!(blurred1, blurred2, block_diff_ac, size,),
                )
                .expect("mask_to_error_mul launch failed");
        }
    }

    /// Precompute diff values for masking
    pub fn diff_precompute(&self, stream: &CuStream, src: *const f32, dst: *mut f32, size: usize) {
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
            cudarse_driver::sys::cuMemsetD32Async(dst as u64, 0, size, stream.raw())
                .result()
                .expect("clear_buffer failed");
        }
    }

    // ============================================================
    // Batched kernel wrappers (coefficient#13)
    // ============================================================

    fn batch_config_1d(size: usize, batch: u32) -> LaunchConfig {
        const THREADS: u32 = 256;
        let blocks = ((size as u32) + THREADS - 1) / THREADS;
        LaunchConfig {
            grid_dim: (blocks, 1, batch),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn batch_config_2d(width: u32, height: u32, batch: u32) -> LaunchConfig {
        const TX: u32 = 32;
        const TY: u32 = 8;
        let bx = (width + TX - 1) / TX;
        let by = (height + TY - 1) / TY;
        LaunchConfig {
            grid_dim: (bx, by, batch),
            block_dim: (TX, TY, 1),
            shared_mem_bytes: 0,
        }
    }

    fn batch_config_malta(width: u32, height: u32, batch: u32) -> LaunchConfig {
        const TILE: u32 = 16;
        let bx = (width + TILE - 1) / TILE;
        let by = (height + TILE - 1) / TILE;
        LaunchConfig {
            grid_dim: (bx, by, batch),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 24 * 24 * 4,
        }
    }

    /// Batched sRGB -> linear. Inputs are packed `Image<u8, C<3>>` views
    /// but we take pointers directly because the batch layout is a simple
    /// concatenation of N packed-RGB images.
    pub fn srgb_to_linear_batch(
        &self,
        stream: &CuStream,
        src: *const u8,
        src_pitch: usize,
        src_image_stride: usize,
        dst: *mut f32,
        dst_pitch: usize,
        dst_image_stride: usize,
        width: u32,
        height: u32,
        batch: u32,
    ) {
        unsafe {
            self.srgb_to_linear_batch
                .launch(
                    &Self::batch_config_2d(width * 3, height, batch),
                    stream,
                    kernel_params!(
                        src,
                        src_pitch,
                        src_image_stride,
                        dst,
                        dst_pitch,
                        dst_image_stride,
                        width as usize,
                        height as usize,
                    ),
                )
                .expect("srgb_to_linear_batch launch failed");
        }
    }

    /// Batched deinterleave. Plane outputs are N contiguous `plane_stride`
    /// chunks per channel.
    pub fn deinterleave_3ch_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        src_pitch: usize,
        src_image_stride: usize,
        dst0: *mut f32,
        dst1: *mut f32,
        dst2: *mut f32,
        plane_stride: usize,
        width: u32,
        height: u32,
        batch: u32,
    ) {
        unsafe {
            self.deinterleave_3ch_batch
                .launch(
                    &Self::batch_config_2d(width, height, batch),
                    stream,
                    kernel_params!(
                        src,
                        src_pitch,
                        src_image_stride,
                        dst0,
                        dst1,
                        dst2,
                        plane_stride,
                        width as usize,
                        height as usize,
                    ),
                )
                .expect("deinterleave_3ch_batch launch failed");
        }
    }

    /// Batched opsin dynamics. Each pointer addresses N concatenated
    /// planes of `plane_stride` f32 each.
    pub fn opsin_dynamics_batch(
        &self,
        stream: &CuStream,
        src_r: *mut f32,
        src_g: *mut f32,
        src_b: *mut f32,
        blur_r: *const f32,
        blur_g: *const f32,
        blur_b: *const f32,
        plane_stride: usize,
        intensity: f32,
        batch: u32,
    ) {
        unsafe {
            self.opsin_dynamics_batch
                .launch(
                    &Self::batch_config_1d(plane_stride, batch),
                    stream,
                    kernel_params!(
                        src_r, src_g, src_b, blur_r, blur_g, blur_b, plane_stride, intensity,
                    ),
                )
                .expect("opsin_dynamics_batch launch failed");
        }
    }

    /// Batched separable Gaussian blur (two-pass, same on-the-fly
    /// weights as the unbatched path).
    pub fn blur_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        temp: *mut f32,
        width: usize,
        height: usize,
        sigma: f32,
        plane_stride: usize,
        batch: u32,
    ) {
        // Horizontal pass: src -> temp
        unsafe {
            self.horizontal_blur_batch
                .launch(
                    &Self::batch_config_1d(width * height, batch),
                    stream,
                    kernel_params!(src, temp, width, height, sigma, plane_stride,),
                )
                .expect("horizontal_blur_batch launch failed");
            self.vertical_blur_batch
                .launch(
                    &Self::batch_config_1d(width * height, batch),
                    stream,
                    kernel_params!(temp, dst, width, height, sigma, plane_stride,),
                )
                .expect("vertical_blur_batch launch failed");
        }
    }

    /// Batched 5x5 mirrored blur (the σ=1.2 opsin blur).
    pub fn blur_mirrored_5x5_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        temp: *mut f32,
        width: usize,
        height: usize,
        w0: f32,
        w1: f32,
        w2: f32,
        plane_stride: usize,
        batch: u32,
    ) {
        unsafe {
            self.blur_mirrored_5x5_h_batch
                .launch(
                    &Self::batch_config_1d(width * height, batch),
                    stream,
                    kernel_params!(src, temp, width, height, w0, w1, w2, plane_stride,),
                )
                .expect("blur_mirrored_5x5_h_batch launch failed");
            self.blur_mirrored_5x5_v_batch
                .launch(
                    &Self::batch_config_1d(width * height, batch),
                    stream,
                    kernel_params!(temp, dst, width, height, w0, w1, w2, plane_stride,),
                )
                .expect("blur_mirrored_5x5_v_batch launch failed");
        }
    }

    pub fn downsample_2x_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        src_plane_stride: usize,
        dst_plane_stride: usize,
        batch: u32,
    ) {
        unsafe {
            self.downsample_2x_batch
                .launch(
                    &Self::batch_config_2d(dst_width as u32, dst_height as u32, batch),
                    stream,
                    kernel_params!(
                        src,
                        dst,
                        src_width,
                        src_height,
                        dst_width,
                        dst_height,
                        src_plane_stride,
                        dst_plane_stride,
                    ),
                )
                .expect("downsample_2x_batch launch failed");
        }
    }

    pub fn add_upsample_2x_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        src_plane_stride: usize,
        dst_plane_stride: usize,
        scale: f32,
        batch: u32,
    ) {
        unsafe {
            self.add_upsample_2x_batch
                .launch(
                    &Self::batch_config_2d(dst_width as u32, dst_height as u32, batch),
                    stream,
                    kernel_params!(
                        dst,
                        src,
                        dst_width,
                        dst_height,
                        src_width,
                        src_height,
                        src_plane_stride,
                        dst_plane_stride,
                        scale,
                    ),
                )
                .expect("add_upsample_2x_batch launch failed");
        }
    }

    pub fn malta_diff_map_batch(
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
        plane_stride: usize,
        batch: u32,
    ) {
        const MULLI_HF: f64 = 0.39905817637;
        const K_WEIGHT0: f64 = 0.5;
        const K_WEIGHT1: f64 = 0.33;
        const LEN2: f64 = 3.75 * 2.0 + 1.0;
        let norm2_0gt1 =
            (MULLI_HF * (K_WEIGHT0 * w_0gt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        let norm2_0lt1 =
            (MULLI_HF * (K_WEIGHT1 * w_0lt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        unsafe {
            self.malta_diff_map_batch
                .launch(
                    &Self::batch_config_malta(width as u32, height as u32, batch),
                    stream,
                    kernel_params!(
                        lum0,
                        lum1,
                        block_diff_ac,
                        width,
                        height,
                        norm2_0gt1,
                        norm2_0lt1,
                        norm1,
                        plane_stride,
                    ),
                )
                .expect("malta_diff_map_batch launch failed");
        }
    }

    pub fn malta_diff_map_lf_batch(
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
        plane_stride: usize,
        batch: u32,
    ) {
        const MULLI_LF: f64 = 0.611612573796;
        const K_WEIGHT0: f64 = 0.5;
        const K_WEIGHT1: f64 = 0.33;
        const LEN2: f64 = 3.75 * 2.0 + 1.0;
        let norm2_0gt1 =
            (MULLI_LF * (K_WEIGHT0 * w_0gt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        let norm2_0lt1 =
            (MULLI_LF * (K_WEIGHT1 * w_0lt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        unsafe {
            self.malta_diff_map_lf_batch
                .launch(
                    &Self::batch_config_malta(width as u32, height as u32, batch),
                    stream,
                    kernel_params!(
                        lum0,
                        lum1,
                        block_diff_ac,
                        width,
                        height,
                        norm2_0gt1,
                        norm2_0lt1,
                        norm1,
                        plane_stride,
                    ),
                )
                .expect("malta_diff_map_lf_batch launch failed");
        }
    }

    pub fn fuzzy_erosion_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        width: usize,
        height: usize,
        plane_stride: usize,
        batch: u32,
    ) {
        unsafe {
            self.fuzzy_erosion_batch
                .launch(
                    &Self::batch_config_1d(width * height, batch),
                    stream,
                    kernel_params!(src, dst, width, height, plane_stride,),
                )
                .expect("fuzzy_erosion_batch launch failed");
        }
    }

    /// Batched max reduction. Produces one u32 (float bits) per image in
    /// `result[0..batch]`. Caller must zero the result slots first.
    pub fn max_reduce_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        result: *mut f32,
        size: usize,
        plane_stride: usize,
        batch: u32,
    ) {
        unsafe {
            cudarse_driver::sys::cuMemsetD32Async(result as u64, 0, batch as usize, stream.raw())
                .result()
                .expect("max_reduce_batch clear failed");
        }
        const THREADS: u32 = 256;
        const BLOCKS: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: (BLOCKS, 1, batch),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.max_reduce_f32_to_u32_batch
                .launch(
                    &cfg,
                    stream,
                    kernel_params!(src, result as *mut u32, size, plane_stride,),
                )
                .expect("max_reduce_batch launch failed");
        }
    }

    /// Batched Malta HF with separate strides for the reference and
    /// distorted input buffers and the output. Passing `lum0_stride = 0`
    /// broadcasts a single cached reference plane across all N batch
    /// slots; `lum1_stride = out_stride = width * height` indexes N
    /// distinct distorted planes and per-slot outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn malta_diff_map_batch_split_stride(
        &self,
        stream: &CuStream,
        lum0: *const f32,
        lum0_stride: usize,
        lum1: *const f32,
        lum1_stride: usize,
        block_diff_ac: *mut f32,
        out_stride: usize,
        width: usize,
        height: usize,
        w_0gt1: f32,
        w_0lt1: f32,
        norm1: f32,
        batch: u32,
    ) {
        const MULLI_HF: f64 = 0.39905817637;
        const K_WEIGHT0: f64 = 0.5;
        const K_WEIGHT1: f64 = 0.33;
        const LEN2: f64 = 3.75 * 2.0 + 1.0;
        let norm2_0gt1 =
            (MULLI_HF * (K_WEIGHT0 * w_0gt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        let norm2_0lt1 =
            (MULLI_HF * (K_WEIGHT1 * w_0lt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        unsafe {
            self.malta_diff_map_batch_split_stride
                .launch(
                    &Self::batch_config_malta(width as u32, height as u32, batch),
                    stream,
                    kernel_params!(
                        lum0,
                        lum0_stride,
                        lum1,
                        lum1_stride,
                        block_diff_ac,
                        out_stride,
                        width,
                        height,
                        norm2_0gt1,
                        norm2_0lt1,
                        norm1,
                    ),
                )
                .expect("malta_diff_map_batch_split_stride launch failed");
        }
    }

    /// Batched Malta LF with separate strides. See `malta_diff_map_batch_split_stride`.
    #[allow(clippy::too_many_arguments)]
    pub fn malta_diff_map_lf_batch_split_stride(
        &self,
        stream: &CuStream,
        lum0: *const f32,
        lum0_stride: usize,
        lum1: *const f32,
        lum1_stride: usize,
        block_diff_ac: *mut f32,
        out_stride: usize,
        width: usize,
        height: usize,
        w_0gt1: f32,
        w_0lt1: f32,
        norm1: f32,
        batch: u32,
    ) {
        const MULLI_LF: f64 = 0.611612573796;
        const K_WEIGHT0: f64 = 0.5;
        const K_WEIGHT1: f64 = 0.33;
        const LEN2: f64 = 3.75 * 2.0 + 1.0;
        let norm2_0gt1 =
            (MULLI_LF * (K_WEIGHT0 * w_0gt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        let norm2_0lt1 =
            (MULLI_LF * (K_WEIGHT1 * w_0lt1 as f64).sqrt() / LEN2 * norm1 as f64) as f32;
        unsafe {
            self.malta_diff_map_lf_batch_split_stride
                .launch(
                    &Self::batch_config_malta(width as u32, height as u32, batch),
                    stream,
                    kernel_params!(
                        lum0,
                        lum0_stride,
                        lum1,
                        lum1_stride,
                        block_diff_ac,
                        out_stride,
                        width,
                        height,
                        norm2_0gt1,
                        norm2_0lt1,
                        norm1,
                    ),
                )
                .expect("malta_diff_map_lf_batch_split_stride launch failed");
        }
    }

    /// Batched mask_to_error_mul with split strides. Pass
    /// `blurred1_stride = 0` to broadcast a single reference blurred-mask
    /// plane across N distorted planes.
    #[allow(clippy::too_many_arguments)]
    pub fn mask_to_error_mul_batch_split_stride(
        &self,
        stream: &CuStream,
        blurred1: *const f32,
        blurred1_stride: usize,
        blurred2: *const f32,
        blurred2_stride: usize,
        block_diff_ac: *mut f32,
        out_stride: usize,
        size: usize,
        batch: u32,
    ) {
        unsafe {
            self.mask_to_error_mul_batch_split_stride
                .launch(
                    &Self::batch_config_1d(size, batch),
                    stream,
                    kernel_params!(
                        blurred1,
                        blurred1_stride,
                        blurred2,
                        blurred2_stride,
                        block_diff_ac,
                        out_stride,
                        size,
                    ),
                )
                .expect("mask_to_error_mul_batch_split_stride launch failed");
        }
    }

    /// Batched compute_diffmap. `size_per_image` is the per-image work
    /// size (plane or half_plane); `plane_stride` is the slot stride in
    /// the concatenated buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_diffmap_batch(
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
        size_per_image: usize,
        plane_stride: usize,
        batch: u32,
    ) {
        unsafe {
            self.compute_diffmap_batch
                .launch(
                    &Self::batch_config_1d(size_per_image, batch),
                    stream,
                    kernel_params!(
                        mask,
                        dc0,
                        dc1,
                        dc2,
                        ac0,
                        ac1,
                        ac2,
                        dst,
                        size_per_image,
                        plane_stride,
                    ),
                )
                .expect("compute_diffmap_batch launch failed");
        }
    }

    /// Broadcast a single `size`-element source plane into N destination
    /// slots laid out with `out_stride`. Used to seed `mask_batch` with
    /// N copies of `ref_mask_final_*` before the batched `compute_diffmap`.
    pub fn broadcast_plane_batch(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *mut f32,
        out_stride: usize,
        size: usize,
        batch: u32,
    ) {
        unsafe {
            self.broadcast_plane_batch
                .launch(
                    &Self::batch_config_1d(size, batch),
                    stream,
                    kernel_params!(src, dst, out_stride, size,),
                )
                .expect("broadcast_plane_batch launch failed");
        }
    }

    /// GPU-side max reduction over a non-negative f32 array.
    ///
    /// `src` is the device pointer to a contiguous f32 buffer of length
    /// `size`. `result` must be a u32-aligned device pointer (interpreted
    /// as f32 bits on return). `_scratch` is unused by the current
    /// atomicMax path but is kept in the signature so a future
    /// two-phase reduction can slot in without API churn.
    ///
    /// The kernel zeros `result` on the provided stream before launching
    /// the reduction kernel, so it's safe to call back-to-back.
    pub fn max_reduce(
        &self,
        stream: &CuStream,
        src: *const f32,
        result: *mut f32,
        _scratch: *mut f32,
        size: usize,
    ) {
        // Zero the u32 result slot.
        unsafe {
            cudarse_driver::sys::cuMemsetD32Async(result as u64, 0, 1, stream.raw())
                .result()
                .expect("max_reduce clear failed");
        }
        // Launch ~4096 threads in total: 16 blocks of 256. Each thread
        // scans a grid-strided slice, computes its local max, then
        // contends on one atomicMax. Contention is low because most
        // threads fail the early-out check or hit the global max once.
        const THREADS: u32 = 256;
        const BLOCKS: u32 = 16;
        let config = LaunchConfig {
            grid_dim: (BLOCKS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.max_reduce_f32_to_u32
                .launch(
                    &config,
                    stream,
                    kernel_params!(src, result as *mut u32, size,),
                )
                .expect("max_reduce launch failed");
        }
    }
}
