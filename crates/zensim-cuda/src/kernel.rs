//! Host-side kernel loader and wrappers for zensim-cuda-kernel PTX.
//!
//! Loads the embedded PTX once, resolves each kernel by name, and
//! exposes ergonomic `launch_*` wrappers with the right grid/block
//! geometry for each kernel shape.

use cudarse_driver::{CuFunction, CuModule, CuStream, LaunchConfig, kernel_params};

/// Container for every zensim CUDA kernel used by the Rust wrapper.
pub struct Kernel {
    _module: CuModule,
    srgb_to_positive_xyb: CuFunction,
    fused_blur_h_ssim: CuFunction,
    fused_vblur_features_ssim: CuFunction,
    downscale_2x_plane: CuFunction,
    pad_mirror_plane: CuFunction,
}

impl Kernel {
    pub fn load() -> Self {
        let module = CuModule::load_ptx(include_str!(concat!(
            env!("OUT_DIR"),
            "/zensim_cuda_kernel.ptx"
        )))
        .expect("Failed to load zensim PTX");

        Self {
            srgb_to_positive_xyb: module
                .function_by_name("srgb_to_positive_xyb_kernel")
                .expect("srgb_to_positive_xyb_kernel not found"),
            fused_blur_h_ssim: module
                .function_by_name("fused_blur_h_ssim_kernel")
                .expect("fused_blur_h_ssim_kernel not found"),
            fused_vblur_features_ssim: module
                .function_by_name("fused_vblur_features_ssim_kernel")
                .expect("fused_vblur_features_ssim_kernel not found"),
            downscale_2x_plane: module
                .function_by_name("downscale_2x_plane_kernel")
                .expect("downscale_2x_plane_kernel not found"),
            pad_mirror_plane: module
                .function_by_name("pad_mirror_plane_kernel")
                .expect("pad_mirror_plane_kernel not found"),
            _module: module,
        }
    }

    fn cfg_2d(width: u32, height: u32) -> LaunchConfig {
        const TX: u32 = 32;
        const TY: u32 = 8;
        let bx = (width + TX - 1) / TX;
        let by = (height + TY - 1) / TY;
        LaunchConfig {
            grid_dim: (bx, by, 1),
            block_dim: (TX, TY, 1),
            shared_mem_bytes: 0,
        }
    }

    /// sRGB packed u8 (3 channels, pitched) → planar positive-XYB f32
    /// at the same dimensions. All three output planes share `dst_pitch`.
    #[allow(clippy::too_many_arguments)]
    pub fn srgb_to_positive_xyb(
        &self,
        stream: &CuStream,
        src: *const u8,
        src_pitch: usize,
        x_out: *mut f32,
        y_out: *mut f32,
        b_out: *mut f32,
        dst_pitch: usize,
        width: usize,
        height: usize,
    ) {
        unsafe {
            self.srgb_to_positive_xyb
                .launch(
                    &Self::cfg_2d(width as u32, height as u32),
                    stream,
                    kernel_params!(
                        src, src_pitch, x_out, y_out, b_out, dst_pitch, width, height,
                    ),
                )
                .expect("srgb_to_positive_xyb launch failed");
        }
    }

    /// Fused H-blur producing 4 output planes from (src, dst).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_blur_h_ssim(
        &self,
        stream: &CuStream,
        src: *const f32,
        dst: *const f32,
        src_pitch: usize,
        h_mu1: *mut f32,
        h_mu2: *mut f32,
        h_sigma_sq: *mut f32,
        h_sigma12: *mut f32,
        dst_pitch: usize,
        width: usize,
        height: usize,
        radius: usize,
    ) {
        unsafe {
            self.fused_blur_h_ssim
                .launch(
                    &Self::cfg_2d(width as u32, height as u32),
                    stream,
                    kernel_params!(
                        src, dst, src_pitch, h_mu1, h_mu2, h_sigma_sq, h_sigma12, dst_pitch, width,
                        height, radius,
                    ),
                )
                .expect("fused_blur_h_ssim launch failed");
        }
    }

    /// Fused V-blur + all SSIM/edge/HF/MSE feature extraction.
    /// `accum_f64` must point at 17 contiguous zeroed f64 slots;
    /// `peak_u32` at 3 contiguous zeroed u32 slots.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_vblur_features_ssim(
        &self,
        stream: &CuStream,
        h_mu1: *const f32,
        h_mu2: *const f32,
        h_sigma_sq: *const f32,
        h_sigma12: *const f32,
        src: *const f32,
        dst: *const f32,
        pitch: usize,
        width: usize,
        height: usize,
        radius: usize,
        accum_f64: *mut f64,
        peak_u32: *mut u32,
    ) {
        const THREADS: u32 = 512;
        let blocks_x = (width as u32 + THREADS - 1) / THREADS;
        let cfg = LaunchConfig {
            grid_dim: (blocks_x, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.fused_vblur_features_ssim
                .launch(
                    &cfg,
                    stream,
                    kernel_params!(
                        h_mu1, h_mu2, h_sigma_sq, h_sigma12, src, dst, pitch, width, height,
                        radius, accum_f64, peak_u32,
                    ),
                )
                .expect("fused_vblur_features_ssim launch failed");
        }
    }

    pub fn downscale_2x_plane(
        &self,
        stream: &CuStream,
        src: *const f32,
        src_pitch: usize,
        dst: *mut f32,
        dst_pitch: usize,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) {
        unsafe {
            self.downscale_2x_plane
                .launch(
                    &Self::cfg_2d(dst_width as u32, dst_height as u32),
                    stream,
                    kernel_params!(
                        src, src_pitch, dst, dst_pitch, src_width, src_height, dst_width,
                        dst_height,
                    ),
                )
                .expect("downscale_2x_plane launch failed");
        }
    }

    /// Fill padding cols `[logical_w..padded_w)` with mirror-reflected
    /// copies of real cols. `mirror_offsets` is a device pointer to
    /// `padded_w - logical_w` u32 values (source-col lookup table).
    /// No-op if `padded_w == logical_w`.
    #[allow(clippy::too_many_arguments)]
    pub fn pad_mirror_plane(
        &self,
        stream: &CuStream,
        plane: *mut f32,
        pitch: usize,
        logical_w: usize,
        padded_w: usize,
        height: usize,
        mirror_offsets: *const u32,
    ) {
        if padded_w == logical_w {
            return;
        }
        let pad_count = (padded_w - logical_w) as u32;
        const TX: u32 = 16;
        const TY: u32 = 16;
        let bx = (pad_count + TX - 1) / TX;
        let by = (height as u32 + TY - 1) / TY;
        let cfg = LaunchConfig {
            grid_dim: (bx, by, 1),
            block_dim: (TX, TY, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.pad_mirror_plane
                .launch(
                    &cfg,
                    stream,
                    kernel_params!(plane, pitch, logical_w, padded_w, height, mirror_offsets,),
                )
                .expect("pad_mirror_plane launch failed");
        }
    }
}
