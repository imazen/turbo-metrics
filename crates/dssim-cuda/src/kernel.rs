//! CUDA kernel wrappers for DSSIM computation

use cudarse_driver::{kernel_params, CuFunction, CuModule, CuStream, LaunchConfig};
use cudarse_npp::image::{Img, ImgMut, C};

/// Loaded DSSIM CUDA kernels
pub struct Kernel {
    _module: CuModule,
    srgb_to_linear: CuFunction,
    linear_to_lab_planar: CuFunction,
    linear_to_lab_packed: CuFunction,
    downscale_plane_by_2: CuFunction,
    downscale_rgb_by_2: CuFunction,
    blur_3x3: CuFunction,
    blur_squared: CuFunction,
    blur_product: CuFunction,
    square: CuFunction,
    multiply: CuFunction,
    compute_ssim: CuFunction,
    compute_ssim_precomputed_ref: CuFunction,
    compute_ssim_lab: CuFunction,
    compute_abs_diff_scalar: CuFunction,
}

impl Kernel {
    /// Load all DSSIM kernels from the compiled PTX
    pub fn load() -> Self {
        let module = CuModule::load_ptx(include_str!(concat!(
            env!("OUT_DIR"),
            "/dssim_cuda_kernel.ptx"
        )))
        .unwrap();

        Self {
            srgb_to_linear: module.function_by_name("srgb_to_linear").unwrap(),
            linear_to_lab_planar: module.function_by_name("linear_to_lab_planar").unwrap(),
            linear_to_lab_packed: module.function_by_name("linear_to_lab_packed").unwrap(),
            downscale_plane_by_2: module.function_by_name("downscale_plane_by_2").unwrap(),
            downscale_rgb_by_2: module.function_by_name("downscale_rgb_by_2").unwrap(),
            blur_3x3: module.function_by_name("blur_3x3").unwrap(),
            blur_squared: module.function_by_name("blur_squared").unwrap(),
            blur_product: module.function_by_name("blur_product").unwrap(),
            square: module.function_by_name("square").unwrap(),
            multiply: module.function_by_name("multiply").unwrap(),
            compute_ssim: module.function_by_name("compute_ssim").unwrap(),
            compute_ssim_precomputed_ref: module
                .function_by_name("compute_ssim_precomputed_ref")
                .unwrap(),
            compute_ssim_lab: module.function_by_name("compute_ssim_lab").unwrap(),
            compute_abs_diff_scalar: module.function_by_name("compute_abs_diff_scalar").unwrap(),
            _module: module,
        }
    }

    /// Convert sRGB u8 to linear f32
    pub fn srgb_to_linear(
        &self,
        stream: &CuStream,
        src: impl Img<u8, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
    ) {
        unsafe {
            self.srgb_to_linear
                .launch(
                    &launch_config_2d(src.width() * 3, src.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch srgb_to_linear kernel");
        }
    }

    /// Convert linear RGB to LAB (planar output: 3 separate planes)
    pub fn linear_to_lab_planar(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        mut dst_l: impl ImgMut<f32, C<1>>,
        mut dst_a: impl ImgMut<f32, C<1>>,
        mut dst_b: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.linear_to_lab_planar
                .launch(
                    &launch_config_2d(src.width(), src.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst_l.device_ptr_mut(),
                        dst_a.device_ptr_mut(),
                        dst_b.device_ptr_mut(),
                        dst_l.pitch() as usize,
                        src.width() as usize,
                        src.height() as usize,
                    ),
                )
                .expect("Could not launch linear_to_lab_planar kernel");
        }
    }

    /// Convert linear RGB to LAB (packed output)
    pub fn linear_to_lab_packed(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
    ) {
        unsafe {
            self.linear_to_lab_packed
                .launch(
                    &launch_config_2d(src.width(), src.height()),
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
                .expect("Could not launch linear_to_lab_packed kernel");
        }
    }

    /// Downscale a single plane by 2x
    pub fn downscale_plane_by_2(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.downscale_plane_by_2
                .launch(
                    &launch_config_2d(dst.width(), dst.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.width() as usize,
                        src.height() as usize,
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.width() as usize,
                        dst.height() as usize,
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch downscale_plane_by_2 kernel");
        }
    }

    /// Downscale packed RGB by 2x
    pub fn downscale_rgb_by_2(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
    ) {
        unsafe {
            self.downscale_rgb_by_2
                .launch(
                    &launch_config_2d(dst.width(), dst.height()),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.width() as usize,
                        src.height() as usize,
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.width() as usize,
                        dst.height() as usize,
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch downscale_rgb_by_2 kernel");
        }
    }

    /// Apply 3x3 Gaussian blur to a single plane
    pub fn blur_3x3(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.blur_3x3
                .launch(
                    &launch_config_2d(src.width(), src.height()),
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
                .expect("Could not launch blur_3x3 kernel");
        }
    }

    /// Compute blur(src^2) - fused square and blur
    pub fn blur_squared(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.blur_squared
                .launch(
                    &launch_config_2d(src.width(), src.height()),
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
                .expect("Could not launch blur_squared kernel");
        }
    }

    /// Compute blur(src1 * src2) - fused multiply and blur
    pub fn blur_product(
        &self,
        stream: &CuStream,
        src1: impl Img<f32, C<1>>,
        src2: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.blur_product
                .launch(
                    &launch_config_2d(src1.width(), src1.height()),
                    stream,
                    kernel_params!(
                        src1.device_ptr(),
                        src1.pitch() as usize,
                        src2.device_ptr(),
                        src2.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                        src1.width() as usize,
                        src1.height() as usize,
                    ),
                )
                .expect("Could not launch blur_product kernel");
        }
    }

    /// Element-wise square: dst = src * src
    pub fn square(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.square
                .launch(
                    &launch_config_2d(src.width(), src.height()),
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
                .expect("Could not launch square kernel");
        }
    }

    /// Element-wise multiply: dst = src1 * src2
    pub fn multiply(
        &self,
        stream: &CuStream,
        src1: impl Img<f32, C<1>>,
        src2: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
    ) {
        unsafe {
            self.multiply
                .launch(
                    &launch_config_2d(src1.width(), src1.height()),
                    stream,
                    kernel_params!(
                        src1.device_ptr(),
                        src1.pitch() as usize,
                        src2.device_ptr(),
                        src2.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                        src1.width() as usize,
                        src1.height() as usize,
                    ),
                )
                .expect("Could not launch multiply kernel");
        }
    }

    /// Compute per-pixel SSIM from blur statistics
    #[allow(clippy::too_many_arguments)]
    pub fn compute_ssim(
        &self,
        stream: &CuStream,
        mu1: impl Img<f32, C<1>>,
        mu2: impl Img<f32, C<1>>,
        img1_sq_blur: impl Img<f32, C<1>>,
        img2_sq_blur: impl Img<f32, C<1>>,
        img12_blur: impl Img<f32, C<1>>,
        mut ssim_out: impl ImgMut<f32, C<1>>,
    ) {
        let width = mu1.width();
        let height = mu1.height();
        unsafe {
            self.compute_ssim
                .launch(
                    &launch_config_2d(width, height),
                    stream,
                    kernel_params!(
                        mu1.device_ptr(),
                        mu1.pitch() as usize,
                        mu2.device_ptr(),
                        mu2.pitch() as usize,
                        img1_sq_blur.device_ptr(),
                        img1_sq_blur.pitch() as usize,
                        img2_sq_blur.device_ptr(),
                        img2_sq_blur.pitch() as usize,
                        img12_blur.device_ptr(),
                        img12_blur.pitch() as usize,
                        ssim_out.device_ptr_mut(),
                        ssim_out.pitch() as usize,
                        width as usize,
                        height as usize,
                    ),
                )
                .expect("Could not launch compute_ssim kernel");
        }
    }

    /// Compute SSIM with precomputed reference statistics
    #[allow(clippy::too_many_arguments)]
    pub fn compute_ssim_precomputed_ref(
        &self,
        stream: &CuStream,
        // Reference (precomputed)
        mu1: impl Img<f32, C<1>>,
        sigma1_sq: impl Img<f32, C<1>>,
        // Distorted
        mu2: impl Img<f32, C<1>>,
        img2_sq_blur: impl Img<f32, C<1>>,
        // Cross term
        img12_blur: impl Img<f32, C<1>>,
        mut ssim_out: impl ImgMut<f32, C<1>>,
    ) {
        let width = mu1.width();
        let height = mu1.height();
        unsafe {
            self.compute_ssim_precomputed_ref
                .launch(
                    &launch_config_2d(width, height),
                    stream,
                    kernel_params!(
                        mu1.device_ptr(),
                        mu1.pitch() as usize,
                        sigma1_sq.device_ptr(),
                        sigma1_sq.pitch() as usize,
                        mu2.device_ptr(),
                        mu2.pitch() as usize,
                        img2_sq_blur.device_ptr(),
                        img2_sq_blur.pitch() as usize,
                        img12_blur.device_ptr(),
                        img12_blur.pitch() as usize,
                        ssim_out.device_ptr_mut(),
                        ssim_out.pitch() as usize,
                        width as usize,
                        height as usize,
                    ),
                )
                .expect("Could not launch compute_ssim_precomputed_ref kernel");
        }
    }

    /// Compute per-pixel SSIM from LAB channel statistics, averaging across channels.
    ///
    /// Takes 15 input buffers (5 statistics × 3 channels) and produces single SSIM map.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_ssim_lab(
        &self,
        stream: &CuStream,
        // Reference mu (L, a, b)
        mu1_l: impl Img<f32, C<1>>,
        mu1_a: impl Img<f32, C<1>>,
        mu1_b: impl Img<f32, C<1>>,
        // Distorted mu (L, a, b)
        mu2_l: impl Img<f32, C<1>>,
        mu2_a: impl Img<f32, C<1>>,
        mu2_b: impl Img<f32, C<1>>,
        // Reference blur(img²) (L, a, b)
        sq1_l: impl Img<f32, C<1>>,
        sq1_a: impl Img<f32, C<1>>,
        sq1_b: impl Img<f32, C<1>>,
        // Distorted blur(img²) (L, a, b)
        sq2_l: impl Img<f32, C<1>>,
        sq2_a: impl Img<f32, C<1>>,
        sq2_b: impl Img<f32, C<1>>,
        // Cross blur(img1*img2) (L, a, b)
        cross_l: impl Img<f32, C<1>>,
        cross_a: impl Img<f32, C<1>>,
        cross_b: impl Img<f32, C<1>>,
        // Output
        mut ssim_out: impl ImgMut<f32, C<1>>,
    ) {
        let width = mu1_l.width();
        let height = mu1_l.height();
        unsafe {
            self.compute_ssim_lab
                .launch(
                    &launch_config_2d(width, height),
                    stream,
                    kernel_params!(
                        mu1_l.device_ptr(),
                        mu1_a.device_ptr(),
                        mu1_b.device_ptr(),
                        mu1_l.pitch() as usize,
                        mu2_l.device_ptr(),
                        mu2_a.device_ptr(),
                        mu2_b.device_ptr(),
                        mu2_l.pitch() as usize,
                        sq1_l.device_ptr(),
                        sq1_a.device_ptr(),
                        sq1_b.device_ptr(),
                        sq1_l.pitch() as usize,
                        sq2_l.device_ptr(),
                        sq2_a.device_ptr(),
                        sq2_b.device_ptr(),
                        sq2_l.pitch() as usize,
                        cross_l.device_ptr(),
                        cross_a.device_ptr(),
                        cross_b.device_ptr(),
                        cross_l.pitch() as usize,
                        ssim_out.device_ptr_mut(),
                        ssim_out.pitch() as usize,
                        width as usize,
                        height as usize,
                    ),
                )
                .expect("Could not launch compute_ssim_lab kernel");
        }
    }

    /// Compute |scalar - value| for each pixel (for MAD computation).
    pub fn compute_abs_diff_scalar(
        &self,
        stream: &CuStream,
        src: impl Img<f32, C<1>>,
        mut dst: impl ImgMut<f32, C<1>>,
        scalar: f32,
    ) {
        let width = src.width();
        let height = src.height();
        unsafe {
            self.compute_abs_diff_scalar
                .launch(
                    &launch_config_2d(width, height),
                    stream,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                        scalar,
                        width as usize,
                        height as usize,
                    ),
                )
                .expect("Could not launch compute_abs_diff_scalar kernel");
        }
    }
}

fn launch_config_2d(width: u32, height: u32) -> LaunchConfig {
    const THREADS_WIDTH: u32 = 32;
    const THREADS_HEIGHT: u32 = 8;
    let num_blocks_w = (width + THREADS_WIDTH - 1) / THREADS_WIDTH;
    let num_blocks_h = (height + THREADS_HEIGHT - 1) / THREADS_HEIGHT;
    LaunchConfig {
        grid_dim: (num_blocks_w, num_blocks_h, 1),
        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
        shared_mem_bytes: 0,
    }
}
