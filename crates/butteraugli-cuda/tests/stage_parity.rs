//! Per-Stage CPU Parity Tests for butteraugli-cuda
//!
//! These tests compare intermediate outputs from GPU and CPU implementations
//! to identify exactly where divergence occurs.
//!
//! Run with: CUDA_PATH=/usr/local/cuda-12.6 cargo test --test stage_parity --release -- --nocapture

use butteraugli::blur::{compute_kernel, gaussian_blur};
use butteraugli::image::{Image3F, ImageF};
use butteraugli::opsin::{srgb_to_linear, opsin_dynamics_image};
use butteraugli::psycho::separate_frequencies;
use butteraugli::malta::malta_diff_map;
use butteraugli::mask::{combine_channels_for_masking, compute_mask};
use butteraugli::{ButteraugliParams, compute_butteraugli};
use butteraugli_cuda::Butteraugli;
use cudarse_driver::{CuBox, CuStream};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut, C};
use cudarse_npp::set_stream;

// ============================================================================
// Test Configuration
// ============================================================================

/// Test sizes - including problematic 32x32 and 64x64
const TEST_SIZES: [(usize, usize); 5] = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)];

/// Blur sigmas used in butteraugli pipeline
const SIGMA_OPSIN: f32 = 1.2;
const SIGMA_LF: f32 = 7.15593339443;
const SIGMA_MF: f32 = 3.22489901262;
const SIGMA_HF: f32 = 1.56416327805;
const SIGMA_UHF: f32 = 1.2;
const SIGMA_MASK: f32 = 2.7;

// ============================================================================
// CUDA Context Setup
// ============================================================================

struct CudaContext {
    stream: CuStream,
    kernel: butteraugli_cuda::Kernel,
}

impl CudaContext {
    fn new() -> Self {
        cudarse_driver::init_cuda_and_primary_ctx().expect("Failed to initialize CUDA");
        let stream = CuStream::new().expect("Failed to create CUDA stream");
        set_stream(stream.inner() as _).expect("Failed to set NPP stream");
        let kernel = butteraugli_cuda::Kernel::load();
        Self { stream, kernel }
    }
}

// ============================================================================
// Image Generation Utilities
// ============================================================================

/// Generate uniform gray image
fn gen_uniform(width: usize, height: usize, val: u8) -> Vec<u8> {
    vec![val; width * height * 3]
}

/// Generate horizontal edge (sharp transition at center)
fn gen_edge_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if y < height / 2 { 50u8 } else { 200u8 };
        for _x in 0..width {
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate vertical edge (sharp transition at center)
fn gen_edge_v(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if x < width / 2 { 50u8 } else { 200u8 };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate gradient image
fn gen_gradient_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = ((x * 255) / (width.max(1) - 1).max(1)) as u8;
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Convert sRGB to linear planar f32 for CPU reference
fn srgb_to_linear_planar(rgb: &[u8], width: usize, height: usize) -> Image3F {
    let mut linear = Image3F::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            linear.plane_mut(0).set(x, y, srgb_to_linear(rgb[idx]));
            linear.plane_mut(1).set(x, y, srgb_to_linear(rgb[idx + 1]));
            linear.plane_mut(2).set(x, y, srgb_to_linear(rgb[idx + 2]));
        }
    }
    linear
}

// ============================================================================
// Comparison Utilities
// ============================================================================

/// Compare two f32 slices and return statistics
fn compare_buffers(cpu: &[f32], gpu: &[f32], name: &str) -> CompareResult {
    assert_eq!(cpu.len(), gpu.len(), "Buffer size mismatch for {}", name);

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut worst_idx = 0;
    let mut worst_cpu = 0.0f32;
    let mut worst_gpu = 0.0f32;

    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let abs_err = (c - g).abs();
        let rel_err = if c.abs() > 1e-6 { abs_err / c.abs() } else { abs_err };

        sum_sq_err += (abs_err as f64).powi(2);

        if abs_err > max_abs_err {
            max_abs_err = abs_err;
            worst_idx = i;
            worst_cpu = c;
            worst_gpu = g;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
    }

    let rmse = (sum_sq_err / cpu.len() as f64).sqrt() as f32;

    CompareResult {
        name: name.to_string(),
        max_abs_err,
        max_rel_err,
        rmse,
        worst_idx,
        worst_cpu,
        worst_gpu,
    }
}

struct CompareResult {
    name: String,
    max_abs_err: f32,
    max_rel_err: f32,
    rmse: f32,
    worst_idx: usize,
    worst_cpu: f32,
    worst_gpu: f32,
}

impl CompareResult {
    fn print(&self) {
        println!("  {}: max_abs={:.6}, max_rel={:.2}%, rmse={:.6}",
            self.name, self.max_abs_err, self.max_rel_err * 100.0, self.rmse);
        if self.max_abs_err > 0.01 {
            println!("    worst@{}: CPU={:.6}, GPU={:.6}, diff={:.6}",
                self.worst_idx, self.worst_cpu, self.worst_gpu,
                self.worst_cpu - self.worst_gpu);
        }
    }

    fn is_ok(&self, abs_tol: f32, rel_tol: f32) -> bool {
        self.max_abs_err <= abs_tol || self.max_rel_err <= rel_tol
    }
}

// ============================================================================
// GPU Buffer Helpers
// ============================================================================

/// Copy data to GPU planar buffer
fn upload_planar(ctx: &CudaContext, cpu_plane: &ImageF) -> CuBox<[f32]> {
    // Flatten the CPU image data
    let mut flat = Vec::with_capacity(cpu_plane.width() * cpu_plane.height());
    for y in 0..cpu_plane.height() {
        flat.extend_from_slice(cpu_plane.row(y));
    }

    CuBox::<[f32]>::new(&flat, &ctx.stream).expect("Failed to upload")
}

/// Upload flat f32 buffer to GPU
fn upload_flat(ctx: &CudaContext, data: &[f32]) -> CuBox<[f32]> {
    CuBox::<[f32]>::new(data, &ctx.stream).expect("Failed to upload")
}

/// Download GPU planar buffer to Vec
fn download_planar(ctx: &CudaContext, gpu_buf: &CuBox<[f32]>, width: usize, height: usize) -> Vec<f32> {
    ctx.stream.sync().expect("Failed to sync");
    let size = width * height;
    let mut cpu_data = vec![0.0f32; size];
    unsafe {
        cudarse_driver::sys::cuMemcpyDtoH_v2(
            cpu_data.as_mut_ptr() as *mut _,
            gpu_buf.ptr(),
            size * 4,
        ).result().expect("Failed to download");
    }
    cpu_data
}

/// Flatten ImageF to Vec
fn imagef_to_vec(img: &ImageF) -> Vec<f32> {
    let mut result = Vec::with_capacity(img.width() * img.height());
    for y in 0..img.height() {
        result.extend_from_slice(img.row(y));
    }
    result
}

// ============================================================================
// Stage Tests
// ============================================================================

/// Test: Gaussian blur at various sizes
#[test]
fn test_blur_parity() {
    let ctx = CudaContext::new();

    println!("\n=== BLUR PARITY TEST ===");

    for (width, height) in TEST_SIZES {
        println!("\nSize: {}x{}", width, height);

        // Create test pattern: edge image (most sensitive to boundary issues)
        let rgb = gen_edge_v(width, height);
        let linear = srgb_to_linear_planar(&rgb, width, height);
        let size = width * height;

        // Test each sigma
        for (sigma_name, sigma) in [
            ("opsin", SIGMA_OPSIN),
            ("hf", SIGMA_HF),
            ("mf", SIGMA_MF),
            ("lf", SIGMA_LF),
            ("mask", SIGMA_MASK),
        ] {
            // CPU blur
            let cpu_blurred = gaussian_blur(linear.plane(0), sigma);
            let cpu_data = imagef_to_vec(&cpu_blurred);

            // GPU blur
            let mut gpu_src = upload_planar(&ctx, linear.plane(0));
            let mut gpu_dst = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();
            let mut gpu_temp = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();

            ctx.kernel.blur(
                &ctx.stream,
                gpu_src.ptr() as *const f32,
                gpu_dst.ptr() as *mut f32,
                gpu_temp.ptr() as *mut f32,
                width, height, sigma,
            );

            let gpu_data = download_planar(&ctx, &gpu_dst, width, height);

            let result = compare_buffers(&cpu_data, &gpu_data, &format!("blur_sigma_{}", sigma_name));
            result.print();

            // Flag problematic cases
            if !result.is_ok(0.05, 0.05) {
                println!("    ⚠️  SIGNIFICANT DIVERGENCE at {}x{} sigma={}", width, height, sigma_name);
            }
        }
    }
}

/// Test: sRGB to linear conversion
#[test]
fn test_srgb_to_linear_parity() {
    let ctx = CudaContext::new();

    println!("\n=== SRGB TO LINEAR PARITY TEST ===");

    for (width, height) in TEST_SIZES {
        println!("\nSize: {}x{}", width, height);

        let rgb = gen_gradient_h(width, height);

        // CPU conversion
        let cpu_linear = srgb_to_linear_planar(&rgb, width, height);
        let cpu_r = imagef_to_vec(cpu_linear.plane(0));
        let cpu_g = imagef_to_vec(cpu_linear.plane(1));
        let cpu_b = imagef_to_vec(cpu_linear.plane(2));

        // GPU conversion
        let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        let mut gpu_linear = Image::<f32, C<3>>::malloc(width as u32, height as u32).unwrap();
        gpu_src.copy_from_cpu(&rgb, ctx.stream.inner() as _).unwrap();

        ctx.kernel.srgb_to_linear(&ctx.stream, gpu_src.full_view(), gpu_linear.full_view_mut());

        // Download and deinterleave
        ctx.stream.sync().unwrap();
        let gpu_interleaved = gpu_linear.full_view().copy_to_cpu(ctx.stream.inner() as _).unwrap();
        ctx.stream.sync().unwrap();

        // Deinterleave
        let mut gpu_r = Vec::with_capacity(width * height);
        let mut gpu_g = Vec::with_capacity(width * height);
        let mut gpu_b = Vec::with_capacity(width * height);
        for i in 0..(width * height) {
            gpu_r.push(gpu_interleaved[i * 3]);
            gpu_g.push(gpu_interleaved[i * 3 + 1]);
            gpu_b.push(gpu_interleaved[i * 3 + 2]);
        }

        compare_buffers(&cpu_r, &gpu_r, "linear_R").print();
        compare_buffers(&cpu_g, &gpu_g, "linear_G").print();
        compare_buffers(&cpu_b, &gpu_b, "linear_B").print();
    }
}

/// Test: Malta filter (core perceptual difference computation)
#[test]
fn test_malta_parity() {
    let ctx = CudaContext::new();

    println!("\n=== MALTA FILTER PARITY TEST ===");

    // Malta weights from butteraugli constants
    const W_UHF_MALTA: f64 = 1.10039032555;
    const HF_ASYMMETRY: f64 = 0.8;
    const NORM1_UHF: f64 = 71.7800275169;

    for (width, height) in TEST_SIZES {
        println!("\nSize: {}x{}", width, height);
        let size = width * height;

        // Create two slightly different edge patterns
        let rgb1 = gen_edge_v(width, height);
        let rgb2: Vec<u8> = rgb1.iter().map(|&v| v.saturating_add(10)).collect();

        // Convert to XYB and get UHF Y
        let linear1 = srgb_to_linear_planar(&rgb1, width, height);
        let linear2 = srgb_to_linear_planar(&rgb2, width, height);

        let xyb1 = opsin_dynamics_image(&linear1, 80.0);
        let xyb2 = opsin_dynamics_image(&linear2, 80.0);

        let ps1 = separate_frequencies(&xyb1);
        let ps2 = separate_frequencies(&xyb2);

        // CPU Malta
        let cpu_malta = malta_diff_map(
            &ps1.uhf[1], // UHF Y image 1
            &ps2.uhf[1], // UHF Y image 2
            W_UHF_MALTA * HF_ASYMMETRY,
            W_UHF_MALTA / HF_ASYMMETRY,
            NORM1_UHF,
            false, // full Malta (not LF variant)
        );
        let cpu_data = imagef_to_vec(&cpu_malta);

        // GPU Malta - upload UHF Y bands
        let gpu_uhf1 = upload_planar(&ctx, &ps1.uhf[1]);
        let gpu_uhf2 = upload_planar(&ctx, &ps2.uhf[1]);
        let mut gpu_diff = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();

        // Clear diff buffer
        ctx.kernel.clear_buffer(&ctx.stream, gpu_diff.ptr() as *mut f32, size);

        ctx.kernel.malta_diff_map(
            &ctx.stream,
            gpu_uhf1.ptr() as *const f32,
            gpu_uhf2.ptr() as *const f32,
            gpu_diff.ptr() as *mut f32,
            width, height,
            (W_UHF_MALTA * HF_ASYMMETRY) as f32,
            (W_UHF_MALTA / HF_ASYMMETRY) as f32,
            NORM1_UHF as f32,
        );

        let gpu_data = download_planar(&ctx, &gpu_diff, width, height);

        let result = compare_buffers(&cpu_data, &gpu_data, "malta_uhf_y");
        result.print();

        if !result.is_ok(0.1, 0.1) {
            println!("    ⚠️  SIGNIFICANT MALTA DIVERGENCE at {}x{}", width, height);
        }
    }
}

/// Test: Combine channels for masking kernel
#[test]
fn test_combine_channels_for_masking_parity() {
    let ctx = CudaContext::new();

    println!("\n=== COMBINE CHANNELS FOR MASKING PARITY TEST ===");

    for (width, height) in TEST_SIZES {
        println!("\nSize: {}x{}", width, height);
        let size = width * height;

        // Create edge pattern
        let rgb = gen_edge_v(width, height);
        let linear = srgb_to_linear_planar(&rgb, width, height);
        let xyb = opsin_dynamics_image(&linear, 80.0);
        let ps = separate_frequencies(&xyb);

        // CPU combine_channels_for_masking
        let mut cpu_mask = ImageF::new(width, height);
        combine_channels_for_masking(
            &[ps.hf[0].clone(), ps.hf[1].clone()],
            &[ps.uhf[0].clone(), ps.uhf[1].clone()],
            &mut cpu_mask,
        );
        let cpu_data = imagef_to_vec(&cpu_mask);

        println!("  CPU mask range: min={:.6}, max={:.6}",
            cpu_data.iter().cloned().fold(f32::INFINITY, f32::min),
            cpu_data.iter().cloned().fold(0.0f32, f32::max));

        // GPU: Upload HF and UHF bands
        let gpu_hf_x = upload_planar(&ctx, &ps.hf[0]);
        let gpu_uhf_x = upload_planar(&ctx, &ps.uhf[0]);
        let gpu_hf_y = upload_planar(&ctx, &ps.hf[1]);
        let gpu_uhf_y = upload_planar(&ctx, &ps.uhf[1]);
        let mut gpu_mask = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();

        // Call combine_channels_for_masking kernel
        ctx.kernel.combine_channels_for_masking(
            &ctx.stream,
            gpu_hf_x.ptr() as *const f32,
            gpu_uhf_x.ptr() as *const f32,
            gpu_hf_y.ptr() as *const f32,
            gpu_uhf_y.ptr() as *const f32,
            gpu_mask.ptr() as *mut f32,
            size,
        );

        let gpu_data = download_planar(&ctx, &gpu_mask, width, height);

        println!("  GPU mask range: min={:.6}, max={:.6}",
            gpu_data.iter().cloned().fold(f32::INFINITY, f32::min),
            gpu_data.iter().cloned().fold(0.0f32, f32::max));

        let result = compare_buffers(&cpu_data, &gpu_data, "combine_channels_mask");
        result.print();

        if !result.is_ok(0.01, 0.01) {
            // Print sample values at corners and center for debugging
            let center = width / 2;
            println!("  Sample values:");
            println!("    [0,0]:     CPU={:.6}, GPU={:.6}", cpu_data[0], gpu_data[0]);
            println!("    [center]:  CPU={:.6}, GPU={:.6}", cpu_data[center * width + center], gpu_data[center * width + center]);
            println!("    [end]:     CPU={:.6}, GPU={:.6}", cpu_data[size-1], gpu_data[size-1]);
        }
    }
}

/// Test: Full mask pipeline step by step
#[test]
fn test_mask_pipeline_step_by_step() {
    use butteraugli::mask::{diff_precompute, fuzzy_erosion};
    use butteraugli::consts::{MASK_MUL, MASK_BIAS};

    let ctx = CudaContext::new();

    println!("\n=== MASK PIPELINE STEP-BY-STEP TEST (32x32) ===");

    let width = 32;
    let height = 32;
    let size = width * height;

    // Create edge pattern
    let rgb = gen_edge_v(width, height);
    let linear = srgb_to_linear_planar(&rgb, width, height);
    let xyb = opsin_dynamics_image(&linear, 80.0);
    let ps = separate_frequencies(&xyb);

    // ===== Step 1: combine_channels_for_masking for image 1 =====
    let mut cpu_mask0 = ImageF::new(width, height);
    combine_channels_for_masking(
        &[ps.hf[0].clone(), ps.hf[1].clone()],
        &[ps.uhf[0].clone(), ps.uhf[1].clone()],
        &mut cpu_mask0,
    );
    let cpu_mask0_data = imagef_to_vec(&cpu_mask0);
    println!("\n[Step 1] combine_channels_for_masking(img1):");
    println!("  CPU: min={:.6}, max={:.6}",
        cpu_mask0_data.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_mask0_data.iter().cloned().fold(0.0f32, f32::max));

    // ===== Step 2: diff_precompute(mask0) =====
    let mut cpu_diff0 = ImageF::new(width, height);
    diff_precompute(&cpu_mask0, MASK_MUL, MASK_BIAS, &mut cpu_diff0);
    let cpu_diff0_data = imagef_to_vec(&cpu_diff0);
    println!("\n[Step 2] diff_precompute(mask0):");
    println!("  CPU: min={:.6}, max={:.6}",
        cpu_diff0_data.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_diff0_data.iter().cloned().fold(0.0f32, f32::max));

    // ===== Step 3: blur(diff0) =====
    let cpu_blurred0 = gaussian_blur(&cpu_diff0, SIGMA_MASK);
    let cpu_blurred0_data = imagef_to_vec(&cpu_blurred0);
    println!("\n[Step 3] blur(diff0) with sigma={:.1}:", SIGMA_MASK);
    println!("  CPU: min={:.6}, max={:.6}",
        cpu_blurred0_data.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_blurred0_data.iter().cloned().fold(0.0f32, f32::max));

    // GPU blur
    let gpu_diff0 = upload_flat(&ctx, &cpu_diff0_data);
    let mut gpu_blurred0 = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();
    let mut gpu_temp = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();

    ctx.kernel.blur(
        &ctx.stream,
        gpu_diff0.ptr() as *const f32,
        gpu_blurred0.ptr() as *mut f32,
        gpu_temp.ptr() as *mut f32,
        width, height, SIGMA_MASK,
    );

    let gpu_blurred0_data = download_planar(&ctx, &gpu_blurred0, width, height);
    println!("  GPU: min={:.6}, max={:.6}",
        gpu_blurred0_data.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_blurred0_data.iter().cloned().fold(0.0f32, f32::max));

    let blur_result = compare_buffers(&cpu_blurred0_data, &gpu_blurred0_data, "blurred0");
    blur_result.print();

    // ===== Step 4: fuzzy_erosion(blurred0) =====
    let mut cpu_eroded = ImageF::new(width, height);
    fuzzy_erosion(&cpu_blurred0, &mut cpu_eroded);
    let cpu_eroded_data = imagef_to_vec(&cpu_eroded);
    println!("\n[Step 4] fuzzy_erosion(blurred0):");
    println!("  CPU: min={:.6}, max={:.6}",
        cpu_eroded_data.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_eroded_data.iter().cloned().fold(0.0f32, f32::max));

    // GPU fuzzy_erosion
    let mut gpu_eroded = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();
    ctx.kernel.fuzzy_erosion(
        &ctx.stream,
        gpu_blurred0.ptr() as *const f32,
        gpu_eroded.ptr() as *mut f32,
        width, height,
    );

    let gpu_eroded_data = download_planar(&ctx, &gpu_eroded, width, height);
    println!("  GPU: min={:.6}, max={:.6}",
        gpu_eroded_data.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_eroded_data.iter().cloned().fold(0.0f32, f32::max));

    let erosion_result = compare_buffers(&cpu_eroded_data, &gpu_eroded_data, "eroded");
    erosion_result.print();

    // Sample value comparison at center
    let center = width / 2;
    let center_idx = center * width + center;
    println!("\n[Sample at center ({}, {})]:", center, center);
    println!("  mask0:    CPU={:.6}", cpu_mask0_data[center_idx]);
    println!("  diff0:    CPU={:.6}", cpu_diff0_data[center_idx]);
    println!("  blurred0: CPU={:.6}, GPU={:.6}", cpu_blurred0_data[center_idx], gpu_blurred0_data[center_idx]);
    println!("  eroded:   CPU={:.6}, GPU={:.6}", cpu_eroded_data[center_idx], gpu_eroded_data[center_idx]);
}

/// Test: Diff precompute kernel
#[test]
fn test_diff_precompute_parity() {
    use butteraugli::mask::diff_precompute;
    use butteraugli::consts::{MASK_MUL, MASK_BIAS};

    let ctx = CudaContext::new();

    println!("\n=== DIFF_PRECOMPUTE PARITY TEST ===");

    for (width, height) in TEST_SIZES {
        println!("\nSize: {}x{}", width, height);
        let size = width * height;

        // Create edge pattern and get mask0 (from combine_channels_for_masking)
        let rgb = gen_edge_v(width, height);
        let linear = srgb_to_linear_planar(&rgb, width, height);
        let xyb = opsin_dynamics_image(&linear, 80.0);
        let ps = separate_frequencies(&xyb);

        let mut cpu_mask0 = ImageF::new(width, height);
        combine_channels_for_masking(
            &[ps.hf[0].clone(), ps.hf[1].clone()],
            &[ps.uhf[0].clone(), ps.uhf[1].clone()],
            &mut cpu_mask0,
        );

        // CPU diff_precompute
        let mut cpu_diff = ImageF::new(width, height);
        diff_precompute(&cpu_mask0, MASK_MUL, MASK_BIAS, &mut cpu_diff);
        let cpu_data = imagef_to_vec(&cpu_diff);

        println!("  CPU diff range: min={:.6}, max={:.6}",
            cpu_data.iter().cloned().fold(f32::INFINITY, f32::min),
            cpu_data.iter().cloned().fold(0.0f32, f32::max));

        // GPU diff_precompute
        let mask0_data = imagef_to_vec(&cpu_mask0);
        let gpu_mask0 = upload_flat(&ctx, &mask0_data);
        let mut gpu_diff = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();

        ctx.kernel.diff_precompute(
            &ctx.stream,
            gpu_mask0.ptr() as *const f32,
            gpu_diff.ptr() as *mut f32,
            size,
        );

        let gpu_data = download_planar(&ctx, &gpu_diff, width, height);

        println!("  GPU diff range: min={:.6}, max={:.6}",
            gpu_data.iter().cloned().fold(f32::INFINITY, f32::min),
            gpu_data.iter().cloned().fold(0.0f32, f32::max));

        let result = compare_buffers(&cpu_data, &gpu_data, "diff_precompute");
        result.print();
    }
}

/// Test: Full frequency separation
#[test]
fn test_frequency_separation_parity() {
    let ctx = CudaContext::new();

    println!("\n=== FREQUENCY SEPARATION PARITY TEST ===");

    for (width, height) in TEST_SIZES {
        println!("\nSize: {}x{}", width, height);
        let size = width * height;

        let rgb = gen_edge_v(width, height);
        let linear = srgb_to_linear_planar(&rgb, width, height);
        let xyb_cpu = opsin_dynamics_image(&linear, 80.0);
        let ps_cpu = separate_frequencies(&xyb_cpu);

        // Test LF band (most affected by blur boundary handling)
        let cpu_lf_y = imagef_to_vec(ps_cpu.lf.plane(1));

        // For GPU, we need to replicate the frequency separation
        // Upload XYB Y channel
        let gpu_xyb_y = upload_planar(&ctx, xyb_cpu.plane(1));
        let mut gpu_lf = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();
        let mut gpu_temp = CuBox::<[f32]>::new_zeroed(size, &ctx.stream).unwrap();

        // LF = blur(src, sigma_lf)
        ctx.kernel.blur(
            &ctx.stream,
            gpu_xyb_y.ptr() as *const f32,
            gpu_lf.ptr() as *mut f32,
            gpu_temp.ptr() as *mut f32,
            width, height, SIGMA_LF,
        );

        let gpu_lf_data = download_planar(&ctx, &gpu_lf, width, height);

        // Note: CPU LF has been scaled by xyb_low_freq_to_vals, GPU LF has not
        // We compare pre-scaling values
        let cpu_lf_prescale = gaussian_blur(xyb_cpu.plane(1), SIGMA_LF);
        let cpu_lf_prescale_data = imagef_to_vec(&cpu_lf_prescale);

        let result = compare_buffers(&cpu_lf_prescale_data, &gpu_lf_data, "lf_y_prescale");
        result.print();

        if !result.is_ok(0.05, 0.05) {
            println!("    ⚠️  LF BLUR DIVERGENCE at {}x{}", width, height);
        }
    }
}

/// Test: Boundary behavior specifically at problem sizes
#[test]
fn test_boundary_behavior() {
    let ctx = CudaContext::new();

    println!("\n=== BOUNDARY BEHAVIOR TEST ===");
    println!("Testing blur boundary handling at problematic sizes...\n");

    // Focus on 32x32 where 97% error occurs
    for size in [16, 32, 48, 64, 128] {
        println!("Size: {}x{}", size, size);

        // Create image with single bright pixel at each corner
        let width = size;
        let height = size;
        let mut data = vec![0.0f32; width * height];

        // Corners
        data[0] = 1.0; // top-left
        data[width - 1] = 1.0; // top-right
        data[(height - 1) * width] = 1.0; // bottom-left
        data[(height - 1) * width + width - 1] = 1.0; // bottom-right

        // Convert to ImageF for CPU
        let mut cpu_img = ImageF::new(width, height);
        for y in 0..height {
            for x in 0..width {
                cpu_img.set(x, y, data[y * width + x]);
            }
        }

        // CPU blur
        let cpu_blurred = gaussian_blur(&cpu_img, SIGMA_MF);
        let cpu_data = imagef_to_vec(&cpu_blurred);

        // GPU blur
        let gpu_src = CuBox::<[f32]>::new(&data, &ctx.stream).unwrap();
        let mut gpu_dst = CuBox::<[f32]>::new_zeroed(width * height, &ctx.stream).unwrap();
        let mut gpu_temp = CuBox::<[f32]>::new_zeroed(width * height, &ctx.stream).unwrap();

        ctx.kernel.blur(
            &ctx.stream,
            gpu_src.ptr() as *const f32,
            gpu_dst.ptr() as *mut f32,
            gpu_temp.ptr() as *mut f32,
            width, height, SIGMA_MF,
        );

        let gpu_data = download_planar(&ctx, &gpu_dst, width, height);

        // Check corner values specifically
        let corners = [
            (0, 0, "TL"),
            (width - 1, 0, "TR"),
            (0, height - 1, "BL"),
            (width - 1, height - 1, "BR"),
        ];

        let mut corner_divergence = false;
        for (x, y, name) in corners {
            let idx = y * width + x;
            let cpu_val = cpu_data[idx];
            let gpu_val = gpu_data[idx];
            let diff = (cpu_val - gpu_val).abs();
            let rel = if cpu_val.abs() > 1e-6 { diff / cpu_val.abs() } else { diff };

            if rel > 0.1 {
                println!("  {} corner: CPU={:.6}, GPU={:.6}, diff={:.2}%",
                    name, cpu_val, gpu_val, rel * 100.0);
                corner_divergence = true;
            }
        }

        if !corner_divergence {
            println!("  ✓ Corner values match within 10%");
        } else {
            println!("  ⚠️  BOUNDARY DIVERGENCE DETECTED");
        }

        // Overall comparison
        let result = compare_buffers(&cpu_data, &gpu_data, "corners_blur");
        println!("  Overall: max_abs={:.6}, max_rel={:.2}%",
            result.max_abs_err, result.max_rel_err * 100.0);
    }
}

/// Test: Full pipeline end-to-end comparison at each size
#[test]
fn test_full_pipeline_by_size() {
    let ctx = CudaContext::new();

    println!("\n=== FULL PIPELINE BY SIZE TEST ===");

    for (width, height) in TEST_SIZES {
        println!("\n{}x{}:", width, height);

        // Test with edge pattern (most sensitive)
        let source = gen_edge_v(width, height);
        let distorted: Vec<u8> = source.iter().map(|&v| v.saturating_add(10)).collect();

        // CPU score
        let cpu_result = compute_butteraugli(&source, &distorted, width, height, &ButteraugliParams::default())
            .expect("CPU butteraugli failed");
        let cpu_score = cpu_result.score;

        // GPU score
        let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        let mut gpu_dst = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        gpu_src.copy_from_cpu(&source, ctx.stream.inner() as _).unwrap();
        gpu_dst.copy_from_cpu(&distorted, ctx.stream.inner() as _).unwrap();
        ctx.stream.sync().unwrap();

        let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();
        let gpu_score = butteraugli.compute(gpu_src.full_view(), gpu_dst.full_view()).unwrap();

        let abs_err = (cpu_score - gpu_score as f64).abs();
        let rel_err = if cpu_score.abs() > 1e-6 { abs_err / cpu_score.abs() } else { abs_err };

        let status = if rel_err < 0.15 { "✓" } else { "✗" };
        println!("  {} edge_v: CPU={:.4}, GPU={:.4}, err={:.2}%",
            status, cpu_score, gpu_score, rel_err * 100.0);

        // Test with gradient
        let source = gen_gradient_h(width, height);
        let distorted: Vec<u8> = source.iter().map(|&v| v.saturating_add(10)).collect();

        let cpu_result = compute_butteraugli(&source, &distorted, width, height, &ButteraugliParams::default())
            .expect("CPU butteraugli failed");
        let cpu_score = cpu_result.score;

        gpu_src.copy_from_cpu(&source, ctx.stream.inner() as _).unwrap();
        gpu_dst.copy_from_cpu(&distorted, ctx.stream.inner() as _).unwrap();
        ctx.stream.sync().unwrap();

        let gpu_score = butteraugli.compute(gpu_src.full_view(), gpu_dst.full_view()).unwrap();

        let abs_err = (cpu_score - gpu_score as f64).abs();
        let rel_err = if cpu_score.abs() > 1e-6 { abs_err / cpu_score.abs() } else { abs_err };

        let status = if rel_err < 0.30 { "✓" } else { "✗" };
        println!("  {} gradient: CPU={:.4}, GPU={:.4}, err={:.2}%",
            status, cpu_score, gpu_score, rel_err * 100.0);
    }
}

/// Test: Kernel size vs image size interaction
#[test]
fn test_kernel_size_interaction() {
    let ctx = CudaContext::new();

    println!("\n=== KERNEL SIZE VS IMAGE SIZE TEST ===");
    println!("Testing if blur kernel size exceeds image dimensions...\n");

    for sigma in [SIGMA_OPSIN, SIGMA_HF, SIGMA_MF, SIGMA_LF, SIGMA_MASK] {
        let kernel = compute_kernel(sigma);
        let kernel_size = kernel.len();
        let half_kernel = kernel_size / 2;

        println!("Sigma {:.3}: kernel_size={}, half={}", sigma, kernel_size, half_kernel);

        for size in [8, 16, 32, 64, 128] {
            let ratio = size as f32 / kernel_size as f32;
            let warning = if ratio < 2.0 { " ⚠️ kernel > half image!" } else { "" };
            println!("  {}x{}: ratio={:.1}{}", size, size, ratio, warning);
        }
    }
}

/// Test: Detailed pipeline comparison showing where GPU diverges from CPU
#[test]
fn test_pipeline_trace() {
    let ctx = CudaContext::new();

    println!("\n=== PIPELINE TRACE TEST ===");
    println!("Tracing where GPU diverges from CPU at 32x32...\n");

    let size = 32;
    let width = size;
    let height = size;

    // Create edge pattern
    let source = gen_edge_v(width, height);
    let distorted: Vec<u8> = source.iter().map(|&v| v.saturating_add(10)).collect();

    // Also get CPU intermediate values for comparison
    let linear1 = srgb_to_linear_planar(&source, width, height);
    let linear2 = srgb_to_linear_planar(&distorted, width, height);
    let xyb1 = opsin_dynamics_image(&linear1, 80.0);
    let xyb2 = opsin_dynamics_image(&linear2, 80.0);
    let ps1 = separate_frequencies(&xyb1);
    let ps2 = separate_frequencies(&xyb2);

    // === CPU Reference ===
    let cpu_result = compute_butteraugli(&source, &distorted, width, height, &ButteraugliParams::default())
        .expect("CPU butteraugli failed");
    println!("CPU final score: {:.4}", cpu_result.score);
    let cpu_diffmap = cpu_result.diffmap.as_ref().expect("diffmap not returned");
    let cpu_diffmap_vec = imagef_to_vec(cpu_diffmap);
    println!("CPU diffmap max: {:.4}", cpu_diffmap_vec.iter().cloned().fold(0.0f32, f32::max));

    // === GPU Pipeline ===
    let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    let mut gpu_dst = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    gpu_src.copy_from_cpu(&source, ctx.stream.inner() as _).unwrap();
    gpu_dst.copy_from_cpu(&distorted, ctx.stream.inner() as _).unwrap();
    ctx.stream.sync().unwrap();

    let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();

    // Run without multi-scale to isolate base pipeline
    let gpu_score = butteraugli.compute_with_options(gpu_src.full_view(), gpu_dst.full_view(), false).unwrap();
    println!("\nGPU final score (no MS): {:.4}", gpu_score);

    // Get GPU diffmap
    let gpu_diffmap = butteraugli.get_diffmap();
    let gpu_diffmap_max = gpu_diffmap.iter().cloned().fold(0.0f32, f32::max);
    println!("GPU diffmap max: {:.4}", gpu_diffmap_max);

    // Compare diffmap values at key positions
    println!("\nDiffmap comparison at key positions:");
    let center = size / 2;
    let edge_x = center;

    let cpu_center = cpu_diffmap_vec[center * width + center];
    let gpu_center = gpu_diffmap[center * width + center];
    println!("  center ({},{}): CPU={:.6}, GPU={:.6}, ratio={:.2}",
        center, center, cpu_center, gpu_center, gpu_center / cpu_center.max(1e-6));

    let cpu_edge = cpu_diffmap_vec[center * width + edge_x];
    let gpu_edge = gpu_diffmap[center * width + edge_x];
    println!("  edge ({},{}): CPU={:.6}, GPU={:.6}, ratio={:.2}",
        edge_x, center, cpu_edge, gpu_edge, gpu_edge / cpu_edge.max(1e-6));

    // Find the pixel with maximum GPU value
    let max_idx = gpu_diffmap.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let max_x = max_idx % width;
    let max_y = max_idx / width;
    println!("\nMax GPU value at ({}, {}):", max_x, max_y);
    println!("  CPU: {:.6}", cpu_diffmap_vec[max_idx]);
    println!("  GPU: {:.6}", gpu_diffmap[max_idx]);

    // Look at block_diff_ac values for all channels
    let gpu_ac_x = butteraugli.get_block_diff_ac(0);
    let gpu_ac_y = butteraugli.get_block_diff_ac(1);
    let gpu_ac_b = butteraugli.get_block_diff_ac(2);
    println!("\nGPU block_diff_ac at max position ({}, {}):", max_x, max_y);
    println!("  X={:.6}, Y={:.6}, B={:.6}",
        gpu_ac_x[max_idx], gpu_ac_y[max_idx], gpu_ac_b[max_idx]);
    println!("  sum={:.6}", gpu_ac_x[max_idx] + gpu_ac_y[max_idx] + gpu_ac_b[max_idx]);

    println!("\nGPU block_diff_ac max values:");
    println!("  X_max={:.6}, Y_max={:.6}, B_max={:.6}",
        gpu_ac_x.iter().cloned().fold(0.0f32, f32::max),
        gpu_ac_y.iter().cloned().fold(0.0f32, f32::max),
        gpu_ac_b.iter().cloned().fold(0.0f32, f32::max));

    // Look at block_diff_dc values
    let gpu_dc_x = butteraugli.get_block_diff_dc(0);
    let gpu_dc_y = butteraugli.get_block_diff_dc(1);
    let gpu_dc_b = butteraugli.get_block_diff_dc(2);
    println!("\nGPU block_diff_dc at max position ({}, {}):", max_x, max_y);
    println!("  X={:.6}, Y={:.6}, B={:.6}",
        gpu_dc_x[max_idx], gpu_dc_y[max_idx], gpu_dc_b[max_idx]);

    // Look at mask values
    let gpu_mask = butteraugli.get_mask();
    let gpu_mask_at_max = gpu_mask[max_idx];
    let gpu_mask_max = gpu_mask.iter().cloned().fold(0.0f32, f32::max);
    let gpu_mask_min = gpu_mask.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("\nGPU mask:");
    println!("  at_max_pos={:.6}, min={:.6}, max={:.6}", gpu_mask_at_max, gpu_mask_min, gpu_mask_max);

    // Manually compute what diffmap SHOULD be at max position
    let ac_sum = gpu_ac_x[max_idx] + gpu_ac_y[max_idx] + gpu_ac_b[max_idx];
    let dc_sum = gpu_dc_x[max_idx] + gpu_dc_y[max_idx] + gpu_dc_b[max_idx];

    // mask_y formula
    let mask_val = gpu_mask[max_idx];
    let offset_ac = 0.829591754942f32;
    let scaler_ac = 0.451936922203f32;
    let mul_ac = 2.5485944793f32;
    let norm_ac = 1.0f32 / (0.79079917404f32 * 17.83f32);
    let c_ac = mul_ac / (scaler_ac * mask_val + offset_ac);
    let retval_ac = (1.0 + c_ac) * norm_ac;
    let maskval_ac = retval_ac * retval_ac;

    // mask_dc_y formula
    let offset_dc = 0.20025578522f32;
    let scaler_dc = 3.87449418804f32;
    let mul_dc = 0.505054525019f32;
    let norm_dc = 1.0f32 / (0.79079917404f32 * 17.83f32);
    let c_dc = mul_dc / (scaler_dc * mask_val + offset_dc);
    let retval_dc = (1.0 + c_dc) * norm_dc;
    let maskval_dc = retval_dc * retval_dc;

    let computed_diffmap = (maskval_ac * ac_sum + maskval_dc * dc_sum).sqrt();
    println!("\nManual diffmap computation at ({}, {}):", max_x, max_y);
    println!("  maskval_ac={:.6}, maskval_dc={:.6}", maskval_ac, maskval_dc);
    println!("  ac_sum={:.6}, dc_sum={:.6}", ac_sum, dc_sum);
    println!("  computed_diffmap={:.6}", computed_diffmap);
    println!("  actual_gpu_diffmap={:.6}", gpu_diffmap[max_idx]);

    // Compare CPU LF values at max position
    println!("\nCPU LF values at ({}, {}):", max_x, max_y);
    let cpu_lf1_x = ps1.lf.plane(0).get(max_x, max_y);
    let cpu_lf1_y = ps1.lf.plane(1).get(max_x, max_y);
    let cpu_lf1_b = ps1.lf.plane(2).get(max_x, max_y);
    let cpu_lf2_x = ps2.lf.plane(0).get(max_x, max_y);
    let cpu_lf2_y = ps2.lf.plane(1).get(max_x, max_y);
    let cpu_lf2_b = ps2.lf.plane(2).get(max_x, max_y);
    println!("  LF1: X={:.6}, Y={:.6}, B={:.6}", cpu_lf1_x, cpu_lf1_y, cpu_lf1_b);
    println!("  LF2: X={:.6}, Y={:.6}, B={:.6}", cpu_lf2_x, cpu_lf2_y, cpu_lf2_b);
    let cpu_lf_diff_x = cpu_lf1_x - cpu_lf2_x;
    let cpu_lf_diff_y = cpu_lf1_y - cpu_lf2_y;
    let cpu_lf_diff_b = cpu_lf1_b - cpu_lf2_b;
    println!("  Diff: X={:.6}, Y={:.6}, B={:.6}", cpu_lf_diff_x, cpu_lf_diff_y, cpu_lf_diff_b);

    // CPU expected block_diff_dc
    const WMUL_LF_X: f32 = 29.2353797994;
    const WMUL_LF_Y: f32 = 0.844626970982;
    const WMUL_LF_B: f32 = 0.703646627719;
    let cpu_dc_x = WMUL_LF_X * cpu_lf_diff_x * cpu_lf_diff_x;
    let cpu_dc_y = WMUL_LF_Y * cpu_lf_diff_y * cpu_lf_diff_y;
    let cpu_dc_b = WMUL_LF_B * cpu_lf_diff_b * cpu_lf_diff_b;
    println!("\nCPU expected block_diff_dc at ({}, {}):", max_x, max_y);
    println!("  X={:.6}, Y={:.6}, B={:.6}", cpu_dc_x, cpu_dc_y, cpu_dc_b);
    println!("  sum={:.6}", cpu_dc_x + cpu_dc_y + cpu_dc_b);

    // Compare with GPU
    println!("\nGPU block_diff_dc at ({}, {}):", max_x, max_y);
    println!("  X={:.6}, Y={:.6}, B={:.6}",
        gpu_dc_x[max_idx], gpu_dc_y[max_idx], gpu_dc_b[max_idx]);
    println!("  sum={:.6}", gpu_dc_x[max_idx] + gpu_dc_y[max_idx] + gpu_dc_b[max_idx]);

    // Compute ratio
    println!("\nGPU/CPU DC ratio:");
    if cpu_dc_x.abs() > 1e-10 {
        println!("  X: {:.2}", gpu_dc_x[max_idx] / cpu_dc_x);
    }
    if cpu_dc_y.abs() > 1e-10 {
        println!("  Y: {:.2}", gpu_dc_y[max_idx] / cpu_dc_y);
    }
    if cpu_dc_b.abs() > 1e-10 {
        println!("  B: {:.2}", gpu_dc_b[max_idx] / cpu_dc_b);
    }

    // Compute CPU mask at max position
    let mut cpu_mask0 = ImageF::new(width, height);
    let mut cpu_mask1 = ImageF::new(width, height);
    combine_channels_for_masking(&[ps1.hf[0].clone(), ps1.hf[1].clone()],
                                  &[ps1.uhf[0].clone(), ps1.uhf[1].clone()],
                                  &mut cpu_mask0);
    combine_channels_for_masking(&[ps2.hf[0].clone(), ps2.hf[1].clone()],
                                  &[ps2.uhf[0].clone(), ps2.uhf[1].clone()],
                                  &mut cpu_mask1);
    let cpu_mask_final = compute_mask(&cpu_mask0, &cpu_mask1, None);
    let cpu_mask_at_max = cpu_mask_final.get(max_x, max_y);
    let cpu_mask_max = (0..width*height).map(|i| cpu_mask_final.get(i % width, i / width)).fold(0.0f32, f32::max);
    let cpu_mask_min = (0..width*height).map(|i| cpu_mask_final.get(i % width, i / width)).fold(f32::INFINITY, f32::min);
    println!("\nCPU mask:");
    println!("  at_max_pos={:.6}, min={:.6}, max={:.6}", cpu_mask_at_max, cpu_mask_min, cpu_mask_max);
    println!("  GPU mask at same pos: {:.6}", gpu_mask[max_idx]);
    println!("  GPU/CPU mask ratio: {:.2}", gpu_mask[max_idx] / cpu_mask_at_max);
}

/// Test: Multi-scale vs single-scale to isolate the 2x error
#[test]
fn test_multiscale_isolation() {
    let ctx = CudaContext::new();

    println!("\n=== MULTI-SCALE ISOLATION TEST ===");
    println!("Comparing with and without multi-scale processing...\n");

    for (width, height) in TEST_SIZES {
        println!("Size: {}x{}", width, height);

        // Edge pattern test
        let source = gen_edge_v(width, height);
        let distorted: Vec<u8> = source.iter().map(|&v| v.saturating_add(10)).collect();

        // CPU score (always with multi-scale)
        let cpu_result = compute_butteraugli(&source, &distorted, width, height, &ButteraugliParams::default())
            .expect("CPU butteraugli failed");
        let cpu_score = cpu_result.score;

        // GPU setup
        let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        let mut gpu_dst = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        gpu_src.copy_from_cpu(&source, ctx.stream.inner() as _).unwrap();
        gpu_dst.copy_from_cpu(&distorted, ctx.stream.inner() as _).unwrap();
        ctx.stream.sync().unwrap();

        let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();

        // GPU WITHOUT multi-scale
        let gpu_no_ms = butteraugli.compute_with_options(gpu_src.full_view(), gpu_dst.full_view(), false).unwrap();

        // GPU WITH multi-scale
        let gpu_with_ms = butteraugli.compute_with_options(gpu_src.full_view(), gpu_dst.full_view(), true).unwrap();

        let err_no_ms = (cpu_score - gpu_no_ms as f64).abs() / cpu_score.max(0.001) * 100.0;
        let err_with_ms = (cpu_score - gpu_with_ms as f64).abs() / cpu_score.max(0.001) * 100.0;

        println!("  CPU:         {:.4}", cpu_score);
        println!("  GPU no MS:   {:.4} (err: {:.1}%)", gpu_no_ms, err_no_ms);
        println!("  GPU with MS: {:.4} (err: {:.1}%)", gpu_with_ms, err_with_ms);

        // Analysis
        if err_no_ms < 15.0 && err_with_ms > 50.0 {
            println!("  ⚠️  Issue is in MULTI-SCALE combination");
        } else if err_no_ms > 50.0 && err_with_ms < 15.0 {
            println!("  ℹ️  Multi-scale COMPENSATES for single-scale error");
        } else if err_no_ms > 50.0 && err_with_ms > 50.0 {
            println!("  ⚠️  Issue is in BASE PIPELINE");
        } else {
            println!("  ✓ Both within 15%");
        }
        println!();
    }
}

/// Test: Compare GPU vs CPU frequency bands at problematic position
#[test]
fn test_frequency_band_comparison() {
    let ctx = CudaContext::new();

    println!("\n=== FREQUENCY BAND COMPARISON TEST ===");
    println!("Comparing GPU vs CPU HF/UHF bands at 32x32...\n");

    let width = 32;
    let height = 32;
    let size = width * height;

    // Create edge pattern
    let source = gen_edge_v(width, height);
    let distorted: Vec<u8> = source.iter().map(|&v| v.saturating_add(10)).collect();

    // CPU pipeline
    let linear1 = srgb_to_linear_planar(&source, width, height);
    let linear2 = srgb_to_linear_planar(&distorted, width, height);
    let xyb1_cpu = opsin_dynamics_image(&linear1, 80.0);
    let xyb2_cpu = opsin_dynamics_image(&linear2, 80.0);
    let ps1 = separate_frequencies(&xyb1_cpu);
    let ps2 = separate_frequencies(&xyb2_cpu);

    // GPU pipeline
    let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    let mut gpu_dst = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    gpu_src.copy_from_cpu(&source, ctx.stream.inner() as _).unwrap();
    gpu_dst.copy_from_cpu(&distorted, ctx.stream.inner() as _).unwrap();
    ctx.stream.sync().unwrap();

    let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();

    // Run pipeline without multiscale to get frequency bands
    let _gpu_score = butteraugli.compute_with_options(gpu_src.full_view(), gpu_dst.full_view(), false).unwrap();

    // Get GPU frequency bands
    let gpu_uhf1_x = butteraugli.get_freq1(0, 0); // UHF X
    let gpu_uhf1_y = butteraugli.get_freq1(0, 1); // UHF Y
    let gpu_hf1_x = butteraugli.get_freq1(1, 0);  // HF X
    let gpu_hf1_y = butteraugli.get_freq1(1, 1);  // HF Y

    // Get CPU frequency bands
    let cpu_uhf1_x = imagef_to_vec(&ps1.uhf[0]);
    let cpu_uhf1_y = imagef_to_vec(&ps1.uhf[1]);
    let cpu_hf1_x = imagef_to_vec(&ps1.hf[0]);
    let cpu_hf1_y = imagef_to_vec(&ps1.hf[1]);

    // Find position with max GPU diffmap
    let gpu_diffmap = butteraugli.get_diffmap();
    let max_idx = gpu_diffmap.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let max_x = max_idx % width;
    let max_y = max_idx / width;

    println!("Position with max GPU diffmap: ({}, {})", max_x, max_y);
    println!();

    // Compare UHF X
    println!("=== UHF X ===");
    println!("CPU range: min={:.6}, max={:.6}",
        cpu_uhf1_x.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_uhf1_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("GPU range: min={:.6}, max={:.6}",
        gpu_uhf1_x.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_uhf1_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("At max_pos: CPU={:.6}, GPU={:.6}", cpu_uhf1_x[max_idx], gpu_uhf1_x[max_idx]);
    let uhf_x_result = compare_buffers(&cpu_uhf1_x, &gpu_uhf1_x, "UHF_X");
    uhf_x_result.print();

    // Compare UHF Y
    println!("\n=== UHF Y ===");
    println!("CPU range: min={:.6}, max={:.6}",
        cpu_uhf1_y.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_uhf1_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("GPU range: min={:.6}, max={:.6}",
        gpu_uhf1_y.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_uhf1_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("At max_pos: CPU={:.6}, GPU={:.6}", cpu_uhf1_y[max_idx], gpu_uhf1_y[max_idx]);
    let uhf_y_result = compare_buffers(&cpu_uhf1_y, &gpu_uhf1_y, "UHF_Y");
    uhf_y_result.print();

    // Compare HF X
    println!("\n=== HF X ===");
    println!("CPU range: min={:.6}, max={:.6}",
        cpu_hf1_x.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_hf1_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("GPU range: min={:.6}, max={:.6}",
        gpu_hf1_x.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_hf1_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("At max_pos: CPU={:.6}, GPU={:.6}", cpu_hf1_x[max_idx], gpu_hf1_x[max_idx]);
    let hf_x_result = compare_buffers(&cpu_hf1_x, &gpu_hf1_x, "HF_X");
    hf_x_result.print();

    // Compare HF Y
    println!("\n=== HF Y ===");
    println!("CPU range: min={:.6}, max={:.6}",
        cpu_hf1_y.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_hf1_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("GPU range: min={:.6}, max={:.6}",
        gpu_hf1_y.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_hf1_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("At max_pos: CPU={:.6}, GPU={:.6}", cpu_hf1_y[max_idx], gpu_hf1_y[max_idx]);
    let hf_y_result = compare_buffers(&cpu_hf1_y, &gpu_hf1_y, "HF_Y");
    hf_y_result.print();

    // Now compute what GPU mask would be with GPU's frequency bands
    // Formula: sqrt((uhf_x + hf_x)² * 2.5² + (uhf_y * 0.4 + hf_y * 0.4)²)
    let gpu_xdiff = (gpu_uhf1_x[max_idx] + gpu_hf1_x[max_idx]) * 2.5;
    let gpu_ydiff = gpu_uhf1_y[max_idx] * 0.4 + gpu_hf1_y[max_idx] * 0.4;
    let gpu_mask_raw = (gpu_xdiff * gpu_xdiff + gpu_ydiff * gpu_ydiff).sqrt();

    let cpu_xdiff = (cpu_uhf1_x[max_idx] + cpu_hf1_x[max_idx]) * 2.5;
    let cpu_ydiff = cpu_uhf1_y[max_idx] * 0.4 + cpu_hf1_y[max_idx] * 0.4;
    let cpu_mask_raw = (cpu_xdiff * cpu_xdiff + cpu_ydiff * cpu_ydiff).sqrt();

    println!("\n=== Mask Computation at ({}, {}) ===", max_x, max_y);
    println!("GPU: xdiff={:.6}, ydiff={:.6}, mask_raw={:.6}", gpu_xdiff, gpu_ydiff, gpu_mask_raw);
    println!("CPU: xdiff={:.6}, ydiff={:.6}, mask_raw={:.6}", cpu_xdiff, cpu_ydiff, cpu_mask_raw);

    // Get actual GPU mask value
    let gpu_mask = butteraugli.get_mask();
    println!("\nActual GPU mask at max_pos: {:.6}", gpu_mask[max_idx]);

    // Also compare XYB values
    println!("\n=== XYB Comparison ===");
    let gpu_xyb1_x = butteraugli.get_xyb1(0);
    let gpu_xyb1_y = butteraugli.get_xyb1(1);
    let cpu_xyb1_x = imagef_to_vec(xyb1_cpu.plane(0));
    let cpu_xyb1_y = imagef_to_vec(xyb1_cpu.plane(1));

    println!("XYB1 X range: CPU=[{:.6}, {:.6}], GPU=[{:.6}, {:.6}]",
        cpu_xyb1_x.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_xyb1_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        gpu_xyb1_x.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_xyb1_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    let xyb_x_result = compare_buffers(&cpu_xyb1_x, &gpu_xyb1_x, "XYB1_X");
    xyb_x_result.print();

    println!("XYB1 Y range: CPU=[{:.6}, {:.6}], GPU=[{:.6}, {:.6}]",
        cpu_xyb1_y.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_xyb1_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        gpu_xyb1_y.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_xyb1_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    let xyb_y_result = compare_buffers(&cpu_xyb1_y, &gpu_xyb1_y, "XYB1_Y");
    xyb_y_result.print();
}

/// Comprehensive summary test
#[test]
fn test_parity_summary() {
    let ctx = CudaContext::new();

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║            BUTTERAUGLI-CUDA STAGE PARITY SUMMARY                  ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");

    let mut failures = Vec::new();

    for (width, height) in TEST_SIZES {
        // Edge pattern test
        let source = gen_edge_v(width, height);
        let distorted: Vec<u8> = source.iter().map(|&v| v.saturating_add(10)).collect();

        let cpu_result = compute_butteraugli(&source, &distorted, width, height, &ButteraugliParams::default())
            .expect("CPU failed");

        let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        let mut gpu_dst = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
        gpu_src.copy_from_cpu(&source, ctx.stream.inner() as _).unwrap();
        gpu_dst.copy_from_cpu(&distorted, ctx.stream.inner() as _).unwrap();
        ctx.stream.sync().unwrap();

        let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();
        let gpu_score = butteraugli.compute(gpu_src.full_view(), gpu_dst.full_view()).unwrap();

        let rel_err = (cpu_result.score - gpu_score as f64).abs() / cpu_result.score.max(0.001);
        let status = if rel_err < 0.15 { "✓" } else { "✗" };

        println!("║ {} {:3}x{:3} edge:     CPU={:7.4} GPU={:7.4} err={:6.2}%        ║",
            status, width, height, cpu_result.score, gpu_score, rel_err * 100.0);

        if rel_err >= 0.15 {
            failures.push(format!("{}x{} edge: {:.1}% error", width, height, rel_err * 100.0));
        }
    }

    println!("╠═══════════════════════════════════════════════════════════════════╣");

    if failures.is_empty() {
        println!("║                    ALL TESTS PASSED                               ║");
    } else {
        println!("║ FAILURES:                                                         ║");
        for f in &failures {
            println!("║   - {}                                              ║", f);
        }
    }

    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Don't assert - we want to see the results even with failures
}
