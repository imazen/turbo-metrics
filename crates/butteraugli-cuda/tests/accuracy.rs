//! Tests comparing GPU Butteraugli implementation against CPU butteraugli crate
//! using real JPEG compression artifacts.
//!
//! Uses the same test images as dssim-cuda (symlinked from test_data/).

use butteraugli::{butteraugli, ButteraugliParams};
use butteraugli_cuda::Butteraugli;
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut, C};
use cudarse_npp::set_stream;
use imgref::ImgVec;
use rgb::RGB8;
use std::path::Path;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

/// CUDA context state (initialized once)
struct CudaContext {
    stream: CuStream,
}

impl CudaContext {
    fn new() -> Self {
        cudarse_driver::init_cuda_and_primary_ctx().expect("Failed to initialize CUDA");
        let stream = CuStream::new().expect("Failed to create CUDA stream");
        set_stream(stream.inner() as _).expect("Failed to set NPP stream");
        Self { stream }
    }
}

/// Load an image from a file and return (width, height, RGB data)
fn load_image(path: &Path) -> (usize, usize, Vec<u8>) {
    use zune_image::traits::DecoderTrait;

    let data = std::fs::read(path).expect("Failed to read image file");

    // Detect format and decode
    let (width, height, raw_data) = if path.extension().map_or(false, |e| e == "png") {
        let mut decoder = zune_image::codecs::png::PngDecoder::new(&data);
        let result = decoder.decode().expect("Failed to decode PNG");
        let (w, h) = decoder.dimensions().unwrap();
        let raw = match result {
            zune_core::result::DecodingResult::U8(v) => v,
            zune_core::result::DecodingResult::U16(v) => v.iter().map(|&x| (x >> 8) as u8).collect(),
            _ => panic!("Unsupported pixel format"),
        };
        (w, h, raw)
    } else if path.extension().map_or(false, |e| e == "jpg" || e == "jpeg") {
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut decoder = zune_image::codecs::jpeg::JpegDecoder::new_with_options(&data, opts);
        let raw = decoder.decode().expect("Failed to decode JPEG");
        let (w, h) = decoder.dimensions().unwrap();
        (w, h, raw)
    } else {
        panic!("Unsupported image format: {:?}", path);
    };

    // Convert to RGB if needed
    let rgb_data = if raw_data.len() == width * height * 4 {
        // RGBA -> RGB
        raw_data
            .chunks_exact(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect()
    } else if raw_data.len() == width * height * 3 {
        raw_data
    } else if raw_data.len() == width * height {
        // Grayscale -> RGB
        raw_data.iter().flat_map(|&g| [g, g, g]).collect()
    } else {
        panic!(
            "Unexpected image data size: {} for {}x{}",
            raw_data.len(),
            width,
            height
        );
    };

    (width, height, rgb_data)
}

/// Convert RGB u8 slice to RGB8 vec for butteraugli
fn to_rgb8_vec(data: &[u8]) -> Vec<RGB8> {
    data.chunks_exact(3)
        .map(|rgb| RGB8::new(rgb[0], rgb[1], rgb[2]))
        .collect()
}

/// Compute Butteraugli score using CPU implementation
fn compute_cpu_score(source: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    let src_rgb = to_rgb8_vec(source);
    let dst_rgb = to_rgb8_vec(distorted);
    let src_img = ImgVec::new(src_rgb, width, height);
    let dst_img = ImgVec::new(dst_rgb, width, height);
    butteraugli(src_img.as_ref(), dst_img.as_ref(), &params)
        .expect("CPU butteraugli failed")
        .score
}

/// Compute Butteraugli score using GPU implementation
fn compute_gpu_score(
    ctx: &CudaContext,
    source: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> f32 {
    // Allocate GPU images
    let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32)
        .expect("Failed to allocate GPU source image");
    let mut gpu_dst = gpu_src
        .malloc_same_size()
        .expect("Failed to allocate GPU distorted image");

    // Upload to GPU
    gpu_src
        .copy_from_cpu(source, ctx.stream.inner() as _)
        .expect("Failed to upload source");
    gpu_dst
        .copy_from_cpu(distorted, ctx.stream.inner() as _)
        .expect("Failed to upload distorted");
    ctx.stream.sync().expect("Failed to sync after upload");

    // Create Butteraugli instance and compute
    let mut butteraugli =
        Butteraugli::new(width as u32, height as u32).expect("Failed to create Butteraugli instance");

    butteraugli
        .compute(gpu_src.full_view(), gpu_dst.full_view())
        .expect("Failed to compute Butteraugli score")
}

/// Test helper for comparing images from files
fn test_image_pair(ref_path: &str, dis_path: &str, tolerance: f64) {
    let ctx = CudaContext::new();

    let test_data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");

    let (ref_w, ref_h, ref_data) = load_image(&test_data_dir.join(ref_path));
    let (dis_w, dis_h, dis_data) = load_image(&test_data_dir.join(dis_path));

    assert_eq!(
        (ref_w, ref_h),
        (dis_w, dis_h),
        "Image dimensions must match"
    );

    let width = ref_w;
    let height = ref_h;

    // CPU Butteraugli
    let cpu_score = compute_cpu_score(&ref_data, &dis_data, width, height);

    // GPU Butteraugli
    let gpu_score = compute_gpu_score(&ctx, &ref_data, &dis_data, width, height);

    println!(
        "{} vs {}: CPU={:.6}, GPU={:.6}",
        ref_path, dis_path, cpu_score, gpu_score
    );

    // Check relative tolerance
    let rel_error = if cpu_score > 0.0 {
        (gpu_score as f64 - cpu_score).abs() / cpu_score
    } else {
        (gpu_score as f64 - cpu_score).abs()
    };

    assert!(
        rel_error < tolerance,
        "GPU Butteraugli {gpu_score:.6} differs from CPU {cpu_score:.6} by {:.2}% (tolerance: {:.2}%)",
        rel_error * 100.0,
        tolerance * 100.0
    );
}

// =============================================================================
// Real JPEG artifact tests
// =============================================================================

#[test]
fn test_jpeg_quality_q90() {
    // Q90 is very high quality, should have low Butteraugli score
    // Actual error: ~0.97%, tolerance allows for GPU/CPU float differences
    test_image_pair("source.png", "q90.jpg", 0.02); // 2% tolerance
}

#[test]
fn test_jpeg_quality_q70() {
    // Q70 is good quality
    // Actual error: ~1.24%, tolerance allows for GPU/CPU float differences
    test_image_pair("source.png", "q70.jpg", 0.02); // 2% tolerance
}

#[test]
fn test_jpeg_quality_q45() {
    // Q45 is medium quality
    // Actual error: ~0.12%, best parity of all quality levels
    test_image_pair("source.png", "q45.jpg", 0.01); // 1% tolerance
}

#[test]
fn test_jpeg_quality_q20() {
    // Q20 is low quality, higher Butteraugli score
    // Actual error: ~0.88%, tolerance allows for GPU/CPU float differences
    test_image_pair("source.png", "q20.jpg", 0.02); // 2% tolerance
}

/// Per-stage comparison test to identify divergence source
#[test]
fn test_jpeg_stage_comparison() {
    use butteraugli::opsin::{srgb_to_linear, opsin_dynamics_image};
    use butteraugli::psycho::separate_frequencies;
    use butteraugli::image::{Image3F, ImageF};

    let ctx = CudaContext::new();
    let test_data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");

    let (width, height, ref_data) = load_image(&test_data_dir.join("source.png"));
    let (_, _, dis_data) = load_image(&test_data_dir.join("q70.jpg"));

    println!("\n=== JPEG Stage Comparison ({}x{}, Q70) ===\n", width, height);

    // CPU pipeline
    fn to_linear_planar(rgb: &[u8], width: usize, height: usize) -> Image3F {
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

    fn imagef_to_vec(img: &ImageF) -> Vec<f32> {
        let mut result = Vec::with_capacity(img.width() * img.height());
        for y in 0..img.height() {
            result.extend_from_slice(img.row(y));
        }
        result
    }

    let linear1 = to_linear_planar(&ref_data, width, height);
    let linear2 = to_linear_planar(&dis_data, width, height);
    let xyb1_cpu = opsin_dynamics_image(&linear1, 80.0);
    let xyb2_cpu = opsin_dynamics_image(&linear2, 80.0);
    let ps1 = separate_frequencies(&xyb1_cpu);
    let _ps2 = separate_frequencies(&xyb2_cpu);

    // GPU pipeline
    let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    let mut gpu_dst = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    gpu_src.copy_from_cpu(&ref_data, ctx.stream.inner() as _).unwrap();
    gpu_dst.copy_from_cpu(&dis_data, ctx.stream.inner() as _).unwrap();
    ctx.stream.sync().unwrap();

    let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();
    // Use compute_with_options(false) to disable multi-scale so xyb1 isn't overwritten
    let gpu_score = butteraugli.compute_with_options(gpu_src.full_view(), gpu_dst.full_view(), false).unwrap();

    // Get CPU score
    let cpu_score = compute_cpu_score(&ref_data, &dis_data, width, height);

    println!("Final scores: CPU={:.6}, GPU={:.6}, error={:.2}%\n",
        cpu_score, gpu_score, (gpu_score as f64 - cpu_score).abs() / cpu_score * 100.0);

    // Compare XYB values
    let gpu_xyb1_x = butteraugli.get_xyb1(0);
    let gpu_xyb1_y = butteraugli.get_xyb1(1);
    let gpu_xyb1_b = butteraugli.get_xyb1(2);
    let cpu_xyb1_x = imagef_to_vec(xyb1_cpu.plane(0));
    let cpu_xyb1_y = imagef_to_vec(xyb1_cpu.plane(1));
    let cpu_xyb1_b = imagef_to_vec(xyb1_cpu.plane(2));

    fn compare_max_rel_err(cpu: &[f32], gpu: &[f32]) -> f32 {
        cpu.iter().zip(gpu.iter())
            .map(|(&c, &g)| if c.abs() > 1e-6 { (c - g).abs() / c.abs() } else { (c - g).abs() })
            .fold(0.0f32, f32::max)
    }

    fn compare_rmse(cpu: &[f32], gpu: &[f32]) -> f32 {
        let sum_sq: f64 = cpu.iter().zip(gpu.iter())
            .map(|(&c, &g)| ((c - g) as f64).powi(2))
            .sum();
        (sum_sq / cpu.len() as f64).sqrt() as f32
    }

    fn range_str(v: &[f32]) -> String {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        format!("[{:.4}, {:.4}]", min, max)
    }

    println!("Stage          | CPU Range              | GPU Range              | RMSE");
    println!("---------------|------------------------|------------------------|--------");

    let xyb_x_rmse = compare_rmse(&cpu_xyb1_x, &gpu_xyb1_x);
    println!("XYB1 X         | {:22} | {:22} | {:.6}", range_str(&cpu_xyb1_x), range_str(&gpu_xyb1_x), xyb_x_rmse);

    let xyb_y_rmse = compare_rmse(&cpu_xyb1_y, &gpu_xyb1_y);
    println!("XYB1 Y         | {:22} | {:22} | {:.6}", range_str(&cpu_xyb1_y), range_str(&gpu_xyb1_y), xyb_y_rmse);

    let xyb_b_rmse = compare_rmse(&cpu_xyb1_b, &gpu_xyb1_b);
    println!("XYB1 B         | {:22} | {:22} | {:.6}", range_str(&cpu_xyb1_b), range_str(&gpu_xyb1_b), xyb_b_rmse);

    // Compare frequency bands
    let gpu_uhf1_x = butteraugli.get_freq1(0, 0);
    let gpu_uhf1_y = butteraugli.get_freq1(0, 1);
    let gpu_hf1_x = butteraugli.get_freq1(1, 0);
    let gpu_hf1_y = butteraugli.get_freq1(1, 1);
    let gpu_mf1_x = butteraugli.get_freq1(2, 0);
    let gpu_mf1_y = butteraugli.get_freq1(2, 1);

    let cpu_uhf1_x = imagef_to_vec(&ps1.uhf[0]);
    let cpu_uhf1_y = imagef_to_vec(&ps1.uhf[1]);
    let cpu_hf1_x = imagef_to_vec(&ps1.hf[0]);
    let cpu_hf1_y = imagef_to_vec(&ps1.hf[1]);
    let cpu_mf1_x = imagef_to_vec(&ps1.mf[0]);
    let cpu_mf1_y = imagef_to_vec(&ps1.mf[1]);

    let uhf_x_rmse = compare_rmse(&cpu_uhf1_x, &gpu_uhf1_x);
    println!("UHF1 X         | {:22} | {:22} | {:.6}", range_str(&cpu_uhf1_x), range_str(&gpu_uhf1_x), uhf_x_rmse);

    let uhf_y_rmse = compare_rmse(&cpu_uhf1_y, &gpu_uhf1_y);
    println!("UHF1 Y         | {:22} | {:22} | {:.6}", range_str(&cpu_uhf1_y), range_str(&gpu_uhf1_y), uhf_y_rmse);

    let hf_x_rmse = compare_rmse(&cpu_hf1_x, &gpu_hf1_x);
    println!("HF1 X          | {:22} | {:22} | {:.6}", range_str(&cpu_hf1_x), range_str(&gpu_hf1_x), hf_x_rmse);

    let hf_y_rmse = compare_rmse(&cpu_hf1_y, &gpu_hf1_y);
    println!("HF1 Y          | {:22} | {:22} | {:.6}", range_str(&cpu_hf1_y), range_str(&gpu_hf1_y), hf_y_rmse);

    let mf_x_rmse = compare_rmse(&cpu_mf1_x, &gpu_mf1_x);
    println!("MF1 X          | {:22} | {:22} | {:.6}", range_str(&cpu_mf1_x), range_str(&gpu_mf1_x), mf_x_rmse);

    let mf_y_rmse = compare_rmse(&cpu_mf1_y, &gpu_mf1_y);
    println!("MF1 Y          | {:22} | {:22} | {:.6}", range_str(&cpu_mf1_y), range_str(&gpu_mf1_y), mf_y_rmse);

    // Compare block_diff values
    let gpu_ac_x = butteraugli.get_block_diff_ac(0);
    let gpu_ac_y = butteraugli.get_block_diff_ac(1);
    let gpu_ac_b = butteraugli.get_block_diff_ac(2);
    let gpu_dc_x = butteraugli.get_block_diff_dc(0);
    let gpu_dc_y = butteraugli.get_block_diff_dc(1);
    let gpu_dc_b = butteraugli.get_block_diff_dc(2);

    println!("\nBlock diff AC ranges:");
    println!("  X: GPU max={:.6}", gpu_ac_x.iter().cloned().fold(0.0f32, f32::max));
    println!("  Y: GPU max={:.6}", gpu_ac_y.iter().cloned().fold(0.0f32, f32::max));
    println!("  B: GPU max={:.6}", gpu_ac_b.iter().cloned().fold(0.0f32, f32::max));

    println!("\nBlock diff DC ranges:");
    println!("  X: GPU max={:.6}", gpu_dc_x.iter().cloned().fold(0.0f32, f32::max));
    println!("  Y: GPU max={:.6}", gpu_dc_y.iter().cloned().fold(0.0f32, f32::max));
    println!("  B: GPU max={:.6}", gpu_dc_b.iter().cloned().fold(0.0f32, f32::max));

    // Compare mask
    let gpu_mask = butteraugli.get_mask();
    println!("\nMask range: GPU min={:.6}, max={:.6}",
        gpu_mask.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_mask.iter().cloned().fold(0.0f32, f32::max));

    // Diffmap
    let gpu_diffmap = butteraugli.get_diffmap();
    println!("\nDiffmap range: GPU min={:.6}, max={:.6}",
        gpu_diffmap.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_diffmap.iter().cloned().fold(0.0f32, f32::max));

    // Find largest error contributors
    println!("\n=== Error Analysis ===");
    if uhf_y_rmse > 0.001 || hf_y_rmse > 0.001 {
        println!("Significant Y-channel frequency band divergence detected!");
        println!("UHF Y RMSE: {:.6}, HF Y RMSE: {:.6}", uhf_y_rmse, hf_y_rmse);
    }
    if mf_x_rmse > 0.001 || mf_y_rmse > 0.001 {
        println!("Significant MF band divergence detected!");
        println!("MF X RMSE: {:.6}, MF Y RMSE: {:.6}", mf_x_rmse, mf_y_rmse);
    }
}

/// Test Q5 (very low quality) - extreme distortion case
#[test]
fn test_jpeg_quality_q5() {
    // Q5 is very low quality with high Butteraugli scores (~15-20)
    // This tests the GPU implementation's handling of extreme distortion
    test_image_pair("source.png", "q5.jpg", 0.10); // 10% tolerance for extreme cases
}

/// Test Q1 (minimal quality) - extreme distortion case
#[test]
fn test_jpeg_quality_q1() {
    // Q1 is minimal quality with very high Butteraugli scores (~40-60)
    // This tests the GPU implementation's numerical stability at extreme distortion levels
    // Note: If this test fails, it indicates a bug in GPU Butteraugli at high distortion levels
    test_image_pair("source.png", "q1.jpg", 0.15); // 15% tolerance for extreme cases
}

/// Summary test that runs all JPEG quality comparisons
#[test]
fn test_jpeg_quality_summary() {
    let ctx = CudaContext::new();
    let test_data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");

    let (ref_w, ref_h, ref_data) = load_image(&test_data_dir.join("source.png"));

    println!("\n=== Butteraugli JPEG Quality Summary ({}x{}) ===", ref_w, ref_h);
    println!("{:<10} {:>12} {:>12} {:>10}", "Quality", "CPU Score", "GPU Score", "Error %");
    println!("{}", "-".repeat(48));

    for quality in ["q90", "q70", "q45", "q20", "q5", "q1"] {
        let dis_path = format!("{}.jpg", quality);
        if !test_data_dir.join(&dis_path).exists() {
            continue;
        }
        let (_, _, dis_data) = load_image(&test_data_dir.join(&dis_path));

        let cpu_score = compute_cpu_score(&ref_data, &dis_data, ref_w, ref_h);
        let gpu_score = compute_gpu_score(&ctx, &ref_data, &dis_data, ref_w, ref_h);

        let rel_error = if cpu_score > 0.0 {
            (gpu_score as f64 - cpu_score).abs() / cpu_score * 100.0
        } else {
            0.0
        };

        println!(
            "{:<10} {:>12.4} {:>12.4} {:>9.2}%",
            quality, cpu_score, gpu_score, rel_error
        );
    }
    println!();
}

// =============================================================================
// Multi-image context reuse tests (catches state management bugs)
// =============================================================================

/// Test that GPU Butteraugli maintains correct state when computing multiple
/// images in sequence using the same Butteraugli instance.
///
/// This test catches bugs where the GPU returns stale results from previous images
/// when the context is reused. The bug manifests as massive divergence (>50% error)
/// on certain images.
#[test]
fn test_context_reuse_multiple_images() {
    const KODAK_PATH: &str = "/home/lilith/work/codec-corpus/kodak";

    // Skip if Kodak corpus not available
    let kodak_dir = Path::new(KODAK_PATH);
    if !kodak_dir.exists() {
        eprintln!("Skipping test: Kodak corpus not found at {}", KODAK_PATH);
        return;
    }

    let ctx = CudaContext::new();

    // Load Kodak images - Kodak corpus alternates between 768x512 and 512x768
    // We'll filter to only use landscape (768x512) images to avoid dimension mismatches
    let all_images: Vec<_> = (1..=24)
        .filter_map(|i| {
            let name = format!("{}.png", i);
            let path = kodak_dir.join(&name);
            if path.exists() {
                let img = load_image(&path);
                // Only keep landscape images (768x512)
                if img.0 == 768 && img.1 == 512 {
                    Some((name, img))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .take(4)  // Take first 4 matching images
        .collect();

    if all_images.len() < 2 {
        eprintln!("Skipping test: Need at least 2 landscape Kodak images");
        return;
    }

    let (width, height) = (all_images[0].1.0, all_images[0].1.1);

    // Create GPU buffers and Butteraugli instance once
    let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32)
        .expect("Failed to allocate GPU source");
    let mut gpu_dst = gpu_src.malloc_same_size()
        .expect("Failed to allocate GPU distorted");
    let mut butteraugli = Butteraugli::new(width as u32, height as u32)
        .expect("Failed to create Butteraugli");

    println!("\n=== Context Reuse Test ({}x{}) ===", width, height);
    println!("{:<15} {:>12} {:>12} {:>10}", "Image", "CPU Score", "GPU Score", "Error %");
    println!("{}", "-".repeat(55));

    let mut max_error = 0.0f64;

    // Process each image multiple times to catch state bugs
    for iteration in 0..3 {
        for (name, (w, h, ref_data)) in &all_images {
            assert_eq!((*w, *h), (width, height), "Image dimensions must match");

            // Create a "distorted" version by adding noise (simple test)
            let dis_data: Vec<u8> = ref_data.iter()
                .enumerate()
                .map(|(i, &v)| {
                    // Add deterministic noise based on position
                    let noise = ((i % 17) as i16 - 8).clamp(-10, 10);
                    (v as i16 + noise).clamp(0, 255) as u8
                })
                .collect();

            // CPU Butteraugli
            let cpu_score = compute_cpu_score(ref_data, &dis_data, width, height);

            // GPU Butteraugli (reusing buffers and instance)
            gpu_src.copy_from_cpu(ref_data, ctx.stream.inner() as _)
                .expect("Failed to upload source");
            gpu_dst.copy_from_cpu(&dis_data, ctx.stream.inner() as _)
                .expect("Failed to upload distorted");
            ctx.stream.sync().expect("Failed to sync");

            let gpu_score = butteraugli
                .compute(gpu_src.full_view(), gpu_dst.full_view())
                .expect("Failed to compute");

            let rel_error = if cpu_score > 0.0 {
                (gpu_score as f64 - cpu_score).abs() / cpu_score * 100.0
            } else {
                (gpu_score as f64 - cpu_score).abs() * 100.0
            };

            max_error = max_error.max(rel_error);

            println!(
                "{:<15} {:>12.4} {:>12.4} {:>9.2}%",
                format!("{}[{}]", name, iteration), cpu_score, gpu_score, rel_error
            );

            // Fail immediately on massive divergence (indicates stale state bug)
            assert!(
                rel_error < 20.0,
                "CRITICAL: GPU returned stale result! {} iteration {}: GPU={:.4} CPU={:.4} error={:.1}%",
                name, iteration, gpu_score, cpu_score, rel_error
            );
        }
    }

    println!("\nMax error across all iterations: {:.2}%", max_error);
    assert!(
        max_error < 10.0,
        "Overall error too high: {:.2}% (max 10% allowed)",
        max_error
    );
}

/// Test high-distortion images (Q1 JPEG) with context reuse.
/// This specifically targets the bug seen in zenjpeg where GPU returns ~1.0
/// when CPU returns ~60 for extreme distortion.
#[test]
fn test_extreme_distortion_context_reuse() {
    const KODAK_PATH: &str = "/home/lilith/work/codec-corpus/kodak";

    let kodak_dir = Path::new(KODAK_PATH);
    if !kodak_dir.exists() {
        eprintln!("Skipping test: Kodak corpus not found at {}", KODAK_PATH);
        return;
    }

    let ctx = CudaContext::new();

    // Load landscape images (768x512) only
    let images: Vec<_> = (1..=24)
        .filter_map(|i| {
            let name = format!("{}.png", i);
            let path = kodak_dir.join(&name);
            if path.exists() {
                let img = load_image(&path);
                if img.0 == 768 && img.1 == 512 {
                    Some((name, img))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .take(4)
        .collect();

    if images.is_empty() {
        eprintln!("Skipping test: No matching Kodak images found");
        return;
    }

    let (width, height) = (images[0].1.0, images[0].1.1);

    // Create buffers
    let mut gpu_src = Image::<u8, C<3>>::malloc(width as u32, height as u32)
        .expect("Failed to allocate GPU source");
    let mut gpu_dst = gpu_src.malloc_same_size()
        .expect("Failed to allocate GPU distorted");
    let mut butteraugli = Butteraugli::new(width as u32, height as u32)
        .expect("Failed to create Butteraugli");

    println!("\n=== Extreme Distortion Context Reuse Test ({}x{}) ===", width, height);
    println!("{:<20} {:>12} {:>12} {:>10}", "Test", "CPU Score", "GPU Score", "Error %");
    println!("{}", "-".repeat(60));

    // Simulate Q1-like extreme distortion by aggressive quantization
    fn create_extreme_distortion(rgb: &[u8]) -> Vec<u8> {
        rgb.iter()
            .map(|&v| {
                // Quantize to 4 levels (like extreme JPEG)
                let level = v / 64;
                level * 64 + 32
            })
            .collect()
    }

    for (name, (w, h, ref_data)) in &images {
        assert_eq!((*w, *h), (width, height));

        let dis_data = create_extreme_distortion(ref_data);

        // CPU score
        let cpu_score = compute_cpu_score(ref_data, &dis_data, *w, *h);

        // GPU score
        gpu_src.copy_from_cpu(ref_data, ctx.stream.inner() as _).unwrap();
        gpu_dst.copy_from_cpu(&dis_data, ctx.stream.inner() as _).unwrap();
        ctx.stream.sync().unwrap();

        let gpu_score = butteraugli
            .compute(gpu_src.full_view(), gpu_dst.full_view())
            .expect("compute failed");

        let rel_error = if cpu_score > 0.0 {
            (gpu_score as f64 - cpu_score).abs() / cpu_score * 100.0
        } else {
            0.0
        };

        println!(
            "{:<20} {:>12.4} {:>12.4} {:>9.2}%",
            name, cpu_score, gpu_score, rel_error
        );

        // The bug manifests as GPU returning a very low value (~1.0) when CPU
        // returns a high value (~60). Check for this specific failure mode.
        if cpu_score > 30.0 && gpu_score < 5.0 {
            panic!(
                "CRITICAL BUG: GPU returned {} but CPU returned {} for {}. \
                 This indicates GPU is returning stale/wrong buffer data.",
                gpu_score, cpu_score, name
            );
        }

        assert!(
            rel_error < 15.0,
            "Error too high for {}: GPU={:.4} CPU={:.4} error={:.1}%",
            name, gpu_score, cpu_score, rel_error
        );
    }
}
