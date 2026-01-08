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
    test_image_pair("source.png", "q90.jpg", 0.05); // 5% tolerance
}

#[test]
fn test_jpeg_quality_q70() {
    // Q70 is good quality
    test_image_pair("source.png", "q70.jpg", 0.05);
}

#[test]
fn test_jpeg_quality_q45() {
    // Q45 is medium quality
    test_image_pair("source.png", "q45.jpg", 0.05);
}

#[test]
fn test_jpeg_quality_q20() {
    // Q20 is low quality, higher Butteraugli score
    test_image_pair("source.png", "q20.jpg", 0.05);
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

    for quality in ["q90", "q70", "q45", "q20"] {
        let dis_path = format!("{}.jpg", quality);
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
