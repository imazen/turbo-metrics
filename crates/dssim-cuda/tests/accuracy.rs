//! Tests comparing GPU DSSIM implementation against CPU dssim-core
//!
//! Uses both synthetic images and real compressed images from ssimulacra2 test data.

use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut};
use cudarse_npp::set_stream;
use dssim_core::{Dssim as DssimCpu, ToRGBAPLU};
use dssim_cuda::Dssim as DssimGpu;
use imgref::ImgVec;
use rgb::RGB;
use std::path::Path;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

/// Create a solid color test image
fn create_solid_image(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        data.push(r);
        data.push(g);
        data.push(b);
    }
    data
}

/// Create a horizontal gradient image
fn create_gradient_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let v = ((x * 255) / width) as u8;
            data.push(v);
            data.push(v);
            data.push(v);
        }
    }
    data
}

/// Create a noisy variant of an image (add noise to each pixel)
fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut result = Vec::with_capacity(data.len());
    // Simple deterministic pseudo-random noise
    let mut seed = Wrapping(12345u32);
    for &v in data {
        seed = seed * Wrapping(1103515245u32) + Wrapping(12345u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        let new_val = (v as i16 + noise).clamp(0, 255) as u8;
        result.push(new_val);
    }
    result
}

/// Convert RGB u8 data to RGB<u8> slice for dssim-core
fn to_rgb_slice(data: &[u8]) -> Vec<RGB<u8>> {
    data.chunks_exact(3)
        .map(|rgb| RGB::new(rgb[0], rgb[1], rgb[2]))
        .collect()
}

/// Compute DSSIM using CPU implementation
fn compute_dssim_cpu(ref_data: &[u8], dis_data: &[u8], width: usize, height: usize) -> f64 {
    let dssim = DssimCpu::new();

    // Use dssim-core's proper sRGB to linear conversion via ToRGBAPLU trait
    let ref_rgb = to_rgb_slice(ref_data);
    let dis_rgb = to_rgb_slice(dis_data);

    // Convert to linear light using dssim-core's LUT-based conversion
    let ref_linear = ref_rgb.to_rgblu();
    let dis_linear = dis_rgb.to_rgblu();

    let ref_img = ImgVec::new(ref_linear, width, height);
    let dis_img = ImgVec::new(dis_linear, width, height);

    let ref_prepared = dssim.create_image(&ref_img).unwrap();
    let dis_prepared = dssim.create_image(&dis_img).unwrap();

    let (score, _) = dssim.compare(&ref_prepared, dis_prepared);
    score.into()
}

/// Compute DSSIM using GPU implementation
fn compute_dssim_gpu(
    ref_data: &[u8],
    dis_data: &[u8],
    width: u32,
    height: u32,
    dssim: &mut DssimGpu,
    ref_gpu: &mut Image<u8, cudarse_npp::image::C<3>>,
    dis_gpu: &mut Image<u8, cudarse_npp::image::C<3>>,
    stream: &CuStream,
) -> f64 {
    // Upload images to GPU
    ref_gpu
        .copy_from_cpu(ref_data, stream.inner() as _)
        .unwrap();
    dis_gpu
        .copy_from_cpu(dis_data, stream.inner() as _)
        .unwrap();
    stream.sync().unwrap();

    // Compute DSSIM
    dssim.compute_sync(ref_gpu, dis_gpu, stream).unwrap()
}

#[test]
fn test_identical_images() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA");

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    let width = 64u32;
    let height = 64u32;

    // Create test image
    let img = create_gradient_image(width as usize, height as usize);

    // CPU DSSIM (should be 0 for identical images)
    let cpu_score = compute_dssim_cpu(&img, &img, width as usize, height as usize);

    // GPU DSSIM
    let mut dssim = DssimGpu::new(width, height, &stream).unwrap();
    let mut ref_gpu: Image<u8, cudarse_npp::image::C<3>> = Image::malloc(width, height).unwrap();
    let mut dis_gpu = ref_gpu.malloc_same_size().unwrap();

    let gpu_score = compute_dssim_gpu(
        &img,
        &img,
        width,
        height,
        &mut dssim,
        &mut ref_gpu,
        &mut dis_gpu,
        &stream,
    );

    println!("Identical images: CPU={cpu_score:.6}, GPU={gpu_score:.6}");

    // Both should be ~0 for identical images
    assert!(
        cpu_score < 0.001,
        "CPU DSSIM for identical images should be ~0, got {cpu_score}"
    );
    assert!(
        gpu_score < 0.001,
        "GPU DSSIM for identical images should be ~0, got {gpu_score}"
    );
}

#[test]
fn test_slightly_different_images() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA");

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    let width = 128u32;
    let height = 128u32;

    // Create reference and slightly noisy distorted
    let ref_img = create_gradient_image(width as usize, height as usize);
    let dis_img = add_noise(&ref_img, 10); // Small noise

    // CPU DSSIM
    let cpu_score = compute_dssim_cpu(&ref_img, &dis_img, width as usize, height as usize);

    // GPU DSSIM
    let mut dssim = DssimGpu::new(width, height, &stream).unwrap();
    let mut ref_gpu: Image<u8, cudarse_npp::image::C<3>> = Image::malloc(width, height).unwrap();
    let mut dis_gpu = ref_gpu.malloc_same_size().unwrap();

    let gpu_score = compute_dssim_gpu(
        &ref_img,
        &dis_img,
        width,
        height,
        &mut dssim,
        &mut ref_gpu,
        &mut dis_gpu,
        &stream,
    );

    println!("Slightly different: CPU={cpu_score:.6}, GPU={gpu_score:.6}");

    // Check relative tolerance - should be very close now
    let rel_error = if cpu_score > 0.0 {
        (gpu_score - cpu_score).abs() / cpu_score
    } else {
        (gpu_score - cpu_score).abs()
    };

    // GPU and CPU should produce nearly identical results (< 1% error)
    assert!(
        rel_error < 0.01,
        "GPU DSSIM {gpu_score:.6} differs from CPU {cpu_score:.6} by {:.2}%",
        rel_error * 100.0
    );
}

#[test]
fn test_very_different_images() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA");

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    let width = 64u32;
    let height = 64u32;

    // Create very different images
    let ref_img = create_solid_image(width as usize, height as usize, 0, 0, 0); // Black
    let dis_img = create_solid_image(width as usize, height as usize, 255, 255, 255); // White

    // CPU DSSIM
    let cpu_score = compute_dssim_cpu(&ref_img, &dis_img, width as usize, height as usize);

    // GPU DSSIM
    let mut dssim = DssimGpu::new(width, height, &stream).unwrap();
    let mut ref_gpu: Image<u8, cudarse_npp::image::C<3>> = Image::malloc(width, height).unwrap();
    let mut dis_gpu = ref_gpu.malloc_same_size().unwrap();

    let gpu_score = compute_dssim_gpu(
        &ref_img,
        &dis_img,
        width,
        height,
        &mut dssim,
        &mut ref_gpu,
        &mut dis_gpu,
        &stream,
    );

    println!("Very different: CPU={cpu_score:.6}, GPU={gpu_score:.6}");

    // Both should show significant difference
    assert!(
        cpu_score > 0.1,
        "CPU DSSIM for black vs white should be significant, got {cpu_score}"
    );
    assert!(
        gpu_score > 0.1,
        "GPU DSSIM for black vs white should be significant, got {gpu_score}"
    );

    // Check relative tolerance - should be very close
    let rel_error = (gpu_score - cpu_score).abs() / cpu_score;
    assert!(
        rel_error < 0.001, // 0.1% tolerance
        "GPU DSSIM {gpu_score:.6} differs from CPU {cpu_score:.6} by {:.3}%",
        rel_error * 100.0
    );
}

// =============================================================================
// Real image tests using ssimulacra2 test data
// =============================================================================

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
            zune_core::result::DecodingResult::U16(v) => {
                v.iter().map(|&x| (x >> 8) as u8).collect()
            }
            _ => panic!("Unsupported pixel format"),
        };
        (w, h, raw)
    } else if path
        .extension()
        .map_or(false, |e| e == "jpg" || e == "jpeg")
    {
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

/// Test helper for comparing images from files
fn test_image_pair(ref_path: &str, dis_path: &str, tolerance: f64) {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA");

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    let test_data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");

    let (ref_w, ref_h, ref_data) = load_image(&test_data_dir.join(ref_path));
    let (dis_w, dis_h, dis_data) = load_image(&test_data_dir.join(dis_path));

    assert_eq!(
        (ref_w, ref_h),
        (dis_w, dis_h),
        "Image dimensions must match"
    );

    let width = ref_w as u32;
    let height = ref_h as u32;

    // CPU DSSIM
    let cpu_score = compute_dssim_cpu(&ref_data, &dis_data, ref_w, ref_h);

    // GPU DSSIM
    let mut dssim = DssimGpu::new(width, height, &stream).unwrap();
    let mut ref_gpu: Image<u8, cudarse_npp::image::C<3>> = Image::malloc(width, height).unwrap();
    let mut dis_gpu = ref_gpu.malloc_same_size().unwrap();

    ref_gpu
        .copy_from_cpu(&ref_data, stream.inner() as _)
        .unwrap();
    dis_gpu
        .copy_from_cpu(&dis_data, stream.inner() as _)
        .unwrap();
    stream.sync().unwrap();

    let gpu_score = dssim.compute_sync(&ref_gpu, &dis_gpu, &stream).unwrap();

    println!(
        "{} vs {}: CPU={:.6}, GPU={:.6}",
        ref_path, dis_path, cpu_score, gpu_score
    );

    // Check relative tolerance
    let rel_error = if cpu_score > 0.0 {
        (gpu_score - cpu_score).abs() / cpu_score
    } else {
        (gpu_score - cpu_score).abs()
    };

    assert!(
        rel_error < tolerance,
        "GPU DSSIM {gpu_score:.6} differs from CPU {cpu_score:.6} by {:.2}% (tolerance: {:.2}%)",
        rel_error * 100.0,
        tolerance * 100.0
    );
}

#[test]
fn test_jpeg_quality_q90() {
    // Q90 is very high quality, should have low DSSIM
    test_image_pair("source.png", "q90.jpg", 0.01);
}

#[test]
fn test_jpeg_quality_q70() {
    // Q70 is good quality, moderate DSSIM
    test_image_pair("source.png", "q70.jpg", 0.01);
}

#[test]
fn test_jpeg_quality_q45() {
    // Q45 is medium quality
    test_image_pair("source.png", "q45.jpg", 0.01);
}

#[test]
fn test_jpeg_quality_q20() {
    // Q20 is low quality, higher DSSIM
    test_image_pair("source.png", "q20.jpg", 0.01);
}
