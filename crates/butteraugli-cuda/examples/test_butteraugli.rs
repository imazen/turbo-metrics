//! Test butteraugli-cuda implementation with actual images.
//!
//! Usage: cargo run --example test_butteraugli -- <reference.png> <distorted.png>
//!
//! If no arguments provided, creates synthetic test images.

use std::env;
use std::time::Instant;

use zune_image::codecs::png::zune_core::options::DecoderOptions;

use butteraugli_cuda::Butteraugli;
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut, C};
use cudarse_npp::set_stream;

fn main() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA API");

    let args: Vec<String> = env::args().collect();

    let (ref_bytes, dis_bytes, width, height) = if args.len() >= 3 {
        // Load from files
        let ref_path = &args[1];
        let dis_path = &args[2];

        let ref_img =
            zune_image::image::Image::open_with_options(ref_path, DecoderOptions::new_fast())
                .expect("Failed to load reference image");

        let dis_img =
            zune_image::image::Image::open_with_options(dis_path, DecoderOptions::new_fast())
                .expect("Failed to load distorted image");

        assert_eq!(
            ref_img.dimensions(),
            dis_img.dimensions(),
            "Image dimensions must match"
        );

        let (width, height) = ref_img.dimensions();
        let ref_bytes = ref_img.flatten_to_u8()[0].clone();
        let dis_bytes = dis_img.flatten_to_u8()[0].clone();

        println!("Loaded images: {}x{}", width, height);
        (ref_bytes, dis_bytes, width, height)
    } else {
        // Create synthetic test images
        let width = 128;
        let height = 128;
        let size = width * height * 3;

        // Reference: white image with some pattern
        let mut ref_bytes = vec![128u8; size];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                ref_bytes[idx] = ((x as f32 / width as f32) * 255.0) as u8; // R
                ref_bytes[idx + 1] = ((y as f32 / height as f32) * 255.0) as u8; // G
                ref_bytes[idx + 2] = 128; // B
            }
        }

        // Distorted: slightly modified
        let mut dis_bytes = ref_bytes.clone();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // Add some noise to red channel
                dis_bytes[idx] = dis_bytes[idx].saturating_add(10);
            }
        }

        println!("Created synthetic test images: {}x{}", width, height);
        (ref_bytes, dis_bytes, width, height)
    };

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    // Allocate GPU images
    let mut gpu_ref = Image::<u8, C<3>>::malloc(width as u32, height as u32).unwrap();
    let mut gpu_dis = gpu_ref.malloc_same_size().unwrap();

    // Upload to GPU
    gpu_ref
        .copy_from_cpu(&ref_bytes, stream.inner() as _)
        .unwrap();
    gpu_dis
        .copy_from_cpu(&dis_bytes, stream.inner() as _)
        .unwrap();
    stream.sync().unwrap();

    // Create Butteraugli instance
    println!("Creating Butteraugli instance...");
    let mut butteraugli = Butteraugli::new(width as u32, height as u32).unwrap();
    println!("Butteraugli dimensions: {:?}", butteraugli.dimensions());

    // Compute score
    println!("Computing Butteraugli score...");
    let start = Instant::now();
    let score = butteraugli
        .compute(gpu_ref.full_view(), gpu_dis.full_view())
        .expect("Failed to compute Butteraugli score");
    let elapsed = start.elapsed();

    println!("Butteraugli score: {:.6}", score);
    println!(
        "Computed in {:.2} ms ({:.1} fps)",
        elapsed.as_nanos() as f64 / 1_000_000.0,
        1_000_000_000.0 / elapsed.as_nanos() as f64
    );

    // For identical images, score should be 0
    if ref_bytes == dis_bytes {
        assert!(
            score < 0.001,
            "Score for identical images should be near 0, got {}",
            score
        );
        println!("✓ Identical images test passed");
    } else {
        println!("Score for different images: {}", score);
        // Score should be positive for different images
        assert!(score >= 0.0, "Score should be non-negative, got {}", score);
        println!("✓ Different images test passed");
    }

    // Exit cleanly (avoid CUDA cleanup crash on some systems)
    std::process::exit(0);
}
