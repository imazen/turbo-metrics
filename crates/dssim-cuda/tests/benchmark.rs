//! Simple benchmark comparing GPU DSSIM performance
//!
//! Run with: CUDA_PATH=/usr/local/cuda-12.6 cargo test -p dssim-cuda --test benchmark --release -- --nocapture

use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut};
use cudarse_npp::set_stream;
use dssim_cuda::Dssim;
use std::time::Instant;

fn create_test_images(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
    let size = (width * height * 3) as usize;
    let mut ref_data = vec![128u8; size];
    let mut dis_data = vec![128u8; size];

    // Create some variation
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            ref_data[idx] = ((x * 255 / width) as u8).wrapping_add((y * 127 / height) as u8);
            ref_data[idx + 1] = ((y * 255 / height) as u8).wrapping_add(64);
            ref_data[idx + 2] = ((x + y) as u8).wrapping_mul(17);

            dis_data[idx] = ref_data[idx].wrapping_add(5);
            dis_data[idx + 1] = ref_data[idx + 1].wrapping_sub(3);
            dis_data[idx + 2] = ref_data[idx + 2].wrapping_add(2);
        }
    }

    (ref_data, dis_data)
}

#[test]
fn benchmark_gpu_throughput() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA");

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    // Test at different resolutions
    let resolutions = [
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 768, "1024x768"),
        (1920, 1080, "1920x1080"),
    ];

    println!("\n=== DSSIM GPU Benchmark ===\n");

    for (width, height, name) in resolutions {
        let (ref_data, dis_data) = create_test_images(width, height);

        let mut dssim = Dssim::new(width, height, &stream).unwrap();
        let mut ref_gpu: Image<u8, cudarse_npp::image::C<3>> =
            Image::malloc(width, height).unwrap();
        let mut dis_gpu = ref_gpu.malloc_same_size().unwrap();

        // Upload images
        ref_gpu
            .copy_from_cpu(&ref_data, stream.inner() as _)
            .unwrap();
        dis_gpu
            .copy_from_cpu(&dis_data, stream.inner() as _)
            .unwrap();
        stream.sync().unwrap();

        // Warmup
        for _ in 0..3 {
            let _ = dssim.compute_sync(&ref_gpu, &dis_gpu, &stream).unwrap();
        }

        // Benchmark
        let iterations = 50;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = dssim.compute_sync(&ref_gpu, &dis_gpu, &stream).unwrap();
        }

        let elapsed = start.elapsed();
        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let imgs_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!(
            "{}: {:.2} ms/image ({:.1} images/sec)",
            name, ms_per_iter, imgs_per_sec
        );
    }

    println!();
}
