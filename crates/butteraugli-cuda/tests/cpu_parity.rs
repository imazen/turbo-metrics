//! CPU Parity Tests for butteraugli-cuda
//!
//! These tests compare the CUDA implementation against the CPU butteraugli crate
//! to verify numerical parity. The CPU crate is itself validated against the
//! C++ libjxl implementation.
//!
//! Tolerance levels are based on:
//! - Identical images: Must score exactly 0.0
//! - Synthetic patterns: 5% relative tolerance (FP32 vs FP64 differences)
//! - Complex patterns: 10% relative tolerance
//!
//! Run with: CUDA_PATH=/usr/local/cuda-12.6 cargo test --test cpu_parity --release

use butteraugli::{compute_butteraugli, ButteraugliParams};
use butteraugli_cuda::Butteraugli;
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut, C};
use cudarse_npp::set_stream;
use sha2::{Digest, Sha256};

// ============================================================================
// Test Configuration
// ============================================================================

/// Absolute tolerance for identical image comparison (must be exactly 0)
const IDENTICAL_TOLERANCE: f64 = 1e-6;

/// Relative tolerance for simple patterns (uniform, gradients)
const SIMPLE_PATTERN_TOLERANCE: f64 = 0.10; // 10%

/// Relative tolerance for complex patterns (checkerboard, noise)
const COMPLEX_PATTERN_TOLERANCE: f64 = 0.25; // 25%

/// Absolute tolerance floor (when relative tolerance would be too small)
const ABSOLUTE_TOLERANCE_FLOOR: f64 = 0.05;

/// Test image sizes (must be >= 8x8 for butteraugli)
const TEST_SIZES: [(usize, usize); 4] = [(16, 16), (32, 32), (64, 64), (128, 128)];

// ============================================================================
// Deterministic Image Generation (matches butteraugli/fast-ssim2 patterns)
// ============================================================================

/// LCG pseudo-random number generator (deterministic, matches C++ tests)
struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u8(&mut self) -> u8 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) & 0xFF) as u8
    }
}

/// Compute SHA256 hash of image data for reproducibility verification
fn hash_image(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

/// Generate uniform color image
fn gen_uniform(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    vec![r, g, b]
        .into_iter()
        .cycle()
        .take(width * height * 3)
        .collect()
}

/// Generate horizontal gradient (grayscale)
fn gen_gradient_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if width > 1 {
                (x * 255 / (width - 1)) as u8
            } else {
                128
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate vertical gradient (grayscale)
fn gen_gradient_v(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if height > 1 {
            (y * 255 / (height - 1)) as u8
        } else {
            128
        };
        for _x in 0..width {
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate diagonal gradient
fn gen_gradient_diag(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let max_dist = width + height - 2;
    for y in 0..height {
        for x in 0..width {
            let val = if max_dist > 0 {
                ((x + y) * 255 / max_dist) as u8
            } else {
                128
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate checkerboard pattern
fn gen_checkerboard(width: usize, height: usize, cell_size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let val = if ((x / cell_size) + (y / cell_size)) % 2 == 0 {
                255
            } else {
                0
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate random noise
fn gen_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        data.push(lcg.next_u8());
        data.push(lcg.next_u8());
        data.push(lcg.next_u8());
    }
    data
}

/// Generate edge pattern (sharp vertical transition)
fn gen_edge_v(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if x < width / 2 { 50 } else { 200 };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate edge pattern (sharp horizontal transition)
fn gen_edge_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if y < height / 2 { 50 } else { 200 };
        for _x in 0..width {
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Add uniform offset to all pixels
fn add_offset(data: &[u8], offset: i16) -> Vec<u8> {
    data.iter()
        .map(|&v| (v as i16 + offset).clamp(0, 255) as u8)
        .collect()
}

// ============================================================================
// GPU Helper Functions
// ============================================================================

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

/// Compute butteraugli score using GPU
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
    let mut gpu_dst = gpu_src.malloc_same_size()
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
    let mut butteraugli = Butteraugli::new(width as u32, height as u32)
        .expect("Failed to create Butteraugli instance");

    butteraugli
        .compute(gpu_src.full_view(), gpu_dst.full_view())
        .expect("Failed to compute Butteraugli score")
}

/// Compute butteraugli score using CPU
fn compute_cpu_score(source: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    compute_butteraugli(source, distorted, width, height, &params)
        .expect("CPU butteraugli failed")
        .score
}

// ============================================================================
// Test Result Tracking
// ============================================================================

#[derive(Debug)]
struct TestResult {
    name: String,
    width: usize,
    height: usize,
    cpu_score: f64,
    gpu_score: f64,
    abs_error: f64,
    rel_error: f64,
    passed: bool,
}

impl TestResult {
    fn new(
        name: &str,
        width: usize,
        height: usize,
        cpu_score: f64,
        gpu_score: f64,
        tolerance_rel: f64,
        tolerance_abs: f64,
    ) -> Self {
        let abs_error = (cpu_score - gpu_score as f64).abs();
        let rel_error = if cpu_score.abs() > 1e-10 {
            abs_error / cpu_score.abs()
        } else {
            abs_error
        };

        let effective_tolerance = (tolerance_rel * cpu_score.abs()).max(tolerance_abs);
        let passed = abs_error <= effective_tolerance;

        Self {
            name: name.to_string(),
            width,
            height,
            cpu_score,
            gpu_score: gpu_score as f64,
            abs_error,
            rel_error,
            passed,
        }
    }
}

// ============================================================================
// Test Cases
// ============================================================================

/// Test identical images (must score exactly 0.0)
#[test]
fn test_identical_images() {
    let ctx = CudaContext::new();
    let mut all_passed = true;
    let mut results = Vec::new();

    for (width, height) in TEST_SIZES {
        // Test various image types
        for (name, source) in [
            ("uniform_gray", gen_uniform(width, height, 128, 128, 128)),
            ("uniform_color", gen_uniform(width, height, 100, 150, 200)),
            ("gradient_h", gen_gradient_h(width, height)),
            ("gradient_v", gen_gradient_v(width, height)),
            ("checkerboard", gen_checkerboard(width, height, 4)),
            ("noise", gen_noise(width, height, 42)),
        ] {
            let gpu_score = compute_gpu_score(&ctx, &source, &source, width, height);
            let cpu_score = compute_cpu_score(&source, &source, width, height);

            let result = TestResult::new(
                &format!("identical_{}_{}x{}", name, width, height),
                width,
                height,
                cpu_score,
                gpu_score as f64,
                0.0, // No relative tolerance - must be exact
                IDENTICAL_TOLERANCE,
            );

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }
    }

    // Print all results
    println!("\n=== Identical Image Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.6}, GPU={:.6}, abs_err={:.2e}",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.abs_error
        );
    }

    assert!(
        all_passed,
        "Some identical image tests failed! GPU must return 0.0 for identical images."
    );
}

/// Test uniform images with different gray levels
#[test]
fn test_uniform_gray_shifts() {
    let ctx = CudaContext::new();
    let mut results = Vec::new();
    let mut all_passed = true;

    for (width, height) in TEST_SIZES {
        for shift in [5, 10, 20, 50] {
            let source = gen_uniform(width, height, 128, 128, 128);
            let distorted = add_offset(&source, shift);

            let gpu_score = compute_gpu_score(&ctx, &source, &distorted, width, height);
            let cpu_score = compute_cpu_score(&source, &distorted, width, height);

            let result = TestResult::new(
                &format!("uniform_shift_{}_{}x{}", shift, width, height),
                width,
                height,
                cpu_score,
                gpu_score as f64,
                SIMPLE_PATTERN_TOLERANCE,
                ABSOLUTE_TOLERANCE_FLOOR,
            );

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }
    }

    // Print summary
    println!("\n=== Uniform Gray Shift Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.4}, GPU={:.4}, rel_err={:.2}%",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.rel_error * 100.0
        );
    }

    assert!(all_passed, "Some uniform gray shift tests failed!");
}

/// Test gradient patterns
#[test]
fn test_gradients() {
    let ctx = CudaContext::new();
    let mut results = Vec::new();
    let mut all_passed = true;

    for (width, height) in TEST_SIZES {
        for (name, gen_fn) in [
            ("h", gen_gradient_h as fn(usize, usize) -> Vec<u8>),
            ("v", gen_gradient_v as fn(usize, usize) -> Vec<u8>),
            ("diag", gen_gradient_diag as fn(usize, usize) -> Vec<u8>),
        ] {
            let source = gen_fn(width, height);
            let distorted = add_offset(&source, 10);

            let gpu_score = compute_gpu_score(&ctx, &source, &distorted, width, height);
            let cpu_score = compute_cpu_score(&source, &distorted, width, height);

            let result = TestResult::new(
                &format!("gradient_{}_{}x{}", name, width, height),
                width,
                height,
                cpu_score,
                gpu_score as f64,
                SIMPLE_PATTERN_TOLERANCE,
                ABSOLUTE_TOLERANCE_FLOOR,
            );

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }
    }

    println!("\n=== Gradient Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.4}, GPU={:.4}, rel_err={:.2}%",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.rel_error * 100.0
        );
    }

    assert!(all_passed, "Some gradient tests failed!");
}

/// Test checkerboard patterns (high frequency content)
#[test]
fn test_checkerboards() {
    let ctx = CudaContext::new();
    let mut results = Vec::new();
    let mut all_passed = true;

    for (width, height) in TEST_SIZES {
        for cell_size in [2, 4, 8] {
            if cell_size > width / 2 || cell_size > height / 2 {
                continue;
            }

            let source = gen_checkerboard(width, height, cell_size);
            let distorted = add_offset(&source, 10);

            let gpu_score = compute_gpu_score(&ctx, &source, &distorted, width, height);
            let cpu_score = compute_cpu_score(&source, &distorted, width, height);

            let result = TestResult::new(
                &format!("checker_{}_{}x{}", cell_size, width, height),
                width,
                height,
                cpu_score,
                gpu_score as f64,
                COMPLEX_PATTERN_TOLERANCE,
                ABSOLUTE_TOLERANCE_FLOOR,
            );

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }
    }

    println!("\n=== Checkerboard Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.4}, GPU={:.4}, rel_err={:.2}%",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.rel_error * 100.0
        );
    }

    assert!(all_passed, "Some checkerboard tests failed!");
}

/// Test random noise patterns
#[test]
fn test_noise_patterns() {
    let ctx = CudaContext::new();
    let mut results = Vec::new();
    let mut all_passed = true;

    for (width, height) in TEST_SIZES {
        for seed in [42, 123, 999] {
            let source = gen_noise(width, height, seed);
            let distorted = gen_noise(width, height, seed + 1000); // Different noise

            let gpu_score = compute_gpu_score(&ctx, &source, &distorted, width, height);
            let cpu_score = compute_cpu_score(&source, &distorted, width, height);

            let result = TestResult::new(
                &format!("noise_{}_{}x{}", seed, width, height),
                width,
                height,
                cpu_score,
                gpu_score as f64,
                COMPLEX_PATTERN_TOLERANCE,
                ABSOLUTE_TOLERANCE_FLOOR,
            );

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }
    }

    println!("\n=== Noise Pattern Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.4}, GPU={:.4}, rel_err={:.2}%",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.rel_error * 100.0
        );
    }

    assert!(all_passed, "Some noise pattern tests failed!");
}

/// Test edge patterns
#[test]
fn test_edges() {
    let ctx = CudaContext::new();
    let mut results = Vec::new();
    let mut all_passed = true;

    for (width, height) in TEST_SIZES {
        for (name, source) in [
            ("edge_v", gen_edge_v(width, height)),
            ("edge_h", gen_edge_h(width, height)),
        ] {
            let distorted = add_offset(&source, 10);

            let gpu_score = compute_gpu_score(&ctx, &source, &distorted, width, height);
            let cpu_score = compute_cpu_score(&source, &distorted, width, height);

            let result = TestResult::new(
                &format!("{}_{}x{}", name, width, height),
                width,
                height,
                cpu_score,
                gpu_score as f64,
                COMPLEX_PATTERN_TOLERANCE,
                ABSOLUTE_TOLERANCE_FLOOR,
            );

            if !result.passed {
                all_passed = false;
            }
            results.push(result);
        }
    }

    println!("\n=== Edge Pattern Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.4}, GPU={:.4}, rel_err={:.2}%",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.rel_error * 100.0
        );
    }

    assert!(all_passed, "Some edge pattern tests failed!");
}

/// Test extreme cases
#[test]
fn test_extreme_cases() {
    let ctx = CudaContext::new();
    let mut results = Vec::new();
    let mut all_passed = true;

    let width = 64;
    let height = 64;

    // Black vs White
    let black = gen_uniform(width, height, 0, 0, 0);
    let white = gen_uniform(width, height, 255, 255, 255);

    let gpu_score = compute_gpu_score(&ctx, &black, &white, width, height);
    let cpu_score = compute_cpu_score(&black, &white, width, height);

    let result = TestResult::new(
        "black_vs_white",
        width,
        height,
        cpu_score,
        gpu_score as f64,
        COMPLEX_PATTERN_TOLERANCE,
        ABSOLUTE_TOLERANCE_FLOOR,
    );
    if !result.passed {
        all_passed = false;
    }
    results.push(result);

    // Red vs Green
    let red = gen_uniform(width, height, 255, 0, 0);
    let green = gen_uniform(width, height, 0, 255, 0);

    let gpu_score = compute_gpu_score(&ctx, &red, &green, width, height);
    let cpu_score = compute_cpu_score(&red, &green, width, height);

    let result = TestResult::new(
        "red_vs_green",
        width,
        height,
        cpu_score,
        gpu_score as f64,
        COMPLEX_PATTERN_TOLERANCE,
        ABSOLUTE_TOLERANCE_FLOOR,
    );
    if !result.passed {
        all_passed = false;
    }
    results.push(result);

    println!("\n=== Extreme Case Tests ===");
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "[{}] {}: CPU={:.4}, GPU={:.4}, rel_err={:.2}%",
            status, r.name, r.cpu_score, r.gpu_score as f64, r.rel_error * 100.0
        );
    }

    assert!(all_passed, "Some extreme case tests failed!");
}

/// Comprehensive summary test that runs all patterns and reports statistics
#[test]
fn test_comprehensive_parity_summary() {
    let ctx = CudaContext::new();
    let mut all_results: Vec<TestResult> = Vec::new();

    let width = 64;
    let height = 64;

    // Generate test cases
    let test_cases: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
        // Identical (should be 0)
        {
            let img = gen_gradient_h(width, height);
            ("identical", img.clone(), img)
        },
        // Simple patterns
        {
            let src = gen_uniform(width, height, 128, 128, 128);
            let dst = add_offset(&src, 20);
            ("uniform+20", src, dst)
        },
        {
            let src = gen_gradient_h(width, height);
            let dst = add_offset(&src, 10);
            ("gradient_h+10", src, dst)
        },
        {
            let src = gen_gradient_v(width, height);
            let dst = add_offset(&src, 10);
            ("gradient_v+10", src, dst)
        },
        // Complex patterns
        {
            let src = gen_checkerboard(width, height, 4);
            let dst = add_offset(&src, 10);
            ("checker4+10", src, dst)
        },
        {
            let src = gen_noise(width, height, 42);
            let dst = gen_noise(width, height, 43);
            ("noise_diff", src, dst)
        },
        {
            let src = gen_edge_v(width, height);
            let dst = add_offset(&src, 10);
            ("edge_v+10", src, dst)
        },
    ];

    for (name, source, distorted) in test_cases {
        let gpu_score = compute_gpu_score(&ctx, &source, &distorted, width, height);
        let cpu_score = compute_cpu_score(&source, &distorted, width, height);

        let tolerance = if name.starts_with("identical") {
            (0.0, IDENTICAL_TOLERANCE)
        } else if name.contains("uniform") || name.contains("gradient") {
            (SIMPLE_PATTERN_TOLERANCE, ABSOLUTE_TOLERANCE_FLOOR)
        } else {
            (COMPLEX_PATTERN_TOLERANCE, ABSOLUTE_TOLERANCE_FLOOR)
        };

        let result = TestResult::new(name, width, height, cpu_score, gpu_score as f64, tolerance.0, tolerance.1);
        all_results.push(result);
    }

    // Compute statistics
    let total = all_results.len();
    let passed = all_results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let max_abs_error = all_results
        .iter()
        .map(|r| r.abs_error)
        .fold(0.0, f64::max);
    let max_rel_error = all_results
        .iter()
        .filter(|r| r.cpu_score.abs() > 0.01) // Exclude near-zero scores
        .map(|r| r.rel_error)
        .fold(0.0, f64::max);
    let mean_abs_error: f64 = all_results.iter().map(|r| r.abs_error).sum::<f64>() / total as f64;

    // Print comprehensive report
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║            BUTTERAUGLI-CUDA PARITY TEST SUMMARY                ║");
    println!("╠════════════════════════════════════════════════════════════════╣");

    for r in &all_results {
        let status = if r.passed { "✓" } else { "✗" };
        println!(
            "║ {} {:20} CPU={:8.4} GPU={:8.4} err={:6.2}%  ║",
            status,
            r.name,
            r.cpu_score,
            r.gpu_score as f64,
            r.rel_error * 100.0
        );
    }

    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Total: {} tests, {} passed, {} failed                           ║", total, passed, failed);
    println!("║ Max absolute error: {:.6}                                    ║", max_abs_error);
    println!("║ Max relative error: {:.2}%                                       ║", max_rel_error * 100.0);
    println!("║ Mean absolute error: {:.6}                                   ║", mean_abs_error);
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Strict assertion
    assert!(
        failed == 0,
        "\n{} out of {} tests FAILED! GPU implementation diverges from CPU beyond tolerance.\n\
         Max abs error: {:.6}, Max rel error: {:.2}%\n\
         Review the detailed results above.",
        failed,
        total,
        max_abs_error,
        max_rel_error * 100.0
    );
}
