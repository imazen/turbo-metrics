# ssimulacra2-cuda

A GPU-accelerated implementation of SSIMULACRA2 using CUDA.

## Features

- Algorithmic parity with the reference implementation for comparable results
- Custom CUDA kernels written in Rust (compiled to PTX via `nvptx64-nvidia-cuda` target)
- CUDA NPP primitives for optimized matrix operations
- CUDA graphs to reduce overhead of 305 kernel launches per image pair
- ~78-131 Mpx/s throughput (3-5ms per 512x768 image on RTX 5070)

## Prerequisites

### 1. CUDA Toolkit

Install CUDA Toolkit 12.x. The `nvcc` and `ptxas` tools must be in your PATH.

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 2. Rust Nightly with NVPTX Target

The CUDA kernels are compiled using Rust's `nvptx64-nvidia-cuda` target, which requires nightly.

```bash
rustup install nightly
rustup +nightly target add nvptx64-nvidia-cuda
rustup +nightly component add llvm-bitcode-linker
```

### 3. WSL2 Users

If running on WSL2, add the WSL CUDA library path to your linker configuration:

```toml
# .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "link-arg=-L/usr/lib/wsl/lib"]
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ssimulacra2-cuda = { git = "https://github.com/Gui-Yom/turbo-metrics" }

# Also needed for CUDA context and image handling
cudarse-driver = { git = "https://github.com/Gui-Yom/turbo-metrics" }
cudarse-npp = { git = "https://github.com/Gui-Yom/turbo-metrics", features = ["isu"] }
```

### Build Configuration

Create `.cargo/config.toml` in your project:

```toml
[target.nvptx64-nvidia-cuda]
rustflags = ["-Z", "unstable-options", "-C", "linker-flavor=llbc"]

# Optional: target specific GPU architecture for better performance
# rustflags = ["-Z", "unstable-options", "-C", "linker-flavor=llbc", "-C", "target-cpu=sm_89"]
```

Add the nvptx release profile to your `Cargo.toml`:

```toml
[profile.release-nvptx]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
overflow-checks = false
```

## Usage

### Basic Example

```rust
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::Image;
use cudarse_npp::set_stream;
use ssimulacra2_cuda::Ssimulacra2;

fn main() {
    // Initialize CUDA
    cudarse_driver::init_cuda_and_primary_ctx()
        .expect("Failed to initialize CUDA");

    // Create a CUDA stream
    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    // Image dimensions
    let width = 1920u32;
    let height = 1080u32;

    // Allocate GPU buffers for sRGB input (u8) and linear RGB (f32)
    let tmp_ref: Image<u8, _> = Image::malloc(width, height).unwrap();
    let tmp_dis: Image<u8, _> = Image::malloc(width, height).unwrap();
    let ref_linear: Image<f32, _> = Image::malloc(width, height).unwrap();
    let dis_linear: Image<f32, _> = Image::malloc(width, height).unwrap();

    // Create the Ssimulacra2 instance (records CUDA graph)
    let mut ssim2 = Ssimulacra2::new(&ref_linear, &dis_linear, &stream).unwrap();

    // Your sRGB image bytes (RGB interleaved, 3 bytes per pixel)
    let ref_bytes: &[u8] = &vec![128u8; (width * height * 3) as usize];
    let dis_bytes: &[u8] = &vec![128u8; (width * height * 3) as usize];

    // Compute SSIMULACRA2 score
    let score = ssim2.compute_from_cpu_srgb_sync(
        ref_bytes,
        dis_bytes,
        &mut tmp_ref.borrowed_mut(),
        &mut tmp_dis.borrowed_mut(),
        ref_linear.borrowed_mut(),
        dis_linear.borrowed_mut(),
        &stream,
    ).unwrap();

    println!("SSIMULACRA2 score: {:.2}", score);
}
```

### Processing Multiple Images

For batch processing, reuse the `Ssimulacra2` instance:

```rust
// Create once (expensive - records CUDA graph)
let mut ssim2 = Ssimulacra2::new(&ref_linear, &dis_linear, &stream).unwrap();

// Process many image pairs (fast - replays graph)
for (ref_bytes, dis_bytes) in image_pairs {
    let score = ssim2.compute_from_cpu_srgb_sync(
        ref_bytes, dis_bytes,
        &mut tmp_ref.borrowed_mut(), &mut tmp_dis.borrowed_mut(),
        ref_linear.borrowed_mut(), dis_linear.borrowed_mut(),
        &stream,
    ).unwrap();
    println!("Score: {:.2}", score);
}
```

### Async Processing (Images Already on GPU)

For images already in GPU memory as linear RGB f32:

```rust
// Start computation (non-blocking)
ssim2.compute(&stream)?;

// Do other work...

// Sync and get score
stream.sync().unwrap();
let score = ssim2.get_score();
```

## API Reference

### `Ssimulacra2`

| Method | Description |
|--------|-------------|
| `new(ref_linear, dis_linear, stream)` | Create instance for given image dimensions. Records CUDA graph. |
| `compute_from_cpu_srgb_sync(...)` | Compute from CPU sRGB bytes. Blocks until complete. |
| `compute_srgb_sync(...)` | Compute from GPU sRGB images. Blocks until complete. |
| `compute_sync(stream)` | Compute from GPU linear images. Blocks until complete. |
| `compute(stream)` | Start async computation. Call `get_score()` after sync. |
| `get_score()` | Get score from last computation. Must sync stream first. |
| `mem_usage()` | Estimated GPU memory usage in bytes. |

## Memory Requirements

The implementation requires approximately `270 * width * height` bytes of GPU memory.

**Note**: The `Ssimulacra2` struct implements `Drop` which synchronizes the CUDA context before freeing GPU buffers. This prevents crashes when pending operations reference buffers being freed.

| Resolution | Approximate Memory |
|------------|-------------------|
| 720p (1280x720) | ~250 MB |
| 1080p (1920x1080) | ~560 MB |
| 4K (3840x2160) | ~2.2 GB |

## Score Interpretation

| Score | Quality |
|-------|---------|
| 90+ | Excellent (nearly lossless) |
| 70-90 | Good (high quality lossy) |
| 50-70 | Fair (noticeable artifacts) |
| 30-50 | Poor (significant degradation) |
| <30 | Bad (severe artifacts) |

## Differences from Reference Implementation

The CUDA implementation uses the same algorithm as the [reference](https://github.com/cloudinary/ssimulacra2):
- Same 108 weights
- Same XYB color space conversion
- Same recursive Gaussian blur (Charalampidis 2016)
- Same error map formulas
- Same 6-scale multi-scale approach

**Precision difference**: The blur pass uses `f32` accumulators (vs `f64` in reference) for GPU efficiency. This typically results in score differences < 0.1 units.

**Other factors affecting score parity**:
- YUV to linear RGB conversion may differ from other tools
- FMA instruction usage affects intermediate results
- GPU floating-point operations may differ from CPU

## Building from Source

```bash
git clone https://github.com/Gui-Yom/turbo-metrics
cd turbo-metrics

# Build the library
cargo build --release -p ssimulacra2-cuda

# Run the comparison example
cargo run --release -p ssimulacra2-cuda --example compare -- ref.png distorted.png
```

## Credits

- Original reference implementation: https://github.com/cloudinary/ssimulacra2
- Rust CPU implementation inspiration: https://github.com/rust-av/ssimulacra2

## License

MIT
