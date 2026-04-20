//! zensim CUDA — GPU port of the zensim perceptual similarity metric.
//!
//! Numerically equivalent (within ~ULP of cross-arch FMA drift) to
//! `zen/zensim/zensim/src` on CPU. Produces 57 features per scale × 4
//! scales = 228 features, matching `WEIGHTS_PREVIEW_V0_2`. The caller
//! applies the weight inner product + `100 - 18·d^0.7` score mapping
//! (provided as [`score_from_features`] here for convenience).
//!
//! Public API is single-image `compute(ref_rgb, dis_rgb, w, h) -> score`
//! for now; a butter-style `set_reference` / `compute_with_reference`
//! + graph capture + batch variants will land in follow-up commits.

mod kernel;
pub use kernel::Kernel;

use cudarse_driver::{CuBox, CuStream};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{C, Image, Img, ImgMut};

/// Number of pyramid scales, same as CPU zensim.
pub const SCALES: usize = 4;

/// Features per channel per scale including peaks (mean/L2/L4 + max/L8 pooled).
pub const FEATURES_PER_CHANNEL: usize = 19;

/// Features per scale (19 per channel × 3 channels).
pub const FEATURES_PER_SCALE: usize = FEATURES_PER_CHANNEL * 3;

/// Total features (4 scales × 57).
pub const TOTAL_FEATURES: usize = FEATURES_PER_SCALE * SCALES;

/// Basic ("scored") features per channel per scale — same as CPU
/// `FEATURES_PER_CHANNEL_BASIC`: 13 mean/L2/L4-pooled features.
pub const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Peak features per channel per scale — matches CPU layout pass 2
/// (max + L8-pooled p95): 6 features.
pub const FEATURES_PER_CHANNEL_PEAKS: usize = 6;

/// Blur radius for the V0.1/V0.2 profiles. CPU zensim's
/// `PROFILE_PREVIEW_V0_{1,2}.blur_radius` is **5** (diameter = 11) —
/// the `WEIGHTS_PREVIEW_V0_2` table this crate scores against was
/// trained at that radius, so any other value produces features that
/// don't pair with the weights (`hf_*` ratios drift catastrophically —
/// a radius-3 pipeline gives `ratio_l2 ≈ 0.46` on a 512² Q=90 JPEG,
/// whereas CPU's radius-5 pipeline gives `ratio_l2 ≈ 0.61`). The prior
/// value (`3`) was a miscount and caused a consistent score gap of
/// ~3.7 points vs CPU zensim. See
/// `zen/zensim/zensim/src/profile.rs` lines 92-108.
pub const BLUR_RADIUS: usize = 5;

/// SIMD-alignment padding applied by CPU zensim at scale 0. Every
/// plane is widened to the next multiple of 16, with an extra 16 cols
/// added when the aligned width ≥ 512 and `aligned/16` is even (cache
/// set avoidance on x86). Mirror-reflected copies of the real columns
/// fill the padding region. The CPU pipeline then treats this widened
/// layout as the logical image for every subsequent operation (H-blur,
/// V-blur, feature accumulation) — so features are summed over
/// `padded_width × height`, not `width × height`. We must mirror this
/// or score parity drifts at the HF-ratio features (padding affects
/// `Σ(src-mu)²` and `Σ(dst-mu)²` asymmetrically, shifting the ratio).
///
/// Must match `zen/zensim/zensim/src/blur.rs::simd_padded_width`.
pub fn simd_padded_width(width: usize) -> usize {
    let aligned = (width + 15) & !15;
    if aligned >= 512 && (aligned / 16) % 2 == 0 {
        aligned + 16
    } else {
        aligned
    }
}

#[derive(Debug)]
pub enum Error {
    Cuda(String),
    Npp(String),
    InvalidDimensions(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Cuda(s) => write!(f, "CUDA error: {}", s),
            Error::Npp(s) => write!(f, "NPP error: {}", s),
            Error::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
        }
    }
}

impl std::error::Error for Error {}

/// Trained weights from zensim's `WEIGHTS_PREVIEW_V0_2` (228 entries).
/// The host side reads this and applies `100 - 18 · d^0.7` on the
/// weighted inner product / n_scales to produce the 0-100 score.
pub fn score_from_features(features: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(features.len(), weights.len());
    let raw: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(&f, &w)| w * f)
        .sum();
    let features_per_scale = FEATURES_PER_SCALE;
    let n_scales = features.len() / features_per_scale;
    let raw_per_scale = raw / n_scales.max(1) as f64;
    if raw_per_scale <= 0.0 {
        100.0
    } else {
        100.0 - 18.0 * raw_per_scale.powf(0.7)
    }
}

/// Per-scale buffer set. We pre-allocate at `padded_w × h` to match
/// CPU zensim's SIMD-padded layout (see [`simd_padded_width`]). The
/// image data occupies cols `[0..logical_w)`; cols `[logical_w..padded_w)`
/// hold mirror-reflected copies of the real columns. All blur + feature
/// kernels are invoked with `width = padded_w` so CPU parity is exact.
struct ScaleBufs {
    /// Three planar XYB planes at `padded_w × h`.
    xyb_ref: [Image<f32, C<1>>; 3],
    xyb_dis: [Image<f32, C<1>>; 3],
    /// 4 H-blur temporaries (mu1, mu2, sigma_sq, sigma12), reused across channels.
    h_mu1: Image<f32, C<1>>,
    h_mu2: Image<f32, C<1>>,
    h_sigma_sq: Image<f32, C<1>>,
    h_sigma12: Image<f32, C<1>>,
    /// 17 f64 + 3 u32 accumulators per channel × 3 channels. We pack
    /// into flat CuBox buffers — 51 f64 + 9 u32 per scale.
    accum_f64: CuBox<[f64]>, // 17 * 3
    peak_u32: CuBox<[u32]>, // 3 * 3
    /// Real image width (for colour conv + right-edge mirror source).
    logical_w: u32,
    /// Padded width (for H/V-blur + feature accumulation).
    padded_w: u32,
    /// Working height (unchanged by padding).
    h: u32,
}

impl ScaleBufs {
    fn alloc(logical_w: u32, padded_w: u32, h: u32, stream: &CuStream) -> Result<Self, Error> {
        let alloc_plane = || -> Result<Image<f32, C<1>>, Error> {
            Image::<f32, C<1>>::malloc(padded_w, h).map_err(|e| Error::Npp(format!("{:?}", e)))
        };
        let three = || -> Result<[Image<f32, C<1>>; 3], Error> {
            Ok([alloc_plane()?, alloc_plane()?, alloc_plane()?])
        };
        let accum_f64 = CuBox::<[f64]>::new_zeroed(17 * 3, stream)
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        let peak_u32 = CuBox::<[u32]>::new_zeroed(3 * 3, stream)
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        Ok(Self {
            xyb_ref: three()?,
            xyb_dis: three()?,
            h_mu1: alloc_plane()?,
            h_mu2: alloc_plane()?,
            h_sigma_sq: alloc_plane()?,
            h_sigma12: alloc_plane()?,
            accum_f64,
            peak_u32,
            logical_w,
            padded_w,
            h,
        })
    }
}

/// GPU-accelerated zensim scorer. Allocates one pyramid's worth of
/// buffers up front for a specific (width, height). Call
/// [`Zensim::compute`] with sRGB packed-u8 RGB images.
pub struct Zensim {
    kernel: Kernel,
    stream: CuStream,
    width: u32,
    height: u32,
    scales: Vec<ScaleBufs>,
    /// Upload staging (one for ref, one for dis).
    ref_u8: Image<u8, C<3>>,
    dis_u8: Image<u8, C<3>>,
    /// Per-scale mirror-offset tables (length `padded_w - logical_w` for
    /// that scale). `mirror_offsets[scale][i]` is the real-column index
    /// whose value should be copied into padding column `logical_w + i`
    /// of `scale`. Empty `CuBox<[u32]>` for scales where padded_w ==
    /// logical_w. Host-computed once at `Zensim::new` time.
    mirror_offsets: Vec<Option<CuBox<[u32]>>>,
}

impl Zensim {
    pub fn new(width: u32, height: u32) -> Result<Self, Error> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidDimensions(
                "zensim requires images at least 8×8".into(),
            ));
        }
        let stream = {
            let range = CuStream::priority_range()
                .map_err(|e| Error::Cuda(format!("priority_range: {:?}", e)))?;
            CuStream::new_with_priority(range.least).map_err(|e| Error::Cuda(format!("{:?}", e)))?
        };
        let kernel = Kernel::load();

        // Scale 0 uses SIMD-padded width to match CPU zensim's layout;
        // subsequent scales inherit padded_w/2 because CPU downscales
        // the padded planes without re-padding. See `simd_padded_width`
        // for why this is load-bearing for score parity.
        let mut scales = Vec::with_capacity(SCALES);
        let mut mirror_offsets: Vec<Option<CuBox<[u32]>>> = Vec::with_capacity(SCALES);
        let mut logical_w = width;
        let mut padded_w = simd_padded_width(width as usize) as u32;
        let mut h = height;
        for _ in 0..SCALES {
            if logical_w < 8 || h < 8 {
                break;
            }
            scales.push(ScaleBufs::alloc(logical_w, padded_w, h, &stream)?);

            // Precompute the mirror-offset table matching CPU zensim
            // (streaming.rs line 591-601):
            //   period = 2 * (logical_w - 1)
            //   for i in 0..pad_count:
            //     m = (logical_w + i) % period
            //     offset = if m < logical_w { m } else { period - m }
            if padded_w > logical_w {
                let pad_count = (padded_w - logical_w) as usize;
                let period = 2 * (logical_w as usize - 1);
                let host: Vec<u32> = (0..pad_count)
                    .map(|i| {
                        let m = (logical_w as usize + i) % period;
                        let off = if m < logical_w as usize {
                            m
                        } else {
                            period - m
                        };
                        off as u32
                    })
                    .collect();
                let buf = CuBox::<[u32]>::new_zeroed(pad_count, &stream)
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                unsafe {
                    cudarse_driver::sys::cuMemcpyHtoDAsync_v2(
                        buf.ptr(),
                        host.as_ptr() as *const _,
                        pad_count * 4,
                        stream.raw(),
                    )
                    .result()
                    .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }
                mirror_offsets.push(Some(buf));
            } else {
                mirror_offsets.push(None);
            }

            logical_w = (logical_w + 1) / 2;
            padded_w = padded_w / 2;
            h = (h + 1) / 2;
        }
        // Ensure mirror-offset HtoD copies complete before any compute.
        stream.sync().map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        let ref_u8 =
            Image::<u8, C<3>>::malloc(width, height).map_err(|e| Error::Npp(format!("{:?}", e)))?;
        let dis_u8 =
            Image::<u8, C<3>>::malloc(width, height).map_err(|e| Error::Npp(format!("{:?}", e)))?;

        Ok(Self {
            kernel,
            stream,
            width,
            height,
            scales,
            ref_u8,
            dis_u8,
            mirror_offsets,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Debug: return the raw per-scale per-channel accumulator sums
    /// (17 f64 + 3 u32 per channel × 3 channels × n_scales). Layout:
    /// `Vec<[ScaleRaw; 3]>` where each `ScaleRaw` has:
    ///   f64[0..17] = the 17 accumulators (see `compute_features`
    ///                 comment for slot meanings)
    ///   u32[0..3]  = ssim_max, art_max, det_max (bits of f32)
    ///
    /// Intended purely for the `gpu_zensim_per_feature_verify`-style
    /// debug harness. Not a stable API.
    pub fn debug_compute_raw(
        &mut self,
        source_rgb: &[u8],
        distorted_rgb: &[u8],
    ) -> Result<Vec<Vec<([f64; 17], [u32; 3])>>, Error> {
        self.compute_features_inner(source_rgb, distorted_rgb, true)
            .map(|(_, raw)| raw)
    }

    /// Score `distorted` against `source`. Both must be tightly packed
    /// sRGB RGB u8 at the (width, height) this instance was created for.
    /// Returns the 228-entry feature vector; apply
    /// [`score_from_features`] with the trained weights to turn this
    /// into a 0-100 score.
    pub fn compute_features(
        &mut self,
        source_rgb: &[u8],
        distorted_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES], Error> {
        self.compute_features_inner(source_rgb, distorted_rgb, false)
            .map(|(feat, _)| feat)
    }

    fn compute_features_inner(
        &mut self,
        source_rgb: &[u8],
        distorted_rgb: &[u8],
        return_raw: bool,
    ) -> Result<([f64; TOTAL_FEATURES], Vec<Vec<([f64; 17], [u32; 3])>>), Error> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if source_rgb.len() != expected || distorted_rgb.len() != expected {
            return Err(Error::InvalidDimensions(format!(
                "expected {} bytes per image, got {} / {}",
                expected,
                source_rgb.len(),
                distorted_rgb.len()
            )));
        }

        // Upload RGB.
        self.ref_u8
            .copy_from_cpu(source_rgb, self.stream.inner() as _)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;
        self.dis_u8
            .copy_from_cpu(distorted_rgb, self.stream.inner() as _)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;

        // Scale 0: sRGB → positive XYB for both images (only the real
        // [0..logical_w) cols are written by the color kernel).
        let s0 = &mut self.scales[0];
        let dst_pitch = s0.xyb_ref[0].pitch() as usize;
        self.kernel.srgb_to_positive_xyb(
            &self.stream,
            self.ref_u8.device_ptr(),
            self.ref_u8.pitch() as usize,
            s0.xyb_ref[0].device_ptr_mut(),
            s0.xyb_ref[1].device_ptr_mut(),
            s0.xyb_ref[2].device_ptr_mut(),
            dst_pitch,
            self.width as usize,
            self.height as usize,
        );
        self.kernel.srgb_to_positive_xyb(
            &self.stream,
            self.dis_u8.device_ptr(),
            self.dis_u8.pitch() as usize,
            s0.xyb_dis[0].device_ptr_mut(),
            s0.xyb_dis[1].device_ptr_mut(),
            s0.xyb_dis[2].device_ptr_mut(),
            dst_pitch,
            self.width as usize,
            self.height as usize,
        );
        // Fill scale-0 padding cols with mirror-reflected copies (matches
        // CPU `convert_source_to_xyb` → streaming.rs line 859-876).
        if let Some(mo) = self.mirror_offsets[0].as_ref() {
            let s0 = &mut self.scales[0];
            let pitch = s0.xyb_ref[0].pitch() as usize;
            let logical = s0.logical_w as usize;
            let padded = s0.padded_w as usize;
            let height = s0.h as usize;
            for ch in 0..3 {
                self.kernel.pad_mirror_plane(
                    &self.stream,
                    s0.xyb_ref[ch].device_ptr_mut(),
                    pitch,
                    logical,
                    padded,
                    height,
                    mo.ptr() as *const u32,
                );
                self.kernel.pad_mirror_plane(
                    &self.stream,
                    s0.xyb_dis[ch].device_ptr_mut(),
                    pitch,
                    logical,
                    padded,
                    height,
                    mo.ptr() as *const u32,
                );
            }
        }

        // Scales 1..N: downscale both XYB pyramids from the previous
        // scale. Use `padded_w` everywhere so the padding cols (which
        // already contain mirror-reflected values at scale 0) propagate
        // naturally down the pyramid — this matches CPU zensim, which
        // downscales the whole padded plane in-place without re-padding.
        for s in 1..self.scales.len() {
            let (prev, curr) = {
                let (left, right) = self.scales.split_at_mut(s);
                (&left[s - 1], &mut right[0])
            };
            for ch in 0..3 {
                self.kernel.downscale_2x_plane(
                    &self.stream,
                    prev.xyb_ref[ch].device_ptr(),
                    prev.xyb_ref[ch].pitch() as usize,
                    curr.xyb_ref[ch].device_ptr_mut(),
                    curr.xyb_ref[ch].pitch() as usize,
                    prev.padded_w as usize,
                    prev.h as usize,
                    curr.padded_w as usize,
                    curr.h as usize,
                );
                self.kernel.downscale_2x_plane(
                    &self.stream,
                    prev.xyb_dis[ch].device_ptr(),
                    prev.xyb_dis[ch].pitch() as usize,
                    curr.xyb_dis[ch].device_ptr_mut(),
                    curr.xyb_dis[ch].pitch() as usize,
                    prev.padded_w as usize,
                    prev.h as usize,
                    curr.padded_w as usize,
                    curr.h as usize,
                );
            }
        }

        // Per scale, per channel: fused H-blur + fused V-blur+features.
        // The kernel writes 17 f64 + 3 u32 per channel into per-scale
        // device buffers. We collect them after running all channels
        // on the stream (one sync at the end of compute).
        for s in 0..self.scales.len() {
            // Borrow-split to get &mut kernel-write buffers + &ref planes.
            let sc = &mut self.scales[s];
            // Zero the accumulators for this scale's 3 channels.
            zero_cubox_f64(&mut sc.accum_f64, &self.stream)?;
            zero_cubox_u32(&mut sc.peak_u32, &self.stream)?;
            let pitch = sc.xyb_ref[0].pitch() as usize;
            for ch in 0..3 {
                // H-blur stage: input src[ch], dst[ch] → h_mu1, h_mu2,
                // h_sigma_sq, h_sigma12. Runs over padded width (same as
                // CPU zensim) so horizontal mirror lands at padded-w
                // boundaries — padding cols already hold mirror-copies.
                self.kernel.fused_blur_h_ssim(
                    &self.stream,
                    sc.xyb_ref[ch].device_ptr(),
                    sc.xyb_dis[ch].device_ptr(),
                    pitch,
                    sc.h_mu1.device_ptr_mut(),
                    sc.h_mu2.device_ptr_mut(),
                    sc.h_sigma_sq.device_ptr_mut(),
                    sc.h_sigma12.device_ptr_mut(),
                    pitch,
                    sc.padded_w as usize,
                    sc.h as usize,
                    BLUR_RADIUS,
                );
                // V-blur + features: writes to accum_f64[ch*17..] and
                // peak_u32[ch*3..]. Iterates over `padded_w` columns so
                // feature sums match CPU, whose accum_n = height * padded_w.
                let accum_ptr = unsafe { (sc.accum_f64.ptr() as *mut f64).add(ch * 17) };
                let peak_ptr = unsafe { (sc.peak_u32.ptr() as *mut u32).add(ch * 3) };
                self.kernel.fused_vblur_features_ssim(
                    &self.stream,
                    sc.h_mu1.device_ptr(),
                    sc.h_mu2.device_ptr(),
                    sc.h_sigma_sq.device_ptr(),
                    sc.h_sigma12.device_ptr(),
                    sc.xyb_ref[ch].device_ptr(),
                    sc.xyb_dis[ch].device_ptr(),
                    pitch,
                    sc.padded_w as usize,
                    sc.h as usize,
                    BLUR_RADIUS,
                    accum_ptr,
                    peak_ptr,
                );
            }
        }

        // Sync + D2H collect per-scale per-channel accumulators.
        self.stream
            .sync()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        // Layout of feature vector must match zensim CPU's
        // `combine_scores` (zen/zensim/src/metric.rs line 1641):
        //   Pass 1 ("scored" block, 13/ch × 3ch × N scales = 156):
        //     for each scale s: for each channel c:
        //       [ssim_mean, ssim_4th, ssim_2nd,
        //        art_mean, art_4th, art_2nd,
        //        det_mean, det_4th, det_2nd,
        //        mse, hf_energy_loss, hf_mag_loss, hf_energy_gain]
        //       (all .abs(); HF ratios clamped to [0, 1] / [0, ∞))
        //   Pass 2 ("peaks" block, 6/ch × 3ch × N = 72):
        //     for each scale s: for each channel c:
        //       [ssim_max, art_max, det_max,
        //        ssim_p95, art_p95, det_p95]
        //       (p95 here is the L8 root pool; see metric.rs line 1666)
        //
        // Total: 228 for N=4 scales, matching WEIGHTS_PREVIEW_V0_2.
        //
        // Per-scale kernel accumulators (per channel):
        //   accum_f64[0]  = Σ sd          (→ ssim_mean after /n)
        //   accum_f64[1]  = Σ sd⁴         (→ ssim_4th = (·/n)^0.25)
        //   accum_f64[2]  = Σ sd²         (→ ssim_2nd = (·/n)^0.5)
        //   accum_f64[3]  = Σ artifact
        //   accum_f64[4]  = Σ artifact⁴
        //   accum_f64[5]  = Σ artifact²
        //   accum_f64[6]  = Σ detail_lost
        //   accum_f64[7]  = Σ detail_lost⁴
        //   accum_f64[8]  = Σ detail_lost²
        //   accum_f64[9]  = Σ (src-dst)²  (→ mse)
        //   accum_f64[10] = Σ (src-mu1)²  (→ hf_L2_src for HF ratios)
        //   accum_f64[11] = Σ (dst-mu2)²  (→ hf_L2_dst)
        //   accum_f64[12] = Σ |src-mu1|   (→ hf_L1_src)
        //   accum_f64[13] = Σ |dst-mu2|   (→ hf_L1_dst)
        //   accum_f64[14] = Σ sd⁸         (→ ssim_p95 = (·/n)^0.125)
        //   accum_f64[15] = Σ artifact⁸   (→ art_p95)
        //   accum_f64[16] = Σ detail_lost⁸(→ det_p95)
        //   peak_u32[0..3] = ssim_max, art_max, det_max

        let mut out = [0.0_f64; TOTAL_FEATURES];
        let mut host_f64 = vec![0.0_f64; 17 * 3];
        let mut host_u32 = vec![0_u32; 3 * 3];
        let n_scales = self.scales.len();
        let basic_total = n_scales * FEATURES_PER_CHANNEL_BASIC * 3;
        let mut raw_out: Vec<Vec<([f64; 17], [u32; 3])>> = if return_raw {
            Vec::with_capacity(n_scales)
        } else {
            Vec::new()
        };

        for s in 0..n_scales {
            let sc = &self.scales[s];
            unsafe {
                cudarse_driver::sys::cuMemcpyDtoH_v2(
                    host_f64.as_mut_ptr() as *mut _,
                    sc.accum_f64.ptr(),
                    17 * 3 * 8,
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                cudarse_driver::sys::cuMemcpyDtoH_v2(
                    host_u32.as_mut_ptr() as *mut _,
                    sc.peak_u32.ptr(),
                    3 * 3 * 4,
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
            // Feature-sum denominator must match CPU's `accum.n`, which
            // is `padded_w × h` (streaming.rs line 1379). GPU sums over
            // the same footprint.
            let n_pixels = (sc.padded_w as usize) * (sc.h as usize);
            let inv_n = 1.0 / n_pixels as f64;

            // HF ratio clamp matches CPU (metric.rs line 85-87): both
            // `hf_energy_loss` and `hf_energy_gain` are `max(0, …)`.
            let safe_ratio =
                |num: f64, den: f64| -> f64 { if den.abs() > 0.0 { num / den } else { 0.0 } };

            if return_raw {
                let mut scale_raw: Vec<([f64; 17], [u32; 3])> = Vec::with_capacity(3);
                for ch in 0..3 {
                    let mut f = [0.0f64; 17];
                    f.copy_from_slice(&host_f64[ch * 17..ch * 17 + 17]);
                    let mut u = [0u32; 3];
                    u.copy_from_slice(&host_u32[ch * 3..ch * 3 + 3]);
                    scale_raw.push((f, u));
                }
                raw_out.push(scale_raw);
            }

            for ch in 0..3 {
                let raw = &host_f64[ch * 17..ch * 17 + 17];
                let peaks = &host_u32[ch * 3..ch * 3 + 3];

                let ratio_l2 = safe_ratio(raw[11], raw[10]);
                let ratio_l1 = safe_ratio(raw[13], raw[12]);

                // Basic (scored) 13-feature block, scales-major, channel-minor.
                let basic_base =
                    s * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC;
                out[basic_base] = (raw[0] * inv_n).abs();
                out[basic_base + 1] = (raw[1] * inv_n).max(0.0).powf(0.25);
                out[basic_base + 2] = (raw[2] * inv_n).max(0.0).sqrt();
                out[basic_base + 3] = (raw[3] * inv_n).abs();
                out[basic_base + 4] = (raw[4] * inv_n).max(0.0).powf(0.25);
                out[basic_base + 5] = (raw[5] * inv_n).max(0.0).sqrt();
                out[basic_base + 6] = (raw[6] * inv_n).abs();
                out[basic_base + 7] = (raw[7] * inv_n).max(0.0).powf(0.25);
                out[basic_base + 8] = (raw[8] * inv_n).max(0.0).sqrt();
                out[basic_base + 9] = raw[9] * inv_n;
                out[basic_base + 10] = (1.0 - ratio_l2).max(0.0);
                out[basic_base + 11] = (1.0 - ratio_l1).max(0.0);
                out[basic_base + 12] = (ratio_l2 - 1.0).max(0.0);

                // Peaks 6-feature block, appended after the full basic
                // region.
                let peak_base = basic_total
                    + s * 3 * FEATURES_PER_CHANNEL_PEAKS
                    + ch * FEATURES_PER_CHANNEL_PEAKS;
                out[peak_base] = f32::from_bits(peaks[0]) as f64;
                out[peak_base + 1] = f32::from_bits(peaks[1]) as f64;
                out[peak_base + 2] = f32::from_bits(peaks[2]) as f64;
                out[peak_base + 3] = (raw[14] * inv_n).max(0.0).powf(0.125);
                out[peak_base + 4] = (raw[15] * inv_n).max(0.0).powf(0.125);
                out[peak_base + 5] = (raw[16] * inv_n).max(0.0).powf(0.125);
            }
        }
        Ok((out, raw_out))
    }
}

impl Drop for Zensim {
    fn drop(&mut self) {
        let _ = cudarse_driver::sync_ctx();
    }
}

fn zero_cubox_f64(b: &mut CuBox<[f64]>, stream: &CuStream) -> Result<(), Error> {
    unsafe {
        // f64 zeros = 0-pattern bits, 2 u32 per f64.
        cudarse_driver::sys::cuMemsetD32Async(b.ptr(), 0, 17 * 3 * 2, stream.raw())
            .result()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
    }
    Ok(())
}

fn zero_cubox_u32(b: &mut CuBox<[u32]>, stream: &CuStream) -> Result<(), Error> {
    unsafe {
        cudarse_driver::sys::cuMemsetD32Async(b.ptr(), 0, 3 * 3, stream.raw())
            .result()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
    }
    Ok(())
}
