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

use cudarse_driver::{CuBox, CuGraphExec, CuStream};
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

    // ---- Reference cache (for set_reference / compute_with_reference) ----
    /// True if [`Self::set_reference`] has been called since the last
    /// [`Self::clear_reference`]. When set, every `scales[s].xyb_ref[ch]`
    /// plane holds the reference-side XYB pyramid, ready to be read by
    /// the per-channel blur + feature kernels.
    reference_cache_valid: bool,

    /// Cached CUDA graph for [`Self::compute_with_reference`]. Captured
    /// lazily on the first call after `set_reference`. Keyed by the
    /// pointer pair `(ref_xyb_ptr, dis_u8_ptr)` so the graph is
    /// invalidated if either the reference buffer pointer or the
    /// distorted upload buffer pointer changes. In the common case
    /// (one `Zensim` instance, reused staging), those pointers are
    /// stable and the graph captures once then replays forever.
    compute_graph: Option<(u64, u64, CuGraphExec)>,
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
            reference_cache_valid: false,
            compute_graph: None,
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
        self.set_reference(source_rgb)?;
        self.compute_with_reference_inner(distorted_rgb, true)
            .map(|(_, raw)| raw)
    }

    /// Score `distorted` against `source`. Both must be tightly packed
    /// sRGB RGB u8 at the (width, height) this instance was created for.
    /// Returns the 228-entry feature vector; apply
    /// [`score_from_features`] with the trained weights to turn this
    /// into a 0-100 score.
    ///
    /// Thin convenience shim over [`Self::set_reference`] +
    /// [`Self::compute_with_reference`]. Callers that score many
    /// distorted variants against one reference should prefer the split
    /// API — `compute_with_reference` skips the reference-side kernels
    /// and replays a cached CUDA graph.
    pub fn compute_features(
        &mut self,
        source_rgb: &[u8],
        distorted_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES], Error> {
        self.set_reference(source_rgb)?;
        self.compute_with_reference(distorted_rgb)
    }

    /// Upload `source_rgb` to the GPU, convert it to positive-XYB, pad
    /// the SIMD mirror columns, and downscale the full reference pyramid.
    /// The cached pyramid lives in `scales[s].xyb_ref[ch]`; every
    /// subsequent [`Self::compute_with_reference`] reads it without
    /// re-running any reference-side kernel.
    ///
    /// Invalidates any cached compute graph (distorted-side pipeline
    /// reads ref pointers which are stable across calls, but we
    /// conservatively drop the graph so the first post-set_reference
    /// call re-captures against fresh state).
    pub fn set_reference(&mut self, source_rgb: &[u8]) -> Result<(), Error> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if source_rgb.len() != expected {
            return Err(Error::InvalidDimensions(format!(
                "expected {} bytes for reference, got {}",
                expected,
                source_rgb.len()
            )));
        }
        // Do NOT drop self.compute_graph here: the graph is keyed on
        // `(ref_xyb_ptr, dis_u8_ptr)` and those pointers are stable
        // across set_reference calls (set_reference only writes new
        // *contents* into the same pre-allocated buffers). If we
        // dropped it, every set_reference would force a graph
        // re-capture on the next compute_with_reference — defeating
        // the entire point of the graph when the caller rotates
        // sources. Instead, we rely on the pointer-pair check in
        // compute_with_reference_inner to decide when to recapture,
        // and on `clear_reference` to explicitly drop it.

        self.ref_u8
            .copy_from_cpu(source_rgb, self.stream.inner() as _)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;

        self.run_reference_pipeline()?;

        // Sync so subsequent compute_with_reference calls — which may
        // be captured into a graph — observe the reference buffers as
        // already-populated.
        self.stream
            .sync()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        self.reference_cache_valid = true;
        Ok(())
    }

    /// Compute the feature vector of `distorted_rgb` against the
    /// reference cached by the most recent [`Self::set_reference`] call.
    /// Captures the distorted-side pipeline (~15 kernels + memcpys)
    /// into a CUDA graph on the first call and replays it on every
    /// subsequent call, dropping per-call launch overhead from ~6 ms
    /// to ~100 μs on WSL2. The graph is keyed on
    /// `(ref_xyb_ptr, dis_u8_ptr)` — both stable when one `Zensim`
    /// instance is reused, so the graph captures once.
    pub fn compute_with_reference(
        &mut self,
        distorted_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES], Error> {
        self.compute_with_reference_inner(distorted_rgb, false)
            .map(|(feat, _)| feat)
    }

    /// True if [`Self::set_reference`] has been called since the last
    /// [`Self::clear_reference`] (or instance creation).
    pub fn has_reference(&self) -> bool {
        self.reference_cache_valid
    }

    /// Invalidate the cached reference. The next
    /// [`Self::compute_with_reference`] call will error until
    /// [`Self::set_reference`] is called again. Also drops the cached
    /// compute graph (it's keyed on pointers which remain stable, but
    /// clearing the cache is a clear signal of intent).
    pub fn clear_reference(&mut self) {
        self.reference_cache_valid = false;
        self.compute_graph = None;
    }

    fn compute_with_reference_inner(
        &mut self,
        distorted_rgb: &[u8],
        return_raw: bool,
    ) -> Result<([f64; TOTAL_FEATURES], Vec<Vec<([f64; 17], [u32; 3])>>), Error> {
        if !self.reference_cache_valid {
            return Err(Error::Cuda(
                "no cached reference; call set_reference() first".into(),
            ));
        }
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if distorted_rgb.len() != expected {
            return Err(Error::InvalidDimensions(format!(
                "expected {} bytes for distorted, got {}",
                expected,
                distorted_rgb.len()
            )));
        }

        // Upload distorted RGB. Stays outside the captured graph —
        // copy_from_cpu goes through NPP's own stream glue and may
        // allocate pinned staging, neither of which is graph-safe. The
        // H2D itself is async on our stream; the kernels inside the
        // graph will serialize behind it automatically since they're
        // on the same stream.
        self.dis_u8
            .copy_from_cpu(distorted_rgb, self.stream.inner() as _)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;

        // Capture-or-replay. Graph key: (ref_xyb ptr, dis_u8 ptr).
        // Both are stable for the life of the instance, so in practice
        // the graph captures on the first call and replays forever.
        let ref_ptr = self.scales[0].xyb_ref[0].device_ptr() as u64;
        let dis_ptr = self.dis_u8.device_ptr() as u64;
        let need_capture = match &self.compute_graph {
            Some((cached_ref, cached_dis, _)) => *cached_ref != ref_ptr || *cached_dis != dis_ptr,
            None => true,
        };

        if need_capture {
            self.compute_graph = None;
            if std::env::var_os("ZENSIM_GRAPH_DEBUG").is_some() {
                eprintln!(
                    "zensim: capturing graph (ref_ptr={:x}, dis_ptr={:x})",
                    ref_ptr, dis_ptr
                );
            }
            self.stream
                .begin_capture_thread_local()
                .map_err(|e| Error::Cuda(format!("begin_capture: {:?}", e)))?;
            let run_result = self.run_distorted_pipeline();
            let end_result = self
                .stream
                .end_capture()
                .map_err(|e| Error::Cuda(format!("end_capture: {:?}", e)));
            let graph = match (run_result, end_result) {
                (Ok(()), Ok(g)) => g,
                (Err(e), _) => return Err(e),
                (_, Err(e)) => return Err(e),
            };
            let exec = graph
                .instantiate()
                .map_err(|e| Error::Cuda(format!("instantiate: {:?}", e)))?;
            self.compute_graph = Some((ref_ptr, dis_ptr, exec));
        }

        // Replay (or freshly-captured graph's first launch).
        self.compute_graph
            .as_ref()
            .expect("compute_graph set above")
            .2
            .launch(&self.stream)
            .map_err(|e| Error::Cuda(format!("graph launch: {:?}", e)))?;

        // D2H is done asynchronously on the same stream as the graph
        // launch, so the 8 per-scale copies (f64 accumulator + u32 peak
        // per scale × 4 scales) pipeline behind the graph without a
        // blocking round trip each. `collect_features` issues them on
        // the stream and syncs once at the end — that single sync cuts
        // warm-path cost from ~8 ms (blocking cuMemcpyDtoH_v2 × 8) to
        // kernel-compute + 1 synchronisation.
        self.collect_features(return_raw)
    }

    /// Run the reference-side pipeline: srgb→XYB, mirror-pad, downscale
    /// pyramid. Writes into `scales[s].xyb_ref[ch]`. Called by
    /// [`Self::set_reference`]; NOT captured into a graph (runs once
    /// per source, graph capture overhead isn't worth it).
    fn run_reference_pipeline(&mut self) -> Result<(), Error> {
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
            }
        }
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
            }
        }
        Ok(())
    }

    /// Run the distorted-side pipeline: srgb→XYB on dis, mirror-pad,
    /// downscale pyramid, then per-scale-per-channel fused H-blur +
    /// fused V-blur+features against the cached reference pyramid.
    /// Writes accumulator sums into `scales[s].accum_f64` and
    /// `scales[s].peak_u32`. Safe to run inside `begin_capture()` /
    /// `end_capture()` — every op is async on `self.stream` and no
    /// host syncs happen here.
    fn run_distorted_pipeline(&mut self) -> Result<(), Error> {
        // Scale 0: sRGB → positive XYB for distorted.
        let s0 = &mut self.scales[0];
        let dst_pitch = s0.xyb_dis[0].pitch() as usize;
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
        if let Some(mo) = self.mirror_offsets[0].as_ref() {
            let s0 = &mut self.scales[0];
            let pitch = s0.xyb_dis[0].pitch() as usize;
            let logical = s0.logical_w as usize;
            let padded = s0.padded_w as usize;
            let height = s0.h as usize;
            for ch in 0..3 {
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
        // Scales 1..N: downscale distorted pyramid only (ref already
        // downsampled by set_reference).
        for s in 1..self.scales.len() {
            let (prev, curr) = {
                let (left, right) = self.scales.split_at_mut(s);
                (&left[s - 1], &mut right[0])
            };
            for ch in 0..3 {
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
        for s in 0..self.scales.len() {
            let sc = &mut self.scales[s];
            zero_cubox_f64(&mut sc.accum_f64, &self.stream)?;
            zero_cubox_u32(&mut sc.peak_u32, &self.stream)?;
            let pitch = sc.xyb_ref[0].pitch() as usize;
            for ch in 0..3 {
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
        Ok(())
    }

    /// D2H the per-scale per-channel accumulators (after the pipeline's
    /// already been synced) and transform into the 228-feature vector
    /// matching `zen/zensim/src/metric.rs::combine_scores`.
    fn collect_features(
        &mut self,
        return_raw: bool,
    ) -> Result<([f64; TOTAL_FEATURES], Vec<Vec<([f64; 17], [u32; 3])>>), Error> {
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

        // Pre-stage one host-side buffer per type spanning ALL scales,
        // so we issue 2*n_scales async D2H copies then sync once. The
        // blocking per-scale version was ~7ms of our 8ms warm path.
        let mut host_f64_all = vec![0.0_f64; 17 * 3 * n_scales];
        let mut host_u32_all = vec![0_u32; 3 * 3 * n_scales];
        for s in 0..n_scales {
            let sc = &self.scales[s];
            unsafe {
                cudarse_driver::sys::cuMemcpyDtoHAsync_v2(
                    host_f64_all.as_mut_ptr().add(s * 17 * 3) as *mut _,
                    sc.accum_f64.ptr(),
                    17 * 3 * 8,
                    self.stream.inner() as _,
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                cudarse_driver::sys::cuMemcpyDtoHAsync_v2(
                    host_u32_all.as_mut_ptr().add(s * 3 * 3) as *mut _,
                    sc.peak_u32.ptr(),
                    3 * 3 * 4,
                    self.stream.inner() as _,
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
        }
        // Single round-trip sync: waits for graph + all 2*n_scales D2H.
        self.stream
            .sync()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        for s in 0..n_scales {
            host_f64.copy_from_slice(&host_f64_all[s * 17 * 3..(s + 1) * 17 * 3]);
            host_u32.copy_from_slice(&host_u32_all[s * 3 * 3..(s + 1) * 3 * 3]);
            let sc = &self.scales[s];
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
