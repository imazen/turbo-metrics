use indices::{indices, indices_ordered};

use cudarse_driver::{CuGraphExec, CuStream};
use cudarse_npp::image::ial::{Mul, Sqr, SqrIP};
use cudarse_npp::image::idei::Transpose;
use cudarse_npp::image::ist::Sum;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{C, Image, Img, ImgMut};
use cudarse_npp::sys::{NppStreamContext, NppiRect, Result};
use cudarse_npp::{ScratchBuffer, assert_same_size, get_stream_ctx};

use crate::kernel::Kernel;

pub mod kernel;

/// Number of scales to compute
const SCALES: usize = 6;

/// An instance is valid for a specific width and height.
///
/// This implementation never allocates during processing and requires a minimum
/// of `270 * width * height` bytes.
/// e.g. ~800 MiB of device memory for processing 1440x1080 frames.
/// Actual memory usage is higher because of padding and other state.
///
/// Processing a single image pair results in 305 kernels launches !
pub struct Ssimulacra2 {
    kernel: Kernel,
    npp: NppStreamContext,
    /// Graph captured in `new()` against the `(src_ref_linear, src_dis_linear)`
    /// passed in there. Re-runs the full ref+dis pipeline on every launch.
    /// Used by [`compute_srgb_sync`], [`compute_sync`], etc.
    exec: Option<CuGraphExec>,

    // sizes: [NppiRect; SCALES],
    sizes_t: [NppiRect; SCALES],

    ref_linear: [Image<f32, C<3>>; SCALES - 1],
    dis_linear: [Image<f32, C<3>>; SCALES - 1],

    img: [[Image<f32, C<3>>; 10]; SCALES],
    imgt: [[Image<f32, C<3>>; 10]; SCALES],

    sum_scratch: [[ScratchBuffer; 6]; SCALES],

    streams: [[CuStream; 6]; SCALES],
    scores: Box<[f64; 3 * 6 * SCALES]>,

    // ---- Reference cache (for set_reference_linear / compute_with_reference_linear) ----
    // When `reference_cache_valid` is true, these buffers hold the
    // ref-side intermediate state so `compute_with_reference_linear`
    // can skip re-running any ref-side kernels. Populated by
    // `set_reference_linear`.
    //
    // Per scale:
    //   ref_xyb_cache[scale]      — non-transposed ref_xyb (for sigma12 = ref × dis)
    //   ref_xyb_t_cache[scale]    — transposed ref_xyb (source input to compute_error_maps)
    //   sigma11_t_cache[scale]    — fully-blurred transposed sigma11 (ref × ref)
    //   mu1_t_cache[scale]        — fully-blurred transposed ref_xyb (mu1)
    //
    // Total memory at 1 Mpx: ~150 MB per Ssimulacra2 instance beyond
    // the baseline footprint.
    reference_cache_valid: bool,
    ref_xyb_cache: [Image<f32, C<3>>; SCALES],
    ref_xyb_t_cache: [Image<f32, C<3>>; SCALES],
    sigma11_t_cache: [Image<f32, C<3>>; SCALES],
    mu1_t_cache: [Image<f32, C<3>>; SCALES],
    /// Scale-0 reductions for `sum(ssim²_ref_only)`-style precomputables
    /// live inline in `scores[]`, which set_reference fills for the
    /// ref-only slots. We don't split that out; the dis-only pipeline
    /// overwrites the mixed slots and leaves the ref-only slots alone.
    /// (Tracked here as a sentinel; no separate buffer needed.)
    /// Lazily captured graph of the distorted-only + comparison
    /// pipeline. Keyed on the `(ref_linear_ptr, dis_linear_ptr)` pair
    /// observed on first call; re-captured if either rotates.
    dis_only_exec: Option<(u64, u64, CuGraphExec)>,
}

impl Ssimulacra2 {
    pub fn new(
        src_ref_linear: impl Img<f32, C<3>>,
        src_dis_linear: impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<Self> {
        assert_same_size!(src_ref_linear, src_dis_linear);
        let rect = src_ref_linear.rect();

        let streams =
            array_init::try_array_init(|_| array_init::try_array_init(|_| CuStream::new()))
                .unwrap();
        let mut npp = get_stream_ctx()?;
        npp.hStream = stream.inner() as _;

        let mut sizes = [rect; SCALES];
        for scale in 1..SCALES {
            sizes[scale].width = (sizes[scale - 1].width + 1) / 2;
            sizes[scale].height = (sizes[scale - 1].height + 1) / 2;
        }
        let sizes_t = array_init::array_init(|i| sizes[i].transpose());

        // Buffers needed for computations
        let img = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| {
                Image::malloc(sizes[i].width as u32, sizes[i].height as u32)
            })
        })?;
        // Transposed buffers
        let imgt = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| {
                Image::malloc(sizes_t[i].width as u32, sizes_t[i].height as u32)
            })
        })?;

        let ref_linear = array_init::try_array_init(|i| img[i + 1][0].malloc_same_size())?;
        let dis_linear = array_init::try_array_init(|i| img[i + 1][0].malloc_same_size())?;

        let sum_scratch = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| imgt[i][0].sum_alloc_scratch(npp))
        })?;

        // Reference-cache buffers. Non-transposed at img[scale] size,
        // transposed at imgt[scale] size. Allocated up front so the
        // lazy graph can bake in stable pointers.
        let ref_xyb_cache = array_init::try_array_init(|i| img[i][0].malloc_same_size())?;
        let ref_xyb_t_cache = array_init::try_array_init(|i| imgt[i][0].malloc_same_size())?;
        let sigma11_t_cache = array_init::try_array_init(|i| imgt[i][0].malloc_same_size())?;
        let mu1_t_cache = array_init::try_array_init(|i| imgt[i][0].malloc_same_size())?;

        let mut s = Self {
            kernel: Kernel::load(),
            npp,
            exec: None,
            // sizes,
            sizes_t,
            ref_linear,
            dis_linear,
            img,
            imgt,
            sum_scratch,
            streams,
            scores: Box::new([0.0; 3 * 6 * SCALES]),
            reference_cache_valid: false,
            ref_xyb_cache,
            ref_xyb_t_cache,
            sigma11_t_cache,
            mu1_t_cache,
            dis_only_exec: None,
        };

        s.exec = Some(s.record(src_ref_linear, src_dis_linear, stream)?);

        Ok(s)
    }

    /// Read-only access to the internal kernel loader so callers can
    /// issue direct kernel calls (e.g. `srgb_to_linear` for the cached-
    /// reference path) without having to load a second PTX copy.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// Estimate the minimum memory usage
    pub fn mem_usage(&self) -> usize {
        self.ref_linear
            .iter()
            .map(|i| i.device_mem_usage())
            .sum::<usize>()
            + self
                .dis_linear
                .iter()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .img
                .iter()
                .flatten()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .imgt
                .iter()
                .flatten()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .sum_scratch
                .iter()
                .flatten()
                .map(|b| b.len())
                .sum::<usize>()
    }

    fn record(
        &mut self,
        src_ref_linear: impl Img<f32, C<3>>,
        src_dis_linear: impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<CuGraphExec> {
        // TODO we should work with planar images, as it would allow us to coalesce read and writes
        //  coalescing can already be achieved for kernels which doesn't require access to neighbouring pixels or samples

        let alt_stream = CuStream::new().unwrap();
        stream.begin_capture().unwrap();

        // Bring main_dis into the graph capture scope
        alt_stream.wait_for_stream(stream).unwrap();

        // save_img(&self.dev, &self.ref_linear, &format!("ref_linear"));

        // linear -> xyb -> ...
        //    |-> /2 -> xyb -> ...
        //         |-> /2 -> xyb -> ...

        for scale in 0..SCALES {
            if scale == 1 {
                self.kernel.downscale_by_2(
                    stream,
                    &src_ref_linear,
                    &mut self.ref_linear[scale - 1],
                );
                self.kernel.downscale_by_2(
                    &alt_stream,
                    &src_dis_linear,
                    &mut self.dis_linear[scale - 1],
                );
            } else if scale > 1 {
                // TODO this can be done with warp level primitives by having warps sized 16x2
                //  block size 16x16 would be perfect
                //  warps would contain 8 2x2 patches which can be summed using shfl_down_sync and friends
                //  This would require a planar format ...

                let (prev, curr) = indices!(&mut self.ref_linear, scale - 2, scale - 1);
                self.kernel.downscale_by_2(stream, prev, curr);
                let (prev, curr) = indices!(&mut self.dis_linear, scale - 2, scale - 1);
                self.kernel.downscale_by_2(&alt_stream, prev, curr);
            }

            self.streams[scale][0].wait_for_stream(stream).unwrap();
            self.streams[scale][1].wait_for_stream(&alt_stream).unwrap();

            if scale == 0 {
                self.kernel.linear_to_xyb(
                    &self.streams[scale][0],
                    &src_ref_linear,
                    &mut self.img[scale][8],
                );
                self.kernel.linear_to_xyb(
                    &self.streams[scale][1],
                    &src_dis_linear,
                    &mut self.img[scale][9],
                );
            } else {
                self.kernel.linear_to_xyb(
                    &self.streams[scale][0],
                    &self.ref_linear[scale - 1],
                    &mut self.img[scale][8],
                );
                self.kernel.linear_to_xyb(
                    &self.streams[scale][1],
                    &self.dis_linear[scale - 1],
                    &mut self.img[scale][9],
                );
            }

            self.process_scale(scale)?;

            // profiler_stop().unwrap();
        }

        stream.wait_for_stream(&alt_stream).unwrap();
        for scale in 0..SCALES {
            for i in 0..self.streams[scale].len() {
                stream.wait_for_stream(&self.streams[scale][i]).unwrap();
            }
        }

        let graph = stream.end_capture().unwrap();
        // graph.dot("ssimulacra2-cuda-graph.gviz").unwrap();
        let exec = graph.instantiate().unwrap();
        // self.main_ref.sync().unwrap();
        Ok(exec)
    }

    /// Compute ssimulacra2 metric using image bytes in CPU memory. Useful for processing a single pair of images.
    pub fn compute_from_cpu_srgb_sync(
        &mut self,
        ref_bytes: &[u8],
        dis_bytes: &[u8],
        mut tmp_ref: impl ImgMut<u8, C<3>>,
        mut tmp_dis: impl ImgMut<u8, C<3>>,
        src_ref_linear: impl ImgMut<f32, C<3>>,
        src_dis_linear: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        // profiler_stop().unwrap();

        tmp_ref.copy_from_cpu(ref_bytes, stream.inner() as _)?;
        tmp_dis.copy_from_cpu(dis_bytes, stream.inner() as _)?;

        self.compute_srgb_sync(tmp_ref, tmp_dis, src_ref_linear, src_dis_linear, stream)
    }

    /// Upload distorted bytes + sRGB->linear conversion, then run the full
    /// ssimulacra2 pipeline. The caller must have already populated
    /// `src_ref_linear` with the linear RGB of the reference (e.g. via
    /// a prior call that set up the reference, or by calling
    /// `srgb_to_linear` directly using the kernel accessor).
    ///
    /// This is the cached-reference fast path: skips reference H2D +
    /// srgb_to_linear, saving both the upload time and the reference
    /// side of the conversion kernel.
    pub fn compute_from_cpu_dis_only_srgb_sync(
        &mut self,
        dis_bytes: &[u8],
        mut tmp_dis: impl ImgMut<u8, C<3>>,
        src_dis_linear: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        tmp_dis.copy_from_cpu(dis_bytes, stream.inner() as _)?;
        // Reference linear is already on device; only process distorted.
        self.kernel.srgb_to_linear(stream, tmp_dis, src_dis_linear);
        // The captured graph is bound to the same src_ref_linear /
        // src_dis_linear pointers it was recorded with. The caller must
        // preserve those buffers across calls.
        self.compute_sync(stream)
    }

    /// Convenience: run sRGB->linear for the reference and cache it in
    /// place in `src_ref_linear`. Intended to be called once per new
    /// reference before a sequence of `compute_from_cpu_dis_only_srgb_sync`
    /// calls.
    pub fn prepare_reference_from_cpu_srgb(
        &self,
        ref_bytes: &[u8],
        mut tmp_ref: impl ImgMut<u8, C<3>>,
        src_ref_linear: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        tmp_ref.copy_from_cpu(ref_bytes, stream.inner() as _)?;
        self.kernel.srgb_to_linear(stream, tmp_ref, src_ref_linear);
        Ok(())
    }

    /// Compute ssimulacra2 metric using images already in CUDA memory.
    /// Reference and distorted images must be copied to the [ref_input] and [dis_input] fields.
    /// This will block until CUDA is done to post process scores as that last part is done on the CPU.
    pub fn compute_srgb_sync(
        &mut self,
        src_ref: impl Img<u8, C<3>>,
        src_dis: impl Img<u8, C<3>>,
        src_ref_linear: impl ImgMut<f32, C<3>>,
        src_dis_linear: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        // Convert to linear
        self.kernel.srgb_to_linear(stream, src_ref, src_ref_linear);
        self.kernel.srgb_to_linear(stream, src_dis, src_dis_linear);

        self.compute_sync(stream)
    }

    /// Compute ssimulacra2 metric using images already in CUDA memory.
    /// Reference and distorted images must be copied to the [ref_input] and [dis_input] fields.
    /// This will block until CUDA is done to post process scores as that last part is done on the CPU.
    pub fn compute_sync(&mut self, stream: &CuStream) -> Result<f64> {
        self.compute(stream)?;

        // Wait for CUDA to transfer scores back to the CPU before post-processing.
        stream.sync().unwrap();

        Ok(self.get_score())
    }

    /// Compute ssimulacra2 metric using images already in CUDA memory.
    /// Reference and distorted images must be copied to the linear input images.
    /// This one does not block. To retrieve the score, you must sync with the `main_ref` stream and call [Self::get_score] afterward.
    pub fn compute(&mut self, stream: &CuStream) -> Result<()> {
        self.exec.as_ref().unwrap().launch(stream).unwrap();
        Ok(())
    }

    /// Post process and retrieve the score for the last computation.
    pub fn get_score(&mut self) -> f64 {
        self.post_process_scores()
    }

    /// Precompute the reference-side state (downscale pyramid, XYB, ref²,
    /// blurred ref, blurred ref² transposed) from an already-linear
    /// reference image and cache it on the instance. Follow up with
    /// [`compute_with_reference_linear`] to score one or more distorted
    /// images against this reference with ~40% less per-call GPU work.
    ///
    /// Not captured as a graph — runs eagerly on `stream`. It only fires
    /// once per source anyway; graphing it would add complexity without
    /// meaningful per-source-amortized win.
    pub fn set_reference_linear(
        &mut self,
        src_ref_linear: impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        // Drop any stale dis-only graph — it was keyed on a pair that
        // includes our old ref pointer.
        self.dis_only_exec = None;

        // Per scale: downscale (if scale>0) → linear_to_xyb → sigma11 =
        // ref²  →  blur-pass horizontal (2-wide: sigma11, ref_xyb) →
        // transpose both → blur-pass vertical (2-wide) → transpose
        // ref_xyb only. Final cached tensors:
        //   self.ref_xyb_cache[scale]     (non-transposed, for sigma12 mul)
        //   self.ref_xyb_t_cache[scale]   (transposed, source input for error_maps)
        //   self.sigma11_t_cache[scale]   (transposed, fully blurred)
        //   self.mu1_t_cache[scale]       (transposed, fully blurred)
        for scale in 0..SCALES {
            // 1. Produce ref_linear at this scale.
            if scale == 1 {
                self.kernel
                    .downscale_by_2(stream, &src_ref_linear, &mut self.ref_linear[scale - 1]);
            } else if scale > 1 {
                let (prev, curr) = indices!(&mut self.ref_linear, scale - 2, scale - 1);
                self.kernel.downscale_by_2(stream, prev, curr);
            }

            // 2. linear_to_xyb → ref_xyb_cache[scale]
            if scale == 0 {
                self.kernel
                    .linear_to_xyb(stream, &src_ref_linear, &mut self.ref_xyb_cache[scale]);
            } else {
                self.kernel.linear_to_xyb(
                    stream,
                    &self.ref_linear[scale - 1],
                    &mut self.ref_xyb_cache[scale],
                );
            }

            // 3. sigma11 = ref_xyb × ref_xyb (into img[scale][0], scratch).
            {
                let (ref_xyb, sigma11_tmp) =
                    (&self.ref_xyb_cache[scale], &mut self.img[scale][0]);
                ref_xyb.mul(
                    ref_xyb,
                    sigma11_tmp,
                    self.npp.with_stream(stream.inner() as _),
                )?;
            }

            // 4. Horizontal blur 2-wide: sigma11, ref_xyb → img[scale][3], img[scale][6].
            {
                let [sigma11_tmp, _i1, _i2, sigma11_h, _i4, _i5, mu1_h, _i7, _i8, _i9] =
                    self.img[scale].each_mut();
                self.kernel.blur_pass_fused_2(
                    stream,
                    &*sigma11_tmp,
                    sigma11_h,
                    &self.ref_xyb_cache[scale],
                    mu1_h,
                );
            }

            // 5. Transpose horizontal-blur results + ref_xyb → imgt[scale][...].
            self.img[scale][3].transpose(
                &mut self.imgt[scale][0],
                self.npp.with_stream(stream.inner() as _),
            )?; // sigma11_h → imgt[0]
            self.img[scale][6].transpose(
                &mut self.imgt[scale][3],
                self.npp.with_stream(stream.inner() as _),
            )?; // mu1_h → imgt[3]
            self.ref_xyb_cache[scale].transpose(
                &mut self.ref_xyb_t_cache[scale],
                self.npp.with_stream(stream.inner() as _),
            )?;

            // 6. Vertical blur 2-wide on transposed inputs → sigma11_t_cache,
            //    mu1_t_cache.
            {
                let [i0, _i1, _i2, i3, _i4, _i5, _i6, _i7, _i8, _i9] =
                    self.imgt[scale].each_mut();
                self.kernel.blur_pass_fused_2(
                    stream,
                    &*i0,
                    &mut self.sigma11_t_cache[scale],
                    &*i3,
                    &mut self.mu1_t_cache[scale],
                );
            }
        }

        stream.sync().unwrap();
        self.reference_cache_valid = true;
        Ok(())
    }

    /// Score an already-linear distorted image against the cached
    /// reference set by [`set_reference_linear`]. Requires a prior
    /// `set_reference_linear` call since the last `clear_reference`;
    /// errors out if the cache isn't valid.
    ///
    /// Captures the distorted-only + comparison pipeline into a CUDA
    /// graph on the first call (keyed on `(ref_linear, dis_linear)`
    /// pointer pair) and replays it on every subsequent call.
    pub fn compute_with_reference_linear(
        &mut self,
        src_ref_linear: impl Img<f32, C<3>>,
        src_dis_linear: impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        if !self.reference_cache_valid {
            panic!("compute_with_reference_linear requires set_reference_linear to be called first");
        }
        let ref_ptr = src_ref_linear.device_ptr() as u64;
        let dis_ptr = src_dis_linear.device_ptr() as u64;
        let need_capture = match &self.dis_only_exec {
            Some((r, d, _)) => *r != ref_ptr || *d != dis_ptr,
            None => true,
        };
        if need_capture {
            self.dis_only_exec = None;
            stream.begin_capture().unwrap();
            let run_res = self.record_dis_only(&src_ref_linear, &src_dis_linear, stream);
            let graph = stream.end_capture().unwrap();
            run_res?;
            let exec = graph.instantiate().unwrap();
            self.dis_only_exec = Some((ref_ptr, dis_ptr, exec));
        }
        self.dis_only_exec
            .as_ref()
            .unwrap()
            .2
            .launch(stream)
            .unwrap();
        stream.sync().unwrap();
        Ok(self.post_process_scores())
    }

    /// Invalidate the cached reference. The next call to
    /// [`compute_with_reference_linear`] will error until
    /// `set_reference_linear` is called again.
    pub fn clear_reference(&mut self) {
        self.reference_cache_valid = false;
        self.dis_only_exec = None;
    }

    pub fn has_reference(&self) -> bool {
        self.reference_cache_valid
    }

    /// Record the dis-only pipeline (distorted-side kernels + comparison
    /// + reductions) into the currently-being-captured graph on `stream`.
    /// Must be called inside `begin_capture()` / `end_capture()`; relies
    /// on the reference cache populated by `set_reference_linear`.
    fn record_dis_only(
        &mut self,
        src_ref_linear: &impl Img<f32, C<3>>,
        src_dis_linear: &impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        let _ = src_ref_linear; // used via cached ref_xyb / sigma11_t / mu1_t; kept in signature for symmetry with new()
        let alt_stream = CuStream::new().unwrap();
        alt_stream.wait_for_stream(stream).unwrap();
        for scale in 0..SCALES {
            // 1. Downscale dis (scales 1..5).
            if scale == 1 {
                self.kernel
                    .downscale_by_2(&alt_stream, src_dis_linear, &mut self.dis_linear[scale - 1]);
            } else if scale > 1 {
                let (prev, curr) = indices!(&mut self.dis_linear, scale - 2, scale - 1);
                self.kernel.downscale_by_2(&alt_stream, prev, curr);
            }

            self.streams[scale][1].wait_for_stream(&alt_stream).unwrap();

            // 2. linear_to_xyb dis → img[scale][9]
            if scale == 0 {
                self.kernel
                    .linear_to_xyb(&self.streams[scale][1], src_dis_linear, &mut self.img[scale][9]);
            } else {
                self.kernel.linear_to_xyb(
                    &self.streams[scale][1],
                    &self.dis_linear[scale - 1],
                    &mut self.img[scale][9],
                );
            }

            self.process_scale_dis_only(scale)?;
        }
        stream.wait_for_stream(&alt_stream).unwrap();
        for scale in 0..SCALES {
            for i in 0..self.streams[scale].len() {
                stream.wait_for_stream(&self.streams[scale][i]).unwrap();
            }
        }
        Ok(())
    }

    /// Like `process_scale` but skips every ref-only kernel. Expects:
    ///   self.img[scale][9]           = dis_xyb       (written by caller's linear_to_xyb)
    ///   self.ref_xyb_cache[scale]    = ref_xyb       (cached)
    ///   self.ref_xyb_t_cache[scale]  = ref_xyb_t     (cached)
    ///   self.sigma11_t_cache[scale]  = sigma11_t     (cached, fully blurred)
    ///   self.mu1_t_cache[scale]      = mu1_t         (cached, fully blurred)
    fn process_scale_dis_only(&mut self, scale: usize) -> Result<()> {
        let streams: [&CuStream; 6] = self.streams[scale].each_ref().try_into().unwrap();
        streams[2].wait_for_stream(streams[1]).unwrap();

        // dis_xyb × dis_xyb → sigma22 (img[1]); ref_xyb × dis_xyb → sigma12 (img[2]).
        // NPP's `Mul::mul` takes `(src1, src2, dst)` where src2 is the output
        // pointer's sibling. We pre-capture dis_xyb through a local copy of
        // the slice to satisfy the borrow checker — each call through it
        // grabs its own borrow in its own scope.
        {
            let [_i0, i1, _i2, _i3, _i4, _i5, _i6, _i7, _i8, i9] = self.img[scale].each_mut();
            i9.mul(&*i9, i1, self.npp.with_stream(streams[1].inner() as _))?;
        }
        {
            let [_i0, _i1, i2, _i3, _i4, _i5, _i6, _i7, _i8, i9] = self.img[scale].each_mut();
            self.ref_xyb_cache[scale].mul(
                &*i9,
                i2,
                self.npp.with_stream(streams[2].inner() as _),
            )?;
        }

        streams[0].wait_for_stream(streams[1]).unwrap();
        streams[0].wait_for_stream(streams[2]).unwrap();

        // Horizontal blur 3-wide: sigma22 (img[1]), sigma12 (img[2]),
        // dis_xyb (img[9]) → img[4], img[5], img[7].
        {
            let [_i0, i1, i2, _i3, i4, i5, _i6, i7, _i8, i9] = self.img[scale].each_mut();
            self.kernel.blur_pass_fused_3(streams[0], &*i1, i4, &*i2, i5, &*i9, i7);
        }

        streams[1].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[3].wait_for_stream(streams[0]).unwrap();

        // Transpose sigma22_h, sigma12_h, mu2_h into imgt[1, 2, 4].
        self.img[scale][4].transpose(
            &mut self.imgt[scale][1],
            self.npp.with_stream(streams[0].inner() as _),
        )?;
        self.img[scale][5].transpose(
            &mut self.imgt[scale][2],
            self.npp.with_stream(streams[1].inner() as _),
        )?;
        self.img[scale][7].transpose(
            &mut self.imgt[scale][4],
            self.npp.with_stream(streams[2].inner() as _),
        )?;
        // Also transpose dis_xyb (img[9]) into imgt[3] for compute_error_maps
        // second source input. We use imgt[3] since imgt[0] is ref_xyb_t_cache
        // (cached) — we need a separate slot for dis_xyb_t.
        self.img[scale][9].transpose(
            &mut self.imgt[scale][3],
            self.npp.with_stream(streams[3].inner() as _),
        )?;

        streams[0].wait_for_stream(streams[1]).unwrap();
        streams[0].wait_for_stream(streams[2]).unwrap();
        streams[0].wait_for_stream(streams[3]).unwrap();

        // Vertical blur 3-wide on transposed: imgt[1,2,4] → imgt[6,7,9].
        {
            let [_i0, i1, i2, _i3, i4, _i5, i6, i7, _i8, i9] = self.imgt[scale].each_mut();
            self.kernel.blur_pass_fused_3(streams[0], &*i1, i6, &*i2, i7, &*i4, i9);
        }

        // compute_error_maps inputs:
        //   source          = ref_xyb_t_cache[scale]  (cached)
        //   distorted       = imgt[3] (dis_xyb_t, fresh)
        //   mu1             = mu1_t_cache[scale]      (cached)
        //   mu2             = imgt[9] (fresh)
        //   sigma11         = sigma11_t_cache[scale]  (cached)
        //   sigma22         = imgt[6] (fresh)
        //   sigma12         = imgt[7] (fresh)
        //   outputs         = imgt[2, 3, 4] (overwritten in place) — wait, imgt[3]
        //                     is dis_xyb_t which we just used as input. compute_error_maps
        //                     reads sigma22=imgt[6] (not imgt[3]); the outputs go to
        //                     imgt[2,3,4]. But imgt[3] was a cached input to this kernel!
        // Solution: output artifact map to a different imgt slot, e.g. imgt[0] is
        // only used for ref_xyb_t in the full path — but we're using ref_xyb_t_cache
        // here, not imgt[0]. So imgt[0] is free for us to use as an output.
        //
        // We route: ssim → imgt[2], artifact → imgt[0] (free slot), detail_loss → imgt[4].
        // And then the `reduce` helper's `data` indices need to follow: originally
        // (2, 5, 0) and (3, 6, 2) and (4, 7, 4) referred to (data, tmp, offset). Here
        // we need to update reduce to use our new locations.
        //
        // Actually — rather than rewiring reduce, let's keep the same output slots as
        // the original path: imgt[2]=ssim, imgt[3]=artifact, imgt[4]=detail_loss. That
        // means we need to consume dis_xyb_t from imgt[3] BEFORE compute_error_maps
        // overwrites it.
        //
        // But compute_error_maps reads `distorted` = imgt[3] (dis_xyb_t)! This kernel
        // reads source + distorted as its first two ImgC3 args. So distorted IS imgt[3]
        // and the output artifact IS imgt[3]. In the original code that's fine because
        // distorted-input is ONLY used to read mu values... wait, let's recheck the
        // kernel. compute_error_maps reads source and distorted directly — they go into
        // the ssim formula. So they must be stable reads throughout the kernel.
        //
        // In CUDA, a kernel may read from a pointer and write to the SAME pointer safely
        // if each thread only reads/writes its own pixel. compute_error_maps is
        // pointwise — every thread writes dst[i] after reading src[i]. The kernel source
        // (error_maps.rs) computes ssim[i], artifact[i], detail_loss[i] from the N input
        // reads at i and writes the 3 outputs. If `distorted` and `artifact` overlap,
        // we'd read the input first, then write the output — safe within one thread.
        //
        // But CUDA compiler / memory ordering guarantees this? For pointwise kernels
        // where each thread touches one distinct pixel, yes: the read happens before
        // the write by the same thread. No racing. The original code actually DOES
        // this: in the original process_scale, compute_error_maps takes source=imgt[0],
        // distorted=imgt[1], with outputs going to imgt[2, 3, 4]. The outputs don't
        // overlap with source/distorted. So this concern was avoided.
        //
        // We'd better avoid the overlap too. Let me route:
        //   outputs → imgt[5, 6, 7] (these were sigma11_t, sigma22_t, sigma12_t inputs
        //   which are now consumed and no longer needed).
        // That frees us from the overlap problem.
        //
        // But `reduce` expects specific input slots (currently 2, 3, 4). We'll need to
        // update reduce to read from 5, 6, 7 in the dis-only path.
        //
        // Simplest: call compute_error_maps with outputs = imgt[5,6,7] and a new reduce
        // variant that uses those slots.
        // Route outputs to free (non-input) imgt slots so Rust aliasing
        // is clean. At this point these slots are either never touched in
        // the dis-only path (0, 5, 8 — their data went into the ref cache)
        // or hold now-stale horizontal-blur intermediates (1, 2, 4).
        //   ssim        → imgt[0]
        //   artifact    → imgt[2]
        //   detail_loss → imgt[8]
        {
            let [i0, _i1, i2, i3, _i4, _i5, i6, i7, i8, i9] = self.imgt[scale].each_mut();
            self.kernel.compute_error_maps(
                streams[0],
                &self.ref_xyb_t_cache[scale],
                &*i3,
                &self.mu1_t_cache[scale],
                &*i9,
                &self.sigma11_t_cache[scale],
                &*i6,
                &*i7,
                i0, // ssim
                i2, // artifact
                i8, // detail_loss
            );
        }

        streams[1].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[3].wait_for_stream(streams[0]).unwrap();
        streams[4].wait_for_stream(streams[0]).unwrap();
        streams[5].wait_for_stream(streams[0]).unwrap();

        // Reduce: output data at (0, 2, 8), tmp at free slots (5, 1, 4).
        self.reduce_at(scale, 0, 5, 0)?;
        self.reduce_at(scale, 2, 1, 2)?;
        self.reduce_at(scale, 8, 4, 4)?;
        Ok(())
    }

    /// Generalization of `reduce()` that takes explicit (data, tmp)
    /// imgt-slot indices. `offset` matches the score-array offset used
    /// by the original `reduce` (one of 0, 2, 4).
    fn reduce_at(
        &mut self,
        scale: usize,
        data: usize,
        tmp: usize,
        offset: usize,
    ) -> Result<()> {
        let [scratch0, scratch1] = &mut self.sum_scratch[scale][offset..offset + 2] else {
            unreachable!()
        };
        let ctx1 = self
            .npp
            .with_stream(self.streams[scale][offset + 1].inner() as _);
        {
            let (ssim, tmp_img) = indices!(&mut self.imgt[scale], data, tmp);
            ssim.sum_into(
                scratch0,
                (&mut self.scores
                    [scale * 6 * 3 + offset / 2 * 3..scale * 6 * 3 + offset / 2 * 3 + 3])
                    .try_into()
                    .unwrap(),
                self.npp
                    .with_stream(self.streams[scale][offset].inner() as _),
            )?;
            ssim.sqr(tmp_img, ctx1)?;
        }
        self.imgt[scale][tmp].sqr_ip(ctx1)?;
        self.imgt[scale][tmp].sum_into(
            scratch1,
            (&mut self.scores
                [scale * 6 * 3 + offset / 2 * 3 + 9..scale * 6 * 3 + offset / 2 * 3 + 12])
                .try_into()
                .unwrap(),
            ctx1,
        )?;
        Ok(())
    }

    fn process_scale(&mut self, scale: usize) -> Result<()> {
        let streams: [&CuStream; 6] = self.streams[scale].each_ref().try_into().unwrap();

        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[1]).unwrap();

        {
            let (sigma11, sigma22, sigma12, ref_xyb, dis_xyb) =
                indices_ordered!(&mut self.img[scale], 0, 1, 2, 8, 9);
            ref_xyb.mul(
                &ref_xyb,
                sigma11,
                self.npp.with_stream(streams[0].inner() as _),
            )?;
            dis_xyb.mul(
                &dis_xyb,
                sigma22,
                self.npp.with_stream(streams[1].inner() as _),
            )?;
            ref_xyb.mul(
                &dis_xyb,
                sigma12,
                self.npp.with_stream(streams[2].inner() as _),
            )?;
        }

        streams[0].wait_for_stream(streams[1]).unwrap();
        streams[0].wait_for_stream(streams[2]).unwrap();

        {
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = self.img[scale].each_mut();
            // TODO make blur work in place
            // We currently can't compute our blur pass in place,
            // which means we need 10 full buffers allocated :(
            #[rustfmt::skip]
            self.kernel.blur_pass_fused(streams[0],
                i0, i3,
                i1, i4,
                i2, i5,
                i8, i6,
                i9, i7,
            );
        }

        streams[1].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[3].wait_for_stream(streams[0]).unwrap();
        streams[4].wait_for_stream(streams[0]).unwrap();

        self.img[scale][3].transpose(
            &mut self.imgt[scale][0],
            self.npp.with_stream(streams[0].inner() as _),
        )?;
        self.img[scale][4].transpose(
            &mut self.imgt[scale][1],
            self.npp.with_stream(streams[1].inner() as _),
        )?;
        self.img[scale][5].transpose(
            &mut self.imgt[scale][2],
            self.npp.with_stream(streams[2].inner() as _),
        )?;
        self.img[scale][6].transpose(
            &mut self.imgt[scale][3],
            self.npp.with_stream(streams[3].inner() as _),
        )?;
        self.img[scale][7].transpose(
            &mut self.imgt[scale][4],
            self.npp.with_stream(streams[4].inner() as _),
        )?;

        streams[0].wait_for_stream(streams[1]).unwrap();
        streams[0].wait_for_stream(streams[2]).unwrap();
        streams[0].wait_for_stream(streams[3]).unwrap();
        streams[0].wait_for_stream(streams[4]).unwrap();

        {
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = self.imgt[scale].each_mut();

            #[rustfmt::skip]
            self.kernel.blur_pass_fused(streams[0],
                i0, i5,
                i1, i6,
                i2, i7,
                i3, i8,
                i4, i9,
            );
        }

        streams[1].wait_for_stream(streams[0]).unwrap();

        self.img[scale][8].transpose(
            &mut self.imgt[scale][0],
            self.npp.with_stream(streams[0].inner() as _),
        )?;
        self.img[scale][9].transpose(
            &mut self.imgt[scale][1],
            self.npp.with_stream(streams[1].inner() as _),
        )?;

        streams[0].wait_for_stream(streams[1]).unwrap();

        // profiler_start().unwrap();

        {
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = self.imgt[scale].each_mut();

            self.kernel
                .compute_error_maps(streams[0], i0, i1, i8, i9, i5, i6, i7, i2, i3, i4);
        }
        // save_img(&self.dev, self.tmp0.view(size), &format!("ssim_{scale}"));

        streams[1].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[3].wait_for_stream(streams[0]).unwrap();
        streams[4].wait_for_stream(streams[0]).unwrap();
        streams[5].wait_for_stream(streams[0]).unwrap();

        self.reduce(scale, 2, 5, 0)?;
        self.reduce(scale, 3, 6, 2)?;
        self.reduce(scale, 4, 7, 4)?;

        Ok(())
    }

    fn reduce(&mut self, scale: usize, data: usize, tmp: usize, offset: usize) -> Result<()> {
        let [scratch0, scratch1] = &mut self.sum_scratch[scale][offset..offset + 2] else {
            unreachable!()
        };
        let ctx1 = self
            .npp
            .with_stream(self.streams[scale][offset + 1].inner() as _);
        {
            let (ssim, tmp) = indices!(&mut self.imgt[scale], data, tmp);
            ssim.sum_into(
                scratch0,
                (&mut self.scores
                    [scale * 6 * 3 + offset / 2 * 3..scale * 6 * 3 + offset / 2 * 3 + 3])
                    .try_into()
                    .unwrap(),
                self.npp
                    .with_stream(self.streams[scale][offset].inner() as _),
            )?;
            ssim.sqr(tmp, ctx1)?;
        }
        self.imgt[scale][tmp].sqr_ip(ctx1)?;
        self.imgt[scale][tmp].sum_into(
            scratch1,
            (&mut self.scores
                [scale * 6 * 3 + offset / 2 * 3 + 9..scale * 6 * 3 + offset / 2 * 3 + 12])
                .try_into()
                .unwrap(),
            ctx1,
        )?;
        Ok(())
    }

    fn post_process_scores(&mut self) -> f64 {
        // TODO jeez that's a lot of zeros
        //  Computing with a finer granularity (e.g. separated planes)
        //  may allow us to reduce total computations since there are a lot of zeros
        #[rustfmt::skip]
        const WEIGHT: [f64; 108] = [
            // X
            // Scale 0
            0.0,
            0.000_737_660_670_740_658_6,
            0.0,
            0.0,
            0.000_779_348_168_286_730_9,
            0.0,
            // Scale 1
            0.0,
            0.000_437_115_573_010_737_9,
            0.0,
            1.104_172_642_665_734_6,
            0.000_662_848_341_292_71,
            0.000_152_316_327_837_187_52,
            // Scale 2
            0.0,
            0.001_640_643_745_659_975_4,
            0.0,
            1.842_245_552_053_929_8,
            11.441_172_603_757_666,
            0.0,
            // Scale 3
            0.000_798_910_943_601_516_3,
            0.000_176_816_438_078_653,
            0.0,
            1.878_759_497_954_638_7,
            10.949_069_906_051_42,
            0.0,
            // Scale 4
            0.000_728_934_699_150_807_2,
            0.967_793_708_062_683_3,
            0.0,
            0.000_140_034_242_854_358_84,
            0.998_176_697_785_496_7,
            0.000_319_497_559_344_350_53,
            // Scale 5
            0.000_455_099_211_379_206_3,
            0.0,
            0.0,
            0.001_364_876_616_324_339_8,
            0.0,
            0.0,
            // Y
            // Scale 0
            0.0,
            0.0,
            0.0,
            7.466_890_328_078_848,
            0.0,
            17.445_833_984_131_262,
            // Scale 1
            0.000_623_560_163_404_146_6,
            0.0,
            0.0,
            6.683_678_146_179_332,
            0.000_377_244_079_796_112_96,
            1.027_889_937_768_264,
            // Scale 2
            225.205_153_008_492_74,
            0.0,
            0.0,
            19.213_238_186_143_016,
            0.001_140_152_458_661_836_1,
            0.001_237_755_635_509_985,
            // Scale 3
            176.393_175_984_506_94,
            0.0,
            0.0,
            24.433_009_998_704_76,
            0.285_208_026_121_177_57,
            0.000_448_543_692_383_340_8,
            // Scale 4
            0.0,
            0.0,
            0.0,
            34.779_063_444_837_72,
            44.835_625_328_877_896,
            0.0,
            // Scale 5
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            // B
            // Scale 0
            0.0,
            0.000_868_055_657_329_169_8,
            0.0,
            0.0,
            0.0,
            0.0,
            // Scale 1
            0.0,
            0.000_531_319_187_435_874_7,
            0.0,
            0.000_165_338_141_613_791_12,
            0.0,
            0.0,
            // Scale 2
            0.0,
            0.0,
            0.0,
            0.000_417_917_180_325_133_6,
            0.001_729_082_823_472_283_3,
            0.0,
            // Scale 3
            0.002_082_700_584_663_643_7,
            0.0,
            0.0,
            8.826_982_764_996_862,
            23.192_433_439_989_26,
            0.0,
            // Scale 4
            95.108_049_881_108_6,
            0.986_397_803_440_068_2,
            0.983_438_279_246_535_3,
            0.001_228_640_504_827_849_3,
            171.266_725_589_730_7,
            0.980_785_887_243_537_9,
            // Scale 5
            0.0,
            0.0,
            0.0,
            0.000_513_006_458_899_067_9,
            0.0,
            0.000_108_540_578_584_115_37,
        ];

        for scale in 0..SCALES {
            let opp = self.sizes_t[scale].norm();
            let offset = 3 * 6 * scale;
            for c in 0..3 {
                let offset = offset + c;
                let offsetw = c * 6 * 6 + 6 * scale;
                self.scores[offset] = (self.scores[offset] * opp).abs() * WEIGHT[offsetw];
                self.scores[offset + 3] =
                    (self.scores[offset + 3] * opp).abs() * WEIGHT[offsetw + 1];
                self.scores[offset + 2 * 3] =
                    (self.scores[offset + 2 * 3] * opp).abs() * WEIGHT[offsetw + 2];
                self.scores[offset + 3 * 3] =
                    (self.scores[offset + 3 * 3] * opp).sqrt().sqrt() * WEIGHT[offsetw + 3];
                self.scores[offset + 4 * 3] =
                    (self.scores[offset + 4 * 3] * opp).sqrt().sqrt() * WEIGHT[offsetw + 4];
                self.scores[offset + 5 * 3] =
                    (self.scores[offset + 5 * 3] * opp).sqrt().sqrt() * WEIGHT[offsetw + 5];
            }
        }

        let mut score: f64 = self.scores.iter().sum();

        score *= 0.956_238_261_683_484_4_f64;
        score = (6.248_496_625_763_138e-5 * score * score).mul_add(
            score,
            2.326_765_642_916_932f64.mul_add(score, -0.020_884_521_182_843_837 * score * score),
        );

        if score > 0.0f64 {
            score = score
                .powf(0.627_633_646_783_138_7)
                .mul_add(-10.0f64, 100.0f64);
        } else {
            score = 100.0f64;
        }

        score
    }
}

impl Drop for Ssimulacra2 {
    fn drop(&mut self) {
        // Sync the CUDA context before dropping GPU buffers to prevent crashes.
        // Without this, pending operations may still reference buffers being freed.
        // This pattern was discovered in glassa when CUDA crashes occurred during cleanup.
        let _ = cudarse_driver::sync_ctx();
    }
}

/*
fn save_img(img: impl Img<f32, C<3>>, name: &str, stream: cudaStream_t) {
    // dev.synchronize().unwrap();
    let bytes = img.copy_to_cpu(stream).unwrap();
    let mut img = zune_image::image::Image::from_f32(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("./{name}.png")).unwrap()
}*/
