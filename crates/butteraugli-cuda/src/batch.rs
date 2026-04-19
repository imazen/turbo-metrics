//! Batched butteraugli scorer (coefficient#13).
//!
//! Runs N distorted variants against a single cached reference in one
//! grid launch per kernel. The per-kernel batched variants live in
//! `kernel.rs`; this module is the pipeline that assembles them into
//! a full butteraugli compute against the reference cache held by
//! [`crate::Butteraugli`].
//!
//! Buffer layout: every per-image intermediate plane is a contiguous
//! `plane_stride × batch_size` f32 run. At full resolution
//! `plane_stride = width * height`; at half resolution the same
//! buffers are reused with `plane_stride = half_width * half_height`
//! (tightly packed in the first `half_plane * batch` elements — we
//! treat the buffer as a smaller-stride array for the half-res pass,
//! then discard everything past the upsample step).
//!
//! Reference-side state (XYB freq bands, pre-computed masks) is a
//! single-image cache on the inner [`Butteraugli`]. The Malta and
//! mask_to_error_mul batch kernels read it with `ref_stride = 0` to
//! broadcast; the final `compute_diffmap` consumes N copies of
//! `ref_mask_final_*` written into `mask_batch` by a tiny broadcast
//! kernel.

use cudarse_driver::{sys::cuMemcpyAsync, CuBox, CuGraphExec};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut, C};

use crate::{consts, Butteraugli, Error};

/// Batched butteraugli scorer. Shares a reference-state cache with an
/// internal [`Butteraugli`] instance; the per-distorted buffers are
/// batch-sized replicas of the single-image buffers.
pub struct ButteraugliBatch {
    pub(crate) inner: Butteraugli,
    batch_size: usize,
    // Packed linear f32 RGB for N distorted images, laid out as a
    // (width, height*batch_size) Image so we can use the normal NPP
    // upload path.
    linear2_batch: Image<f32, C<3>>,
    // Planar XYB / intermediates: each buffer holds N × plane_stride f32s.
    // At full-res: plane_stride = width*height. At half-res: the same
    // buffer is reused with plane_stride = half_width*half_height
    // (tightly packed in the first half_plane*batch elements).
    xyb2_batch: [CuBox<[f32]>; 3],
    linear_blur2_batch: [CuBox<[f32]>; 3],
    linear_planar2_batch: [CuBox<[f32]>; 3], // half-res linear RGB, half_plane stride
    freq2_batch: [[CuBox<[f32]>; 3]; 4],     // [band][channel]
    block_diff_dc_batch: [CuBox<[f32]>; 3],
    block_diff_ac_batch: [CuBox<[f32]>; 3],
    mask_batch: CuBox<[f32]>,
    mask_temp_batch: CuBox<[f32]>,
    temp1_batch: CuBox<[f32]>,
    temp2_batch: CuBox<[f32]>,
    diffmap_batch: CuBox<[f32]>,
    diffmap_half_batch: CuBox<[f32]>,
    /// Per-image f32 max result (stored as u32 bits for atomicMax).
    max_results: CuBox<[u32]>,
    /// Staging Image for N packed-RGB u8 uploads.
    dis_staging: Image<u8, C<3>>,
    /// Lazily captured CUDA graph of the compute pipeline. All the
    /// pointers the graph references (dis_staging, linear2_batch,
    /// xyb2_batch, freq2_batch, block_diff_*_batch, mask_*_batch,
    /// diffmap*_batch, max_results) are owned by this struct and
    /// stable for its lifetime. The H2D upload of distorted bytes
    /// happens OUTSIDE the graph (see compute_batch_with_reference)
    /// because the host-side source pointer changes per call. The
    /// graph covers everything from srgb_to_linear_batch through
    /// max_reduce_batch; the CPU-side sync + D2H of the 4N-byte
    /// result then happens after graph replay.
    compute_graph: Option<CuGraphExec>,
}

impl ButteraugliBatch {
    /// Create a batched scorer for images of the given dimensions and
    /// the given batch size.
    pub fn new(width: u32, height: u32, batch_size: usize) -> Result<Self, Error> {
        if batch_size == 0 {
            return Err(Error::InvalidDimensions("batch_size must be > 0".into()));
        }
        let inner = Butteraugli::new(width, height)?;
        let stream_ref = inner.stream();
        let w = width as usize;
        let h = height as usize;
        let plane = w * h;
        let half_w = (w + 1) / 2;
        let half_h = (h + 1) / 2;
        let half_plane = half_w * half_h;

        let alloc_n = |size: usize| -> Result<CuBox<[f32]>, Error> {
            CuBox::<[f32]>::new_zeroed(size * batch_size, stream_ref)
                .map_err(|e| Error::Cuda(format!("{:?}", e)))
        };
        let three = |size: usize| -> Result<[CuBox<[f32]>; 3], Error> {
            Ok([alloc_n(size)?, alloc_n(size)?, alloc_n(size)?])
        };

        let dis_staging: Image<u8, C<3>> = Image::malloc(width, height * batch_size as u32)
            .map_err(|e| Error::Npp(format!("{:?}", e)))?;
        let linear2_batch: Image<f32, C<3>> =
            Image::malloc(width, height * batch_size as u32)
                .map_err(|e| Error::Npp(format!("{:?}", e)))?;

        let xyb2_batch = three(plane)?;
        let linear_blur2_batch = three(plane)?;
        // linear_planar2_batch holds downsampled (half-res) linear RGB.
        // Allocated at half_plane per slot; that's the slot stride we
        // pass to downsample_2x_batch / the half-res copy.
        let linear_planar2_batch = three(half_plane)?;
        let freq2_batch = [three(plane)?, three(plane)?, three(plane)?, three(plane)?];
        let block_diff_dc_batch = three(plane)?;
        let block_diff_ac_batch = three(plane)?;
        let mask_batch = alloc_n(plane)?;
        let mask_temp_batch = alloc_n(plane)?;
        let temp1_batch = alloc_n(plane)?;
        let temp2_batch = alloc_n(plane)?;
        let diffmap_batch = alloc_n(plane)?;
        let diffmap_half_batch = alloc_n(half_plane)?;
        let max_results = CuBox::<[u32]>::new_zeroed(batch_size, stream_ref)
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        Ok(Self {
            inner,
            batch_size,
            linear2_batch,
            xyb2_batch,
            linear_blur2_batch,
            linear_planar2_batch,
            freq2_batch,
            block_diff_dc_batch,
            block_diff_ac_batch,
            mask_batch,
            mask_temp_batch,
            temp1_batch,
            temp2_batch,
            diffmap_batch,
            diffmap_half_batch,
            max_results,
            dis_staging,
            compute_graph: None,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.inner.dimensions()
    }

    /// Cache the reference image on the GPU. Bit-identical to
    /// [`Butteraugli::set_reference`] — just delegates.
    ///
    /// Does NOT invalidate the captured compute graph: the graph
    /// references the cached `ref_*` buffers by pointer, and those
    /// pointers are stable across set_reference calls (only the
    /// contents change). The graph replays against the new reference
    /// state transparently.
    pub fn set_reference(&mut self, reference: impl Img<u8, C<3>>) -> Result<(), Error> {
        self.inner.set_reference(reference)
    }

    pub fn has_reference(&self) -> bool {
        self.inner.has_reference()
    }

    pub fn clear_reference(&mut self) {
        self.inner.clear_reference();
        // Drop the captured graph as well — clear_reference signals an
        // intentional cache break, so we toss the captured pipeline
        // too. Next compute_batch_with_reference will recapture.
        self.compute_graph = None;
    }

    #[inline]
    fn cptr(buf: &CuBox<[f32]>) -> *const f32 {
        buf.ptr() as *const f32
    }

    #[inline]
    fn mptr(buf: &CuBox<[f32]>) -> *mut f32 {
        // CuBox::ptr takes &self; the device memory address is a
        // raw pointer with no Rust-side aliasing, so we can hand out
        // both *const and *mut from a shared borrow. All safety is at
        // the kernel-launch layer.
        buf.ptr() as *mut f32
    }

    /// Score N distorted variants against the cached reference.
    ///
    /// `distorted_bytes` must be a tightly packed sequence of N
    /// `width*height*3` byte images in sRGB u8 RGB layout.
    /// Returns one score per image in order.
    ///
    /// The first call captures the full compute pipeline (everything
    /// from `srgb_to_linear_batch` through `max_reduce_batch`) into a
    /// CUDA graph and replays it on every subsequent call. The H2D
    /// upload of `distorted_bytes` stays outside the graph because the
    /// host source pointer changes per call. At N=8, 512×512 this
    /// drops the ~75 per-call kernel launches down to one graph replay.
    pub fn compute_batch_with_reference(
        &mut self,
        distorted_bytes: &[u8],
    ) -> Result<Vec<f32>, Error> {
        let n = self.batch_size;
        let w = self.inner.width;
        let h = self.inner.height;

        let expected_bytes = w * h * 3 * n;
        if distorted_bytes.len() != expected_bytes {
            return Err(Error::InvalidDimensions(format!(
                "expected {} bytes for batch of {} × {}x{}, got {}",
                expected_bytes,
                n,
                w,
                h,
                distorted_bytes.len()
            )));
        }
        if !self.inner.has_reference() {
            return Err(Error::Cuda(
                "no cached reference; call set_reference() first".into(),
            ));
        }

        // Upload: one H2D copy that lands a tall-stacked Image<u8,C<3>>.
        // OUTSIDE the graph — the source Vec<u8> rotates between calls.
        {
            let stream_inner = self.inner.stream.inner() as _;
            self.dis_staging
                .copy_from_cpu(distorted_bytes, stream_inner)
                .map_err(|e| Error::Npp(format!("{:?}", e)))?;
        }

        // Lazy-capture the post-upload pipeline into a graph.
        if self.compute_graph.is_none() {
            self.inner
                .stream
                .begin_capture_thread_local()
                .map_err(|e| Error::Cuda(format!("begin_capture: {:?}", e)))?;
            let run_result = self.run_pipeline_after_upload();
            let end_result = self
                .inner
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
                .map_err(|e| Error::Cuda(format!("graph instantiate: {:?}", e)))?;
            self.compute_graph = Some(exec);
        }

        // Replay.
        self.compute_graph
            .as_ref()
            .expect("graph set above")
            .launch(&self.inner.stream)
            .map_err(|e| Error::Cuda(format!("graph launch: {:?}", e)))?;

        self.inner
            .stream
            .sync()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;

        // D2H copy of N u32s, reinterpret as f32 bits.
        let mut out_bits = vec![0u32; n];
        unsafe {
            cudarse_driver::sys::cuMemcpyDtoH_v2(
                out_bits.as_mut_ptr() as *mut _,
                self.max_results.ptr(),
                n * 4,
            )
            .result()
            .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
        }
        Ok(out_bits.into_iter().map(f32::from_bits).collect())
    }

    /// Everything from `srgb_to_linear_batch` through `max_reduce_batch`.
    /// Designed to be graph-capturable: no CPU syncs, no synchronous
    /// memcpys, every op goes on `self.inner.stream`. The H2D upload
    /// of distorted bytes into `dis_staging` must be done by the
    /// caller beforehand — that step cannot live inside the graph
    /// because its host source pointer changes between invocations.
    fn run_pipeline_after_upload(&self) -> Result<(), Error> {
        let n = self.batch_size;
        let batch = n as u32;
        let w = self.inner.width;
        let h = self.inner.height;
        let plane = self.inner.size;
        let half_w = self.inner.half_width;
        let half_h = self.inner.half_height;
        let half_plane = self.inner.half_size;

        // Grab pitches + raw pointers — these are stable for the life
        // of the ButteraugliBatch instance.
        let dis_pitch = self.dis_staging.pitch() as usize;
        let lin2_pitch = self.linear2_batch.pitch() as usize;
        let dis_src = self.dis_staging.device_ptr();
        // device_ptr_mut would require &mut self, but the underlying
        // raw device address is stable and safe to reinterpret mut:
        let lin2_dst = self.linear2_batch.device_ptr() as *mut f32;

        // Full-resolution pass.
        self.run_distorted_pass_full(w, h, plane, batch, dis_src, dis_pitch, lin2_dst, lin2_pitch)?;

        // Copy per-image half-res linear RGB into the xyb2 slots.
        for ch in 0..3 {
            for i in 0..n {
                let slot_bytes = (i * half_plane * 4) as u64;
                let dst = self.xyb2_batch[ch].ptr() + slot_bytes;
                let src = self.linear_planar2_batch[ch].ptr() + slot_bytes;
                unsafe {
                    cuMemcpyAsync(dst, src, half_plane * 4, self.inner.stream.raw())
                        .result()
                        .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
                }
            }
        }

        // Half-resolution pass.
        self.run_distorted_pass_half(half_w, half_h, half_plane, batch)?;

        // Combine full + half diffmaps.
        self.inner.kernel.add_upsample_2x_batch(
            &self.inner.stream,
            Self::cptr(&self.diffmap_half_batch),
            Self::mptr(&self.diffmap_batch),
            half_w,
            half_h,
            w,
            h,
            half_plane,
            plane,
            0.5,
            batch,
        );

        // Per-image max reduction (writes into self.max_results).
        self.inner.kernel.max_reduce_batch(
            &self.inner.stream,
            Self::cptr(&self.diffmap_batch),
            self.max_results.ptr() as *mut f32,
            plane,
            plane,
            batch,
        );
        Ok(())
    }

    /// Full-resolution distorted-side pass.
    ///
    /// Writes `diffmap_batch` and leaves `linear_planar2_batch` populated
    /// with half-res linear RGB for the half-res pass. Image source
    /// pointers are passed in so we don't have to juggle Image borrows.
    #[allow(clippy::too_many_arguments)]
    fn run_distorted_pass_full(
        &self,
        w: usize,
        h: usize,
        plane: usize,
        batch: u32,
        dis_src: *const u8,
        dis_pitch: usize,
        lin2_dst: *mut f32,
        lin2_pitch: usize,
    ) -> Result<(), Error> {
        let n = batch as usize;
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;

        // ---- sRGB -> linear (batched) ----
        k.srgb_to_linear_batch(
            stream,
            dis_src,
            dis_pitch,
            h * dis_pitch,
            lin2_dst,
            lin2_pitch,
            h * lin2_pitch,
            w as u32,
            h as u32,
            batch,
        );

        // ---- Deinterleave -> planar ----
        k.deinterleave_3ch_batch(
            stream,
            lin2_dst as *const f32,
            lin2_pitch,
            h * lin2_pitch,
            Self::mptr(&self.xyb2_batch[0]),
            Self::mptr(&self.xyb2_batch[1]),
            Self::mptr(&self.xyb2_batch[2]),
            plane,
            w as u32,
            h as u32,
            batch,
        );

        // ---- Downsample 2x to linear_planar2_batch (for later half-res) ----
        for ch in 0..3 {
            k.downsample_2x_batch(
                stream,
                Self::cptr(&self.xyb2_batch[ch]),
                Self::mptr(&self.linear_planar2_batch[ch]),
                w,
                h,
                self.inner.half_width,
                self.inner.half_height,
                plane,
                self.inner.half_size,
                batch,
            );
        }

        // ---- Opsin dynamics (in-place on xyb2_batch) ----
        self.opsin_dynamics_batch(w, h, plane, batch)?;

        // ---- Frequency separation ----
        self.separate_frequencies_batch(w, h, plane, batch)?;

        // ---- Clear block_diff accumulators ----
        for ch in 0..3 {
            k.clear_buffer(stream, Self::mptr(&self.block_diff_dc_batch[ch]), plane * n);
            k.clear_buffer(stream, Self::mptr(&self.block_diff_ac_batch[ch]), plane * n);
        }

        // ---- Compute differences (Malta + L2) ----
        self.compute_differences_batch(w, h, plane, batch, false)?;

        // ---- Masking ----
        self.apply_distorted_masking_batch(w, h, plane, batch, false)?;

        // ---- Broadcast ref_mask_final_full → mask_batch ----
        k.broadcast_plane_batch(
            stream,
            self.inner.ref_mask_final_full.ptr() as *const f32,
            Self::mptr(&self.mask_batch),
            plane,
            plane,
            batch,
        );

        // ---- Final diffmap (full-res) ----
        self.compute_diffmap_batch(plane, plane, batch, false)
    }

    /// Half-resolution distorted-side pass. Reuses the full-res buffers
    /// with half_plane stride (first half_plane * batch elements of
    /// each buffer). `plane` here is `half_plane`.
    fn run_distorted_pass_half(
        &self,
        w: usize,
        h: usize,
        plane: usize,
        batch: u32,
    ) -> Result<(), Error> {
        let n = batch as usize;
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;

        // ---- Opsin dynamics ----
        self.opsin_dynamics_batch(w, h, plane, batch)?;

        // ---- Frequency separation ----
        self.separate_frequencies_batch(w, h, plane, batch)?;

        // ---- Clear half-res slot of block_diff accumulators ----
        for ch in 0..3 {
            k.clear_buffer(stream, Self::mptr(&self.block_diff_dc_batch[ch]), plane * n);
            k.clear_buffer(stream, Self::mptr(&self.block_diff_ac_batch[ch]), plane * n);
        }

        // ---- Compute differences ----
        self.compute_differences_batch(w, h, plane, batch, true)?;

        // ---- Masking ----
        self.apply_distorted_masking_batch(w, h, plane, batch, true)?;

        // ---- Broadcast ref_mask_final_half → mask_batch ----
        k.broadcast_plane_batch(
            stream,
            self.inner.ref_mask_final_half.ptr() as *const f32,
            Self::mptr(&self.mask_batch),
            plane,
            plane,
            batch,
        );

        // ---- Final diffmap → diffmap_half_batch ----
        self.compute_diffmap_batch(plane, plane, batch, true)
    }

    /// Opsin dynamics for the N-batch xyb2 planes in-place. Uses
    /// `temp1_batch` as the mirrored-blur scratch.
    fn opsin_dynamics_batch(
        &self,
        w: usize,
        h: usize,
        plane: usize,
        batch: u32,
    ) -> Result<(), Error> {
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;

        // Blur each channel with sigma=1.2 (opsin blur), mirrored.
        for ch in 0..3 {
            k.blur_mirrored_5x5_batch(
                stream,
                Self::cptr(&self.xyb2_batch[ch]),
                Self::mptr(&self.linear_blur2_batch[ch]),
                Self::mptr(&self.temp1_batch),
                w,
                h,
                consts::OPSIN_BLUR_W0,
                consts::OPSIN_BLUR_W1,
                consts::OPSIN_BLUR_W2,
                plane,
                batch,
            );
        }

        k.opsin_dynamics_batch(
            stream,
            Self::mptr(&self.xyb2_batch[0]),
            Self::mptr(&self.xyb2_batch[1]),
            Self::mptr(&self.xyb2_batch[2]),
            Self::cptr(&self.linear_blur2_batch[0]),
            Self::cptr(&self.linear_blur2_batch[1]),
            Self::cptr(&self.linear_blur2_batch[2]),
            plane,
            consts::INTENSITY_TARGET,
            batch,
        );
        Ok(())
    }

    /// Batched cascaded frequency separation on xyb2_batch -> freq2_batch.
    /// Mirrors [`Butteraugli::separate_frequencies_for_image`] (img_idx=1).
    fn separate_frequencies_batch(
        &self,
        w: usize,
        h: usize,
        plane: usize,
        batch: u32,
    ) -> Result<(), Error> {
        let n = batch as usize;
        let size = plane * n;
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;

        // Step 1: LF = blur(src); MF_raw = src - LF
        for ch in 0..3 {
            let src = Self::cptr(&self.xyb2_batch[ch]);
            let lf_ptr = Self::mptr(&self.freq2_batch[3][ch]);
            let temp1 = Self::mptr(&self.temp1_batch);
            k.blur_batch(stream, src, lf_ptr, temp1, w, h, consts::SIGMA_LF, plane, batch);

            let mf_ptr = Self::mptr(&self.freq2_batch[2][ch]);
            k.subtract_arrays(stream, src, lf_ptr as *const f32, mf_ptr, size);
        }

        // Step 2: blur MF_raw; HF = MF_raw - blur(MF_raw) for ch<2.
        for ch in 0..3 {
            let mf_raw_const = Self::cptr(&self.freq2_batch[2][ch]);
            let temp1 = Self::mptr(&self.temp1_batch);
            let temp2 = Self::mptr(&self.temp2_batch);
            k.blur_batch(
                stream,
                mf_raw_const,
                temp1,
                temp2,
                w,
                h,
                consts::SIGMA_MF,
                plane,
                batch,
            );

            if ch < 2 {
                let hf_ptr = Self::mptr(&self.freq2_batch[1][ch]);
                k.subtract_arrays(stream, mf_raw_const, temp1 as *const f32, hf_ptr, size);
            }

            unsafe {
                cuMemcpyAsync(
                    self.freq2_batch[2][ch].ptr(),
                    self.temp1_batch.ptr(),
                    size * 4,
                    stream.raw(),
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
        }

        k.remove_range(
            stream,
            Self::mptr(&self.freq2_batch[2][0]),
            size,
            consts::REMOVE_MF_RANGE,
        );
        k.amplify_range(
            stream,
            Self::mptr(&self.freq2_batch[2][1]),
            size,
            consts::ADD_MF_RANGE,
        );

        k.suppress_x_by_y(
            stream,
            Self::mptr(&self.freq2_batch[1][0]),
            Self::cptr(&self.freq2_batch[1][1]),
            size,
            consts::SUPPRESS_XY,
        );

        // Step 3a: HF/UHF for X channel.
        {
            let hf_raw_const = Self::cptr(&self.freq2_batch[1][0]);
            let temp1 = Self::mptr(&self.temp1_batch);
            let temp2 = Self::mptr(&self.temp2_batch);
            k.blur_batch(
                stream,
                hf_raw_const,
                temp1,
                temp2,
                w,
                h,
                consts::SIGMA_HF,
                plane,
                batch,
            );

            let uhf_ptr = Self::mptr(&self.freq2_batch[0][0]);
            k.subtract_arrays(stream, hf_raw_const, temp1 as *const f32, uhf_ptr, size);

            unsafe {
                cuMemcpyAsync(
                    self.freq2_batch[1][0].ptr(),
                    self.temp1_batch.ptr(),
                    size * 4,
                    stream.raw(),
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
            k.remove_range(
                stream,
                Self::mptr(&self.freq2_batch[1][0]),
                size,
                consts::REMOVE_HF_RANGE,
            );
            k.remove_range(
                stream,
                Self::mptr(&self.freq2_batch[0][0]),
                size,
                consts::REMOVE_UHF_RANGE,
            );
        }

        // Step 3b: HF/UHF for Y channel.
        {
            // UHF Y = HF_raw (copy freq[1][1] into freq[0][1]).
            unsafe {
                cuMemcpyAsync(
                    self.freq2_batch[0][1].ptr(),
                    self.freq2_batch[1][1].ptr(),
                    size * 4,
                    stream.raw(),
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
            let hf_raw_const = Self::cptr(&self.freq2_batch[0][1]);
            let temp1 = Self::mptr(&self.temp1_batch);
            let temp2 = Self::mptr(&self.temp2_batch);
            k.blur_batch(
                stream,
                hf_raw_const,
                temp1,
                temp2,
                w,
                h,
                consts::SIGMA_HF,
                plane,
                batch,
            );
            unsafe {
                cuMemcpyAsync(
                    self.freq2_batch[1][1].ptr(),
                    self.temp1_batch.ptr(),
                    size * 4,
                    stream.raw(),
                )
                .result()
                .map_err(|e| Error::Cuda(format!("{:?}", e)))?;
            }
            k.separate_hf_uhf(
                stream,
                Self::mptr(&self.freq2_batch[1][1]),
                Self::mptr(&self.freq2_batch[0][1]),
                size,
            );
        }

        k.xyb_low_freq_to_vals(
            stream,
            Self::mptr(&self.freq2_batch[3][0]),
            Self::mptr(&self.freq2_batch[3][1]),
            Self::mptr(&self.freq2_batch[3][2]),
            size,
        );
        Ok(())
    }

    /// Compute differences (Malta split-stride + L2) between the cached
    /// reference freq bands and the N distorted freq bands.
    fn compute_differences_batch(
        &self,
        w: usize,
        h: usize,
        plane: usize,
        batch: u32,
        is_half: bool,
    ) -> Result<(), Error> {
        let n = batch as usize;
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;
        let ref_freq: &[[CuBox<[f32]>; 3]; 4] = if is_half {
            &self.inner.ref_freq_half
        } else {
            &self.inner.ref_freq_full
        };
        let ref_ptr = |band: usize, ch: usize| ref_freq[band][ch].ptr() as *const f32;

        // UHF Malta (HF kernel). lum0 = ref (stride=0), lum1 = dis (stride=plane).
        k.malta_diff_map_batch_split_stride(
            stream,
            ref_ptr(0, 1),
            0,
            Self::cptr(&self.freq2_batch[0][1]),
            plane,
            Self::mptr(&self.block_diff_ac_batch[1]),
            plane,
            w,
            h,
            consts::W_UHF_MALTA_0GT1,
            consts::W_UHF_MALTA_0LT1,
            consts::NORM1_UHF,
            batch,
        );
        k.malta_diff_map_batch_split_stride(
            stream,
            ref_ptr(0, 0),
            0,
            Self::cptr(&self.freq2_batch[0][0]),
            plane,
            Self::mptr(&self.block_diff_ac_batch[0]),
            plane,
            w,
            h,
            consts::W_UHF_MALTA_X_0GT1,
            consts::W_UHF_MALTA_X_0LT1,
            consts::NORM1_UHF_X,
            batch,
        );

        // HF Malta (LF kernel).
        k.malta_diff_map_lf_batch_split_stride(
            stream,
            ref_ptr(1, 1),
            0,
            Self::cptr(&self.freq2_batch[1][1]),
            plane,
            Self::mptr(&self.block_diff_ac_batch[1]),
            plane,
            w,
            h,
            consts::W_HF_MALTA_0GT1,
            consts::W_HF_MALTA_0LT1,
            consts::NORM1_HF,
            batch,
        );
        k.malta_diff_map_lf_batch_split_stride(
            stream,
            ref_ptr(1, 0),
            0,
            Self::cptr(&self.freq2_batch[1][0]),
            plane,
            Self::mptr(&self.block_diff_ac_batch[0]),
            plane,
            w,
            h,
            consts::W_HF_MALTA_X_0GT1,
            consts::W_HF_MALTA_X_0LT1,
            consts::NORM1_HF_X,
            batch,
        );

        // MF Malta (LF kernel, symmetric weights).
        k.malta_diff_map_lf_batch_split_stride(
            stream,
            ref_ptr(2, 1),
            0,
            Self::cptr(&self.freq2_batch[2][1]),
            plane,
            Self::mptr(&self.block_diff_ac_batch[1]),
            plane,
            w,
            h,
            consts::W_MF_MALTA,
            consts::W_MF_MALTA,
            consts::NORM1_MF,
            batch,
        );
        k.malta_diff_map_lf_batch_split_stride(
            stream,
            ref_ptr(2, 0),
            0,
            Self::cptr(&self.freq2_batch[2][0]),
            plane,
            Self::mptr(&self.block_diff_ac_batch[0]),
            plane,
            w,
            h,
            consts::W_MF_MALTA_X,
            consts::W_MF_MALTA_X,
            consts::NORM1_MF_X,
            batch,
        );

        // L2 differences. The 1D kernels read both inputs with unit stride,
        // so to broadcast the reference we loop N times. These are cheap
        // (8 kernel launches each covers one N-slot), and only run at the
        // very end of compute_differences.
        for i in 0..n {
            // byte offset per slot for pointer-arith on CUdeviceptr (u64).
            let off = (i * plane * 4) as u64;
            let out_ac_0 = (self.block_diff_ac_batch[0].ptr() + off) as *mut f32;
            let out_ac_1 = (self.block_diff_ac_batch[1].ptr() + off) as *mut f32;
            let out_ac_2 = (self.block_diff_ac_batch[2].ptr() + off) as *mut f32;
            let out_dc_0 = (self.block_diff_dc_batch[0].ptr() + off) as *mut f32;
            let out_dc_1 = (self.block_diff_dc_batch[1].ptr() + off) as *mut f32;
            let out_dc_2 = (self.block_diff_dc_batch[2].ptr() + off) as *mut f32;
            let dis_1_0 = (self.freq2_batch[1][0].ptr() + off) as *const f32;
            let dis_1_1 = (self.freq2_batch[1][1].ptr() + off) as *const f32;
            let dis_2_0 = (self.freq2_batch[2][0].ptr() + off) as *const f32;
            let dis_2_1 = (self.freq2_batch[2][1].ptr() + off) as *const f32;
            let dis_2_2 = (self.freq2_batch[2][2].ptr() + off) as *const f32;
            let dis_3_0 = (self.freq2_batch[3][0].ptr() + off) as *const f32;
            let dis_3_1 = (self.freq2_batch[3][1].ptr() + off) as *const f32;
            let dis_3_2 = (self.freq2_batch[3][2].ptr() + off) as *const f32;

            k.l2_asym_diff(
                stream,
                ref_ptr(1, 0),
                dis_1_0,
                out_ac_0,
                plane,
                consts::WMUL[0] * consts::HF_ASYMMETRY,
                consts::WMUL[0] / consts::HF_ASYMMETRY,
            );
            k.l2_asym_diff(
                stream,
                ref_ptr(1, 1),
                dis_1_1,
                out_ac_1,
                plane,
                consts::WMUL[1] * consts::HF_ASYMMETRY,
                consts::WMUL[1] / consts::HF_ASYMMETRY,
            );
            k.l2_diff(stream, ref_ptr(2, 0), dis_2_0, out_ac_0, plane, consts::WMUL[3]);
            k.l2_diff(stream, ref_ptr(2, 1), dis_2_1, out_ac_1, plane, consts::WMUL[4]);
            k.l2_diff(stream, ref_ptr(2, 2), dis_2_2, out_ac_2, plane, consts::WMUL[5]);
            k.l2_diff(stream, ref_ptr(3, 0), dis_3_0, out_dc_0, plane, consts::WMUL[6]);
            k.l2_diff(stream, ref_ptr(3, 1), dis_3_1, out_dc_1, plane, consts::WMUL[7]);
            k.l2_diff(stream, ref_ptr(3, 2), dis_3_2, out_dc_2, plane, consts::WMUL[8]);
        }

        Ok(())
    }

    /// Distorted-only masking path (reference mask precomputed and cached).
    /// 1. combine_channels_for_masking on freq2 → mask_temp_batch
    /// 2. diff_precompute → temp2_batch
    /// 3. blur → mask_temp_batch (distorted blurred)
    /// 4. mask_to_error_mul with ref blurred (broadcast) vs mask_temp
    ///    → adds to block_diff_ac[1].
    fn apply_distorted_masking_batch(
        &self,
        w: usize,
        h: usize,
        plane: usize,
        batch: u32,
        is_half: bool,
    ) -> Result<(), Error> {
        let n = batch as usize;
        let size = plane * n;
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;

        k.combine_channels_for_masking(
            stream,
            Self::cptr(&self.freq2_batch[1][0]),
            Self::cptr(&self.freq2_batch[0][0]),
            Self::cptr(&self.freq2_batch[1][1]),
            Self::cptr(&self.freq2_batch[0][1]),
            Self::mptr(&self.mask_temp_batch),
            size,
        );
        k.diff_precompute(
            stream,
            Self::cptr(&self.mask_temp_batch),
            Self::mptr(&self.temp2_batch),
            size,
        );
        k.blur_batch(
            stream,
            Self::cptr(&self.temp2_batch),
            Self::mptr(&self.mask_temp_batch),
            Self::mptr(&self.temp1_batch),
            w,
            h,
            consts::SIGMA_MASK,
            plane,
            batch,
        );

        let ref_blurred_ptr = if is_half {
            self.inner.ref_mask_blurred_half.ptr() as *const f32
        } else {
            self.inner.ref_mask_blurred_full.ptr() as *const f32
        };
        k.mask_to_error_mul_batch_split_stride(
            stream,
            ref_blurred_ptr,
            0,
            Self::cptr(&self.mask_temp_batch),
            plane,
            Self::mptr(&self.block_diff_ac_batch[1]),
            plane,
            plane,
            batch,
        );
        Ok(())
    }

    /// Final compute_diffmap on the current batched state. Writes to
    /// `diffmap_batch` at full-res or `diffmap_half_batch` at half-res.
    fn compute_diffmap_batch(
        &self,
        size_per_image: usize,
        plane_stride: usize,
        batch: u32,
        is_half: bool,
    ) -> Result<(), Error> {
        let k = &self.inner.kernel;
        let stream = &self.inner.stream;
        let dst_ptr = if is_half {
            self.diffmap_half_batch.ptr() as *mut f32
        } else {
            self.diffmap_batch.ptr() as *mut f32
        };
        k.compute_diffmap_batch(
            stream,
            Self::cptr(&self.mask_batch),
            Self::cptr(&self.block_diff_dc_batch[0]),
            Self::cptr(&self.block_diff_dc_batch[1]),
            Self::cptr(&self.block_diff_dc_batch[2]),
            Self::cptr(&self.block_diff_ac_batch[0]),
            Self::cptr(&self.block_diff_ac_batch[1]),
            Self::cptr(&self.block_diff_ac_batch[2]),
            dst_ptr,
            size_per_image,
            plane_stride,
            batch,
        );
        Ok(())
    }
}

impl Drop for ButteraugliBatch {
    fn drop(&mut self) {
        let _ = cudarse_driver::sync_ctx();
    }
}
