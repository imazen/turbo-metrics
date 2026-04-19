//! Batched butteraugli scaffolding (coefficient#13, phase 2 in progress).
//!
//! This module is the plumbing for running N distorted variants against a
//! single cached reference in one grid launch per kernel. The per-kernel
//! batched variants (`*_batch_kernel`) and their wrappers in
//! `kernel.rs` are already shipping. What's still missing before the
//! public `compute_batch_with_reference` can produce bit-exact scores:
//!
//! 1. A Malta-diff-map batch kernel that takes **separate** stride
//!    parameters for the reference vs the distorted pointer (reference
//!    is a single image we want to broadcast across N batch slots,
//!    distorted is N slots).
//! 2. The same stride-split treatment for `combine_channels_for_masking`
//!    + `mask_to_error_mul` on the masking side.
//! 3. An `N × ref_mask_final` broadcast (or zero-stride trick) so the
//!    final mask read from `combine_diffmap` addresses the single
//!    ref_mask_final regardless of batch index.
//!
//! Until those land, [`ButteraugliBatch::compute_batch_with_reference`]
//! returns an explicit WIP error. All per-plane batch allocations,
//! `set_reference` cache sharing with the inner [`Butteraugli`], and
//! H2D/D2H plumbing are already in place so the follow-up work is
//! confined to kernel + wrapper additions.

use cudarse_driver::CuBox;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, C};

use crate::{Butteraugli, Error};

/// Batched butteraugli scorer. Shares a reference-state cache with an
/// internal [`Butteraugli`] instance; the per-distorted buffers are
/// batch-sized replicas of the single-image buffers.
pub struct ButteraugliBatch {
    pub(crate) inner: Butteraugli,
    batch_size: usize,
    // Packed linear f32 RGB for N distorted images, laid out as a
    // (width, height*batch_size) Image so we can use the normal NPP
    // upload path.
    _linear2_batch: Image<f32, C<3>>,
    // Planar XYB / intermediates: each buffer holds N × plane_stride f32s
    _xyb2_batch: [CuBox<[f32]>; 3],
    _linear_blur2_batch: [CuBox<[f32]>; 3],
    _linear_planar2_batch: [CuBox<[f32]>; 3], // half-res linear RGB
    _freq2_batch: [[CuBox<[f32]>; 3]; 4],     // [band][channel]
    _block_diff_dc_batch: [CuBox<[f32]>; 3],
    _block_diff_ac_batch: [CuBox<[f32]>; 3],
    _mask_batch: CuBox<[f32]>,
    _mask_temp_batch: CuBox<[f32]>,
    _temp1_batch: CuBox<[f32]>,
    _temp2_batch: CuBox<[f32]>,
    _diffmap_batch: CuBox<[f32]>,
    _diffmap_half_batch: CuBox<[f32]>,
    /// Per-image f32 max result (stored as u32 bits for atomicMax).
    _max_results: CuBox<[u32]>,
    /// Staging Image for N packed-RGB u8 uploads.
    _dis_staging: Image<u8, C<3>>,
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

        let dis_staging: Image<u8, C<3>> =
            Image::malloc(width, height * batch_size as u32)
                .map_err(|e| Error::Npp(format!("{:?}", e)))?;
        let linear2_batch: Image<f32, C<3>> =
            Image::malloc(width, height * batch_size as u32)
                .map_err(|e| Error::Npp(format!("{:?}", e)))?;

        let xyb2_batch = three(plane)?;
        let linear_blur2_batch = three(plane)?;
        let linear_planar2_batch = three(half_plane)?;
        let freq2_batch = [
            three(plane)?,
            three(plane)?,
            three(plane)?,
            three(plane)?,
        ];
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
            _linear2_batch: linear2_batch,
            _xyb2_batch: xyb2_batch,
            _linear_blur2_batch: linear_blur2_batch,
            _linear_planar2_batch: linear_planar2_batch,
            _freq2_batch: freq2_batch,
            _block_diff_dc_batch: block_diff_dc_batch,
            _block_diff_ac_batch: block_diff_ac_batch,
            _mask_batch: mask_batch,
            _mask_temp_batch: mask_temp_batch,
            _temp1_batch: temp1_batch,
            _temp2_batch: temp2_batch,
            _diffmap_batch: diffmap_batch,
            _diffmap_half_batch: diffmap_half_batch,
            _max_results: max_results,
            _dis_staging: dis_staging,
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
    pub fn set_reference(&mut self, reference: impl Img<u8, C<3>>) -> Result<(), Error> {
        self.inner.set_reference(reference)
    }

    pub fn has_reference(&self) -> bool {
        self.inner.has_reference()
    }

    pub fn clear_reference(&mut self) {
        self.inner.clear_reference()
    }

    /// Score N distorted variants against the cached reference.
    ///
    /// **WIP (coefficient#13):** returns an error until the
    /// stride-split Malta + mask batch kernels land. See module doc.
    pub fn compute_batch_with_reference(
        &mut self,
        distorted_bytes: &[u8],
    ) -> Result<Vec<f32>, Error> {
        let (w, h) = self.dimensions();
        let expected = (w as usize) * (h as usize) * 3 * self.batch_size;
        if distorted_bytes.len() != expected {
            return Err(Error::InvalidDimensions(format!(
                "expected {} bytes for batch of {} × {}x{}, got {}",
                expected,
                self.batch_size,
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
        Err(Error::Cuda(
            "ButteraugliBatch::compute_batch_with_reference — work in progress \
             (coefficient#13). Batched per-plane buffers + all single-image \
             reference-cache machinery is allocated; what's missing is Malta \
             and mask batch kernels that take separate strides for the single \
             reference plane (broadcast) vs the N distorted planes. Expected \
             to land in a follow-up commit."
                .into(),
        ))
    }
}

impl Drop for ButteraugliBatch {
    fn drop(&mut self) {
        let _ = cudarse_driver::sync_ctx();
    }
}
