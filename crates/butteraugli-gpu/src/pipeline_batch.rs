//! Batched butteraugli scorer: one cached reference vs `N` distorted
//! variants per `compute_batch_with_reference` call. Mirrors
//! `butteraugli_cuda::ButteraugliBatch` in API shape and intent.
//!
//! Distorted-side planes are packed contiguously as `[image_0,
//! image_1, …, image_{N-1}]` so a single launch per kernel covers all
//! `N` images. Reference state lives in the inner [`Butteraugli`] and
//! is broadcast into the batched accumulator/mask buffers via the
//! `*_broadcast_batched_kernel` family.
//!
//! Multi-resolution (matches CPU butteraugli's default) is always on:
//! a half-resolution batched sibling computes its own diffmap and is
//! supersample-added into the full-res diffmap before reduction.

use cubecl::prelude::*;

use crate::kernels::{blur, colors, diffmap, downscale, frequency, malta, masking, reduction};
use crate::{Butteraugli, ButteraugliParams, Result};

const SIGMA_LF: f32 = 7.155_933_4;
const SIGMA_OPSIN: f32 = 1.2;
const SIGMA_HF: f32 = 3.224_899_0;
const SIGMA_UHF: f32 = 1.564_163_3;
const REMOVE_MF_RANGE: f32 = 0.29;
const ADD_MF_RANGE: f32 = 0.1;
const REMOVE_HF_RANGE: f32 = 1.5;
const REMOVE_UHF_RANGE: f32 = 0.04;
const SUPPRESS_XY: f32 = 46.0;
// (Default HF asymmetry = 1.0; runtime-overridable via ButteraugliParams on the inner Butteraugli.)
const W_MF_MALTA: f64 = 37.081_987_039_9;
const NORM1_MF: f64 = 130_262_059.556;
const W_MF_MALTA_X: f64 = 8_246.753_213_53;
const NORM1_MF_X: f64 = 1_009_002.705_82;
const W_HF_MALTA: f64 = 18.723_741_438_7;
const NORM1_HF: f64 = 4_498_534.452_32;
const W_HF_MALTA_X: f64 = 6_923.994_761_09;
const NORM1_HF_X: f64 = 8_051.158_332_47;
const W_UHF_MALTA: f64 = 1.100_390_325_55;
const NORM1_UHF: f64 = 71.780_027_516_9;
const W_UHF_MALTA_X: f64 = 173.5;
const NORM1_UHF_X: f64 = 5.0;
const WMUL: [f64; 9] = [
    400.0,
    1.508_157_031_18,
    0.0,
    2_150.0,
    10.619_543_323_9,
    16.217_604_315_2,
    29.235_379_799_4,
    0.844_626_970_982,
    0.703_646_627_719,
];
const MASK_RADIUS: f32 = 2.7;
// (Default intensity_target = 80.0 — read at runtime from inner.params().)
const MIN_SIZE_FOR_SUBSAMPLE: u32 = 16;

fn malta_norm(w_0gt1: f64, w_0lt1: f64, norm1: f64, use_lf: bool) -> (f32, f32, f32) {
    const K_WEIGHT0: f64 = 0.5;
    const K_WEIGHT1: f64 = 0.33;
    const LEN: f64 = 3.75;
    let mulli = if use_lf {
        0.611_612_573_796
    } else {
        0.399_058_176_37
    };
    let w_pre0gt1 = mulli * (K_WEIGHT0 * w_0gt1).sqrt() / (LEN * 2.0 + 1.0);
    let w_pre0lt1 = mulli * (K_WEIGHT1 * w_0lt1).sqrt() / (LEN * 2.0 + 1.0);
    (
        (w_pre0gt1 * norm1) as f32,
        (w_pre0lt1 * norm1) as f32,
        norm1 as f32,
    )
}

struct BatchBuffers<R: Runtime> {
    width: u32,
    height: u32,
    plane: usize,
    batch_n: usize,

    src_u8_batch: cubecl::server::Handle,
    lin_b_batch: [cubecl::server::Handle; 3],
    blur_b_batch: [cubecl::server::Handle; 3],
    freq_b_batch: [[cubecl::server::Handle; 3]; 4],
    block_diff_dc_batch: [cubecl::server::Handle; 3],
    block_diff_ac_batch: [cubecl::server::Handle; 3],
    mask_batch: cubecl::server::Handle,
    mask_scratch_batch: cubecl::server::Handle,
    diffmap_batch: cubecl::server::Handle,
    temp1_batch: cubecl::server::Handle,
    temp2_batch: cubecl::server::Handle,
    _runtime: std::marker::PhantomData<R>,
}

fn alloc_b<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

fn alloc_b3<R: Runtime>(client: &ComputeClient<R>, n: usize) -> [cubecl::server::Handle; 3] {
    [alloc_b(client, n), alloc_b(client, n), alloc_b(client, n)]
}

impl<R: Runtime> BatchBuffers<R> {
    fn new(client: &ComputeClient<R>, width: u32, height: u32, batch_n: usize) -> Self {
        let plane = (width * height) as usize;
        let total = plane * batch_n;
        // sRGB bytes uploaded as u32 — see colors.rs / pipeline.rs
        // for the wgpu Array<u8> caveat.
        let src_u8_batch = client.create_from_slice(u32::as_bytes(&vec![0_u32; total * 3]));
        Self {
            width,
            height,
            plane,
            batch_n,
            src_u8_batch,
            lin_b_batch: alloc_b3(client, total),
            blur_b_batch: alloc_b3(client, total),
            freq_b_batch: [
                alloc_b3(client, total),
                alloc_b3(client, total),
                alloc_b3(client, total),
                alloc_b3(client, total),
            ],
            block_diff_dc_batch: alloc_b3(client, total),
            block_diff_ac_batch: alloc_b3(client, total),
            mask_batch: alloc_b(client, total),
            mask_scratch_batch: alloc_b(client, total),
            diffmap_batch: alloc_b(client, total),
            temp1_batch: alloc_b(client, total),
            temp2_batch: alloc_b(client, total),
            _runtime: std::marker::PhantomData,
        }
    }

    fn total(&self) -> usize {
        self.plane * self.batch_n
    }
}

/// Batched butteraugli scorer. Construct with [`ButteraugliBatch::new`],
/// register a reference image once with [`set_reference`], then call
/// [`compute_batch_with_reference`] with a packed `N × W × H × 3` u8
/// buffer to get back `N` butteraugli scores per round-trip.
pub struct ButteraugliBatch<R: Runtime> {
    inner: Butteraugli<R>,
    batch_size: usize,
    full: BatchBuffers<R>,
    half: Option<BatchBuffers<R>>,
    client: ComputeClient<R>,
    width: u32,
    height: u32,
}

impl<R: Runtime> ButteraugliBatch<R> {
    pub fn new(client: ComputeClient<R>, width: u32, height: u32, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        let inner = Butteraugli::<R>::new_multires(client.clone(), width, height);
        let full = BatchBuffers::new(&client, width, height, batch_size);
        let half = if width >= MIN_SIZE_FOR_SUBSAMPLE && height >= MIN_SIZE_FOR_SUBSAMPLE {
            Some(BatchBuffers::new(
                &client,
                width.div_ceil(2),
                height.div_ceil(2),
                batch_size,
            ))
        } else {
            None
        };
        Self {
            inner,
            batch_size,
            full,
            half,
            client,
            width,
            height,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn set_reference(&mut self, ref_srgb: &[u8]) {
        self.inner.set_reference(ref_srgb);
    }

    /// Cache the reference image with custom [`ButteraugliParams`].
    pub fn set_reference_with_options(
        &mut self,
        ref_srgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<()> {
        self.inner.set_reference_with_options(ref_srgb, params)
    }

    /// Drop the cached reference state.
    pub fn clear_reference(&mut self) {
        self.inner.clear_reference();
    }

    /// Whether the reference cache is populated.
    pub fn has_reference(&self) -> bool {
        self.inner.has_cached_reference()
    }

    /// Active comparison params (last set via
    /// [`set_reference_with_options`]).
    pub fn params(&self) -> &ButteraugliParams {
        self.inner.params()
    }

    /// Score N distorted variants against the cached reference.
    /// Returns one max-norm per image. To also get the libjxl 3-norm,
    /// use [`compute_batch_with_reference_full`].
    pub fn compute_batch_with_reference(&mut self, dist_batch: &[u8]) -> Vec<f32> {
        self.run_batch_pipeline(dist_batch);
        reduction::reduce_batched::<R>(
            &self.client,
            self.full.diffmap_batch.clone(),
            (self.width * self.height) as u32,
            self.batch_size as u32,
        )
    }

    /// Same as [`compute_batch_with_reference`] but returns
    /// `Vec<GpuButteraugliResult>` — `score` (max-norm) plus `pnorm_3`
    /// (libjxl 3-norm aggregation) per image. The reduction kernel is
    /// fused, so the extra sums are essentially free.
    pub fn compute_batch_with_reference_full(
        &mut self,
        dist_batch: &[u8],
    ) -> Vec<crate::GpuButteraugliResult> {
        self.run_batch_pipeline(dist_batch);
        reduction::reduce_batched_with_pnorm::<R>(
            &self.client,
            self.full.diffmap_batch.clone(),
            (self.width * self.height) as u32,
            self.batch_size as u32,
        )
    }

    /// Internal: run everything from sRGB upload through the final
    /// full-res diffmap_batch. Both reduction variants share this.
    fn run_batch_pipeline(&mut self, dist_batch: &[u8]) {
        let n = self.batch_size;
        let bytes_per_image = (self.width * self.height * 3) as usize;
        assert_eq!(
            dist_batch.len(),
            n * bytes_per_image,
            "batch length mismatch"
        );
        assert!(self.inner.has_cached_reference(), "set_reference first");
        // Widen each byte to u32 so wgpu/Metal can read it natively.
        let widened: Vec<u32> = dist_batch.iter().map(|&b| b as u32).collect();
        self.full.src_u8_batch = self.client.create_from_slice(u32::as_bytes(&widened));

        // Phase 1: sRGB → linear; downsample linear before opsin
        // overwrites lin_b_batch with XYB.
        self.full_srgb_to_linear();
        if self.half.is_some() {
            self.populate_half_linear_from_full();
        }
        // Phase 2: full-res pipeline.
        self.run_full_pipeline_from_linear();
        // Phase 3: half-res mix.
        if self.half.is_some() {
            self.run_half_distorted();
            self.add_supersampled_half_to_full();
        }
    }

    /// Just the sRGB → linear stage, populating `lin_b_batch`.
    fn full_srgb_to_linear(&self) {
        let buf = &self.full;
        let total = buf.total();
        let dim_total = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        unsafe {
            colors::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total,
                block,
                ArrayArg::from_raw_parts(buf.src_u8_batch.clone(), total * 3),
                ArrayArg::from_raw_parts(buf.lin_b_batch[0].clone(), total),
                ArrayArg::from_raw_parts(buf.lin_b_batch[1].clone(), total),
                ArrayArg::from_raw_parts(buf.lin_b_batch[2].clone(), total),
            );
        }
    }

    // ─── pipelines ───

    /// Full-res pipeline starting from linear RGB in `lin_b_batch`
    /// (pre-opsin). Runs opsin → freq separation → psycho → mask →
    /// compute_diffmap, leaving the per-image diffmap in
    /// `full.diffmap_batch`.
    fn run_full_pipeline_from_linear(&self) {
        let buf = &self.full;
        let plane = buf.plane as u32;
        let total = buf.total();
        let dim_total = self.cube_count(total);
        let block = CubeDim::new_1d(256);

        // Opsin sensitivity blur (σ=1.2)
        for ch in 0..3 {
            self.batched_blur(
                &buf.lin_b_batch[ch].clone(),
                &buf.blur_b_batch[ch].clone(),
                &buf.temp1_batch.clone(),
                buf.width,
                buf.height,
                plane,
                SIGMA_OPSIN,
            );
        }

        // Opsin (pointwise; in place)
        unsafe {
            colors::opsin_dynamics_planar_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.lin_b_batch[0].clone(), total),
                ArrayArg::from_raw_parts(buf.lin_b_batch[1].clone(), total),
                ArrayArg::from_raw_parts(buf.lin_b_batch[2].clone(), total),
                ArrayArg::from_raw_parts(buf.blur_b_batch[0].clone(), total),
                ArrayArg::from_raw_parts(buf.blur_b_batch[1].clone(), total),
                ArrayArg::from_raw_parts(buf.blur_b_batch[2].clone(), total),
                self.inner.params().intensity_target,
            );
        }

        self.batched_separate_frequencies(buf, plane);

        // Zero accumulators
        for ch in 0..3 {
            self.zero_plane(&buf.block_diff_ac_batch[ch], total);
        }
        // Psycho diff (Malta + L2 broadcast against cached reference)
        self.batched_psycho_diff(buf, plane, true);
        // DC diff (broadcast)
        self.batched_dc_diff(buf, plane, true);
        // Mask distorted side + mask_to_error against cached blurred_a
        self.batched_mask_distorted(buf, plane, true);
        // Broadcast cached mask + compute_diffmap
        self.broadcast_cached_mask(buf, true);
        self.batched_compute_diffmap(buf);
    }

    fn run_half_distorted(&self) {
        let half = self.half.as_ref().unwrap();
        let plane = half.plane as u32;
        let total = half.total();
        let dim_total = self.cube_count(total);
        let block = CubeDim::new_1d(256);

        // half.lin_b_batch already populated by populate_half_linear_from_full.
        for ch in 0..3 {
            self.batched_blur(
                &half.lin_b_batch[ch].clone(),
                &half.blur_b_batch[ch].clone(),
                &half.temp1_batch.clone(),
                half.width,
                half.height,
                plane,
                SIGMA_OPSIN,
            );
        }
        unsafe {
            colors::opsin_dynamics_planar_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(half.lin_b_batch[0].clone(), total),
                ArrayArg::from_raw_parts(half.lin_b_batch[1].clone(), total),
                ArrayArg::from_raw_parts(half.lin_b_batch[2].clone(), total),
                ArrayArg::from_raw_parts(half.blur_b_batch[0].clone(), total),
                ArrayArg::from_raw_parts(half.blur_b_batch[1].clone(), total),
                ArrayArg::from_raw_parts(half.blur_b_batch[2].clone(), total),
                self.inner.params().intensity_target,
            );
        }
        self.batched_separate_frequencies(half, plane);
        for ch in 0..3 {
            self.zero_plane(&half.block_diff_ac_batch[ch], total);
        }
        self.batched_psycho_diff(half, plane, false);
        self.batched_dc_diff(half, plane, false);
        self.batched_mask_distorted(half, plane, false);
        self.broadcast_cached_mask(half, false);
        self.batched_compute_diffmap(half);
    }

    fn populate_half_linear_from_full(&self) {
        let half = self.half.as_ref().unwrap();
        let total = half.total();
        let dim = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        for ch in 0..3 {
            unsafe {
                downscale::downsample_2x_batched_kernel::launch_unchecked::<R>(
                    &self.client,
                    dim.clone(),
                    block.clone(),
                    ArrayArg::from_raw_parts(self.full.lin_b_batch[ch].clone(), self.full.total()),
                    ArrayArg::from_raw_parts(half.lin_b_batch[ch].clone(), total),
                    self.width,
                    self.height,
                    half.width,
                    half.height,
                    self.full.plane as u32,
                    half.plane as u32,
                );
            }
        }
    }

    fn add_supersampled_half_to_full(&self) {
        let half = self.half.as_ref().unwrap();
        let total = self.full.total();
        let dim = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        unsafe {
            downscale::add_upsample_2x_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim,
                block,
                ArrayArg::from_raw_parts(self.full.diffmap_batch.clone(), total),
                ArrayArg::from_raw_parts(half.diffmap_batch.clone(), half.total()),
                self.width,
                self.height,
                half.width,
                half.plane as u32,
                self.full.plane as u32,
                0.5_f32,
            );
        }
    }

    // ─── helpers ───

    fn cube_count(&self, total: usize) -> CubeCount {
        let cubes = ((total as u32) + 255) / 256;
        CubeCount::Static(cubes, 1, 1)
    }

    fn zero_plane(&self, dst: &cubecl::server::Handle, total: usize) {
        unsafe {
            frequency::zero_plane_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count(total),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(dst.clone(), total),
            );
        }
    }

    fn batched_blur(
        &self,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        scratch: &cubecl::server::Handle,
        w: u32,
        h: u32,
        plane: u32,
        sigma: f32,
    ) {
        let n = self.batch_size as u32;
        let total = (plane * n) as usize;
        let dim = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        unsafe {
            blur::horizontal_blur_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(src.clone(), total),
                ArrayArg::from_raw_parts(scratch.clone(), total),
                w,
                h * n,
                sigma,
            );
            blur::vertical_blur_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim,
                block,
                ArrayArg::from_raw_parts(scratch.clone(), total),
                ArrayArg::from_raw_parts(dst.clone(), total),
                w,
                h,
                sigma,
                plane,
                n,
            );
        }
    }

    fn batched_separate_frequencies(&self, buf: &BatchBuffers<R>, plane: u32) {
        let total = buf.total();
        let dim = self.cube_count(total);
        let block = CubeDim::new_1d(256);

        // LF + MF
        for ch in 0..3 {
            self.batched_blur(
                &buf.lin_b_batch[ch].clone(),
                &buf.freq_b_batch[3][ch].clone(),
                &buf.temp1_batch.clone(),
                buf.width,
                buf.height,
                plane,
                SIGMA_LF,
            );
            unsafe {
                frequency::subtract_arrays_kernel::launch_unchecked::<R>(
                    &self.client,
                    dim.clone(),
                    block.clone(),
                    ArrayArg::from_raw_parts(buf.lin_b_batch[ch].clone(), total),
                    ArrayArg::from_raw_parts(buf.freq_b_batch[3][ch].clone(), total),
                    ArrayArg::from_raw_parts(buf.freq_b_batch[2][ch].clone(), total),
                );
            }
        }
        unsafe {
            frequency::xyb_low_freq_to_vals_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[3][0].clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[3][1].clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[3][2].clone(), total),
            );
        }

        // MF/HF for X, Y, B (B → just blur in place)
        // X
        self.batched_blur(
            &buf.freq_b_batch[2][0].clone(),
            &buf.temp1_batch.clone(),
            &buf.temp2_batch.clone(),
            buf.width,
            buf.height,
            plane,
            SIGMA_HF,
        );
        unsafe {
            frequency::split_band_remove_inplace_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[2][0].clone(), total),
                ArrayArg::from_raw_parts(buf.temp1_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][0].clone(), total),
                REMOVE_MF_RANGE,
            );
        }
        // Y
        self.batched_blur(
            &buf.freq_b_batch[2][1].clone(),
            &buf.temp1_batch.clone(),
            &buf.temp2_batch.clone(),
            buf.width,
            buf.height,
            plane,
            SIGMA_HF,
        );
        unsafe {
            frequency::split_band_amplify_inplace_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[2][1].clone(), total),
                ArrayArg::from_raw_parts(buf.temp1_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][1].clone(), total),
                ADD_MF_RANGE,
            );
        }
        // B
        self.batched_blur(
            &buf.freq_b_batch[2][2].clone(),
            &buf.temp1_batch.clone(),
            &buf.temp2_batch.clone(),
            buf.width,
            buf.height,
            plane,
            SIGMA_HF,
        );
        unsafe {
            frequency::copy_plane_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.temp1_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[2][2].clone(), total),
            );
            frequency::suppress_x_by_y_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][0].clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][1].clone(), total),
                SUPPRESS_XY,
            );
        }

        // HF/UHF X
        self.batched_blur(
            &buf.freq_b_batch[1][0].clone(),
            &buf.temp1_batch.clone(),
            &buf.mask_scratch_batch.clone(),
            buf.width,
            buf.height,
            plane,
            SIGMA_UHF,
        );
        unsafe {
            frequency::split_uhf_hf_x_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][0].clone(), total),
                ArrayArg::from_raw_parts(buf.temp1_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[0][0].clone(), total),
                ArrayArg::from_raw_parts(buf.temp2_batch.clone(), total),
                REMOVE_UHF_RANGE,
                REMOVE_HF_RANGE,
            );
            frequency::copy_plane_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.temp2_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][0].clone(), total),
            );
        }
        // HF/UHF Y
        self.batched_blur(
            &buf.freq_b_batch[1][1].clone(),
            &buf.temp1_batch.clone(),
            &buf.mask_scratch_batch.clone(),
            buf.width,
            buf.height,
            plane,
            SIGMA_UHF,
        );
        unsafe {
            frequency::split_uhf_hf_y_kernel::launch_unchecked::<R>(
                &self.client,
                dim.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][1].clone(), total),
                ArrayArg::from_raw_parts(buf.temp1_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[0][1].clone(), total),
                ArrayArg::from_raw_parts(buf.temp2_batch.clone(), total),
            );
            frequency::copy_plane_kernel::launch_unchecked::<R>(
                &self.client,
                dim,
                block,
                ArrayArg::from_raw_parts(buf.temp2_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][1].clone(), total),
            );
        }
    }

    fn batched_psycho_diff(&self, buf: &BatchBuffers<R>, plane: u32, is_full: bool) {
        let n = self.batch_size as u32;
        let p = self.inner.params();
        let asym = p.hf_asymmetry as f64;
        let sqrt_asym = asym.sqrt();
        let cached_inner = if is_full {
            &self.inner
        } else {
            self.inner.half_res().expect("half_res cached")
        };
        let total = buf.total();
        let dim_total = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        let cube_count_2d = CubeCount::Static((buf.width + 15) / 16, (buf.height + 15) / 16, n);
        let cube_dim_2d = CubeDim::new_2d(16, 16);

        // ── 6 batched Malta calls (broadcast reference) ──
        let malta_calls: &[(bool, usize, f64, f64, f64)] = &[
            // (use_lf=false → HF kernel, ch, w_gt, w_lt, norm1)
            (false, 1, W_UHF_MALTA * asym, W_UHF_MALTA / asym, NORM1_UHF),
            (
                false,
                0,
                W_UHF_MALTA_X * asym,
                W_UHF_MALTA_X / asym,
                NORM1_UHF_X,
            ),
            (
                true,
                1,
                W_HF_MALTA * sqrt_asym,
                W_HF_MALTA / sqrt_asym,
                NORM1_HF,
            ),
            (
                true,
                0,
                W_HF_MALTA_X * sqrt_asym,
                W_HF_MALTA_X / sqrt_asym,
                NORM1_HF_X,
            ),
            (true, 1, W_MF_MALTA, W_MF_MALTA, NORM1_MF),
            (true, 0, W_MF_MALTA_X, W_MF_MALTA_X, NORM1_MF_X),
        ];
        for (i, &(use_lf, ch, w_gt, w_lt, n1)) in malta_calls.iter().enumerate() {
            let (g, l, n1f) = malta_norm(w_gt, w_lt, n1, use_lf);
            // Band: i 0,1 → UHF (band=0); i 2,3 → HF (band=1); i 4,5 → MF (band=2).
            let band = match i {
                0 | 1 => 0,
                2 | 3 => 1,
                _ => 2,
            };
            let ref_h = cached_inner.cached_freq(band, ch).clone();
            let dis = buf.freq_b_batch[band][ch].clone();
            let acc = buf.block_diff_ac_batch[ch].clone();
            unsafe {
                if use_lf {
                    malta::malta_diff_map_lf_batched_kernel::launch_unchecked::<R>(
                        &self.client,
                        cube_count_2d.clone(),
                        cube_dim_2d.clone(),
                        ArrayArg::from_raw_parts(ref_h, buf.plane),
                        ArrayArg::from_raw_parts(dis, total),
                        ArrayArg::from_raw_parts(acc, total),
                        buf.width,
                        buf.height,
                        g,
                        l,
                        n1f,
                        plane,
                    );
                } else {
                    malta::malta_diff_map_hf_batched_kernel::launch_unchecked::<R>(
                        &self.client,
                        cube_count_2d.clone(),
                        cube_dim_2d.clone(),
                        ArrayArg::from_raw_parts(ref_h, buf.plane),
                        ArrayArg::from_raw_parts(dis, total),
                        ArrayArg::from_raw_parts(acc, total),
                        buf.width,
                        buf.height,
                        g,
                        l,
                        n1f,
                        plane,
                    );
                }
            }
        }

        // ── L2 diffs: 5 broadcast-batched launches (ac side) ──
        unsafe {
            // l2_asym HF X (WMUL[0]) → ac[0]
            diffmap::l2_asym_diff_broadcast_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(cached_inner.cached_freq(1, 0).clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][0].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[0].clone(), total),
                plane,
                (WMUL[0] as f32) * p.hf_asymmetry,
                (WMUL[0] as f32) / p.hf_asymmetry,
            );
            // l2_asym HF Y (WMUL[1]) → ac[1]
            diffmap::l2_asym_diff_broadcast_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(cached_inner.cached_freq(1, 1).clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][1].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[1].clone(), total),
                plane,
                (WMUL[1] as f32) * p.hf_asymmetry,
                (WMUL[1] as f32) / p.hf_asymmetry,
            );
            // l2 MF X (WMUL[3]) → ac[0]
            diffmap::l2_diff_broadcast_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(cached_inner.cached_freq(2, 0).clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.freq_b_batch[2][0].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[0].clone(), total),
                plane,
                WMUL[3] as f32,
            );
            // l2 MF Y (WMUL[4]) → ac[1]
            diffmap::l2_diff_broadcast_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(cached_inner.cached_freq(2, 1).clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.freq_b_batch[2][1].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[1].clone(), total),
                plane,
                WMUL[4] as f32,
            );
            // l2_diff_write MF B (WMUL[5]) → ac[2]
            diffmap::l2_diff_write_broadcast_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total,
                block,
                ArrayArg::from_raw_parts(cached_inner.cached_freq(2, 2).clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.freq_b_batch[2][2].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[2].clone(), total),
                plane,
                WMUL[5] as f32,
            );
        }
    }

    fn batched_dc_diff(&self, buf: &BatchBuffers<R>, plane: u32, is_full: bool) {
        let cached_inner = if is_full {
            &self.inner
        } else {
            self.inner.half_res().expect("half_res cached")
        };
        let total = buf.total();
        let dim_total = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        for ch in 0..3 {
            unsafe {
                diffmap::l2_diff_write_broadcast_batched_kernel::launch_unchecked::<R>(
                    &self.client,
                    dim_total.clone(),
                    block.clone(),
                    ArrayArg::from_raw_parts(cached_inner.cached_freq(3, ch).clone(), buf.plane),
                    ArrayArg::from_raw_parts(buf.freq_b_batch[3][ch].clone(), total),
                    ArrayArg::from_raw_parts(buf.block_diff_dc_batch[ch].clone(), total),
                    plane,
                    WMUL[6 + ch] as f32,
                );
            }
        }
    }

    fn batched_mask_distorted(&self, buf: &BatchBuffers<R>, plane: u32, is_full: bool) {
        let total = buf.total();
        let dim_total = self.cube_count(total);
        let block = CubeDim::new_1d(256);

        unsafe {
            masking::combine_channels_for_masking_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][0].clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[0][0].clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[1][1].clone(), total),
                ArrayArg::from_raw_parts(buf.freq_b_batch[0][1].clone(), total),
                ArrayArg::from_raw_parts(buf.mask_scratch_batch.clone(), total),
            );
            masking::diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total.clone(),
                block.clone(),
                ArrayArg::from_raw_parts(buf.mask_scratch_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.temp2_batch.clone(), total),
            );
        }
        // Blur σ=2.7 batched (use diffmap_batch as scratch — overwritten by compute_diffmap later)
        self.batched_blur(
            &buf.temp2_batch.clone(),
            &buf.mask_scratch_batch.clone(),
            &buf.diffmap_batch.clone(),
            buf.width,
            buf.height,
            plane,
            MASK_RADIUS,
        );
        let cached_inner = if is_full {
            &self.inner
        } else {
            self.inner.half_res().expect("half_res cached")
        };
        unsafe {
            masking::mask_to_error_mul_batched_kernel::launch_unchecked::<R>(
                &self.client,
                dim_total,
                block,
                ArrayArg::from_raw_parts(cached_inner.cached_blurred_a().clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.mask_scratch_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[1].clone(), total),
                plane,
            );
        }
    }

    fn broadcast_cached_mask(&self, buf: &BatchBuffers<R>, is_full: bool) {
        let total = buf.total();
        let dim = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        let cached_inner = if is_full {
            &self.inner
        } else {
            self.inner.half_res().expect("half_res cached")
        };
        unsafe {
            frequency::broadcast_plane_kernel::launch_unchecked::<R>(
                &self.client,
                dim,
                block,
                ArrayArg::from_raw_parts(cached_inner.cached_mask().clone(), buf.plane),
                ArrayArg::from_raw_parts(buf.mask_batch.clone(), total),
                buf.plane as u32,
            );
        }
    }

    fn batched_compute_diffmap(&self, buf: &BatchBuffers<R>) {
        let total = buf.total();
        let dim = self.cube_count(total);
        let block = CubeDim::new_1d(256);
        unsafe {
            diffmap::compute_diffmap_kernel::launch_unchecked::<R>(
                &self.client,
                dim,
                block,
                ArrayArg::from_raw_parts(buf.mask_batch.clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_dc_batch[0].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_dc_batch[1].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_dc_batch[2].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[0].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[1].clone(), total),
                ArrayArg::from_raw_parts(buf.block_diff_ac_batch[2].clone(), total),
                ArrayArg::from_raw_parts(buf.diffmap_batch.clone(), total),
                self.inner.params().xmul,
            );
        }
    }
}
