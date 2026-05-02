//! Batched SSIMULACRA2 — score N distorted images against one cached
//! reference in fewer kernel launches.
//!
//! Mirrors the `butteraugli-gpu::ButteraugliBatch` pattern: the
//! reference side is cached as single-image planes inside an embedded
//! `Ssim2`, and the distorted side is packed contiguously per scale
//! (`batch_size` planes × `plane_stride` floats apart). Most kernels
//! get either:
//!   - **batched** variants (per-image clamp inside the kernel, e.g.
//!     `blur_pass_batched_kernel`, `transpose_batched_kernel`,
//!     `downscale_2x_plane_batched_kernel`, `fused_sum_p4_batched_kernel`); or
//!   - **broadcast-batched** variants (one ref-side input read at
//!     `idx % plane_stride`, batched dis-side at `idx`, e.g.
//!     `pointwise_mul_broadcast_batched_kernel`,
//!     `error_maps_broadcast_batched_kernel`).
//! Pointwise kernels with no per-image boundary semantics (sRGB→linear,
//! linear→XYB, plain `pointwise_mul` of two same-shape inputs) reuse
//! the existing single-image kernels — they only see flat f32 arrays
//! and don't care about batch structure.
//!
//! Throughput vs the sequential wrapper grows fastest at small images
//! where launch overhead dominates per-call cost.

use cubecl::prelude::*;

use crate::kernels::{blur, downscale, error_maps, reduction, srgb, transpose, xyb};
use crate::pipeline::{Ssim2, score_from_stats};
use crate::{Error, GpuSsim2Result, NUM_SCALES, Result};

/// Per-scale batched buffer set. Each plane is `batch_size · n_pixels`
/// f32, stored as `batch_size` contiguous planes (stride =
/// `plane_stride = n_pixels`).
struct BatchScale {
    width: u32,
    height: u32,
    /// Pixels per single-image plane.
    n: usize,
    /// `n_pixels`; storage stride between consecutive image planes.
    plane_stride: u32,

    dis_lin: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],
    dis_xyb_t: [cubecl::server::Handle; 3],
    sigma22_in: [cubecl::server::Handle; 3],
    sigma12_in: [cubecl::server::Handle; 3],
    sigma22_full: [cubecl::server::Handle; 3],
    sigma12_full: [cubecl::server::Handle; 3],
    mu2_full: [cubecl::server::Handle; 3],
    /// Rolling scratch buffers reused across the three two-pass blurs.
    v_scratch: [cubecl::server::Handle; 3],
    t_scratch: [cubecl::server::Handle; 3],
    /// Error map outputs — one batched plane each.
    ssim: [cubecl::server::Handle; 3],
    artifact: [cubecl::server::Handle; 3],
    detail: [cubecl::server::Handle; 3],
}

fn alloc_plane<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}
fn alloc_3<R: Runtime>(client: &ComputeClient<R>, n: usize) -> [cubecl::server::Handle; 3] {
    [alloc_plane(client, n), alloc_plane(client, n), alloc_plane(client, n)]
}

impl BatchScale {
    fn new<R: Runtime>(
        client: &ComputeClient<R>,
        width: u32,
        height: u32,
        batch_size: u32,
    ) -> Self {
        let n = (width as usize) * (height as usize);
        let n_total = n * (batch_size as usize);
        Self {
            width,
            height,
            n,
            plane_stride: n as u32,
            dis_lin: alloc_3(client, n_total),
            dis_xyb: alloc_3(client, n_total),
            dis_xyb_t: alloc_3(client, n_total),
            sigma22_in: alloc_3(client, n_total),
            sigma12_in: alloc_3(client, n_total),
            sigma22_full: alloc_3(client, n_total),
            sigma12_full: alloc_3(client, n_total),
            mu2_full: alloc_3(client, n_total),
            v_scratch: alloc_3(client, n_total),
            t_scratch: alloc_3(client, n_total),
            ssim: alloc_3(client, n_total),
            artifact: alloc_3(client, n_total),
            detail: alloc_3(client, n_total),
        }
    }
}

/// Per-image stats laid out flat. 6 scales × 3 channels × 3 map types
/// × 2 stats = 108 f32 per image. Matches the layout `Ssim2`'s
/// `read_and_aggregate` consumes (so the score-folding helper drops in
/// per-image-by-per-image with no slice gymnastics).
const STATS_PER_IMAGE_SLOTS: u32 = (NUM_SCALES * 3 * 3) as u32; // 54
const STATS_PER_IMAGE_FLOATS: usize = (STATS_PER_IMAGE_SLOTS as usize) * 2; // 108

/// Score N distorted images against a cached reference using batched
/// kernel launches.
///
/// ## Lifecycle
///
/// 1. `Ssim2Batch::new(client, w, h, batch_size)` — allocates per-scale
///    batched buffers for `batch_size` distorted images at `w × h`.
/// 2. `set_reference(ref_srgb)` — populates the embedded `Ssim2`'s
///    cache; equivalent to `Ssim2::set_reference`.
/// 3. `compute_batch(&[Vec<u8>])` — score up to `batch_size` images at
///    once; pads the unused slots with zeros if fewer are passed in,
///    and returns `Vec<GpuSsim2Result>` of length = inputs.len().
pub struct Ssim2Batch<R: Runtime> {
    inner: Ssim2<R>,
    batch_size: u32,

    bscales: Vec<BatchScale>,

    /// sRGB u8 staging for the batched dis bytes (concat'd: image-0
    /// bytes, image-1 bytes, … image-{N-1} bytes — each `n_pixels × 3`
    /// bytes). Re-uploaded per `compute_batch`.
    src_u8_batch: cubecl::server::Handle,
    /// Per-image flat sums: `batch_size × 108` floats.
    sums: cubecl::server::Handle,
}

impl<R: Runtime> Ssim2Batch<R> {
    /// Allocate per-instance buffers for scoring `batch_size` images
    /// of `width × height`. Returns `Err(InvalidImageSize)` for
    /// images smaller than 8×8 (matches `Ssim2::new`), or
    /// `Err(InvalidBatchSize)` for `batch_size == 0`.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2Batch;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let b = Ssim2Batch::<WgpuRuntime>::new(client, 256, 256, 4)?;
    /// assert_eq!(b.batch_size(), 4);
    /// assert_eq!(b.dimensions(), (256, 256));
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn new(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        batch_size: u32,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidBatchSize { got: 0, max: 0 });
        }
        let inner = Ssim2::new(client.clone(), width, height)?;

        let bscales: Vec<BatchScale> = (0..inner.n_scales())
            .map(|s| {
                let (w, h, _) = inner.scale_dims(s);
                BatchScale::new(&client, w, h, batch_size)
            })
            .collect();

        let n_full = (width as usize) * (height as usize);
        let src_u8_batch = client.create_from_slice(&vec![0_u8; n_full * 3 * (batch_size as usize)]);
        let sums = client.create_from_slice(f32::as_bytes(&vec![
            0.0_f32;
            STATS_PER_IMAGE_FLOATS * (batch_size as usize)
        ]));

        Ok(Self {
            inner,
            batch_size,
            bscales,
            src_u8_batch,
            sums,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.inner.dimensions()
    }
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Cache the reference image. Required before any
    /// `compute_batch` call.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.inner.set_reference(ref_srgb)
    }

    pub fn clear_reference(&mut self) {
        self.inner.clear_reference();
    }

    pub fn has_cached_reference(&self) -> bool {
        self.inner.has_cached_reference()
    }

    /// Score up to `batch_size` distorted images in one batched
    /// pipeline launch.
    ///
    /// Returns:
    /// - `Err(NoCachedReference)` if `set_reference` hasn't been called.
    /// - `Err(DimensionMismatch)` if any distorted image's byte
    ///   length doesn't match the configured size.
    /// - `Err(InvalidBatchSize)` if `dis.len() > batch_size`.
    ///
    /// Fewer than `batch_size` images is fine — the unused slots get
    /// zero-padded; the returned `Vec` only contains scores for the
    /// images the caller actually passed.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2Batch;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut batch = Ssim2Batch::<WgpuRuntime>::new(client, 256, 256, 8)?;
    /// let r = vec![0_u8; 256 * 256 * 3];
    /// batch.set_reference(&r)?;
    ///
    /// // Pass 3 images even though batch_size=8 — fine.
    /// let candidates: Vec<Vec<u8>> = (0..3).map(|_| r.clone()).collect();
    /// let scores = batch.compute_batch(&candidates)?;
    /// assert_eq!(scores.len(), 3);
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn compute_batch(&mut self, dis: &[Vec<u8>]) -> Result<Vec<GpuSsim2Result>> {
        if !self.inner.has_cached_reference() {
            return Err(Error::NoCachedReference);
        }
        let n_in = dis.len();
        if (n_in as u32) > self.batch_size {
            return Err(Error::InvalidBatchSize {
                got: n_in,
                max: self.batch_size as usize,
            });
        }
        if n_in == 0 {
            return Ok(Vec::new());
        }

        let (w, h) = self.inner.dimensions();
        let n_full = (w as usize) * (h as usize);
        let bytes_per_image = n_full * 3;

        // Concatenate per-image byte buffers and zero-pad up to batch_size.
        let total_bytes = bytes_per_image * (self.batch_size as usize);
        let mut packed = Vec::with_capacity(total_bytes);
        for d in dis {
            if d.len() != bytes_per_image {
                return Err(Error::DimensionMismatch {
                    expected: bytes_per_image,
                    got: d.len(),
                });
            }
            packed.extend_from_slice(d);
        }
        let pad = total_bytes - packed.len();
        if pad > 0 {
            packed.resize(total_bytes, 0);
        }

        // Upload + sRGB → linear into bscales[0].dis_lin.
        let client = self.inner.client().clone();
        self.src_u8_batch = client.create_from_slice(&packed);
        let n_total_full = n_full * (self.batch_size as usize);
        unsafe {
            srgb::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &client,
                cube_count_1d(n_total_full),
                cube_dim_1d(),
                ArrayArg::from_raw_parts(self.src_u8_batch.clone(), n_total_full * 3),
                ArrayArg::from_raw_parts(self.bscales[0].dis_lin[0].clone(), n_total_full),
                ArrayArg::from_raw_parts(self.bscales[0].dis_lin[1].clone(), n_total_full),
                ArrayArg::from_raw_parts(self.bscales[0].dis_lin[2].clone(), n_total_full),
            );
        }

        // Build dis_lin pyramid (batched downscale).
        for s in 1..self.bscales.len() {
            let prev = &self.bscales[s - 1];
            let curr = &self.bscales[s];
            let prev_w = prev.width;
            let prev_h = prev.height;
            let curr_w = curr.width;
            let curr_h = curr.height;
            let prev_pl = prev.plane_stride;
            let curr_pl = curr.plane_stride;
            let n_curr_total = curr.n * (self.batch_size as usize);
            for ch in 0..3 {
                unsafe {
                    downscale::downscale_2x_plane_batched_kernel::launch_unchecked::<R>(
                        &client,
                        cube_count_1d(n_curr_total),
                        cube_dim_1d(),
                        ArrayArg::from_raw_parts(
                            self.bscales[s - 1].dis_lin[ch].clone(),
                            prev.n * (self.batch_size as usize),
                        ),
                        ArrayArg::from_raw_parts(
                            self.bscales[s].dis_lin[ch].clone(),
                            n_curr_total,
                        ),
                        prev_w,
                        prev_h,
                        curr_w,
                        curr_h,
                        prev_pl,
                        curr_pl,
                    );
                }
            }
        }

        // Per-scale: XYB → products → blurs → transposes → error_maps → reductions.
        for s in 0..self.bscales.len() {
            self.process_scale_batched(s);
        }

        // Read sums and compute per-image scores.
        let bytes = client
            .read_one(self.sums.clone())
            .expect("read batched sums");
        let raw = f32::from_bytes(&bytes);
        debug_assert_eq!(raw.len(), STATS_PER_IMAGE_FLOATS * (self.batch_size as usize));

        // Reset sums for next call.
        self.sums = client.create_from_slice(f32::as_bytes(&vec![
            0.0_f32;
            STATS_PER_IMAGE_FLOATS * (self.batch_size as usize)
        ]));

        let mut results = Vec::with_capacity(n_in);
        for img_idx in 0..n_in {
            let img_off = img_idx * STATS_PER_IMAGE_FLOATS;
            let block = &raw[img_off..img_off + STATS_PER_IMAGE_FLOATS];
            results.push(self.fold_score(block));
        }
        Ok(results)
    }

    /// Per-scale batched processing: one image's worth of work, expanded
    /// across `batch_size` slots in parallel via the batched kernels.
    fn process_scale_batched(&self, s: usize) {
        let bs = &self.bscales[s];
        let plane_stride = bs.plane_stride;
        let n_total = bs.n * (self.batch_size as usize);
        let ref_n = bs.n; // single-image ref plane length
        let client = self.inner.client();

        // 1. linear → positive XYB (pointwise on a flat batched array).
        unsafe {
            xyb::linear_to_xyb_planar_kernel::launch_unchecked::<R>(
                client,
                cube_count_1d(n_total),
                cube_dim_1d(),
                ArrayArg::from_raw_parts(bs.dis_lin[0].clone(), n_total),
                ArrayArg::from_raw_parts(bs.dis_lin[1].clone(), n_total),
                ArrayArg::from_raw_parts(bs.dis_lin[2].clone(), n_total),
                ArrayArg::from_raw_parts(bs.dis_xyb[0].clone(), n_total),
                ArrayArg::from_raw_parts(bs.dis_xyb[1].clone(), n_total),
                ArrayArg::from_raw_parts(bs.dis_xyb[2].clone(), n_total),
            );
        }

        // 2. sigma22_in = dis · dis (plain pointwise on flat array).
        for ch in 0..3 {
            unsafe {
                crate::pipeline::pointwise_mul_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_1d(n_total),
                    cube_dim_1d(),
                    ArrayArg::from_raw_parts(bs.dis_xyb[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.dis_xyb[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.sigma22_in[ch].clone(), n_total),
                );
            }
        }

        // 3. sigma12_in = ref_xyb_broadcast · dis_xyb_batched.
        let ref_xyb = self.inner.cached_ref_xyb(s);
        for ch in 0..3 {
            unsafe {
                error_maps::pointwise_mul_broadcast_batched_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_1d(n_total),
                    cube_dim_1d(),
                    ArrayArg::from_raw_parts(ref_xyb[ch].clone(), ref_n),
                    ArrayArg::from_raw_parts(bs.dis_xyb[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.sigma12_in[ch].clone(), n_total),
                    plane_stride,
                );
            }
        }

        // 4. Blur each of {sigma22_in, sigma12_in, dis_xyb} via
        //    batched-vpass → batched-transpose → batched-vpass into
        //    {sigma22_full, sigma12_full, mu2_full}. Three blurs reuse
        //    one (v_scratch, t_scratch) per channel.
        for ch in 0..3 {
            self.blur_batched_two_pass(
                s,
                ch,
                bs.sigma22_in[ch].clone(),
                bs.sigma22_full[ch].clone(),
            );
            self.blur_batched_two_pass(
                s,
                ch,
                bs.sigma12_in[ch].clone(),
                bs.sigma12_full[ch].clone(),
            );
            self.blur_batched_two_pass(
                s,
                ch,
                bs.dis_xyb[ch].clone(),
                bs.mu2_full[ch].clone(),
            );
        }

        // 5. Transpose raw dis_xyb → dis_xyb_t (so error_maps reads
        //    `distorted` in the same orientation as the blurred mu2/
        //    sigma22/sigma12 buffers).
        for ch in 0..3 {
            unsafe {
                transpose::transpose_batched_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_1d(n_total),
                    cube_dim_1d(),
                    ArrayArg::from_raw_parts(bs.dis_xyb[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.dis_xyb_t[ch].clone(), n_total),
                    bs.width,
                    bs.height,
                    plane_stride,
                );
            }
        }

        // 6. error_maps: ref-side broadcast (source=ref_xyb_t cached,
        //    mu1=mu1_full cached, sigma11=sigma11_full cached);
        //    dis-side batched (distorted=dis_xyb_t, mu2=mu2_full,
        //    sigma22=sigma22_full, sigma12=sigma12_full).
        let ref_xyb_t = self.inner.cached_ref_xyb_t(s);
        let mu1_full = self.inner.cached_mu1_full(s);
        let sigma11_full = self.inner.cached_sigma11_full(s);
        for ch in 0..3 {
            unsafe {
                error_maps::error_maps_broadcast_batched_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_1d(n_total),
                    cube_dim_1d(),
                    ArrayArg::from_raw_parts(ref_xyb_t[ch].clone(), ref_n),
                    ArrayArg::from_raw_parts(bs.dis_xyb_t[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(mu1_full[ch].clone(), ref_n),
                    ArrayArg::from_raw_parts(bs.mu2_full[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(sigma11_full[ch].clone(), ref_n),
                    ArrayArg::from_raw_parts(bs.sigma22_full[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.sigma12_full[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.ssim[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.artifact[ch].clone(), n_total),
                    ArrayArg::from_raw_parts(bs.detail[ch].clone(), n_total),
                    plane_stride,
                );
            }
        }

        // 7. Per-image batched reductions. Slot encoding matches
        //    `Ssim2::run_reductions`: slot = scale*9 + ch*3 + map_type,
        //    so per-image stats line up byte-for-byte with the
        //    single-image layout.
        for ch in 0..3 {
            let plane_handles = [&bs.ssim[ch], &bs.artifact[ch], &bs.detail[ch]];
            for map_type in 0..3 {
                let slot = (s as u32) * 9 + (ch as u32) * 3 + map_type as u32;
                reduction::launch_sum_p4_batched::<R>(
                    client,
                    plane_handles[map_type].clone(),
                    plane_stride,
                    self.batch_size,
                    self.sums.clone(),
                    STATS_PER_IMAGE_FLOATS * (self.batch_size as usize),
                    STATS_PER_IMAGE_SLOTS,
                    slot,
                );
            }
        }
    }

    /// Two-pass batched blur: `vpass(src) → v_scratch; transpose →
    /// t_scratch; vpass(t_scratch) → dst`. Output stays in transposed
    /// orientation (consumed by error_maps without a final transpose).
    fn blur_batched_two_pass(
        &self,
        s: usize,
        ch: usize,
        src: cubecl::server::Handle,
        dst: cubecl::server::Handle,
    ) {
        let bs = &self.bscales[s];
        let plane_stride = bs.plane_stride;
        let n_total = bs.n * (self.batch_size as usize);
        let client = self.inner.client();
        let v = bs.v_scratch[ch].clone();
        let t = bs.t_scratch[ch].clone();

        unsafe {
            // 1. v-pass (per-image columns of width × height).
            blur::blur_pass_batched_kernel::launch_unchecked::<R>(
                client,
                blur_cube_count(bs.width, self.batch_size),
                blur_cube_dim(),
                ArrayArg::from_raw_parts(src, n_total),
                ArrayArg::from_raw_parts(v.clone(), n_total),
                bs.width,
                bs.height,
                plane_stride,
            );
            // 2. transpose to height × width.
            transpose::transpose_batched_kernel::launch_unchecked::<R>(
                client,
                cube_count_1d(n_total),
                cube_dim_1d(),
                ArrayArg::from_raw_parts(v, n_total),
                ArrayArg::from_raw_parts(t.clone(), n_total),
                bs.width,
                bs.height,
                plane_stride,
            );
            // 3. v-pass on transposed: width swapped with height.
            blur::blur_pass_batched_kernel::launch_unchecked::<R>(
                client,
                blur_cube_count(bs.height, self.batch_size),
                blur_cube_dim(),
                ArrayArg::from_raw_parts(t, n_total),
                ArrayArg::from_raw_parts(dst, n_total),
                bs.height,
                bs.width,
                plane_stride,
            );
        }
    }

    /// Fold one image's 108-stat block into a final score. Mirrors
    /// `Ssim2::read_and_aggregate`'s per-scale 1/N normalisation and
    /// the published 108-weight WEIGHT table.
    fn fold_score(&self, block: &[f32]) -> GpuSsim2Result {
        debug_assert_eq!(block.len(), STATS_PER_IMAGE_FLOATS);
        let mut avg_ssim = vec![[0.0_f64; 6]; NUM_SCALES];
        let mut avg_edgediff = vec![[0.0_f64; 12]; NUM_SCALES];

        for s in 0..self.bscales.len() {
            let n_pix = self.bscales[s].n as f64;
            let one_per_pixels = 1.0 / n_pix;
            for ch in 0..3 {
                // Slot indexing matches Ssim2::run_reductions:
                //   off = scale*18 + ch*6 + map*2
                let base = s * 18 + ch * 6;
                let s_sum = block[base] as f64;
                let s_p4 = block[base + 1] as f64;
                let a_sum = block[base + 2] as f64;
                let a_p4 = block[base + 3] as f64;
                let d_sum = block[base + 4] as f64;
                let d_p4 = block[base + 5] as f64;

                avg_ssim[s][ch * 2] = one_per_pixels * s_sum;
                avg_ssim[s][ch * 2 + 1] = (one_per_pixels * s_p4).sqrt().sqrt();
                avg_edgediff[s][ch * 4] = one_per_pixels * a_sum;
                avg_edgediff[s][ch * 4 + 1] = (one_per_pixels * a_p4).sqrt().sqrt();
                avg_edgediff[s][ch * 4 + 2] = one_per_pixels * d_sum;
                avg_edgediff[s][ch * 4 + 3] = (one_per_pixels * d_p4).sqrt().sqrt();
            }
        }

        GpuSsim2Result {
            score: score_from_stats(&avg_ssim, &avg_edgediff, self.bscales.len()),
        }
    }
}

fn cube_count_1d(n: usize) -> CubeCount {
    const TPB: u32 = 256;
    let cubes = ((n as u32) + TPB - 1) / TPB;
    CubeCount::Static(cubes.max(1), 1, 1)
}
fn cube_dim_1d() -> CubeDim {
    CubeDim::new_1d(256)
}
fn blur_cube_count(width: u32, batch_size: u32) -> CubeCount {
    let cubes = (width + blur::BLOCK_WIDTH - 1) / blur::BLOCK_WIDTH;
    CubeCount::Static(cubes.max(1), batch_size, 1)
}
fn blur_cube_dim() -> CubeDim {
    CubeDim::new_1d(blur::BLOCK_WIDTH)
}
