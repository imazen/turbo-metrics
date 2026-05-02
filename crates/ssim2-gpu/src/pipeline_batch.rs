//! Batched SSIMULACRA2 pipeline — score N distorted images against one
//! cached reference in fewer kernel launches.
//!
//! Mirrors `butteraugli-gpu`'s `ButteraugliBatch` shape (per-slot buffers,
//! broadcast reference). Filled in after the single-image pipeline is
//! validated end-to-end; this stub keeps `lib.rs` pointing somewhere.

use cubecl::prelude::*;

use crate::pipeline::Ssim2;
use crate::{Error, GpuSsim2Result, Result};

/// Score many distorted images against a fixed reference. Currently a
/// thin wrapper around [`Ssim2`]; the proper batched-kernel
/// implementation is tracked separately.
pub struct Ssim2Batch<R: Runtime> {
    inner: Ssim2<R>,
}

impl<R: Runtime> Ssim2Batch<R> {
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            inner: Ssim2::new(client, width, height)?,
        })
    }

    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.inner.set_reference(ref_srgb)
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.inner.dimensions()
    }

    /// Score `dist_srgb` (a single image's bytes; one comparison per
    /// call) against the cached reference. Loop this for batched use.
    pub fn compute_one(&mut self, dist_srgb: &[u8]) -> Result<GpuSsim2Result> {
        self.inner.compute_with_reference(dist_srgb)
    }

    /// Score N distorted images in sequence; returns one result per
    /// input. `dis` must be `[w·h·3 ; N]`-flat.
    pub fn compute_many(&mut self, dis: &[Vec<u8>]) -> Result<Vec<GpuSsim2Result>> {
        if !self.inner.has_cached_reference() {
            return Err(Error::NoCachedReference);
        }
        let mut out = Vec::with_capacity(dis.len());
        for d in dis {
            out.push(self.inner.compute_with_reference(d)?);
        }
        Ok(out)
    }
}
