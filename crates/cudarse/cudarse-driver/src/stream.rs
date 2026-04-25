use cudarse_driver_sys::cuStreamIsCapturing;
use std::ffi::c_void;
use std::ptr::{NonNull, null_mut};
use sys::{
    CUstream_flags, CUstreamCaptureMode_enum, CuError, CuResult, cuCtxGetStreamPriorityRange,
    cuStreamBeginCapture_v2, cuStreamCreate, cuStreamCreateWithPriority, cuStreamDestroy_v2,
    cuStreamEndCapture, cuStreamGetPriority, cuStreamQuery, cuStreamSynchronize, cuStreamWaitEvent,
};

/// Stream priority as returned by [`CuStream::priority_range`].
///
/// CUDA stream priorities are integers where smaller is higher priority.
/// The two common values are the highest (most-preferred) and lowest
/// (least-preferred) priorities supported by the current context.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StreamPriorityRange {
    /// Least-preferred priority — use for background compute that should
    /// yield to interactive workloads (like display compositors).
    pub least: i32,
    /// Most-preferred priority — use for latency-critical work.
    pub greatest: i32,
}

use crate::{CuEvent, CuGraph, sys};

#[repr(transparent)]
pub struct CuStream(pub(crate) sys::CUstream);

impl Default for CuStream {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Default for &CuStream {
    fn default() -> Self {
        CuStream::DEFAULT_
    }
}

impl CuStream {
    pub const DEFAULT: Self = CuStream(null_mut());
    pub const DEFAULT_: &'static Self = &CuStream(null_mut());

    /// Create a new CUDA stream.
    pub fn new() -> CuResult<Self> {
        let mut stream = null_mut();
        unsafe {
            cuStreamCreate(&mut stream, CUstream_flags::CU_STREAM_NON_BLOCKING as _).result()?;
        }
        Ok(Self(stream))
    }

    /// Create a new CUDA stream with an explicit priority.
    ///
    /// Priorities are integers where smaller is higher priority. Use
    /// [`CuStream::priority_range`] to discover the valid range on the
    /// current context and pass `range.least` for background compute
    /// that should yield to higher-priority streams (including the
    /// Windows display compositor on WSL2).
    ///
    /// On context types that don't support priorities, CUDA silently
    /// ignores the value.
    pub fn new_with_priority(priority: i32) -> CuResult<Self> {
        let mut stream = null_mut();
        unsafe {
            cuStreamCreateWithPriority(
                &mut stream,
                CUstream_flags::CU_STREAM_NON_BLOCKING as _,
                priority,
            )
            .result()?;
        }
        Ok(Self(stream))
    }

    /// Query the least- and greatest-preferred stream priorities
    /// supported by the current CUDA context. The range may be a single
    /// value on devices without priority support.
    pub fn priority_range() -> CuResult<StreamPriorityRange> {
        let mut least: i32 = 0;
        let mut greatest: i32 = 0;
        unsafe {
            cuCtxGetStreamPriorityRange(&mut least as *mut _, &mut greatest as *mut _).result()?;
        }
        Ok(StreamPriorityRange { least, greatest })
    }

    /// Return the priority of this stream.
    pub fn priority(&self) -> CuResult<i32> {
        let mut prio: i32 = 0;
        unsafe {
            cuStreamGetPriority(self.0, &mut prio as *mut _).result()?;
        }
        Ok(prio)
    }

    pub fn raw(&self) -> sys::CUstream {
        self.0 as _
    }

    pub fn inner(&self) -> *mut c_void {
        self.0 as _
    }

    /// Wait for any work on this stream to complete.
    pub fn sync(&self) -> CuResult<()> {
        unsafe { cuStreamSynchronize(self.0).result() }
    }

    /// Return true if this stream has finished any submitted work.
    pub fn completed(&self) -> CuResult<bool> {
        unsafe {
            match cuStreamQuery(self.0) {
                sys::CUresult::CUDA_SUCCESS => Ok(true),
                sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
                other => Err(CuError(other)),
            }
        }
    }

    /// Make this stream wait for an event ot complete.
    pub fn wait_for_evt(&self, evt: &CuEvent) -> CuResult<()> {
        unsafe { cuStreamWaitEvent(self.0, evt.0, 0).result() }
    }

    /// Make this stream wait for the work in another stream to complete
    pub fn wait_for_stream(&self, other: &Self) -> CuResult<()> {
        let evt = CuEvent::new()?;
        evt.record(other)?;
        self.wait_for_evt(&evt)
    }

    /// Join two streams by making each one wait for the other.
    /// After this point, both streams are synchronized with each other.
    pub fn join(&self, other: &Self) -> CuResult<()> {
        self.wait_for_stream(other)?;
        other.wait_for_stream(self)
    }

    pub fn begin_capture(&self) -> CuResult<()> {
        unsafe {
            cuStreamBeginCapture_v2(
                self.0,
                CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .result()
        }
    }

    /// Begin graph capture in thread-local mode so concurrent workers
    /// on other threads are not serialized during the capture. This is
    /// the right mode for capturing background-metric streams from
    /// multi-threaded workers.
    pub fn begin_capture_thread_local(&self) -> CuResult<()> {
        unsafe {
            cuStreamBeginCapture_v2(
                self.0,
                CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
        }
    }

    pub fn is_capturing(&self) -> CuResult<bool> {
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe {
            cuStreamIsCapturing(self.0, &mut status).result()?;
        }
        Ok(status == sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    }

    pub fn end_capture(&self) -> CuResult<CuGraph> {
        let mut graph = null_mut();
        unsafe { cuStreamEndCapture(self.0, &mut graph).result()? };
        Ok(CuGraph(NonNull::new(graph).expect("Invalid graph")))
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                // Never panic in Drop — a panic-in-destructor after the
                // primary panic (e.g., a CUDA OOM upstream) is aborted
                // by the runtime with no unwind, killing the process
                // before any in-flight work can flush. Log and leak
                // instead; the OS reclaims everything when the process
                // exits.
                if let Err(e) = cuStreamDestroy_v2(self.0).result() {
                    eprintln!(
                        "[cudarse] CuStream::drop cuStreamDestroy_v2 failed (leaking): {:?}",
                        e
                    );
                }
            }
        }
    }
}
