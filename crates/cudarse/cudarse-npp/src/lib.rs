use std::cell::Cell;
use std::ffi::{CStr, CString, c_void};
use std::fmt::Debug;
use std::mem;
use std::ptr::null_mut;
use std::sync::OnceLock;

pub use cudarse_npp_sys as sys;
use cudarse_npp_sys::{cudaMemcpyAsync, cudaMemcpyKind};
use sys::{
    NppStreamContext, Result, cudaDeviceGetAttribute, cudaFreeAsync, cudaGetDevice,
    cudaGetDeviceProperties_v2, cudaMallocAsync, cudaStream_t, nppGetLibVersion,
};

// CUDA 13 removed NPP's old global-state queries (`nppGetGpuName`,
// `nppGetGpuNumSMs`, `nppGetMaxThreadsPerBlock`, `nppGetMaxThreadsPerSM`,
// `nppGetStream`, `nppGetStreamContext`, `nppSetStream`). Modern NPP
// requires an explicit `NppStreamContext` per call. To keep our existing
// API, we maintain the "global" stream context as a thread-local that
// `set_stream` populates from the active CUDA device, and feed that
// through `get_stream_ctx()` to the `_Ctx` NPP entry points.
thread_local! {
    static NPP_CTX: Cell<NppStreamContext> = const { Cell::new(empty_ctx()) };
}

const fn empty_ctx() -> NppStreamContext {
    NppStreamContext {
        hStream: null_mut(),
        nCudaDeviceId: -1,
        nMultiProcessorCount: 0,
        nMaxThreadsPerMultiProcessor: 0,
        nMaxThreadsPerBlock: 0,
        nSharedMemPerBlock: 0,
        nCudaDevAttrComputeCapabilityMajor: 0,
        nCudaDevAttrComputeCapabilityMinor: 0,
        nStreamFlags: 0,
        nReserved0: 0,
    }
}

fn populate_ctx_for_device(stream: cudaStream_t) -> Result<NppStreamContext> {
    let mut device: i32 = 0;
    unsafe { cudaGetDevice(&mut device).result()? };
    let mut ctx = empty_ctx();
    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    let mut tmp: i32 = 0;
    unsafe {
        cudaDeviceGetAttribute(
            &mut tmp,
            sys::cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
            device,
        )
        .result()?;
        ctx.nMultiProcessorCount = tmp;
        cudaDeviceGetAttribute(
            &mut tmp,
            sys::cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor,
            device,
        )
        .result()?;
        ctx.nMaxThreadsPerMultiProcessor = tmp;
        cudaDeviceGetAttribute(
            &mut tmp,
            sys::cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock,
            device,
        )
        .result()?;
        ctx.nMaxThreadsPerBlock = tmp;
        let mut shared: i32 = 0;
        cudaDeviceGetAttribute(
            &mut shared,
            sys::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlock,
            device,
        )
        .result()?;
        ctx.nSharedMemPerBlock = shared as usize;
        cudaDeviceGetAttribute(
            &mut tmp,
            sys::cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor,
            device,
        )
        .result()?;
        ctx.nCudaDevAttrComputeCapabilityMajor = tmp;
        cudaDeviceGetAttribute(
            &mut tmp,
            sys::cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor,
            device,
        )
        .result()?;
        ctx.nCudaDevAttrComputeCapabilityMinor = tmp;
    }
    Ok(ctx)
}

pub mod image;

mod __priv {
    /// So people don't implement child trait for themselves.
    /// Hacky way of doing closed polymorphism with traits.
    pub trait Sealed {}

    impl<T: Sealed> Sealed for &T {}

    impl<T: Sealed> Sealed for &mut T {}

    impl Sealed for u8 {}
    impl Sealed for i8 {}

    impl Sealed for u16 {}

    impl Sealed for i16 {}

    impl Sealed for i32 {}

    impl Sealed for f32 {}
}

/// Return the NPP lib version
pub fn version() -> (i32, i32, i32) {
    let ver = unsafe { &*nppGetLibVersion() };
    (ver.major, ver.minor, ver.build)
}

/// Return the name of the active CUDA device. Cached after the first
/// query.
///
/// Replaces NPP's removed `nppGetGpuName` (gone in CUDA 13). Falls back
/// to "<unknown-device>" if `cudaGetDeviceProperties` fails.
pub fn gpu_name() -> &'static CStr {
    static CACHED: OnceLock<CString> = OnceLock::new();
    CACHED.get_or_init(|| {
        let mut props = unsafe { mem::zeroed::<sys::cudaDeviceProp>() };
        let mut device: i32 = 0;
        let cstr = unsafe {
            if cudaGetDevice(&mut device).result().is_ok()
                && cudaGetDeviceProperties_v2(&mut props, device)
                    .result()
                    .is_ok()
            {
                CStr::from_ptr(props.name.as_ptr()).to_owned()
            } else {
                CString::new("<unknown-device>").unwrap()
            }
        };
        cstr
    })
}

/// Return the number of streaming multiprocessors of the active device.
/// Replaces removed `nppGetGpuNumSMs`.
pub fn gpu_num_sm() -> u32 {
    let mut device: i32 = 0;
    let mut count: i32 = 0;
    unsafe {
        let _ = cudaGetDevice(&mut device);
        let _ = cudaDeviceGetAttribute(
            &mut count,
            sys::cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
            device,
        );
    }
    count as u32
}

pub fn max_threads_per_block() -> u32 {
    let mut device: i32 = 0;
    let mut count: i32 = 0;
    unsafe {
        let _ = cudaGetDevice(&mut device);
        let _ = cudaDeviceGetAttribute(
            &mut count,
            sys::cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock,
            device,
        );
    }
    count as u32
}

pub fn max_threads_per_sm() -> u32 {
    let mut device: i32 = 0;
    let mut count: i32 = 0;
    unsafe {
        let _ = cudaGetDevice(&mut device);
        let _ = cudaDeviceGetAttribute(
            &mut count,
            sys::cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor,
            device,
        );
    }
    count as u32
}

/// Return the per-thread NPP stream context. The first call (or any
/// call after [`set_stream`]) populates the device-derived fields.
pub fn get_stream_ctx() -> Result<NppStreamContext> {
    let cur = NPP_CTX.with(|c| c.get());
    if cur.nCudaDeviceId < 0 {
        // Lazily initialise on first use with a null stream (== default
        // stream). Subsequent `set_stream` calls update only `hStream`
        // without re-querying, since device identity rarely changes per
        // thread.
        let ctx = populate_ctx_for_device(null_mut())?;
        NPP_CTX.with(|c| c.set(ctx));
        Ok(ctx)
    } else {
        Ok(cur)
    }
}

/// The CUDA stream used by the current thread for NPP operations.
pub fn get_stream() -> cudaStream_t {
    NPP_CTX.with(|c| c.get().hStream)
}

/// Set the stream to use for subsequent NPP calls on this thread.
/// Mirrors the removed `nppSetStream` API but stores the stream in a
/// thread-local instead of NPP's old global state.
pub fn set_stream(stream: cudaStream_t) -> Result<()> {
    let mut ctx = NPP_CTX.with(|c| c.get());
    if ctx.nCudaDeviceId < 0 {
        ctx = populate_ctx_for_device(stream)?;
    } else {
        ctx.hStream = stream;
    }
    NPP_CTX.with(|c| c.set(ctx));
    Ok(())
}

/// An opaque scratch buffer on device needed by some npp routines.
/// Uses stream ordered cuda malloc and free.
#[derive(Debug)]
pub struct ScratchBuffer {
    /// Device ptr !
    pub ptr: *mut c_void,
    pub len: usize,
}

impl ScratchBuffer {
    /// Allocates enough memory to hold at least `len` bytes.
    pub fn alloc_len(len: usize, stream: cudaStream_t) -> Result<Self> {
        let mut ptr = null_mut();
        unsafe { cudaMallocAsync(&mut ptr, len, stream).result_with(Self { ptr, len }) }
    }

    /// Allocates enough memory to hold at least `T`.
    pub fn alloc<T: Sized>(stream: cudaStream_t) -> Result<Self> {
        Self::alloc_len(size_of::<T>(), stream)
    }

    pub fn alloc_from_host<T: Sized>(data: &T, stream: cudaStream_t) -> Result<Self> {
        let mut self_ = Self::alloc::<T>(stream)?;
        self_.copy_from_cpu(data, stream)?;
        Ok(self_)
    }

    /// Size of the allocation on device
    pub fn len(&self) -> usize {
        self.len
    }

    /// The drop impl will free memory on the currently set NPP global stream *at the time of drop*.
    /// I suppose this can be impractical to deal with, so this function here will drop the buffer on an explicit stream.
    pub fn manual_drop(self, stream: cudaStream_t) -> Result<()> {
        unsafe {
            cudaFreeAsync(self.ptr, stream).result()?;
        }
        // Do not free a second time (with the drop impl)
        mem::forget(self);
        Ok(())
    }

    /// Asynchronously copy data from the device buffer to a place in host memory.
    pub fn copy_to_cpu<T: Sized>(&self, out: &mut T, stream: cudaStream_t) -> Result<()> {
        assert_eq!(size_of::<T>(), self.len);
        unsafe {
            cudaMemcpyAsync(
                out as *mut T as *mut c_void,
                self.ptr.cast_const(),
                self.len,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            )
            .result()
        }
    }

    /// Asynchronously copy data from the device buffer to a buffer in host memory.
    pub fn copy_to_cpu_buf<T: Sized>(&self, out: &mut [T], stream: cudaStream_t) -> Result<()> {
        assert_eq!(out.len() * size_of::<T>(), self.len);
        unsafe {
            cudaMemcpyAsync(
                out.as_mut_ptr().cast(),
                self.ptr.cast_const(),
                self.len,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            )
            .result()
        }
    }

    pub fn copy_from_cpu<T>(&mut self, data: &T, stream: cudaStream_t) -> Result<()> {
        assert_eq!(size_of_val(data), self.len);
        unsafe {
            cudaMemcpyAsync(
                self.ptr,
                data as *const T as _,
                self.len,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            )
            .result()
        }
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        // Drop must never panic — a panic during stack unwinding from a
        // sibling alloc failure would abort the whole process. If the
        // context is in a sticky error state, we log and leak rather than
        // escalating the error into a SIGABRT. Same rationale as the
        // `Image` Drop impl.
        let status = unsafe { cudaFreeAsync(self.ptr, get_stream()) };
        if let Err(e) = status.result() {
            eprintln!(
                "[cudarse-npp] ScratchBuffer({} B) drop: cudaFreeAsync failed ({:?}); leaking. \
                 Usually a sticky context error from a prior CUDA failure.",
                self.len, e,
            );
        }
    }
}
