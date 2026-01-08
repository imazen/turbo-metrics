#![no_std]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use core::ffi::CStr;

mod helpers;
/// Replacements for some math functions that are only present in std using libdevice.
pub mod math;

#[cfg(all(feature = "panic-handler", not(feature = "minimal-panic")))]
mod panic;

// Minimal panic handler that just traps - avoids pulling in core::fmt (saves ~3000 lines of PTX)
#[cfg(feature = "minimal-panic")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

pub mod prelude {
    pub use crate::helpers::*;
    pub use crate::math::*;
}

/// You can use the standard C printf template arguments.
/// `T` must be a `#[repr(C)]` struct.
#[inline]
pub unsafe fn print<T>(fmt: &CStr, params: &T) {
    core::arch::nvptx::vprintf(fmt.as_ptr().cast(), params as *const T as _);
}

#[macro_export]
macro_rules! printf {
    ($fmt:literal, $($ty:ty),*; $($p:expr),*) => {
        {
            #[repr(C)]
            struct __Fmt($($ty),*);
            $crate::print($fmt, &__Fmt($($p),*));
        }
    };
}
