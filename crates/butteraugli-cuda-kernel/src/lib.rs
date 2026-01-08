//! Butteraugli CUDA kernels
//!
//! GPU implementation of the Butteraugli perceptual image quality metric.
//! Based on the Vship GPU implementation.

#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
#![feature(array_ptr_get)]

mod colors;
mod blur;
mod downscale;
mod malta;
mod frequency;
mod masking;
mod diffmap;
