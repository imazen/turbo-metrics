//! Butteraugli CUDA kernels
//!
//! GPU implementation of the Butteraugli perceptual image quality metric.
//! Based on the Vship GPU implementation.

#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(static_mut_refs)]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
#![feature(array_ptr_get)]

mod blur;
mod colors;
mod diffmap;
mod downscale;
mod frequency;
mod malta;
mod masking;
