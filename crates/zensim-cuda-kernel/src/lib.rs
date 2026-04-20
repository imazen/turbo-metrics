//! zensim CUDA kernels (device code).
//!
//! These PTX kernels target numerical equivalence (within ~ULP of FMA
//! contraction differences) with the CPU zensim implementation in
//! `../../../../zen/zensim/zensim/src`. Feature-extraction math and
//! constants match the CPU scalar paths — the host wrapper is
//! responsible for running them in the right order per scale.

#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(static_mut_refs)]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

mod blur;
mod color;
mod downscale;
mod features;
mod pad;
