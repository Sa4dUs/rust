//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

#[offload_kernel]
pub fn k1(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}
#[offload_kernel]
pub fn k2(x: *mut [f64; 4]) {
    unsafe { (*x)[1] = 2.0; }
}

// EMIT_MIR no_merge_diff_args.diff_args.OffloadMovement.after.mir
// Different payloads (`a` vs `b`): the two transfers must stay separate.
pub fn diff_args(a: *mut [f64; 4], b: *mut [f64; 4]) {
    std::offload::offload! { kernel = k1, args = (a,) }
    std::offload::offload! { kernel = k2, args = (b,) }
}