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

// EMIT_MIR no_licm_multi_chain.multi.OffloadMovement.after.mir
// A loop containing two offload chains (two begins/launches/ends) cannot be
// hoisted as a unit: LICM requires a single begin -> launch -> end chain, so
// both chains must stay inside the loop.
pub fn multi(a: *mut [f64; 4], b: *mut [f64; 4], n: u32) {
    let mut i = 0;
    while i < n {
        std::offload::offload! { kernel = k1, args = (a,) }
        std::offload::offload! { kernel = k2, args = (b,) }
        i += 1;
    }
}
