//@ needs-offload
//@ compile-flags: -Copt-level=0 -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

#[offload_kernel]
pub fn k(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}

// EMIT_MIR licm_opt0.looped.OffloadMovement.after.mir
// The begin/end calls must be hoisted out of the loop: begin to the preheader, end to
// the exit, launch stays inside. Same scenario as `licm_offload.rs`, pinned to
// `-Copt-level=0`, where the loop keeps the checked `AddWithOverflow` induction form.
pub fn looped(a: *mut [f64; 4], n: u32) {
    let mut i = 0;
    while i < n {
        std::offload::offload! {
            kernel = k,
            args = (a,),
        }
        i += 1;
    }
}