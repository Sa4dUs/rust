//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

// A kernel that writes a single element.
#[offload_kernel]
pub fn k(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}

// EMIT_MIR licm_offload.looped.OffloadMovement.after.mir
//
// The begin/end calls must be hoisted out of the loop: begin to the preheader, end to
// the loop exit, and only the launch remains inside. The storage markers of the
// hoisted locals (and of any local whose marker sat in a block that becomes
// unreachable) are dropped so the resulting MIR stays valid.
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