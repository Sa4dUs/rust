//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

#[offload_kernel]
pub fn k(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}

// EMIT_MIR no_licm_varying.varying.OffloadMovement.after.mir
// The offloaded pointer depends on the loop counter, so the begin/launch/end
// chain must stay inside the loop (no hoisting).
pub fn varying(a: *mut [f64; 4], n: u32) {
    let mut i = 0u32;
    while i < n {
        let p = unsafe { a.add(i as usize) };
        std::offload::offload! {
            kernel = k,
            args = (p,),
        }
        i += 1;
    }
}