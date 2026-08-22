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

// EMIT_MIR no_licm_no_exit.noexit.OffloadMovement.after.mir
// An infinite loop has no exit to sink the copy-back into, so LICM must leave the
// whole begin -> launch -> end chain inside the loop.
pub fn noexit(a: *mut [f64; 4]) -> ! {
    loop {
        std::offload::offload! { kernel = k, args = (a,) }
    }
}
