//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

// A minimal kernel. In the host pass the `offload_kernel` attribute macro lowers the
// device body away, so the actual kernel body here is deliberately trivial.
#[offload_kernel]
pub fn k(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}

// EMIT_MIR split_offload.host1.OffloadMovement.after.mir
// The host-side `offload!` call must be split into begin/launch/end.
pub fn host1(a: *mut [f64; 4]) {
    std::offload::offload! {
        kernel = k,
        args = (a,),
    }
}