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

// EMIT_MIR no_licm_conditional.cond.OffloadMovement.after.mir
// The offload runs only on the even iterations, so its begin does not dominate
// every latch: hoisting the transfers out of the loop would copy data in/out on
// iterations that skip the kernel entirely. The chain must stay inside the loop.
pub fn cond(a: *mut [f64; 4], n: u32) {
    let mut i = 0;
    while i < n {
        if i % 2 == 0 {
            std::offload::offload! { kernel = k, args = (a,) }
        }
        i += 1;
    }
}
