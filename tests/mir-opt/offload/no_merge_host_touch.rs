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

// EMIT_MIR no_merge_host_touch.host_touch.OffloadMovement.after.mir
// The host writes to the mapped payload between the two launches, so the round
// trip cannot be removed: both begin/end pairs must remain. (A host *read* would
// be a dead non-volatile load and would be eliminated before the pass runs, so
// the test uses a write, which survives and is caught by the pass's
// `between_access_ok` check of the second begin's block.)
pub fn host_touch(a: *mut [f64; 4]) {
    std::offload::offload! { kernel = k1, args = (a,) }
    unsafe { (*a)[0] = 5.0; }
    std::offload::offload! { kernel = k2, args = (a,) }
}
