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

// EMIT_MIR licm_nested_loops.nested.OffloadMovement.after.mir
// The offload is in the inner loop. The inner loop can run zero iterations
// (when entered with `j >= m`), in which case the outer loop would copy data
// in/out without ever launching the kernel, so hoisting across the outer loop
// would be unsound. The pass hoists exactly as far as the inner loop allows:
// the begin moves to the inner preheader, the end to the inner exit, and only
// the launch stays inside the inner loop.
pub fn nested(a: *mut [f64; 4], n: u32, m: u32) {
    let mut i = 0;
    while i < n {
        let mut j = 0;
        while j < m {
            std::offload::offload! { kernel = k, args = (a,) }
            j += 1;
        }
        i += 1;
    }
}
