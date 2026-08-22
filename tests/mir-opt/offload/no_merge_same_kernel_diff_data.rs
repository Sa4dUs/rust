//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck

#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

#[offload_kernel]
pub fn k(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}

// The same kernel launched twice, but over two *different* payloads. Merging the
// transfers would keep only the first begin's copy-in and the second end's
// copy-back, leaving `b`'s data never copied in and `a`'s never copied back. The
// merge is blocked on the differing argument tuples and both begin/end pairs
// must remain.
//
// CHECK-LABEL: fn same_kernel_diff_data(
// CHECK: offload_begin::{{.*}}k
// CHECK: offload_launch::{{.*}}k
// CHECK: offload_end::{{.*}}k
// CHECK: offload_begin::{{.*}}k
// CHECK: offload_launch::{{.*}}k
// CHECK: offload_end::{{.*}}k
pub fn same_kernel_diff_data(a: *mut [f64; 4], b: *mut [f64; 4]) {
    std::offload::offload! { kernel = k, args = (a,) }
    std::offload::offload! { kernel = k, args = (b,) }
}
