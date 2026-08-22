//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
#![feature(gpu_offload)]
#![allow(internal_features)]

use std::offload::offload_kernel;

// Two kernels with the same access pattern over the same payload.
#[offload_kernel]
pub fn k1(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}
#[offload_kernel]
pub fn k2(x: *mut [f64; 4]) {
    unsafe { (*x)[1] = 2.0; }
}

// Two consecutive launches that map the same arguments must merge their data
// transfers: `begin(k1); launch(k1); launch(k2); end(k2)` with no round trip
// through the host in between. The args tuple and dims are hoisted to locals
// *before* the calls so nothing touches the payload between the two launches.
//
// CHECK-LABEL: fn two_named(
// CHECK: offload_begin::{{.*}}k1
// CHECK: offload_launch::{{.*}}k1
// CHECK-NOT: offload_end
// CHECK-NOT: offload_begin
// CHECK: offload_launch::{{.*}}k2
// CHECK: offload_end::{{.*}}k2
// CHECK-NOT: offload_begin
pub fn two_named(a: *mut [f64; 4]) {
    let args = (a,);
    let w = [1u32, 1, 1];
    let t = [1u32, 1, 1];
    std::offload::offload! {
        kernel = k1,
        workgroup_dim = w,
        thread_dim = t,
        args = args,
    }
    std::offload::offload! {
        kernel = k2,
        workgroup_dim = w,
        thread_dim = t,
        args = args,
    }
}