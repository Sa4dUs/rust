//@ needs-offload
//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat
//@ skip-filecheck

#![feature(gpu_offload)]
#![allow(internal_features)]

// Two plain kernels (not `#[offload_kernel]`, which would stub the bodies out) so
// that the `offload_kernel_arg_access` analysis sees their real access patterns:
// `k1` only writes, `k2` only reads.
#[inline(never)]
fn k1(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}
#[inline(never)]
fn k2(x: *mut [f64; 4]) {
    unsafe { let _ = core::hint::black_box((*x)[0]); }
}

// Both launches map the same payload, but the kernels' access patterns are
// incompatible: the merged region would use kernel 1's begin array for the
// copy-in, which does not cover kernel 2's read (kernel 1 never reads, so its
// begin drops the `TO` transfer), and kernel 1's own copy-back is not covered
// by kernel 2's end array (kernel 2 never writes). The merge must be blocked
// and both begin/end pairs must remain.
//
// CHECK-LABEL: fn access_mismatch(
// CHECK: offload_begin::{{.*}}k1
// CHECK: offload_launch::{{.*}}k1
// CHECK: offload_end::{{.*}}k1
// CHECK: offload_begin::{{.*}}k2
// CHECK: offload_launch::{{.*}}k2
// CHECK: offload_end::{{.*}}k2
pub fn access_mismatch(a: *mut [f64; 4]) {
    std::offload::offload! { kernel = k1, args = (a,) }
    std::offload::offload! { kernel = k2, args = (a,) }
}
