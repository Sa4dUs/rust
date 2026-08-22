//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// Verifies that the `offload_kernel_arg_access` analysis refines the mapping for
// shared-reference arguments (`&T` / `*const T`) and for whole-store `&mut T` writes:
// - a shared reference that is only read keeps the copy-in (`TO`) but needs no
//   copy-back (`FROM`).
// - a `&mut` that is overwritten in full (a whole-`*p` store on every path) needs no
//   copy-in.
//
// `i64 1` = TO, `i64 2` = FROM, `i64 0` = nothing.

#![feature(gpu_offload)]
#![feature(rustc_attrs)]
#![no_main]

// Shared reference, only read: copy-in stays, copy-back is dropped.
// CHECK: @.offload_maptypes.[[S:[^ ]*k_shared]].begin = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @.offload_maptypes.[[S]].end = private unnamed_addr constant [1 x i64] zeroinitializer
#[inline(never)]
pub fn k_shared(x: &f64) {
    // A plain `let _ = *x;` is a dead non-volatile load and is eliminated before the
    // analysis runs; `black_box` keeps the read observable.
    let _ = core::hint::black_box(*x);
}

// `*const`, only read: same as the shared reference above.
// CHECK: @.offload_maptypes.[[P:[^ ]*k_const]].begin = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @.offload_maptypes.[[P]].end = private unnamed_addr constant [1 x i64] zeroinitializer
#[inline(never)]
pub fn k_const(x: *const f64) {
    let _ = core::hint::black_box(unsafe { *x });
}

// A `&mut` overwritten in full needs no copy-in, only the copy-back.
// CHECK: @.offload_maptypes.[[W:[^ ]*k_mut_write]].begin = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK: @.offload_maptypes.[[W]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_mut_write(x: &mut f64) {
    *x = 1.0;
}

#[unsafe(no_mangle)]
fn main() {
    let mut a = 0.0f64;
    let mut b = 0.0f64;
    let mut c = 0.0f64;
    core::offload::offload! {
        kernel = k_shared,
        args = (&a,),
    }
    core::offload::offload! {
        kernel = k_const,
        args = (&b as *const f64,),
    }
    core::offload::offload! {
        kernel = k_mut_write,
        args = (&mut c,),
    }
}
