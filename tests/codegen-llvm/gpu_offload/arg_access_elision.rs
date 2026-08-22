//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// Verifies that the `offload_kernel_arg_access` MIR analysis refines the per-argument
// mapping types in the `.offload_maptypes` globals:
// - a `*mut` argument that is only read keeps the copy-in (`TO`) but drops the
//   copy-back (`FROM`).
// - a `*mut` argument that is only written in full (whole-`*p` store on every path)
//   drops the copy-in entirely (`FULL_OVERWRITE`).
// - an untouched argument drops both transfers.
// - a partially written array payload keeps both transfers (the untouched elements
//   must retain their host values).
//
// The type-based mapping for `*mut f64` is `TO | FROM` (1 | 2), and for `*mut [f64; 4]`
// it is also `TO | FROM`; the begin maptype uses only `TO`-ish bits, the end maptype
// only `FROM`. `i64 1` = TO, `i64 2` = FROM, `i64 0` = nothing.

#![feature(gpu_offload)]
#![feature(rustc_attrs)]
#![no_main]

// Read-only `*mut f64` (first arg) and write-only `*mut f64` (second arg).
// CHECK: @.offload_maptypes.[[K:[^ ]*k_copy]].begin = private unnamed_addr constant [2 x i64] [i64 1, i64 0]
// CHECK: @.offload_maptypes.[[K]].end = private unnamed_addr constant [2 x i64] [i64 0, i64 2]
#[inline(never)]
pub fn k_copy(x: *mut f64, y: *mut f64) {
    unsafe { *y = *x; }
}

// Write-only whole-`*p` store: no copy-in needed.
// CHECK: @.offload_maptypes.[[W:[^ ]*k_write]].begin = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK: @.offload_maptypes.[[W]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_write(x: *mut f64) {
    unsafe { *x = 1.0; }
}

// Argument never touched: no transfers at all.
// CHECK: @.offload_maptypes.[[U:[^ ]*k_unused]].begin = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK: @.offload_maptypes.[[U]].end = private unnamed_addr constant [1 x i64] zeroinitializer
#[inline(never)]
pub fn k_unused(x: *mut f64) {
    let _ = x;
}

// Partial element write: both transfers must remain (copy-in keeps untouched bytes).
// CHECK: @.offload_maptypes.[[E:[^ ]*k_elem]].begin = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @.offload_maptypes.[[E]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_elem(x: *mut [f64; 4]) {
    unsafe { (*x)[0] = 1.0; }
}

#[unsafe(no_mangle)]
fn main() {
    let mut a = 0.0f64;
    let mut b = [0.0f64; 4];
    core::offload::offload! {
        kernel = k_copy,
        args = (&mut a as *mut f64, &mut a as *mut f64),
    }
    core::offload::offload! {
        kernel = k_write,
        args = (&mut a as *mut f64,),
    }
    core::offload::offload! {
        kernel = k_unused,
        args = (&mut a as *mut f64,),
    }
    core::offload::offload! {
        kernel = k_elem,
        args = (&mut b as *mut [f64; 4],),
    }
}