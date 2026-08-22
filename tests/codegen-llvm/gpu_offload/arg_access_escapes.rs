//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// The `offload_kernel_arg_access` analysis is deliberately conservative: whenever it
// cannot fully track what happens to a pointer it assumes both reads and writes may
// occur, which keeps the mapping identical to the type-based default (`TO | FROM`,
// `i64 1 | i64 2`). These cases must therefore keep both the copy-in and the
// copy-back:
// - the pointer is cast to an integer (it could be turned back into a pointer).
// - the pointer escapes into an unknown callee (which may read and write through it).

#![feature(gpu_offload)]
#![feature(rustc_attrs)]
#![no_main]

// A pointer-to-int cast loses track of the pointer: both transfers must remain.
// CHECK: @.offload_maptypes.[[C:[^ ]*k_cast]].begin = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @.offload_maptypes.[[C]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_cast(x: *mut f64) {
    let a = x as usize;
    let _ = a;
}

#[inline(never)]
fn helper(x: *mut f64) {}

// The pointer escapes into an opaque callee: both transfers must remain.
// CHECK: @.offload_maptypes.[[L:[^ ]*k_call]].begin = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @.offload_maptypes.[[L]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_call(x: *mut f64) {
    helper(x);
}

#[unsafe(no_mangle)]
fn main() {
    let mut a = 0.0f64;
    let mut b = 0.0f64;
    core::offload::offload! {
        kernel = k_cast,
        args = (&mut a as *mut f64,),
    }
    core::offload::offload! {
        kernel = k_call,
        args = (&mut b as *mut f64,),
    }
}
