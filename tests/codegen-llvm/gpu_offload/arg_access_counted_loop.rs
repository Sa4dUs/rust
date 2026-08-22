//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=0 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// Verifies that the `offload_kernel_arg_access` analysis recognizes a counted loop
// (`for i in 0..N { arr[i] = v }`, here written with an explicit `usize` induction
// variable) as overwriting the whole array payload, dropping the copy-in (`TO`).
// The loop-indexed store shape is what the analysis is structurally written for.

#![feature(gpu_offload)]
#![feature(rustc_attrs)]
#![no_main]

// CHECK: @.offload_maptypes.[[K:[^ ]*k_loop]].begin = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK: @.offload_maptypes.[[K]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_loop(x: *mut [f64; 4]) {
    unsafe {
        let mut i = 0usize;
        while i < 4 {
            (*x)[i] = 1.0;
            i += 1;
        }
    }
}

#[unsafe(no_mangle)]
fn main() {
    let mut x = [0.0f64; 4];
    core::offload::offload! {
        kernel = k_loop,
        args = (&mut x as *mut [f64; 4],),
    }
}