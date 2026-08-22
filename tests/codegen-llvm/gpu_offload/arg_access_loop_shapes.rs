//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=0 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// The `offload_kernel_arg_access` module documentation claims the whole-payload
// full-overwrite detection covers two more loop shapes beyond the explicit
// `(*x)[i] = v` while-loop form (see `arg_access_counted_loop.rs`):
//   - a `*p = v` loop where `p` is recomputed each iteration as
//     `p = Offset(base, iv)`; and
//   - a `for i in 0..N { arr[i] = v }` range loop.
//
// The first shape is recognized: the store writes one element at the induction
// variable's index and the array argument is overwritten in full, so the copy-in
// (`TO`) is dropped and only the copy-back (`FROM`) remains.
//
// The second shape is *not* recognized: at the MIR level a `for` loop over a range
// is lowered to the iterator protocol (`Range::next` calls, `Option` discriminant
// switches) rather than a `switchInt` on an induction variable, which the structural
// recognizer requires. The kernel therefore keeps the conservative type-based
// mapping (`begin = [i64 1]`, `end = [i64 2]`).

#![feature(core_intrinsics, gpu_offload)]
#![feature(rustc_attrs)]
#![no_main]

// A `*p = v` loop with `p = Offset(base, iv)`: the whole array payload is
// provably overwritten, so no copy-in is needed.
// CHECK: @.offload_maptypes.[[P:[^ ]*k_ptr]].begin = private unnamed_addr constant [1 x i64] zeroinitializer
// CHECK: @.offload_maptypes.[[P]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_ptr(x: *mut [f64; 4]) {
    unsafe {
        let mut i = 0usize;
        while i < 4 {
            let p = core::intrinsics::offset(x as *mut f64, i);
            *p = 1.0;
            i += 1;
        }
    }
}

// A range loop lowers to the iterator protocol and is not recognized as a counted
// loop: the copy-in is kept (the mapping stays at the conservative type-based
// `TO | FROM`).
// CHECK: @.offload_maptypes.[[F:[^ ]*k_for]].begin = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @.offload_maptypes.[[F]].end = private unnamed_addr constant [1 x i64] [i64 2]
#[inline(never)]
pub fn k_for(x: *mut [f64; 4]) {
    unsafe { for i in 0..4usize { (*x)[i] = 1.0; } }
}

#[unsafe(no_mangle)]
fn main() {
    let mut x = [0.0f64; 4];
    core::offload::offload! {
        kernel = k_ptr,
        args = (&mut x as *mut [f64; 4],),
    }
    core::offload::offload! {
        kernel = k_for,
        args = (&mut x as *mut [f64; 4],),
    }
}
