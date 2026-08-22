//@ compile-flags: -Zoffload=Test -Zunstable-options -C opt-level=1 -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// The `offload_kernel_arg_access` analysis tracks argument access with a bitset
// limited to `MAX_ARGS = 32` arguments; a kernel with more arguments is left at the
// conservative type-based mapping (every `*mut` argument keeps both `TO` and `FROM`).
//
// The kernel below writes its first argument in full, which the analysis would
// otherwise recognize as a whole-payload overwrite (dropping the copy-in); with 33
// arguments the analysis does not run, so `x0` keeps `[i64 1]` (`TO`) instead of
// becoming `zeroinitializer`.

#![feature(gpu_offload)]
#![feature(rustc_attrs)]
#![no_main]

// CHECK: @.offload_maptypes.[[K:[^ ]*k_many]].begin = private unnamed_addr constant [33 x i64] [i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1]
// CHECK: @.offload_maptypes.[[K]].end = private unnamed_addr constant [33 x i64] [i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn k_many(
    x0: *mut f64,
    x1: *mut f64,
    x2: *mut f64,
    x3: *mut f64,
    x4: *mut f64,
    x5: *mut f64,
    x6: *mut f64,
    x7: *mut f64,
    x8: *mut f64,
    x9: *mut f64,
    x10: *mut f64,
    x11: *mut f64,
    x12: *mut f64,
    x13: *mut f64,
    x14: *mut f64,
    x15: *mut f64,
    x16: *mut f64,
    x17: *mut f64,
    x18: *mut f64,
    x19: *mut f64,
    x20: *mut f64,
    x21: *mut f64,
    x22: *mut f64,
    x23: *mut f64,
    x24: *mut f64,
    x25: *mut f64,
    x26: *mut f64,
    x27: *mut f64,
    x28: *mut f64,
    x29: *mut f64,
    x30: *mut f64,
    x31: *mut f64,
    x32: *mut f64,
) {
    unsafe {
        *x0 = 1.0;
        let _ = x32;
    }
}

#[unsafe(no_mangle)]
fn main() {
    let mut a = 0.0f64;
    core::offload::offload! {
        kernel = k_many,
        args = (
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
            &mut a as *mut f64, &mut a as *mut f64, &mut a as *mut f64,
        ),
    }
}
