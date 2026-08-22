//! MIR analysis pass computing per-argument data movement for offload kernels.
//!
//! For each argument of an `#[rustc_offload_kernel]` function we determine whether the
//! kernel body reads from and/or writes to the argument's mapped payload. The result is
//! exposed through the `offload_kernel_arg_access` query and consumed by codegen to emit
//! only the OpenMP data transfers a kernel actually needs:
//!
//! - an argument whose payload is never touched needs no `TO`/`FROM` transfer at all.
//! - a `&mut`/`*mut` argument that is only ever read does not need the copy back
//!   (`FROM`), the "immutable use" case mentioned in `gpu_offload.rs`.
//! - a write-only `&mut`/`*mut` argument whose whole payload is provably overwritten on
//!   every path (a whole-`*p` store dominating all exits, or a counted loop writing
//!   every element of the array payload, see `loop_full_write_bits`) needs no copy in
//!   (`TO`), leaving a single copy back.
//! - a by-value argument that is never read does not need the copy in (`TO`).
//!
//! The analysis is deliberately conservative: it only ever *removes* transfers, and
//! whenever it cannot fully track what happens to a pointer (it escapes into an unknown
//! callee, is cast to an integer, multi-level dereference, ...) it assumes both reads
//! and writes may occur, which keeps the mapping identical to the type-based default. A
//! write-only `&mut`/`*mut` argument whose overwrite cannot be proven to cover the whole
//! payload also keeps its copy-in: the kernel may only overwrite part of the payload,
//! and the untouched bytes must retain their host values.
//!
//! The analysis is flow-insensitive: it computes, for every local, the set of kernel
//! arguments whose payload its value may be derived from (by copying pointers or values,
//! reborrowing, or pointer arithmetic), then records every read/write through such a
//! value. Since only "may happen" information is gathered, the result is a sound
//! over-approximation of the actual accesses.
//!
//! Rust semantics make this more precise than the type-based mapping:
//! - `&T`/`*const T` arguments are shared, so writes through them are UB and only reads
//!   can occur, matching the type-based `TO`-only mapping.
//! - `&mut T` arguments give unique access: if the kernel only reads through them no
//!   copy back is needed.
//! - reading a fat pointer's metadata (`len`, `size_of_val`, ...) does not touch the
//!   mapped payload, since that metadata is passed to the kernel as a separate scalar.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, BinOp, Body, CastKind, Const, ConstOperand, Local, LocalDecl,
    NonDivergingIntrinsic, Operand, Place, ProjectionElem, RETURN_PLACE, Rvalue, START_BLOCK,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::offload_meta::ArgAccess;
use rustc_middle::ty::{self, GenericArgsRef, Instance, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::{Spanned, Symbol, sym};

use crate::{PassPolicy, simplify};

/// The largest number of kernel arguments the bitset-based tracking supports. Kernels
/// with more arguments are simply left at the conservative type-based mapping.
const MAX_ARGS: usize = 32;

/// Returns the blocks reachable from the entry block. The pass can run on MIR whose
/// unreachable blocks have not been removed yet (e.g. with `-Zmir-opt-level=0` plus a
/// subset of passes), and the dominator analysis has no results for such blocks, so
/// all dominator-based reasoning must be restricted to the reachable subgraph.
fn reachable_blocks<'tcx>(body: &Body<'tcx>) -> FxHashSet<BasicBlock> {
    let mut reachable = FxHashSet::default();
    let mut stack = vec![START_BLOCK];
    while let Some(bb) = stack.pop() {
        if reachable.insert(bb) {
            stack.extend(body.basic_blocks[bb].terminator().successors());
        }
    }
    reachable
}

/// The query provider for `offload_kernel_arg_access`.
pub(crate) fn offload_kernel_arg_access<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
) -> Option<&'tcx Vec<ArgAccess>> {
    let def_id = instance.def_id();
    if !tcx.is_mir_available(def_id) {
        return None;
    }
    let Ok(body) = instance.try_instantiate_mir_and_normalize_erasing_regions(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        ty::EarlyBinder::bind(tcx, tcx.optimized_mir(def_id).clone()),
    ) else {
        return None;
    };
    Some(tcx.arena.alloc(analyze(tcx, &body)))
}

/// How a place relates to the payloads of the kernel arguments its base derives from.
enum PlaceClass {
    /// The place is `*local.proj` (a single deref of a local): it accesses the pointee,
    /// which is exactly the mapped payload for pointer/reference arguments.
    Pointee,
    /// The place has no deref: it accesses a value held in a local (the value itself for
    /// by-value arguments, or just a pointer value for pointer arguments).
    Value,
    /// A multi-level dereference or otherwise un-analyzable place: assume both reads and
    /// writes may occur through it.
    Escaped,
}

/// Well-known intrinsics whose effect on pointer arguments we model precisely.
enum KnownIntrinsic {
    /// `offset`/`arith_offset`: the result still points into the same payload.
    PointerArithmetic,
    /// `copy`/`copy_nonoverlapping`: read from the source, write to the destination.
    Copy,
    /// `write_bytes`: writes to the destination.
    WriteBytes,
    /// `volatile_load`: reads from the source.
    VolatileLoad,
    /// `volatile_store`: writes to the destination.
    VolatileStore,
}

struct Analysis<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    /// `origins[local]` is a bitset of kernel argument indices (0-based) whose payload
    /// `local`'s value may be derived from.
    origins: IndexVec<Local, u32>,
    /// Reads/writes through a deref of a derived value: accesses the mapped pointee.
    payload_read: u32,
    payload_write: u32,
    /// Reads/writes of values held in locals: accesses by-value argument payloads.
    value_read: u32,
    value_write: u32,
    /// Per basic block: arguments whose payload is written in its entirety (a write to
    /// the whole `*p`, with no field/element projections) in that block.
    full_write: IndexVec<BasicBlock, u32>,
}

impl<'a, 'tcx> Analysis<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>) -> Self {
        let mut origins = IndexVec::from_elem_n(0u32, body.local_decls.len());
        for (i, local) in body.args_iter().enumerate() {
            origins[local] = 1 << i;
        }
        Analysis {
            tcx,
            body,
            origins,
            payload_read: 0,
            payload_write: 0,
            value_read: 0,
            value_write: 0,
            full_write: IndexVec::from_elem_n(0u32, body.basic_blocks.len()),
        }
    }

    /// Records that the payloads of the given arguments may be both read and written.
    fn escape(&mut self, set: u32) {
        self.payload_read |= set;
        self.payload_write |= set;
    }

    /// Propagates `set` into `local`'s origins; returns whether anything changed.
    fn propagate(&mut self, local: Local, set: u32) -> bool {
        let old = self.origins[local];
        let new = old | set;
        self.origins[local] = new;
        new != old
    }

    fn classify_place(&self, place: &Place<'tcx>) -> PlaceClass {
        match place.projection.first() {
            None => PlaceClass::Value,
            Some(ProjectionElem::Deref) => {
                // In post-cleanup MIR a `Deref` can only appear as the first projection,
                // so any further `Deref` is a multi-level dereference.
                let multi = place
                    .projection
                    .iter()
                    .skip(1)
                    .any(|elem| matches!(elem, ProjectionElem::Deref));
                if multi { PlaceClass::Escaped } else { PlaceClass::Pointee }
            }
            Some(_) => PlaceClass::Value,
        }
    }

    fn process_place_read(&mut self, place: &Place<'tcx>) -> bool {
        let set = self.origins[place.local];
        match self.classify_place(place) {
            PlaceClass::Pointee => self.payload_read |= set,
            PlaceClass::Value => self.value_read |= set,
            PlaceClass::Escaped => self.escape(set),
        }
        false
    }

    fn process_place_write(&mut self, bb: BasicBlock, place: &Place<'tcx>) -> bool {
        let set = self.origins[place.local];
        match self.classify_place(place) {
            PlaceClass::Pointee => {
                self.payload_write |= set;
                // A write to the bare `*p` (no further projections) overwrites the whole
                // pointee. If it happens on every path to the kernel's exits, the host
                // values are never needed and the copy-in can be dropped.
                if place.projection.len() == 1 {
                    self.full_write[bb] |= set;
                }
            }
            PlaceClass::Value => self.value_write |= set,
            PlaceClass::Escaped => self.escape(set),
        }
        false
    }

    fn process_operand_read(&mut self, op: &Operand<'tcx>) -> bool {
        match op {
            Operand::Copy(place) | Operand::Move(place) => self.process_place_read(place),
            Operand::Constant(_) | Operand::RuntimeChecks(_) => false,
        }
    }

    fn process_operand_write(&mut self, bb: BasicBlock, op: &Operand<'tcx>) -> bool {
        match op {
            Operand::Copy(place) | Operand::Move(place) => self.process_place_write(bb, place),
            Operand::Constant(_) | Operand::RuntimeChecks(_) => false,
        }
    }

    fn process_statement(&mut self, bb: BasicBlock, stmt: &Statement<'tcx>) -> bool {
        match &stmt.kind {
            StatementKind::Assign((place, rvalue)) => {
                let mut changed = self.process_place_write(bb, place);
                changed |= self.process_rvalue(rvalue, *place);
                changed
            }
            StatementKind::FakeRead((_, place)) => self.process_place_read(place),
            StatementKind::SetDiscriminant { place, .. } => self.process_place_write(bb, place),
            StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(c)) => {
                let mut changed = false;
                changed |= self.process_operand_read(&c.src);
                changed |= self.process_operand_write(bb, &c.dst);
                changed |= self.process_operand_read(&c.count);
                changed
            }
            _ => false,
        }
    }

    fn process_rvalue(&mut self, rvalue: &Rvalue<'tcx>, dest: Place<'tcx>) -> bool {
        match rvalue {
            Rvalue::Use(op, _) | Rvalue::WrapUnsafeBinder(op, _) => {
                let mut changed = self.process_operand_read(op);
                if let Operand::Copy(place) | Operand::Move(place) = op {
                    // Loading a value from a plain local keeps deriving from the same
                    // arguments. Values loaded through a deref are device data and the
                    // pointer they may contain is not mapped, so tracking stops there.
                    if !place.is_indirect() {
                        changed |= self.propagate(dest.local, self.origins[place.local]);
                    }
                }
                changed
            }
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) | Rvalue::Reborrow(_, _, place) => {
                // A (re)borrow of `place` is a pointer into the same payload. Creating the
                // borrow itself does not touch the payload; accesses happen when the new
                // pointer is dereferenced.
                self.propagate(dest.local, self.origins[place.local])
            }
            Rvalue::Cast(kind, op, _) => {
                let mut changed = self.process_operand_read(op);
                let preserves_ptr = matches!(
                    kind,
                    CastKind::PtrToPtr | CastKind::PointerCoercion(..) | CastKind::Transmute
                );
                if preserves_ptr {
                    // Pointer-preserving reinterpretations: the result still points into
                    // the same payload.
                    if let Operand::Copy(place) | Operand::Move(place) = op {
                        changed |= self.propagate(dest.local, self.origins[place.local]);
                    }
                } else if let Operand::Copy(place) | Operand::Move(place) = op {
                    let set = self.origins[place.local];
                    if set != 0 {
                        // E.g. a pointer-to-int cast: the integer may later be turned back
                        // into a pointer, so assume the worst.
                        self.escape(set);
                    }
                }
                changed
            }
            Rvalue::BinaryOp(binop, (a, b)) => {
                let mut changed = self.process_operand_read(a);
                changed |= self.process_operand_read(b);
                if matches!(binop, BinOp::Offset)
                    && let Operand::Copy(place) | Operand::Move(place) = a
                {
                    // `ptr.offset(n)`: the result still points into the same payload.
                    changed |= self.propagate(dest.local, self.origins[place.local]);
                }
                changed
            }
            Rvalue::UnaryOp(_, op) => self.process_operand_read(op),
            Rvalue::Discriminant(place) => self.process_place_read(place),
            Rvalue::Repeat(op, _) => self.process_operand_read(op),
            Rvalue::Aggregate(_, operands) => {
                let mut changed = false;
                for op in operands {
                    changed |= self.process_operand_read(op);
                    if let Operand::Copy(place) | Operand::Move(place) = op {
                        // Values (including pointers) flow into the aggregate; keep
                        // tracking them so that later field projections still derive
                        // from the same arguments.
                        changed |= self.propagate(dest.local, self.origins[place.local]);
                    }
                }
                changed
            }
            Rvalue::CopyForDeref(place) => self.process_place_read(place),
            Rvalue::ThreadLocalRef(_) => false,
        }
    }

    fn known_intrinsic(&self, def_id: DefId) -> Option<KnownIntrinsic> {
        let name = self.tcx.intrinsic(def_id)?.name;
        match name {
            sym::offset | sym::arith_offset => Some(KnownIntrinsic::PointerArithmetic),
            sym::copy | sym::copy_nonoverlapping => Some(KnownIntrinsic::Copy),
            sym::write_bytes => Some(KnownIntrinsic::WriteBytes),
            sym::volatile_load => Some(KnownIntrinsic::VolatileLoad),
            sym::volatile_store => Some(KnownIntrinsic::VolatileStore),
            _ => None,
        }
    }

    fn process_terminator(&mut self, bb: BasicBlock, term: &Terminator<'tcx>) -> bool {
        match &term.kind {
            TerminatorKind::Call { func, args, destination, .. } => {
                let mut changed = self.process_place_write(bb, destination);

                let intrinsic = match func {
                    Operand::Constant(c) => match c.const_.ty().kind() {
                        ty::FnDef(def_id, _) => self.known_intrinsic(*def_id),
                        _ => None,
                    },
                    _ => None,
                };

                match intrinsic {
                    Some(KnownIntrinsic::PointerArithmetic) => {
                        // `offset(p, n)`: the result still points into the same payload.
                        if let Operand::Copy(place) | Operand::Move(place) = &args[0].node {
                            changed |= self.propagate(destination.local, self.origins[place.local]);
                        }
                        changed |= self.process_operand_read(&args[1].node);
                    }
                    Some(KnownIntrinsic::Copy) => {
                        changed |= self.process_operand_read(&args[0].node);
                        changed |= self.process_operand_write(bb, &args[1].node);
                        changed |= self.process_operand_read(&args[2].node);
                    }
                    Some(KnownIntrinsic::WriteBytes) => {
                        changed |= self.process_operand_write(bb, &args[0].node);
                        changed |= self.process_operand_read(&args[1].node);
                        changed |= self.process_operand_read(&args[2].node);
                    }
                    Some(KnownIntrinsic::VolatileLoad) => {
                        changed |= self.process_operand_read(&args[0].node);
                    }
                    Some(KnownIntrinsic::VolatileStore) => {
                        changed |= self.process_operand_write(bb, &args[0].node);
                        changed |= self.process_operand_read(&args[1].node);
                    }
                    None => {
                        // Unknown callee: arguments are read, and any pointer-typed
                        // argument escapes into the callee, which may read and write
                        // through it.
                        for arg in args.iter() {
                            changed |= self.process_operand_read(&arg.node);
                            if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                                let ty = place.ty(self.body, self.tcx).ty;
                                if matches!(ty.kind(), ty::RawPtr(..) | ty::Ref(..)) {
                                    self.escape(self.origins[place.local]);
                                }
                            }
                        }
                    }
                }
                changed
            }
            TerminatorKind::Drop { place, .. } => {
                // Dropping a value reads it (to run its destructor).
                self.process_place_read(place)
            }
            TerminatorKind::SwitchInt { discr, .. } => self.process_operand_read(discr),
            TerminatorKind::Assert { cond, .. } => self.process_operand_read(cond),
            TerminatorKind::Return => {
                // A return value derived from a kernel argument escapes to the caller.
                let set = self.origins[RETURN_PLACE];
                if set != 0 {
                    self.escape(set);
                }
                false
            }
            TerminatorKind::InlineAsm { .. } => {
                // Inline assembly may access arbitrary memory.
                self.escape((1u32 << self.body.arg_count) - 1);
                false
            }
            _ => false,
        }
    }
}

fn analyze<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Vec<ArgAccess> {
    let n = body.arg_count;
    if n == 0 || n > MAX_ARGS {
        // No arguments to optimize, or too many to track with a bitset: keep the
        // conservative type-based mapping (treat every argument as read and written).
        return vec![ArgAccess::READ | ArgAccess::WRITE; n];
    }

    let mut analysis = Analysis::new(tcx, body);

    // Flow-insensitive fixpoint: origins only ever grow, so a bounded number of sweeps
    // over the body suffices.
    let mut changed = true;
    while changed {
        changed = false;
        for (bb, data) in body.basic_blocks.iter_enumerated() {
            for stmt in &data.statements {
                changed |= analysis.process_statement(bb, stmt);
            }
            if let Some(term) = &data.terminator {
                changed |= analysis.process_terminator(bb, term);
            }
        }
    }

    // A whole-payload write only lets us drop the copy-in if it happens on *every* path
    // from the kernel entry to its exits (otherwise a path that skips it would copy
    // garbage back). A block that writes the whole payload and dominates every exit
    // satisfies this.
    let reachable = reachable_blocks(body);
    let exits: Vec<BasicBlock> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|(bb, data)| {
            reachable.contains(bb)
                && matches!(
                    &data.terminator().kind,
                    TerminatorKind::Return
                        | TerminatorKind::Unreachable
                        | TerminatorKind::UnwindResume
                        | TerminatorKind::UnwindTerminate(_)
                )
        })
        .map(|(bb, _)| bb)
        .collect();
    let dominators = body.basic_blocks.dominators();
    let mut fully_overwritten: u32 = if exits.is_empty() {
        0
    } else {
        analysis
            .full_write
            .iter_enumerated()
            .filter(|(w, _)| exits.iter().all(|&e| dominators.dominates(*w, e)))
            .fold(0, |acc, (_, bits)| acc | bits)
    };
    // A counted loop that provably writes every element of an array payload (`for i in
    // 0..N { arr[i] = v }`) also overwrites the whole payload; see `loop_full_write_bits`.
    fully_overwritten |= loop_full_write_bits(tcx, body, &analysis.origins, &exits);

    // Classify each argument based on its type: pointer/reference arguments map their
    // pointee, by-value arguments map the value itself. By-value arguments also count
    // derefs of derived values, since those access the local copy of the value.
    let mut accesses = Vec::with_capacity(n);
    for (i, local) in body.args_iter().enumerate() {
        let bit = 1u32 << i;
        let arg_ty = body.local_decls[local].ty;
        let is_ptr = matches!(arg_ty.kind(), ty::RawPtr(..) | ty::Ref(..));
        let read = if is_ptr {
            analysis.payload_read & bit
        } else {
            (analysis.payload_read | analysis.value_read) & bit
        };
        let write = if is_ptr {
            analysis.payload_write & bit
        } else {
            (analysis.payload_write | analysis.value_write) & bit
        };
        let mut acc = ArgAccess::NONE;
        if read != 0 {
            acc |= ArgAccess::READ;
        }
        if write != 0 {
            acc |= ArgAccess::WRITE;
        }
        // A write-only argument whose whole payload is provably overwritten on every
        // path does not need its host values copied in.
        if write != 0 && read == 0 && (fully_overwritten & bit) != 0 {
            acc |= ArgAccess::FULL_OVERWRITE;
        }
        accesses.push(acc);
    }
    accesses
}

// Loop coverage: whole-payload writes written element by element
//
// A whole-`*p` store is not the only way a write-only buffer can be fully overwritten:
// a counted loop `for i in 0..N { arr[i] = v }` writes every element of the array, so
// the host values are never needed either. This is recognized structurally:
//
// - the loop is a natural loop whose only exit is the header's bound check.
// - it has an induction variable `iv` initialized to 0 before the loop, incremented by
//   1 on every iteration, and bounded by a constant N equal to the array's length
//   (the header switches on `iv < N` or `iv != N`).
// - every iteration stores a whole element at the current index: either directly as
//   `(*base)[Index(iv)] = ...` with `base` an array pointer, or as `*p = ...` where
//   `p = Offset(base, iv)` is defined in the loop.
// - the storing block dominates every latch (so no iteration can bypass the store, a
//   conditional store or `continue` would fail this check) and the loop header
//   dominates every kernel exit (so the loop always runs, a conditional loop would
//   fail this check).
//
// `Rust` semantics do the heavy lifting: the induction variable is recognized from the
// MIR's ownership/def-use structure (a single whole-local definition in the loop and
// a constant 0 outside), and only mutable `&mut [T; N]`/`*mut [T; N]` arguments whose
// array length and element type match the loop's bound and store type are covered.

/// Resolves a local through single whole-definition `Use(Copy/Move(other))` chains to
/// its root local, so plain copies of a value (e.g. of the induction variable) are
/// seen through.
fn resolve_copy_chain<'tcx>(body: &Body<'tcx>, local: Local) -> Local {
    let mut cur = local;
    let mut seen = FxHashSet::default();
    while seen.insert(cur) {
        let Some((_, rvalue)) = single_whole_def(body, cur) else { break };
        if let Rvalue::Use(op, _) = rvalue
            && let Operand::Copy(p) | Operand::Move(p) = op
            && p.projection.is_empty()
        {
            cur = p.local;
        } else {
            break;
        }
    }
    cur
}

/// Evaluates an operand to a `usize` if it is a known constant, either directly or
/// through a copy chain ending in a constant.
fn operand_const<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, op: &Operand<'tcx>) -> Option<u64> {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let c = match op {
        Operand::Constant(c) => c.const_,
        Operand::Copy(p) | Operand::Move(p) => {
            let root = resolve_copy_chain(body, p.local);
            let Some((_, rvalue)) = single_whole_def(body, root) else { return None };
            let Rvalue::Use(op, _) = rvalue else { return None };
            let Operand::Constant(c) = op else { return None };
            c.const_
        }
        _ => return None,
    };
    c.try_eval_target_usize(tcx, typing_env)
}

/// If `rvalue` is an increment of `iv` by a constant (`iv = Add(iv, c)`, possibly
/// unchecked, or the checked `iv = (AddWithOverflow(iv, c)).0` form used at lower opt
/// levels), returns the increment; otherwise `None`.
fn increment_of<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    iv: Local,
    rvalue: &'a Rvalue<'tcx>,
) -> Option<u64> {
    let (a, b) = match rvalue {
        Rvalue::BinaryOp(BinOp::Add | BinOp::AddUnchecked, (a, b)) => (a, b),
        // Checked form: `iv = move (t.0)` where `t = AddWithOverflow(iv, c)`.
        Rvalue::Use(op, _) => {
            let (Operand::Copy(p) | Operand::Move(p)) = op else { return None };
            let [ProjectionElem::Field(f, _)] = p.projection.as_ref() else { return None };
            if f.as_usize() != 0 {
                return None;
            }
            let Some((_, t_rv)) = single_whole_def(body, p.local) else { return None };
            let Rvalue::BinaryOp(BinOp::AddWithOverflow, (a, b)) = t_rv else { return None };
            (a, b)
        }
        _ => return None,
    };
    // One side must be a constant, the other must resolve to `iv` itself.
    let (step, iv_side) = match (operand_const(tcx, body, a), operand_const(tcx, body, b)) {
        (Some(c), None) => (c, b),
        (None, Some(c)) => (c, a),
        _ => return None,
    };
    if let Operand::Copy(p) | Operand::Move(p) = iv_side
        && resolve_copy_chain(body, p.local) == iv
    {
        Some(step)
    } else {
        None
    }
}

/// Recognizes the induction variable of a counted loop: a local with exactly one
/// whole-local definition inside the loop (`iv += c`) and exactly one whole-local
/// definition outside the loop (`iv = 0`) dominating the header, so the loop is always
/// entered with the value 0. Returns `(iv, step)`.
fn loop_induction_variable<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    loop_blocks: &[BasicBlock],
    header: BasicBlock,
) -> Option<(Local, u64)> {
    let doms = body.basic_blocks.dominators();
    // Whole-local definitions inside the loop, in deterministic block order.
    let mut candidates: Vec<(Local, &Rvalue<'tcx>)> = Vec::new();
    for &bb in loop_blocks {
        for stmt in &body.basic_blocks[bb].statements {
            if let StatementKind::Assign((p, rv)) = &stmt.kind
                && p.projection.is_empty()
                && !candidates.iter().any(|(l, _)| *l == p.local)
            {
                candidates.push((p.local, rv));
            }
        }
    }
    for (iv, rvalue) in candidates {
        let Some(step) = increment_of(tcx, body, iv, rvalue) else { continue };
        // Exactly one whole-local definition outside the loop, `iv = 0`, dominating
        // the header (so the loop is always entered with 0).
        let mut entry: Option<(BasicBlock, &Rvalue<'tcx>)> = None;
        let mut bad = false;
        for (bb, data) in body.basic_blocks.iter_enumerated() {
            if loop_blocks.contains(&bb) {
                continue;
            }
            for stmt in &data.statements {
                if let StatementKind::Assign((p, rv)) = &stmt.kind
                    && p.local == iv
                    && p.projection.is_empty()
                {
                    if entry.is_some() {
                        bad = true;
                    }
                    entry = Some((bb, rv));
                }
            }
        }
        if bad {
            continue;
        }
        let Some((bb, rv)) = entry else { continue };
        if !doms.dominates(bb, header) {
            continue;
        }
        if let Rvalue::Use(op, _) = rv
            && let Operand::Constant(c) = op
            && c.const_.try_eval_target_usize(tcx, ty::TypingEnv::fully_monomorphized()) == Some(0)
        {
            return Some((iv, step));
        }
    }
    None
}

/// Returns the trip count of a counted loop, if the loop header's terminator is a
/// `switchInt` on `iv < N` or `iv != N` with a constant N, whose false/equal edge is
/// the loop's only exit and whose true edge stays inside the loop.
fn loop_bound<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    loop_blocks: &[BasicBlock],
    header: BasicBlock,
    iv: Local,
) -> Option<u64> {
    let TerminatorKind::SwitchInt { discr, targets } = &body.basic_blocks[header].terminator().kind
    else {
        return None;
    };
    // The discr must be a local whose single definition is `iv < N` / `iv != N`,
    // recomputed inside the loop (it reads the current value of `iv`).
    let (Operand::Copy(p) | Operand::Move(p)) = discr else { return None };
    let cmp = resolve_copy_chain(body, p.local);
    let Some((cmp_bb, rvalue)) = single_whole_def(body, cmp) else { return None };
    if !loop_blocks.contains(&cmp_bb) {
        return None;
    }
    let Rvalue::BinaryOp(binop, (a, b)) = rvalue else { return None };
    let bound = match binop {
        BinOp::Lt | BinOp::Ne => operand_const(tcx, body, b)?,
        _ => return None,
    };
    // The comparison must be `iv op N` (the constant on the right).
    if let Operand::Copy(pa) | Operand::Move(pa) = a
        && resolve_copy_chain(body, pa.local) == iv
    {
    } else {
        return None;
    }
    // `switchInt(c) -> [0: exit, otherwise: body]`: the false/equal edge (value 0)
    // must leave the loop, and the true edge (the otherwise target) must stay inside.
    let mut exit_target = None;
    for (v, t) in targets.iter() {
        if v == 0 {
            exit_target = Some(t);
        }
    }
    let exit_target = exit_target?;
    if loop_blocks.contains(&exit_target) || !loop_blocks.contains(&targets.otherwise()) {
        return None;
    }
    Some(bound)
}

/// If `local`'s type is a mutable pointer/reference to an array, returns its element
/// type and length constant.
fn array_pointee<'tcx>(body: &Body<'tcx>, local: Local) -> Option<(Ty<'tcx>, ty::Const<'tcx>)> {
    let ty = body.local_decls[local].ty;
    let inner = match ty.kind() {
        ty::RawPtr(inner, mutbl) if *mutbl == ty::Mutability::Mut => inner,
        ty::Ref(_, inner, mutbl) if *mutbl == ty::Mutability::Mut => inner,
        _ => return None,
    };
    let ty::Array(elem, n) = inner.kind() else { return None };
    Some((*elem, *n))
}

/// Whether the argument's type is a mutable pointer/reference to `[elem; len]`.
fn arg_is_mut_array<'tcx>(
    body: &Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    arg_local: Local,
    len: u64,
    elem: Option<Ty<'tcx>>,
) -> bool {
    let ty = body.local_decls[arg_local].ty;
    let (mutbl, inner) = match ty.kind() {
        ty::RawPtr(inner, mutbl) => (*mutbl, inner),
        ty::Ref(_, inner, mutbl) => (*mutbl, inner),
        _ => return false,
    };
    if mutbl != ty::Mutability::Mut {
        return false;
    }
    let ty::Array(elem_ty, n) = inner.kind() else { return false };
    if (*n).try_to_target_usize(tcx) != Some(len) {
        return false;
    }
    if let Some(expected) = elem
        && expected != *elem_ty
    {
        return false;
    }
    true
}

/// Computes the argument bits whose whole payload is provably written by a counted
/// loop `for i in 0..N { arr[i] = v }` (or `*arr.add(i) = v`): entry value 0, step 1,
/// bound N equal to the array's length, no early exits, a store of a whole element on
/// every iteration (dominating every latch), and a loop header that dominates every
/// kernel exit. See the section comment at the top of this block for the recognition
/// rules.
fn loop_full_write_bits<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    origins: &IndexVec<Local, u32>,
    exits: &[BasicBlock],
) -> u32 {
    if exits.is_empty() {
        return 0;
    }
    let doms = body.basic_blocks.dominators();
    let mut covered = 0u32;
    for (header, loop_blocks, latches) in find_loops(body) {
        // The loop must have no early exits: every edge leaving the loop must be the
        // header's bound check, otherwise a `break` could skip iterations (and thus
        // elements).
        let no_early_exits = loop_blocks.iter().all(|&bb| {
            bb == header
                || body.basic_blocks[bb].terminator().successors().all(|s| loop_blocks.contains(&s))
        });
        if !no_early_exits {
            continue;
        }
        let Some((iv, step)) = loop_induction_variable(tcx, body, &loop_blocks, header) else {
            continue;
        };
        if step != 1 {
            continue;
        }
        let Some(bound) = loop_bound(tcx, body, &loop_blocks, header, iv) else { continue };

        // Find the stores that write exactly one element at the current iteration's
        // index, and the argument bits they derive from.
        let mut loop_bits = 0u32;
        for &bb in &loop_blocks {
            for stmt in &body.basic_blocks[bb].statements {
                let StatementKind::Assign((place, _)) = &stmt.kind else { continue };
                let (bits, elem): (u32, Option<Ty<'tcx>>) = match place.projection.as_ref() {
                    // `(*base)[Index(iv)] = ...`: `base` is a mutable array pointer.
                    [ProjectionElem::Deref, ProjectionElem::Index(idx)]
                        if resolve_copy_chain(body, *idx) == iv =>
                    {
                        let Some((e, n)) = array_pointee(body, place.local) else {
                            continue;
                        };
                        if n.try_to_target_usize(tcx) != Some(bound) {
                            continue;
                        }
                        (origins[place.local], Some(e))
                    }
                    // `*p = ...` with `p = Offset(base, iv)` defined in the loop:
                    // `base` points to the array start, `iv` advances one element.
                    [ProjectionElem::Deref] => {
                        // The pointer's own definition: the unique whole-local
                        // assignment (writes *through* the pointer do not redefine
                        // it, so the store statement above is not a definition).
                        let mut def: Option<(BasicBlock, &Rvalue<'tcx>)> = None;
                        let mut ambiguous = false;
                        for (b2, d2) in body.basic_blocks.iter_enumerated() {
                            for s2 in &d2.statements {
                                if let StatementKind::Assign((p2, rv2)) = &s2.kind
                                    && p2.local == place.local
                                    && p2.projection.is_empty()
                                {
                                    if def.is_some() {
                                        ambiguous = true;
                                    }
                                    def = Some((b2, rv2));
                                }
                            }
                            if let TerminatorKind::Call { destination, .. } = &d2.terminator().kind
                                && destination.local == place.local
                            {
                                ambiguous = true;
                            }
                        }
                        if ambiguous {
                            continue;
                        }
                        let Some((def_bb, rvalue)) = def else { continue };
                        if !loop_blocks.contains(&def_bb) {
                            continue;
                        }
                        let Rvalue::BinaryOp(BinOp::Offset, (base, idx)) = rvalue else {
                            continue;
                        };
                        let (Operand::Copy(bp) | Operand::Move(bp)) = base else { continue };
                        let (Operand::Copy(ip) | Operand::Move(ip)) = idx else { continue };
                        if resolve_copy_chain(body, ip.local) != iv {
                            continue;
                        }
                        // The pointee of `base` is the element type written each step.
                        let base_ty = bp.ty(body, tcx).ty;
                        let (ty::RawPtr(inner, _) | ty::Ref(_, inner, _)) = base_ty.kind() else {
                            continue;
                        };
                        (origins[place.local], Some(*inner))
                    }
                    _ => continue,
                };
                if bits == 0 {
                    continue;
                }
                // The store must run on every iteration (it dominates every latch, so
                // no `continue`-style path can skip it), and the loop must be on every
                // path to a kernel exit (the header dominates every exit, so the loop
                // cannot be skipped entirely), leaving no path that observes unwritten
                // elements.
                if !latches.iter().all(|&l| doms.dominates(bb, l))
                    || !exits.iter().all(|&e| doms.dominates(header, e))
                {
                    continue;
                }
                // Only arguments whose payload is a mutable array of exactly `bound`
                // elements of the stored element type can be covered by this loop.
                let mut arg_bits = 0;
                for (i, arg_local) in body.args_iter().enumerate() {
                    let bit = 1u32 << i;
                    if bits & bit != 0 && arg_is_mut_array(body, tcx, arg_local, bound, elem) {
                        arg_bits |= bit;
                    }
                }
                loop_bits |= arg_bits;
            }
        }
        covered |= loop_bits;
    }
    covered
}

// Host-side data-movement optimization
//
// The offload runtime calls (data copy-in/out and the kernel launch) are only emitted
// by codegen at the `offload` intrinsic's call site, so they can neither be moved out
// of loops nor shared between consecutive kernel launches. This pass splits every
// `offload` call into three calls -- `offload_begin` (copy-in), `offload_launch`
// (kernel launch) and `offload_end` (copy-back) -- and then optimizes the resulting
// transfer calls directly in MIR:
//
// - **merge**: two consecutive launches that map the same arguments no longer round
//   trip their data through the host (`begin; launch1; end; begin; launch2; end`
//   becomes `begin; launch1; launch2; end`), provided the host does not touch the
//   mapped payloads in between and both kernels refine to the same per-argument
//   access pattern (so the merged begin/end maptype arrays still cover both kernels).
// - **LICM**: a kernel launched in a loop with loop-invariant arguments only crosses
//   the host/device boundary once: the copy-in is hoisted to the loop preheader and
//   the copy-back is sunk to the loop exits. The OpenMP semantics allow this because
//   touching the mapped variables on the host between begin/end is UB anyway.

/// Which part of an offload region an intrinsic call emits.
#[derive(Copy, Clone, PartialEq, Eq)]
enum OffloadPhase {
    Begin,
    Launch,
    End,
}

/// The canonical origin of a tuple field of an offload call: either a host place (a
/// borrow/raw pointer's pointee or a by-value value) or a constant.
#[derive(Copy, Clone)]
enum FieldOrigin<'tcx> {
    Place(Place<'tcx>),
    Const(ConstOperand<'tcx>),
}

/// The defining value of a plain local, as far as offload tuple resolution cares.
#[derive(Clone)]
enum Def<'tcx> {
    Ref(Place<'tcx>),
    RawPtr(Place<'tcx>),
    CopyChain(Local),
    Aggregate(Vec<Operand<'tcx>>),
}

/// Host-side offload data-movement optimization (see the module docs above).
pub(super) struct OffloadMovement;

impl<'tcx> crate::MirPass<'tcx> for OffloadMovement {
    fn policy(&self, _sess: &Session) -> PassPolicy {
        // Always run: the split is part of the offload lowering (codegen expects the
        // three calls), and the merge/LICM steps only remove redundant runtime calls.
        PassPolicy::optional_non_optimization(true)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        split_offload_calls(tcx, body);
        // Merge and LICM run to a fixpoint: hoisting transfers out of an inner loop
        // can make an outer loop eligible, and a hoisted transfer can become
        // mergeable with an adjacent one.
        loop {
            let merged = merge_offload_regions(tcx, body);
            let licm = licm_offload_regions(tcx, body);
            if !merged && !licm {
                break;
            }
        }
    }
}

fn is_offload_intrinsic<'tcx>(tcx: TyCtxt<'tcx>, func: &Operand<'tcx>, name: Symbol) -> bool {
    match func {
        Operand::Constant(c) => match c.const_.ty().kind() {
            ty::FnDef(def_id, _) => tcx.is_intrinsic(*def_id, name),
            _ => false,
        },
        _ => false,
    }
}

/// Returns the phase of the offload call terminating `bb`, if any.
fn offload_phase<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    bb: BasicBlock,
) -> Option<OffloadPhase> {
    let TerminatorKind::Call { func, .. } = &body.basic_blocks[bb].terminator().kind else {
        return None;
    };
    let Operand::Constant(c) = func else { return None };
    let ty::FnDef(def_id, _) = c.const_.ty().kind() else { return None };
    if tcx.is_intrinsic(*def_id, sym::offload_begin) {
        Some(OffloadPhase::Begin)
    } else if tcx.is_intrinsic(*def_id, sym::offload_launch) {
        Some(OffloadPhase::Launch)
    } else if tcx.is_intrinsic(*def_id, sym::offload_end) {
        Some(OffloadPhase::End)
    } else {
        None
    }
}

/// Returns `(func, successor)` if `bb`'s terminator is a call with a successor.
fn block_call<'a, 'tcx>(
    body: &'a Body<'tcx>,
    bb: BasicBlock,
) -> Option<(&'a Operand<'tcx>, BasicBlock)> {
    match &body.basic_blocks[bb].terminator().kind {
        TerminatorKind::Call { func, target: Some(t), .. } => Some((func, *t)),
        _ => None,
    }
}

/// Resolves the kernel function behind an offload intrinsic call.
fn kernel_instance<'tcx>(tcx: TyCtxt<'tcx>, func: &Operand<'tcx>) -> Option<Instance<'tcx>> {
    let Operand::Constant(c) = func else { return None };
    let ty::FnDef(_, args) = c.const_.ty().kind() else { return None };
    let args = args.skip_binder();
    let ty::FnDef(kernel_def_id, kernel_args) = args.type_at(0).kind() else { return None };
    Instance::try_resolve(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        *kernel_def_id,
        kernel_args.skip_binder(),
    )
    .ok()
    .flatten()
}

/// Finds the `DefId` of an intrinsic declared as a sibling of `offload_def_id` (e.g.
/// `core::intrinsics::offload_begin`) by walking to the enclosing module and
/// enumerating its children.
fn sibling_intrinsic_def_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    offload_def_id: DefId,
    name: Symbol,
) -> Option<DefId> {
    // The parent of `offload` is the `core::intrinsics` module; build its `DefId` from
    // the parent's `DefIndex` in the same crate.
    let parent_index = tcx.def_key(offload_def_id).parent?;
    let module = DefId { krate: offload_def_id.krate, index: parent_index };
    tcx.module_children(module).iter().find_map(|child| {
        if child.ident.name == name
            && let Res::Def(DefKind::Fn, def_id) = child.res
            && tcx.intrinsic(def_id).is_some()
        {
            Some(def_id)
        } else {
            None
        }
    })
}

/// Builds the function operand for an offload phase intrinsic (`offload_begin`,
/// `offload_launch`, `offload_end`): the same function-item constant as the original
/// `offload` call, but with the phase intrinsic's `DefId`. Function items are
/// zero-sized, so the constant value is carried over unchanged.
fn intrinsic_func_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    orig: &ConstOperand<'tcx>,
    offload_def_id: DefId,
    args_binder: ty::Binder<'tcx, GenericArgsRef<'tcx>>,
    name: Symbol,
) -> Option<Operand<'tcx>> {
    let def_id = sibling_intrinsic_def_id(tcx, offload_def_id, name)?;
    let fn_ty = Ty::new_fn_def(tcx, def_id, args_binder.map_bound(|args| args.to_vec()));
    let Const::Val(value, _) = orig.const_ else { return None };
    Some(Operand::Constant(Box::new(ConstOperand {
        span: orig.span,
        user_ty: orig.user_ty,
        const_: Const::Val(value, fn_ty),
    })))
}

/// Splits a single `offload` intrinsic call into `offload_begin` + `offload_launch` +
/// `offload_end`, arranged as a straight-line chain `begin -> launch -> end`. Executing
/// the three calls in order is equivalent to the original call; the original result
/// place is preserved on the end call.
fn split_offload_calls<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let offload_blocks: Vec<BasicBlock> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|(_, data)| {
            matches!(
                &data.terminator().kind,
                TerminatorKind::Call { func, .. } if is_offload_intrinsic(tcx, func, sym::offload)
            )
        })
        .map(|(bb, _)| bb)
        .collect();

    for bb in offload_blocks {
        let term = body.basic_blocks_mut()[bb].terminator.take().expect("invalid terminator");

        // Resolve the three phase intrinsics (siblings of `offload`) while the
        // terminator is still intact; if any cannot be found, keep the call unsplit
        // (codegen still handles a single `offload` call).
        let phase_funcs = (|| {
            let TerminatorKind::Call { func, .. } = &term.kind else { return None };
            let Operand::Constant(fc) = func else { return None };
            let ty::FnDef(offload_def_id, args_binder) = fc.const_.ty().kind() else {
                return None;
            };
            Some((
                intrinsic_func_operand(tcx, fc, *offload_def_id, *args_binder, sym::offload_begin)?,
                intrinsic_func_operand(
                    tcx,
                    fc,
                    *offload_def_id,
                    *args_binder,
                    sym::offload_launch,
                )?,
                intrinsic_func_operand(tcx, fc, *offload_def_id, *args_binder, sym::offload_end)?,
            ))
        })();
        let Some((begin_func, launch_func, end_func)) = phase_funcs else {
            body.basic_blocks_mut()[bb].terminator = Some(term);
            continue;
        };

        // The three phase calls each evaluate the same arguments, so a `Move` operand
        // would transfer ownership three times. Convert them to `Copy` where the type
        // allows; a moved non-`Copy` argument cannot be duplicated, in which case the
        // call is left unsplit (codegen still handles a single `offload` call).
        let args: Option<Vec<Spanned<Operand<'tcx>>>> = match &term.kind {
            TerminatorKind::Call { args, .. } => args
                .iter()
                .cloned()
                .map(|mut arg| {
                    if let Operand::Move(p) = &arg.node {
                        let ty = p.ty(body, tcx).ty;
                        if !tcx
                            .type_is_copy_modulo_regions(ty::TypingEnv::fully_monomorphized(), ty)
                        {
                            return None;
                        }
                        arg.node = Operand::Copy(*p);
                    }
                    Some(arg)
                })
                .collect(),
            _ => unreachable!(),
        };
        let Some(args) = args else {
            body.basic_blocks_mut()[bb].terminator = Some(term);
            continue;
        };

        let TerminatorKind::Call { destination, target, unwind, call_source, fn_span, .. } =
            term.kind
        else {
            unreachable!()
        };
        let target = target.expect("an offload call cannot diverge");

        // Fresh destinations for the begin and launch calls; the original destination
        // (of the unit-typed kernel result) is preserved on the end call.
        let ret_ty = body.local_decls[destination.local].ty;
        let begin_dest: Place<'tcx> = body.local_decls.push(LocalDecl::new(ret_ty, fn_span)).into();
        let launch_dest: Place<'tcx> =
            body.local_decls.push(LocalDecl::new(ret_ty, fn_span)).into();

        let call = |func: Operand<'tcx>, dest: Place<'tcx>, target: BasicBlock| Terminator {
            source_info: term.source_info,
            kind: TerminatorKind::Call {
                func,
                args: args.clone().into_boxed_slice(),
                destination: dest,
                target: Some(target),
                unwind,
                call_source,
                fn_span,
            },
            attributes: term.attributes.clone(),
        };

        // begin -> launch -> end -> original successor
        let end_block = body
            .basic_blocks_mut()
            .push(BasicBlockData::new(Some(call(end_func, destination, target)), false));
        let launch_block = body
            .basic_blocks_mut()
            .push(BasicBlockData::new(Some(call(launch_func, launch_dest, end_block)), false));
        // The original block now holds the begin call.
        body.basic_blocks_mut()[bb].terminator = Some(call(begin_func, begin_dest, launch_block));
    }
}

/// Returns every block that belongs to any natural loop.
fn loop_blocks_of<'tcx>(body: &Body<'tcx>) -> Vec<BasicBlock> {
    let mut all = Vec::new();
    for (_, blocks, _) in find_loops(body) {
        for b in blocks {
            if !all.contains(&b) {
                all.push(b);
            }
        }
    }
    all
}

/// Returns the natural loops of the CFG as `(header, body_blocks, latches)`, deduplicated
/// by header.
fn find_loops<'tcx>(body: &Body<'tcx>) -> Vec<(BasicBlock, Vec<BasicBlock>, Vec<BasicBlock>)> {
    let doms = body.basic_blocks.dominators();
    let preds = body.basic_blocks.predecessors();
    let reachable = reachable_blocks(body);

    // header -> latches, from back edges (latch -> header where header dominates latch).
    // Unreachable latches have no dominator information, so they cannot form a back edge.
    let mut latches_by_header: FxHashMap<BasicBlock, Vec<BasicBlock>> = FxHashMap::default();
    for (latch, data) in body.basic_blocks.iter_enumerated() {
        let Some(term) = &data.terminator else { continue };
        if !reachable.contains(&latch) {
            continue;
        }
        for succ in term.successors() {
            if succ != latch && doms.dominates(succ, latch) {
                latches_by_header.entry(succ).or_default().push(latch);
            }
        }
    }

    let mut loops = Vec::new();
    // Iterate in a deterministic block order (the map is only used for deduplication).
    let headers: Vec<BasicBlock> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|(bb, _)| latches_by_header.contains_key(bb))
        .map(|(bb, _)| bb)
        .collect();
    for header in headers {
        let latches = &latches_by_header[&header];
        // Natural loop body: the header plus every block that can reach a latch without
        // passing through the header.
        let mut body_blocks = Vec::new();
        let mut stack: Vec<BasicBlock> = latches.clone();
        while let Some(b) = stack.pop() {
            if b == header || body_blocks.contains(&b) {
                continue;
            }
            body_blocks.push(b);
            for &p in &preds[b] {
                if p != header {
                    stack.push(p);
                }
            }
        }
        body_blocks.push(header);
        loops.push((header, body_blocks, latches.clone()));
    }
    loops
}

/// Builds a map from plain locals to their defining value, plus the set of locals that
/// are assigned more than once (ambiguous, so never resolved).
fn build_defs<'tcx>(body: &Body<'tcx>) -> (FxHashMap<Local, Def<'tcx>>, FxHashSet<Local>) {
    let mut defs: FxHashMap<Local, Def<'tcx>> = FxHashMap::default();
    let mut multi: FxHashSet<Local> = FxHashSet::default();
    for data in body.basic_blocks.iter() {
        for stmt in &data.statements {
            if let StatementKind::Assign((place, rvalue)) = &stmt.kind
                && place.projection.is_empty()
            {
                let def = match rvalue {
                    Rvalue::Ref(_, _, p) => Def::Ref(*p),
                    Rvalue::RawPtr(_, p) => Def::RawPtr(*p),
                    Rvalue::Use(op, _) => match op {
                        Operand::Copy(p) | Operand::Move(p) if p.projection.is_empty() => {
                            Def::CopyChain(p.local)
                        }
                        _ => continue,
                    },
                    Rvalue::Aggregate(_, ops) => Def::Aggregate(ops.iter().cloned().collect()),
                    _ => continue,
                };
                if defs.insert(place.local, def).is_some() {
                    multi.insert(place.local);
                }
            }
        }
    }
    (defs, multi)
}

/// The result of resolving a place through copies/borrows.
#[derive(Clone)]
enum Resolved<'tcx> {
    /// The place is a tuple aggregate (an offload argument tuple).
    Aggregate(Vec<Operand<'tcx>>),
    /// The place is a plain value or borrow.
    Plain(Place<'tcx>),
}

fn resolve_place<'tcx>(
    body: &Body<'tcx>,
    defs: &FxHashMap<Local, Def<'tcx>>,
    multi: &FxHashSet<Local>,
    place: Place<'tcx>,
    seen: &mut FxHashSet<Local>,
) -> Option<Resolved<'tcx>> {
    if !place.projection.is_empty() || multi.contains(&place.local) {
        return Some(Resolved::Plain(place));
    }
    if !seen.insert(place.local) {
        return None; // copy cycle
    }
    match defs.get(&place.local) {
        Some(Def::Aggregate(ops)) => Some(Resolved::Aggregate(ops.clone())),
        Some(Def::CopyChain(other)) => resolve_place(body, defs, multi, Place::from(*other), seen),
        Some(Def::Ref(p)) | Some(Def::RawPtr(p)) => Some(Resolved::Plain(*p)),
        None => Some(Resolved::Plain(place)),
    }
}

/// Resolves an offload call's tuple operand to the canonical origins of its fields,
/// following copies and borrows back to the underlying host places. Returns `None` for
/// tuples that cannot be fully resolved (nested aggregates, copy cycles, ...).
fn resolve_tuple<'tcx>(body: &Body<'tcx>, op: &Operand<'tcx>) -> Option<Vec<FieldOrigin<'tcx>>> {
    let (defs, multi) = build_defs(body);
    let place = match op {
        Operand::Copy(p) | Operand::Move(p) => *p,
        _ => return None,
    };
    let mut seen = FxHashSet::default();
    match resolve_place(body, &defs, &multi, place, &mut seen)? {
        Resolved::Aggregate(ops) => ops
            .iter()
            .map(|op| match op {
                Operand::Constant(c) => Some(FieldOrigin::Const(**c)),
                Operand::Copy(p) | Operand::Move(p) => {
                    match resolve_place(body, &defs, &multi, *p, &mut seen)? {
                        Resolved::Plain(p) => Some(FieldOrigin::Place(p)),
                        Resolved::Aggregate(_) => None, // nested tuple: unsupported
                    }
                }
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn fields_equal<'tcx>(a: &[FieldOrigin<'tcx>], b: &[FieldOrigin<'tcx>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (FieldOrigin::Place(p1), FieldOrigin::Place(p2)) => p1 == p2,
            (FieldOrigin::Const(c1), FieldOrigin::Const(c2)) => c1.const_ == c2.const_,
            _ => false,
        })
}

/// Returns the host locals whose mapped payload must not be touched between two
/// kernels for their transfers to be mergeable: the fields that correspond to
/// pointer/reference kernel arguments.
fn payload_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    inst: Instance<'tcx>,
    fields: &[FieldOrigin<'tcx>],
) -> Option<Vec<Local>> {
    let sig = tcx.fn_sig(inst.def_id()).instantiate(tcx, inst.args).skip_norm_wip();
    let sig = tcx.instantiate_bound_regions_with_erased(sig);
    let inputs = sig.inputs();
    if inputs.len() != fields.len() {
        return None;
    }
    let mut locals = Vec::new();
    for (field, ty) in fields.iter().zip(inputs) {
        if matches!(ty.kind(), ty::RawPtr(..) | ty::Ref(..)) {
            match field {
                FieldOrigin::Place(p) => locals.push(p.local),
                FieldOrigin::Const(_) => return None,
            }
        }
    }
    Some(locals)
}

fn place_has_deref(place: &Place<'_>) -> bool {
    place.projection.iter().any(|e| matches!(e, ProjectionElem::Deref))
}

/// Whether accessing `place` could touch a mapped payload: writes rooted at a mapped
/// local, or any deref (which may alias a mapped pointer through another local).
fn place_touches_payload(place: &Place<'_>, payload_locals: &[Local]) -> bool {
    payload_locals.contains(&place.local) || place_has_deref(place)
}

fn operand_touches_payload(op: &Operand<'_>, payload_locals: &[Local]) -> bool {
    match op {
        Operand::Copy(p) | Operand::Move(p) => place_touches_payload(p, payload_locals),
        _ => false,
    }
}

fn rvalue_touches_payload(rvalue: &Rvalue<'_>, payload_locals: &[Local]) -> bool {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::WrapUnsafeBinder(op, _)
        | Rvalue::Cast(_, op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Repeat(op, _) => operand_touches_payload(op, payload_locals),
        Rvalue::BinaryOp(_, (a, b)) => {
            operand_touches_payload(a, payload_locals) || operand_touches_payload(b, payload_locals)
        }
        Rvalue::Discriminant(place) | Rvalue::CopyForDeref(place) => {
            place_touches_payload(place, payload_locals)
        }
        Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) | Rvalue::Reborrow(_, _, place) => {
            // Creating a (re)borrow does not read the pointee; only an aliased deref
            // through it would matter.
            place_has_deref(place)
        }
        Rvalue::Aggregate(_, ops) => {
            ops.iter().any(|op| operand_touches_payload(op, payload_locals))
        }
        Rvalue::ThreadLocalRef(_) => false,
    }
}

/// Checks that the statements between two consecutive offload calls (the statements of
/// the block whose terminator is the second call's begin) do not touch the mapped
/// payloads: the host must neither observe nor modify them, otherwise removing the
/// end/begin round trip would change semantics.
fn between_access_ok<'tcx>(
    body: &Body<'tcx>,
    begin2: BasicBlock,
    payload_locals: &[Local],
) -> bool {
    for stmt in &body.basic_blocks[begin2].statements {
        let touches = match &stmt.kind {
            StatementKind::Assign((place, rvalue)) => {
                place_touches_payload(place, payload_locals)
                    || rvalue_touches_payload(rvalue, payload_locals)
            }
            StatementKind::FakeRead((_, place)) => place_has_deref(place),
            StatementKind::SetDiscriminant { place, .. } => {
                place_touches_payload(place, payload_locals)
            }
            StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(c)) => {
                operand_touches_payload(&c.src, payload_locals)
                    || operand_touches_payload(&c.dst, payload_locals)
                    || operand_touches_payload(&c.count, payload_locals)
            }
            _ => false,
        };
        if touches {
            return false;
        }
    }
    true
}

/// Whether an argument's copy-in (`TO`) bit is set in the kernel's begin maptype
/// array. Pointer/reference arguments copy in when read, or when written without a
/// proven full overwrite (the untouched bytes must keep their host values); by-value
/// arguments copy in when read.
fn begin_needs(acc: ArgAccess, is_ptr: bool) -> bool {
    if is_ptr {
        acc.contains(ArgAccess::READ)
            || (acc.contains(ArgAccess::WRITE) && !acc.contains(ArgAccess::FULL_OVERWRITE))
    } else {
        acc.contains(ArgAccess::READ)
    }
}

/// Whether an argument's copy-back (`FROM`) bit is set in the kernel's end maptype
/// array: pointer/reference arguments that are written.
fn end_needs(acc: ArgAccess, is_ptr: bool) -> bool {
    is_ptr && acc.contains(ArgAccess::WRITE)
}

/// Whether two kernels' refined accesses allow merging their transfers, given that the
/// merged region uses kernel 1's begin array for the copy-in and kernel 2's end array
/// for the copy-back. Kernel 1's own copy-in and kernel 2's own copy-back are covered
/// by construction; the merge is sound iff kernel 2's copy-in needs are covered by
/// kernel 1's begin array and kernel 1's copy-back needs are covered by kernel 2's end
/// array.
fn accesses_mergeable(acc1: &[ArgAccess], acc2: &[ArgAccess], types: &[Ty<'_>]) -> bool {
    acc1.len() == acc2.len()
        && acc1.len() == types.len()
        && acc1.iter().zip(acc2).zip(types).all(|((a1, a2), ty)| {
            let is_ptr = matches!(ty.kind(), ty::RawPtr(..) | ty::Ref(..));
            // kernel 2's copy-in must be covered by kernel 1's begin array
            (!begin_needs(*a2, is_ptr) || begin_needs(*a1, is_ptr))
                // kernel 1's copy-back must be covered by kernel 2's end array
                && (!end_needs(*a1, is_ptr) || end_needs(*a2, is_ptr))
        })
}

/// Merges the data transfers of two consecutive kernel launches that map the same
/// arguments: `begin(A); launch1; end(A); begin(A); launch2; end(A)` becomes
/// `begin(A); launch1; launch2; end(A)`, removing a redundant round trip of the mapped
/// data through the host. Returns whether anything changed.
fn merge_offload_regions<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> bool {
    let all_loop_blocks = loop_blocks_of(body);

    let begins: Vec<BasicBlock> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|(_, data)| {
            matches!(
                &data.terminator().kind,
                TerminatorKind::Call { func, .. }
                    if is_offload_intrinsic(tcx, func, sym::offload_begin)
            )
        })
        .map(|(bb, _)| bb)
        .collect();

    let mut changed = false;
    for begin1 in begins {
        // Walk begin1 -> launch1 -> end1 -> begin2 -> launch2 -> end2.
        let Some((_, launch1)) = block_call(body, begin1) else { continue };
        if offload_phase(tcx, body, launch1) != Some(OffloadPhase::Launch) {
            continue;
        }
        let Some((_, end1)) = block_call(body, launch1) else { continue };
        if offload_phase(tcx, body, end1) != Some(OffloadPhase::End) {
            continue;
        }
        let Some((_, begin2)) = block_call(body, end1) else { continue };
        if offload_phase(tcx, body, begin2) != Some(OffloadPhase::Begin) {
            continue;
        }
        let Some((_, launch2)) = block_call(body, begin2) else { continue };
        if offload_phase(tcx, body, launch2) != Some(OffloadPhase::Launch) {
            continue;
        }
        let Some((_, end2)) = block_call(body, launch2) else { continue };
        if offload_phase(tcx, body, end2) != Some(OffloadPhase::End) {
            continue;
        }

        // The whole chain must be straight-line code outside any loop (LICM handles
        // the loop case).
        let chain = [begin1, launch1, end1, begin2, launch2, end2];
        if chain.iter().any(|bb| all_loop_blocks.contains(bb)) {
            continue;
        }

        // Both calls must map the same arguments. The launch call's arguments are
        // `(workgroup_dim, thread_dim, dyn_cache, device_id, args_tuple)` (the kernel is
        // the callee), so the mapped argument tuple is the last one.
        let TerminatorKind::Call { args: args1, .. } =
            &body.basic_blocks[launch1].terminator().kind
        else {
            unreachable!()
        };
        let TerminatorKind::Call { args: args2, .. } =
            &body.basic_blocks[launch2].terminator().kind
        else {
            unreachable!()
        };
        if args1.len() != args2.len() || args1.is_empty() {
            continue;
        }
        let tuple_idx = args1.len() - 1;
        let (Some(fields1), Some(fields2)) = (
            resolve_tuple(body, &args1[tuple_idx].node),
            resolve_tuple(body, &args2[tuple_idx].node),
        ) else {
            continue;
        };
        if !fields_equal(&fields1, &fields2) {
            continue;
        }

        // The merged region keeps kernel 1's begin maptype array (copy-in) and
        // kernel 2's end maptype array (copy-back), so kernel 2's copy-in needs must
        // be covered by kernel 1's begin array and kernel 1's copy-back needs by
        // kernel 2's end array (see `accesses_mergeable`).
        let (func1, _) = block_call(body, launch1).unwrap();
        let (func2, _) = block_call(body, launch2).unwrap();
        let (Some(inst1), Some(inst2)) = (kernel_instance(tcx, func1), kernel_instance(tcx, func2))
        else {
            continue;
        };
        let (Some(acc1), Some(acc2)) =
            (tcx.offload_kernel_arg_access(inst1), tcx.offload_kernel_arg_access(inst2))
        else {
            continue;
        };
        let sig = tcx.fn_sig(inst1.def_id()).instantiate(tcx, inst1.args).skip_norm_wip();
        let sig = tcx.instantiate_bound_regions_with_erased(sig);
        if !accesses_mergeable(acc1, acc2, sig.inputs()) {
            continue;
        }

        // Nothing in between may touch the mapped payloads.
        let Some(payload_locals) = payload_locals(tcx, inst1, &fields1) else { continue };
        if !between_access_ok(body, begin2, &payload_locals) {
            continue;
        }

        // Merge: launch1 flows straight into launch2; the redundant end/begin pair
        // becomes unreachable.
        body.basic_blocks_mut()[launch1].terminator_mut().successors_mut(|s| *s = launch2);
        body.basic_blocks_mut()[end1].terminator_mut().kind = TerminatorKind::Unreachable;
        body.basic_blocks_mut()[begin2].terminator_mut().kind = TerminatorKind::Unreachable;
        changed = true;
    }

    if changed {
        simplify::remove_dead_blocks(body);
    }
    changed
}

/// Whether `local` is written (or has its storage invalidated) anywhere in the loop.
fn local_written_in_loop<'tcx>(
    body: &Body<'tcx>,
    loop_blocks: &[BasicBlock],
    local: Local,
) -> bool {
    for &bb in loop_blocks {
        let data = &body.basic_blocks[bb];
        for stmt in &data.statements {
            match &stmt.kind {
                StatementKind::Assign(pair) => {
                    if pair.0.local == local {
                        return true;
                    }
                }
                StatementKind::SetDiscriminant { place, .. } => {
                    if place.local == local {
                        return true;
                    }
                }
                StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(c)) => {
                    if let Operand::Copy(p) | Operand::Move(p) = &c.dst
                        && p.local == local
                    {
                        return true;
                    }
                }
                StatementKind::StorageDead(p) | StatementKind::StorageLive(p) => {
                    if *p == local {
                        return true;
                    }
                }
                _ => {}
            }
        }
        if let TerminatorKind::Call { destination, .. } = &data.terminator().kind
            && destination.local == local
        {
            return true;
        }
    }
    false
}

/// Where `local` is defined, for the purpose of proving a hoisted read is well-defined.
#[derive(Copy, Clone, PartialEq, Eq)]
enum DefSite {
    /// Never assigned (e.g. a function argument).
    None,
    /// Assigned exactly once, in this block.
    Single(BasicBlock),
    /// Assigned more than once (or by a terminator we do not model): cannot prove.
    Multiple,
}

fn def_site<'tcx>(body: &Body<'tcx>, local: Local) -> DefSite {
    let mut site = DefSite::None;
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        let assigned = data
            .statements
            .iter()
            .any(|s| matches!(&s.kind, StatementKind::Assign((p, _)) if p.local == local))
            || matches!(
                &data.terminator().kind,
                TerminatorKind::Call { destination, .. } if destination.local == local
            );
        if assigned {
            match site {
                DefSite::None => site = DefSite::Single(bb),
                _ => return DefSite::Multiple,
            }
        }
    }
    site
}

/// Returns the unique whole-local assignment defining `local`, or `None` when it is
/// assigned more than once, only partially (via projections), or by a terminator
/// (function-call result, which is never loop-invariant).
fn single_whole_def<'a, 'tcx>(
    body: &'a Body<'tcx>,
    local: Local,
) -> Option<(BasicBlock, &'a Rvalue<'tcx>)> {
    let mut found: Option<(BasicBlock, &'a Rvalue<'tcx>)> = None;
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        for stmt in &data.statements {
            if let StatementKind::Assign((p, rvalue)) = &stmt.kind
                && p.local == local
            {
                if !p.projection.is_empty() || found.is_some() {
                    return None;
                }
                found = Some((bb, rvalue));
            }
        }
        if let TerminatorKind::Call { destination, .. } = &data.terminator().kind
            && destination.local == local
        {
            return None;
        }
    }
    found
}

fn statement_for_def<'tcx>(body: &Body<'tcx>, local: Local) -> Option<Statement<'tcx>> {
    for data in body.basic_blocks.iter() {
        for stmt in &data.statements {
            if let StatementKind::Assign((p, _)) = &stmt.kind
                && p.local == local
            {
                return Some(stmt.clone());
            }
        }
    }
    None
}

/// Whether the value of `op` is loop-invariant, and if so collects the in-loop
/// whole-local definitions it depends on into `out` in dependency order (so replaying
/// them in the preheader makes the hoisted call well-defined). `done` prevents
/// hoisting the same definition twice; `seen` breaks definition cycles.
fn operand_hoistable<'tcx>(
    body: &Body<'tcx>,
    loop_blocks: &[BasicBlock],
    preheader: BasicBlock,
    op: &Operand<'tcx>,
    out: &mut Vec<Statement<'tcx>>,
    done: &mut FxHashSet<Local>,
    seen: &mut FxHashSet<Local>,
) -> bool {
    let (Operand::Copy(p) | Operand::Move(p)) = op else { return true };
    if !seen.insert(p.local) {
        return false; // cyclic definition: cannot prove invariance
    }
    let result = if !local_written_in_loop(body, loop_blocks, p.local) {
        // Defined outside the loop (or never): its value is fixed before the loop
        // runs, provided its definition dominates the preheader.
        let doms = body.basic_blocks.dominators();
        match def_site(body, p.local) {
            DefSite::None => true,
            DefSite::Single(bb) => doms.dominates(bb, preheader),
            DefSite::Multiple => false,
        }
    } else {
        // Assigned inside the loop: invariant only if the assignment is a single
        // whole-local definition of an invariant rvalue.
        let Some((bb, rvalue)) = single_whole_def(body, p.local) else {
            return false;
        };
        if !loop_blocks.contains(&bb)
            || !rvalue_hoistable(body, loop_blocks, preheader, rvalue, out, done, seen)
        {
            return false;
        }
        if done.insert(p.local) {
            out.push(statement_for_def(body, p.local).expect("single_whole_def found it"));
        }
        true
    };
    seen.remove(&p.local);
    result
}

/// Like `operand_hoistable`, for a place: the base local must be invariant (field
/// projections of an invariant value are invariant too).
fn place_hoistable<'tcx>(
    body: &Body<'tcx>,
    loop_blocks: &[BasicBlock],
    preheader: BasicBlock,
    place: &Place<'tcx>,
    out: &mut Vec<Statement<'tcx>>,
    done: &mut FxHashSet<Local>,
    seen: &mut FxHashSet<Local>,
) -> bool {
    operand_hoistable(body, loop_blocks, preheader, &Operand::Copy(*place), out, done, seen)
}

/// Whether the rvalue is loop-invariant (and collects the definitions it depends on,
/// see `operand_hoistable`). Anything that reads memory or is otherwise exotic is
/// conservatively not invariant.
fn rvalue_hoistable<'tcx>(
    body: &Body<'tcx>,
    loop_blocks: &[BasicBlock],
    preheader: BasicBlock,
    rvalue: &Rvalue<'tcx>,
    out: &mut Vec<Statement<'tcx>>,
    done: &mut FxHashSet<Local>,
    seen: &mut FxHashSet<Local>,
) -> bool {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::WrapUnsafeBinder(op, _)
        | Rvalue::Cast(_, op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Repeat(op, _) => {
            operand_hoistable(body, loop_blocks, preheader, op, out, done, seen)
        }
        Rvalue::Ref(_, _, p) | Rvalue::RawPtr(_, p) | Rvalue::Reborrow(_, _, p) => {
            // A borrow is invariant if the borrowed place is (its value is fixed
            // before the loop); the borrow itself does not read the pointee.
            place_hoistable(body, loop_blocks, preheader, p, out, done, seen)
        }
        Rvalue::BinaryOp(_, (a, b)) => {
            operand_hoistable(body, loop_blocks, preheader, a, out, done, seen)
                && operand_hoistable(body, loop_blocks, preheader, b, out, done, seen)
        }
        Rvalue::Aggregate(_, ops) => ops
            .iter()
            .all(|op| operand_hoistable(body, loop_blocks, preheader, op, out, done, seen)),
        Rvalue::Discriminant(_) | Rvalue::CopyForDeref(_) | Rvalue::ThreadLocalRef(_) => false,
    }
}

/// Returns the single non-loop predecessor of the loop header (the preheader), or
/// creates one when the header has several. Returns `None` if the loop is the whole
/// function (nothing to hoist into).
fn find_preheader<'tcx>(
    body: &mut Body<'tcx>,
    loop_blocks: &[BasicBlock],
    header: BasicBlock,
) -> Option<BasicBlock> {
    let preds = body.basic_blocks.predecessors();
    let non_loop: Vec<BasicBlock> =
        preds[header].iter().copied().filter(|p| !loop_blocks.contains(p)).collect();
    if non_loop.is_empty() {
        return None;
    }
    if non_loop.len() == 1 {
        return Some(non_loop[0]);
    }
    let source_info = SourceInfo::outermost(body.span);
    let ph = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator {
            source_info,
            kind: TerminatorKind::Goto { target: header },
            attributes: ThinVec::new(),
        }),
        false,
    ));
    for p in &non_loop {
        body.basic_blocks_mut()[*p].terminator_mut().successors_mut(|s| {
            if *s == header {
                *s = ph;
            }
        });
    }
    Some(ph)
}

/// Loop-invariant code motion for offload data transfers: a loop whose body contains a
/// single `begin -> launch -> end` chain that runs unconditionally every iteration with
/// loop-invariant arguments gets its copy-in hoisted to the preheader and its copy-back
/// sunk to the loop exits, so the data crosses the host/device boundary once instead of
/// once per iteration. Returns whether anything changed.
fn licm_offload_regions<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> bool {
    let loops = find_loops(body);
    let mut changed = false;

    for (header, loop_blocks, latches) in loops {
        // Find the single begin/launch/end chain in the loop.
        let mut begin_bb = None;
        let mut launch_bb = None;
        let mut end_bb = None;
        let mut ok = true;
        for &bb in &loop_blocks {
            match offload_phase(tcx, body, bb) {
                Some(OffloadPhase::Begin) => {
                    if begin_bb.replace(bb).is_some() {
                        ok = false;
                    }
                }
                Some(OffloadPhase::Launch) => {
                    if launch_bb.replace(bb).is_some() {
                        ok = false;
                    }
                }
                Some(OffloadPhase::End) => {
                    if end_bb.replace(bb).is_some() {
                        ok = false;
                    }
                }
                None => {}
            }
        }
        let (Some(begin_bb), Some(launch_bb), Some(end_bb)) = (begin_bb, launch_bb, end_bb) else {
            continue;
        };
        if !ok {
            continue;
        }

        // The chain must be begin -> launch -> end, and none of its blocks may be the
        // loop header or a latch (moving those would break the loop structure).
        let Some((_, chain_launch)) = block_call(body, begin_bb) else { continue };
        if chain_launch != launch_bb {
            continue;
        }
        let Some((_, chain_end)) = block_call(body, launch_bb) else { continue };
        if chain_end != end_bb {
            continue;
        }
        if begin_bb == header
            || launch_bb == header
            || end_bb == header
            || latches.contains(&begin_bb)
            || latches.contains(&launch_bb)
            || latches.contains(&end_bb)
        {
            continue;
        }

        // The begin must dominate every latch so that the whole chain runs on every
        // iteration (only then can the transfers be moved out).
        let doms = body.basic_blocks.dominators();
        if !latches.iter().all(|&l| doms.dominates(begin_bb, l)) {
            continue;
        }

        // The loop must have at least one exit to sink the copy-back into.
        let mut exit_edges: Vec<(BasicBlock, BasicBlock)> = Vec::new();
        for &bb in &loop_blocks {
            let Some(term) = &body.basic_blocks[bb].terminator else { continue };
            for succ in term.successors() {
                if !loop_blocks.contains(&succ) {
                    exit_edges.push((bb, succ));
                }
            }
        }
        if exit_edges.is_empty() {
            continue;
        }

        // All arguments of the begin/end calls must be loop-invariant (transitively:
        // in-loop definitions of invariant rvalues are hoisted along with the call).
        let args = match &body.basic_blocks[begin_bb].terminator().kind {
            TerminatorKind::Call { args, .. } => args.clone(),
            _ => unreachable!(),
        };
        let Some(preheader) = find_preheader(body, &loop_blocks, header) else { continue };
        let mut hoisted: Vec<Statement<'tcx>> = Vec::new();
        let mut hoisted_done = FxHashSet::default();
        let mut hoisted_seen = FxHashSet::default();
        if !args.iter().all(|arg| {
            operand_hoistable(
                body,
                &loop_blocks,
                preheader,
                &arg.node,
                &mut hoisted,
                &mut hoisted_done,
                &mut hoisted_seen,
            )
        }) {
            continue;
        }

        // The hoisted locals are now defined in the preheader, but their `StorageLive`/
        // `StorageDead` markers stay in the loop (the `StorageLive` in the block that
        // becomes unreachable, the `StorageDead` on the latch path), so the calls that
        // remain in the loop would use dead storage and the hoisted definitions would
        // use storage that was never made live. Unreachable blocks can also hold
        // markers of unrelated locals (e.g. a loop-condition temp whose `StorageDead`
        // sat in the begin block); those markers vanish with the block and unbalance
        // that local's storage state everywhere else. Drop the markers of every
        // affected local from the whole body: storage markers are just an optimization
        // hint, and a local without any marker is treated as live throughout the
        // function.
        let mut drop_markers = hoisted_done;
        for bb in [begin_bb, end_bb] {
            for stmt in &body.basic_blocks[bb].statements {
                if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = &stmt.kind
                {
                    drop_markers.insert(*l);
                }
            }
        }
        for bb in body.basic_blocks.indices() {
            body.basic_blocks_mut()[bb].statements.retain(|stmt| {
                !matches!(
                    &stmt.kind,
                    StatementKind::StorageLive(l) | StatementKind::StorageDead(l)
                        if drop_markers.contains(l)
                )
            });
        }

        // The predecessor sets below must be snapshotted before mutating the CFG.
        let preds = body.basic_blocks.predecessors().clone();
        let end_target = match &body.basic_blocks[end_bb].terminator().kind {
            TerminatorKind::Call { target: Some(t), .. } => *t,
            _ => unreachable!(),
        };

        // -- hoist the begin call into a new block between preheader and header --
        let mut begin_term = body.basic_blocks[begin_bb].terminator().clone();
        if let TerminatorKind::Call { target, .. } = &mut begin_term.kind {
            *target = Some(header);
        }
        let hoisted_begin = body.basic_blocks_mut().push(BasicBlockData::new_stmts(
            hoisted,
            Some(begin_term),
            false,
        ));
        body.basic_blocks_mut()[preheader].terminator_mut().successors_mut(|s| {
            if *s == header {
                *s = hoisted_begin;
            }
        });
        for p in &preds[begin_bb] {
            body.basic_blocks_mut()[*p].terminator_mut().successors_mut(|s| {
                if *s == begin_bb {
                    *s = launch_bb;
                }
            });
        }
        body.basic_blocks_mut()[begin_bb].terminator_mut().kind = TerminatorKind::Unreachable;

        // -- sink the end call onto every exit edge --
        body.basic_blocks_mut()[launch_bb].terminator_mut().successors_mut(|s| {
            if *s == end_bb {
                *s = end_target;
            }
        });
        for (exit_bb, exit_target) in &exit_edges {
            let mut end_term = body.basic_blocks[end_bb].terminator().clone();
            if let TerminatorKind::Call { target, .. } = &mut end_term.kind {
                *target = Some(*exit_target);
            }
            let sunk_end = body.basic_blocks_mut().push(BasicBlockData::new(Some(end_term), false));
            body.basic_blocks_mut()[*exit_bb].terminator_mut().successors_mut(|s| {
                if *s == *exit_target {
                    *s = sunk_end;
                }
            });
        }
        body.basic_blocks_mut()[end_bb].terminator_mut().kind = TerminatorKind::Unreachable;
        changed = true;
    }

    if changed {
        simplify::remove_dead_blocks(body);
    }
    changed
}
