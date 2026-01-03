//! IR Instructions
//!
//! Instruction definitions for the Genesis IR.

use super::types::{BlockId, Constant, IrType, VReg};
use std::fmt;

/// An instruction in the IR
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Result register (None for void instructions)
    pub result: Option<VReg>,
    /// The instruction kind
    pub kind: InstrKind,
}

impl Instruction {
    pub fn new(result: Option<VReg>, kind: InstrKind) -> Self {
        Self { result, kind }
    }
}

/// Kinds of instructions
#[derive(Debug, Clone)]
pub enum InstrKind {
    // ============ Constants ============
    /// Load a constant value
    Const(Constant),

    // ============ Arithmetic ============
    /// Integer addition
    Add(VReg, VReg),
    /// Integer subtraction
    Sub(VReg, VReg),
    /// Integer multiplication
    Mul(VReg, VReg),
    /// Signed integer division
    SDiv(VReg, VReg),
    /// Unsigned integer division
    UDiv(VReg, VReg),
    /// Signed integer remainder
    SRem(VReg, VReg),
    /// Unsigned integer remainder
    URem(VReg, VReg),
    /// Integer negation
    Neg(VReg),

    // ============ Floating Point ============
    /// Float addition
    FAdd(VReg, VReg),
    /// Float subtraction
    FSub(VReg, VReg),
    /// Float multiplication
    FMul(VReg, VReg),
    /// Float division
    FDiv(VReg, VReg),
    /// Float negation
    FNeg(VReg),

    // ============ Bitwise ============
    /// Bitwise AND
    And(VReg, VReg),
    /// Bitwise OR
    Or(VReg, VReg),
    /// Bitwise XOR
    Xor(VReg, VReg),
    /// Shift left
    Shl(VReg, VReg),
    /// Arithmetic shift right (sign-extending)
    AShr(VReg, VReg),
    /// Logical shift right (zero-extending)
    LShr(VReg, VReg),
    /// Bitwise NOT
    Not(VReg),

    // ============ Comparison ============
    /// Integer comparison
    ICmp(CmpOp, VReg, VReg),
    /// Float comparison
    FCmp(CmpOp, VReg, VReg),

    // ============ Conversions ============
    /// Sign extend
    SExt(VReg, IrType),
    /// Zero extend
    ZExt(VReg, IrType),
    /// Truncate
    Trunc(VReg, IrType),
    /// Float to signed int
    FPToSI(VReg, IrType),
    /// Float to unsigned int
    FPToUI(VReg, IrType),
    /// Signed int to float
    SIToFP(VReg, IrType),
    /// Unsigned int to float
    UIToFP(VReg, IrType),
    /// Float extend/truncate
    FPCast(VReg, IrType),
    /// Pointer to int
    PtrToInt(VReg, IrType),
    /// Int to pointer
    IntToPtr(VReg, IrType),
    /// Bitcast (reinterpret bits)
    Bitcast(VReg, IrType),

    // ============ Memory ============
    /// Allocate stack space
    Alloca(IrType),
    /// Allocate heap space (malloc) - returns pointer
    Malloc(IrType),
    /// Allocate array on heap (malloc) - type and count
    MallocArray(IrType, VReg),
    /// Free heap memory
    Free(VReg),
    /// Reallocate heap memory (ptr, new_size_in_bytes)
    Realloc(VReg, VReg),
    /// Allocate heap space by byte count (for dynamic sizes)
    MallocBytes(VReg),
    /// Allocate zero-initialized heap space by byte count (calloc)
    Calloc(VReg),

    // ============ Reference Counting (HARC) ============
    /// Allocate with reference count header (returns ptr to data, not header)
    /// Layout: [refcount: i64 | type_id: i64 | data...]
    RcAlloc {
        /// Type being allocated
        ty: IrType,
        /// Type ID for destructor dispatch
        type_id: u64,
    },
    /// Increment reference count (ptr to data)
    RcRetain(VReg),
    /// Decrement reference count, call destructor if zero (ptr to data)
    RcRelease(VReg),
    /// Get current reference count (for debugging)
    RcGetCount(VReg),
    /// Call type-specific destructor (ptr to data, type_id)
    Drop {
        ptr: VReg,
        type_id: u64,
    },

    /// Memory copy (dst, src, len) - copies len bytes from src to dst
    Memcpy(VReg, VReg, VReg),
    /// Memory set (dst, val, len) - sets len bytes to val (val is i8)
    Memset(VReg, VReg, VReg),
    /// Load from memory
    Load(VReg),
    /// Store to memory (ptr, value)
    Store(VReg, VReg),
    /// Get pointer to struct field (ptr, field_index)
    GetFieldPtr(VReg, u32),
    /// Get pointer to array element (ptr, index)
    GetElementPtr(VReg, VReg),
    /// Get byte pointer (ptr + byte_offset) - for i8 array access
    GetBytePtr(VReg, VReg),
    /// Load single byte from pointer
    LoadByte(VReg),

    // ============ Function Calls ============
    /// Call a function
    Call {
        func: String,
        args: Vec<VReg>,
    },
    /// Call a function pointer
    CallPtr {
        ptr: VReg,
        args: Vec<VReg>,
    },

    // ============ Trait Objects ============
    /// Create a trait object (fat pointer) from data pointer and vtable name
    /// Result is a struct { data_ptr: *void, vtable_ptr: *VTable }
    MakeTraitObject {
        /// Pointer to the concrete data (will be auto-boxed if needed)
        data_ptr: VReg,
        /// Name of the vtable global constant (e.g., "__vtable_Animal_Dog")
        vtable: String,
    },
    /// Extract data pointer from trait object (field 0)
    GetDataPtr(VReg),
    /// Extract vtable pointer from trait object (field 1)
    GetVTablePtr(VReg),
    /// Call method through vtable
    VTableCall {
        /// The trait object (fat pointer)
        trait_obj: VReg,
        /// Method index in vtable (0=drop, 1=size, 2=align, 3+=methods)
        method_idx: u32,
        /// Arguments (data_ptr is automatically prepended as first arg)
        args: Vec<VReg>,
    },

    // ============ Misc ============
    /// Phi node for SSA form
    Phi(Vec<(VReg, BlockId)>),
    /// Select (ternary): condition, true_val, false_val
    Select(VReg, VReg, VReg),
    /// Reference to a global (returns pointer to global)
    GlobalRef(String),
    /// Reference to a function (returns function pointer)
    FuncRef(String),
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Signed less than
    Slt,
    /// Signed less than or equal
    Sle,
    /// Signed greater than
    Sgt,
    /// Signed greater than or equal
    Sge,
    /// Unsigned less than
    Ult,
    /// Unsigned less than or equal
    Ule,
    /// Unsigned greater than
    Ugt,
    /// Unsigned greater than or equal
    Uge,
    /// Unordered (true if either operand is NaN)
    Uno,
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "eq"),
            CmpOp::Ne => write!(f, "ne"),
            CmpOp::Slt => write!(f, "slt"),
            CmpOp::Sle => write!(f, "sle"),
            CmpOp::Sgt => write!(f, "sgt"),
            CmpOp::Sge => write!(f, "sge"),
            CmpOp::Ult => write!(f, "ult"),
            CmpOp::Ule => write!(f, "ule"),
            CmpOp::Ugt => write!(f, "ugt"),
            CmpOp::Uge => write!(f, "uge"),
            CmpOp::Uno => write!(f, "uno"),
        }
    }
}

/// Block terminators
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return from function
    Ret(Option<VReg>),
    /// Unconditional branch
    Br(BlockId),
    /// Conditional branch
    CondBr {
        cond: VReg,
        then_block: BlockId,
        else_block: BlockId,
    },
    /// Switch statement
    Switch {
        value: VReg,
        default: BlockId,
        cases: Vec<(Constant, BlockId)>,
    },
    /// Unreachable (for optimization)
    Unreachable,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(result) = self.result {
            write!(f, "{} = ", result)?;
        }
        match &self.kind {
            InstrKind::Const(c) => write!(f, "const {}", c),
            InstrKind::Add(a, b) => write!(f, "add {}, {}", a, b),
            InstrKind::Sub(a, b) => write!(f, "sub {}, {}", a, b),
            InstrKind::Mul(a, b) => write!(f, "mul {}, {}", a, b),
            InstrKind::SDiv(a, b) => write!(f, "sdiv {}, {}", a, b),
            InstrKind::UDiv(a, b) => write!(f, "udiv {}, {}", a, b),
            InstrKind::SRem(a, b) => write!(f, "srem {}, {}", a, b),
            InstrKind::URem(a, b) => write!(f, "urem {}, {}", a, b),
            InstrKind::Neg(v) => write!(f, "neg {}", v),
            InstrKind::FAdd(a, b) => write!(f, "fadd {}, {}", a, b),
            InstrKind::FSub(a, b) => write!(f, "fsub {}, {}", a, b),
            InstrKind::FMul(a, b) => write!(f, "fmul {}, {}", a, b),
            InstrKind::FDiv(a, b) => write!(f, "fdiv {}, {}", a, b),
            InstrKind::FNeg(v) => write!(f, "fneg {}", v),
            InstrKind::And(a, b) => write!(f, "and {}, {}", a, b),
            InstrKind::Or(a, b) => write!(f, "or {}, {}", a, b),
            InstrKind::Xor(a, b) => write!(f, "xor {}, {}", a, b),
            InstrKind::Shl(a, b) => write!(f, "shl {}, {}", a, b),
            InstrKind::AShr(a, b) => write!(f, "ashr {}, {}", a, b),
            InstrKind::LShr(a, b) => write!(f, "lshr {}, {}", a, b),
            InstrKind::Not(v) => write!(f, "not {}", v),
            InstrKind::ICmp(op, a, b) => write!(f, "icmp {} {}, {}", op, a, b),
            InstrKind::FCmp(op, a, b) => write!(f, "fcmp {} {}, {}", op, a, b),
            InstrKind::SExt(v, ty) => write!(f, "sext {} to {}", v, ty),
            InstrKind::ZExt(v, ty) => write!(f, "zext {} to {}", v, ty),
            InstrKind::Trunc(v, ty) => write!(f, "trunc {} to {}", v, ty),
            InstrKind::FPToSI(v, ty) => write!(f, "fptosi {} to {}", v, ty),
            InstrKind::FPToUI(v, ty) => write!(f, "fptoui {} to {}", v, ty),
            InstrKind::SIToFP(v, ty) => write!(f, "sitofp {} to {}", v, ty),
            InstrKind::UIToFP(v, ty) => write!(f, "uitofp {} to {}", v, ty),
            InstrKind::FPCast(v, ty) => write!(f, "fpcast {} to {}", v, ty),
            InstrKind::PtrToInt(v, ty) => write!(f, "ptrtoint {} to {}", v, ty),
            InstrKind::IntToPtr(v, ty) => write!(f, "inttoptr {} to {}", v, ty),
            InstrKind::Bitcast(v, ty) => write!(f, "bitcast {} to {}", v, ty),
            InstrKind::Alloca(ty) => write!(f, "alloca {}", ty),
            InstrKind::Malloc(ty) => write!(f, "malloc {}", ty),
            InstrKind::MallocArray(ty, count) => write!(f, "malloc_array {}, {}", ty, count),
            InstrKind::Free(ptr) => write!(f, "free {}", ptr),
            InstrKind::Realloc(ptr, size) => write!(f, "realloc {}, {}", ptr, size),
            InstrKind::MallocBytes(size) => write!(f, "malloc_bytes {}", size),
            InstrKind::Calloc(size) => write!(f, "calloc {}", size),
            InstrKind::RcAlloc { ty, type_id } => write!(f, "rc_alloc {} (type_id={})", ty, type_id),
            InstrKind::RcRetain(ptr) => write!(f, "rc_retain {}", ptr),
            InstrKind::RcRelease(ptr) => write!(f, "rc_release {}", ptr),
            InstrKind::RcGetCount(ptr) => write!(f, "rc_getcount {}", ptr),
            InstrKind::Drop { ptr, type_id } => write!(f, "drop {} (type_id={})", ptr, type_id),
            InstrKind::Memcpy(dst, src, len) => write!(f, "memcpy {}, {}, {}", dst, src, len),
            InstrKind::Memset(dst, val, len) => write!(f, "memset {}, {}, {}", dst, val, len),
            InstrKind::Load(ptr) => write!(f, "load {}", ptr),
            InstrKind::Store(ptr, val) => write!(f, "store {}, {}", ptr, val),
            InstrKind::GetFieldPtr(ptr, idx) => write!(f, "getfieldptr {}, {}", ptr, idx),
            InstrKind::GetElementPtr(ptr, idx) => write!(f, "getelementptr {}, {}", ptr, idx),
            InstrKind::GetBytePtr(ptr, offset) => write!(f, "getbyteptr {}, {}", ptr, offset),
            InstrKind::LoadByte(ptr) => write!(f, "loadbyte {}", ptr),
            InstrKind::Call { func, args } => {
                write!(f, "call {}(", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            InstrKind::CallPtr { ptr, args } => {
                write!(f, "callptr {}(", ptr)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            InstrKind::MakeTraitObject { data_ptr, vtable } => {
                write!(f, "make_trait_object {}, @{}", data_ptr, vtable)
            }
            InstrKind::GetDataPtr(v) => write!(f, "get_data_ptr {}", v),
            InstrKind::GetVTablePtr(v) => write!(f, "get_vtable_ptr {}", v),
            InstrKind::VTableCall { trait_obj, method_idx, args } => {
                write!(f, "vtable_call {}[{}](", trait_obj, method_idx)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            InstrKind::Phi(preds) => {
                write!(f, "phi ")?;
                for (i, (val, block)) in preds.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[{}, {}]", val, block)?;
                }
                Ok(())
            }
            InstrKind::Select(cond, t, e) => write!(f, "select {}, {}, {}", cond, t, e),
            InstrKind::GlobalRef(name) => write!(f, "globalref @{}", name),
            InstrKind::FuncRef(name) => write!(f, "funcref @{}", name),
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Ret(None) => write!(f, "ret void"),
            Terminator::Ret(Some(v)) => write!(f, "ret {}", v),
            Terminator::Br(block) => write!(f, "br {}", block),
            Terminator::CondBr { cond, then_block, else_block } => {
                write!(f, "br {}, {}, {}", cond, then_block, else_block)
            }
            Terminator::Switch { value, default, cases } => {
                write!(f, "switch {}, {} [", value, default)?;
                for (val, block) in cases {
                    write!(f, " {}: {}", val, block)?;
                }
                write!(f, " ]")
            }
            Terminator::Unreachable => write!(f, "unreachable"),
        }
    }
}
