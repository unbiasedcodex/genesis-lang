//! IR Optimization Passes
//!
//! This module implements various optimization passes for the Genesis IR.
//! These are applied before lowering to LLVM IR.

use std::collections::HashMap;
use super::{Module, Function, BasicBlock, Instruction, InstrKind, Constant, VReg, CmpOp};

/// Optimization pass that evaluates constant expressions at compile time.
///
/// This pass:
/// 1. Tracks which VRegs hold constant values
/// 2. When an operation has all constant operands, evaluates it
/// 3. Replaces the operation with a constant load
pub struct ConstantFolder {
    /// Maps VReg to its constant value (if known)
    constants: HashMap<VReg, Constant>,
}

impl ConstantFolder {
    pub fn new() -> Self {
        Self {
            constants: HashMap::new(),
        }
    }

    /// Run constant folding on an entire module
    pub fn fold_module(&mut self, module: &mut Module) {
        for func in &mut module.functions {
            if !func.is_external {
                self.fold_function(func);
            }
        }
    }

    /// Run constant folding on a single function
    pub fn fold_function(&mut self, func: &mut Function) {
        self.constants.clear();

        for block in &mut func.blocks {
            self.fold_block(block);
        }
    }

    /// Run constant folding on a basic block
    fn fold_block(&mut self, block: &mut BasicBlock) {
        for instr in &mut block.instructions {
            self.fold_instruction(instr);
        }
    }

    /// Try to fold a single instruction
    fn fold_instruction(&mut self, instr: &mut Instruction) {
        // First, check if this is a constant load - record it
        if let InstrKind::Const(ref c) = instr.kind {
            if let Some(result) = instr.result {
                self.constants.insert(result, c.clone());
            }
            return;
        }

        // Try to fold the instruction
        let folded = match &instr.kind {
            // Integer arithmetic
            InstrKind::Add(a, b) => self.fold_int_binop(*a, *b, |x, y| x.wrapping_add(y)),
            InstrKind::Sub(a, b) => self.fold_int_binop(*a, *b, |x, y| x.wrapping_sub(y)),
            InstrKind::Mul(a, b) => self.fold_int_binop(*a, *b, |x, y| x.wrapping_mul(y)),
            InstrKind::SDiv(a, b) => self.fold_int_binop_checked(*a, *b, |x, y| if y != 0 { Some(x / y) } else { None }),
            InstrKind::UDiv(a, b) => self.fold_uint_binop_checked(*a, *b, |x, y| if y != 0 { Some(x / y) } else { None }),
            InstrKind::SRem(a, b) => self.fold_int_binop_checked(*a, *b, |x, y| if y != 0 { Some(x % y) } else { None }),
            InstrKind::URem(a, b) => self.fold_uint_binop_checked(*a, *b, |x, y| if y != 0 { Some(x % y) } else { None }),
            InstrKind::Neg(a) => self.fold_int_unop(*a, |x| x.wrapping_neg()),

            // Floating point arithmetic
            InstrKind::FAdd(a, b) => self.fold_float_binop(*a, *b, |x, y| x + y),
            InstrKind::FSub(a, b) => self.fold_float_binop(*a, *b, |x, y| x - y),
            InstrKind::FMul(a, b) => self.fold_float_binop(*a, *b, |x, y| x * y),
            InstrKind::FDiv(a, b) => self.fold_float_binop(*a, *b, |x, y| x / y),
            InstrKind::FNeg(a) => self.fold_float_unop(*a, |x| -x),

            // Bitwise operations
            InstrKind::And(a, b) => self.fold_int_binop(*a, *b, |x, y| x & y),
            InstrKind::Or(a, b) => self.fold_int_binop(*a, *b, |x, y| x | y),
            InstrKind::Xor(a, b) => self.fold_int_binop(*a, *b, |x, y| x ^ y),
            InstrKind::Shl(a, b) => self.fold_int_binop(*a, *b, |x, y| x.wrapping_shl(y as u32)),
            InstrKind::AShr(a, b) => self.fold_int_binop(*a, *b, |x, y| x.wrapping_shr(y as u32)),
            InstrKind::LShr(a, b) => self.fold_uint_binop(*a, *b, |x, y| x.wrapping_shr(y as u32)),
            InstrKind::Not(a) => self.fold_int_unop(*a, |x| !x),

            // Integer comparison
            InstrKind::ICmp(op, a, b) => self.fold_icmp(*op, *a, *b),

            // Float comparison
            InstrKind::FCmp(op, a, b) => self.fold_fcmp(*op, *a, *b),

            // Select (ternary)
            InstrKind::Select(cond, t, f) => self.fold_select(*cond, *t, *f),

            _ => None,
        };

        // If we folded to a constant, update the instruction
        if let Some(constant) = folded {
            if let Some(result) = instr.result {
                self.constants.insert(result, constant.clone());
            }
            instr.kind = InstrKind::Const(constant);
        }
    }

    // Helper functions for folding

    fn get_int(&self, reg: VReg) -> Option<i64> {
        match self.constants.get(&reg)? {
            Constant::Int(v) => Some(*v),
            Constant::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    fn get_uint(&self, reg: VReg) -> Option<u64> {
        match self.constants.get(&reg)? {
            Constant::Int(v) => Some(*v as u64),
            _ => None,
        }
    }

    fn get_float(&self, reg: VReg) -> Option<f64> {
        match self.constants.get(&reg)? {
            Constant::Float(v) => Some(*v),
            Constant::Float32(v) => Some(*v as f64),
            _ => None,
        }
    }

    fn get_bool(&self, reg: VReg) -> Option<bool> {
        match self.constants.get(&reg)? {
            Constant::Bool(v) => Some(*v),
            Constant::Int(v) => Some(*v != 0),
            _ => None,
        }
    }

    fn fold_int_binop<F>(&self, a: VReg, b: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(i64, i64) -> i64,
    {
        let va = self.get_int(a)?;
        let vb = self.get_int(b)?;
        Some(Constant::Int(op(va, vb)))
    }

    fn fold_int_binop_checked<F>(&self, a: VReg, b: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(i64, i64) -> Option<i64>,
    {
        let va = self.get_int(a)?;
        let vb = self.get_int(b)?;
        op(va, vb).map(Constant::Int)
    }

    fn fold_uint_binop<F>(&self, a: VReg, b: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(u64, u64) -> u64,
    {
        let va = self.get_uint(a)?;
        let vb = self.get_uint(b)?;
        Some(Constant::Int(op(va, vb) as i64))
    }

    fn fold_uint_binop_checked<F>(&self, a: VReg, b: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(u64, u64) -> Option<u64>,
    {
        let va = self.get_uint(a)?;
        let vb = self.get_uint(b)?;
        op(va, vb).map(|v| Constant::Int(v as i64))
    }

    fn fold_int_unop<F>(&self, a: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(i64) -> i64,
    {
        let va = self.get_int(a)?;
        Some(Constant::Int(op(va)))
    }

    fn fold_float_binop<F>(&self, a: VReg, b: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(f64, f64) -> f64,
    {
        let va = self.get_float(a)?;
        let vb = self.get_float(b)?;
        Some(Constant::Float(op(va, vb)))
    }

    fn fold_float_unop<F>(&self, a: VReg, op: F) -> Option<Constant>
    where
        F: FnOnce(f64) -> f64,
    {
        let va = self.get_float(a)?;
        Some(Constant::Float(op(va)))
    }

    fn fold_icmp(&self, op: CmpOp, a: VReg, b: VReg) -> Option<Constant> {
        let va = self.get_int(a)?;
        let vb = self.get_int(b)?;
        let result = match op {
            CmpOp::Eq => va == vb,
            CmpOp::Ne => va != vb,
            CmpOp::Slt => va < vb,
            CmpOp::Sle => va <= vb,
            CmpOp::Sgt => va > vb,
            CmpOp::Sge => va >= vb,
            CmpOp::Ult => (va as u64) < (vb as u64),
            CmpOp::Ule => (va as u64) <= (vb as u64),
            CmpOp::Ugt => (va as u64) > (vb as u64),
            CmpOp::Uge => (va as u64) >= (vb as u64),
            CmpOp::Uno => false, // Not a float, no NaN
        };
        Some(Constant::Bool(result))
    }

    fn fold_fcmp(&self, op: CmpOp, a: VReg, b: VReg) -> Option<Constant> {
        let va = self.get_float(a)?;
        let vb = self.get_float(b)?;
        let result = match op {
            CmpOp::Eq => va == vb,
            CmpOp::Ne => va != vb,
            CmpOp::Slt | CmpOp::Ult => va < vb,
            CmpOp::Sle | CmpOp::Ule => va <= vb,
            CmpOp::Sgt | CmpOp::Ugt => va > vb,
            CmpOp::Sge | CmpOp::Uge => va >= vb,
            CmpOp::Uno => va.is_nan() || vb.is_nan(),
        };
        Some(Constant::Bool(result))
    }

    fn fold_select(&self, cond: VReg, t: VReg, f: VReg) -> Option<Constant> {
        let cond_val = self.get_bool(cond)?;
        let result_reg = if cond_val { t } else { f };
        self.constants.get(&result_reg).cloned()
    }
}

impl Default for ConstantFolder {
    fn default() -> Self {
        Self::new()
    }
}

/// Dead code elimination - removes instructions whose results are never used.
pub struct DeadCodeEliminator;

impl DeadCodeEliminator {
    pub fn new() -> Self {
        Self
    }

    /// Run DCE on a function
    pub fn eliminate_function(&self, func: &mut Function) {
        use std::collections::HashSet;

        // First pass: collect all used VRegs
        let mut used: HashSet<VReg> = HashSet::new();

        for block in &func.blocks {
            // Collect uses from instructions
            for instr in &block.instructions {
                self.collect_uses(&instr.kind, &mut used);
            }

            // Collect uses from terminators
            if let Some(ref term) = block.terminator {
                self.collect_terminator_uses(term, &mut used);
            }
        }

        // Second pass: remove dead instructions
        // An instruction is dead if:
        // 1. It has a result that is never used
        // 2. It has no side effects
        for block in &mut func.blocks {
            block.instructions.retain(|instr| {
                if let Some(result) = instr.result {
                    // If result is used, keep the instruction
                    if used.contains(&result) {
                        return true;
                    }
                    // If result is unused, check for side effects
                    !self.is_pure(&instr.kind)
                } else {
                    // No result, probably has side effects - keep it
                    true
                }
            });
        }
    }

    fn collect_uses(&self, kind: &InstrKind, used: &mut std::collections::HashSet<VReg>) {
        match kind {
            InstrKind::Const(_) => {}
            InstrKind::Add(a, b) | InstrKind::Sub(a, b) | InstrKind::Mul(a, b) |
            InstrKind::SDiv(a, b) | InstrKind::UDiv(a, b) | InstrKind::SRem(a, b) |
            InstrKind::URem(a, b) | InstrKind::FAdd(a, b) | InstrKind::FSub(a, b) |
            InstrKind::FMul(a, b) | InstrKind::FDiv(a, b) | InstrKind::And(a, b) |
            InstrKind::Or(a, b) | InstrKind::Xor(a, b) | InstrKind::Shl(a, b) |
            InstrKind::AShr(a, b) | InstrKind::LShr(a, b) |
            InstrKind::ICmp(_, a, b) | InstrKind::FCmp(_, a, b) => {
                used.insert(*a);
                used.insert(*b);
            }
            InstrKind::Neg(a) | InstrKind::FNeg(a) | InstrKind::Not(a) |
            InstrKind::SExt(a, _) | InstrKind::ZExt(a, _) | InstrKind::Trunc(a, _) |
            InstrKind::FPToSI(a, _) | InstrKind::FPToUI(a, _) | InstrKind::SIToFP(a, _) |
            InstrKind::UIToFP(a, _) | InstrKind::FPCast(a, _) | InstrKind::PtrToInt(a, _) |
            InstrKind::IntToPtr(a, _) | InstrKind::Bitcast(a, _) |
            InstrKind::Load(a) | InstrKind::VolatileLoad(a) | InstrKind::LoadByte(a) |
            InstrKind::Free(a) | InstrKind::RcRetain(a) | InstrKind::RcRelease(a) |
            InstrKind::RcGetCount(a) | InstrKind::MallocBytes(a) | InstrKind::Calloc(a) |
            InstrKind::GetDataPtr(a) | InstrKind::GetVTablePtr(a) => {
                used.insert(*a);
            }
            InstrKind::Store(ptr, val) | InstrKind::VolatileStore(ptr, val) |
            InstrKind::Realloc(ptr, val) => {
                used.insert(*ptr);
                used.insert(*val);
            }
            InstrKind::GetFieldPtr(ptr, _) => {
                used.insert(*ptr);
            }
            InstrKind::GetElementPtr(ptr, idx) | InstrKind::GetBytePtr(ptr, idx) => {
                used.insert(*ptr);
                used.insert(*idx);
            }
            InstrKind::MallocArray(_, count) => {
                used.insert(*count);
            }
            InstrKind::Memcpy(a, b, c) | InstrKind::Memset(a, b, c) |
            InstrKind::Select(a, b, c) => {
                used.insert(*a);
                used.insert(*b);
                used.insert(*c);
            }
            InstrKind::Call { args, .. } => {
                for arg in args {
                    used.insert(*arg);
                }
            }
            InstrKind::CallPtr { ptr, args } => {
                used.insert(*ptr);
                for arg in args {
                    used.insert(*arg);
                }
            }
            InstrKind::VTableCall { trait_obj, args, .. } => {
                used.insert(*trait_obj);
                for arg in args {
                    used.insert(*arg);
                }
            }
            InstrKind::MakeTraitObject { data_ptr, .. } => {
                used.insert(*data_ptr);
            }
            InstrKind::Drop { ptr, .. } => {
                used.insert(*ptr);
            }
            InstrKind::Phi(preds) => {
                for (val, _) in preds {
                    used.insert(*val);
                }
            }
            InstrKind::InlineAsm { inputs, .. } => {
                for input in inputs {
                    used.insert(*input);
                }
            }
            InstrKind::Alloca(_) | InstrKind::Malloc(_) | InstrKind::RcAlloc { .. } |
            InstrKind::GlobalRef(_) | InstrKind::FuncRef(_) | InstrKind::SizeOf(_) |
            InstrKind::AlignOf(_) => {}
        }
    }

    fn collect_terminator_uses(&self, term: &super::Terminator, used: &mut std::collections::HashSet<VReg>) {
        match term {
            super::Terminator::Ret(Some(v)) => { used.insert(*v); }
            super::Terminator::CondBr { cond, .. } => { used.insert(*cond); }
            super::Terminator::Switch { value, .. } => { used.insert(*value); }
            _ => {}
        }
    }

    /// Check if an instruction is pure (no side effects)
    fn is_pure(&self, kind: &InstrKind) -> bool {
        match kind {
            // Pure operations
            InstrKind::Const(_) |
            InstrKind::Add(_, _) | InstrKind::Sub(_, _) | InstrKind::Mul(_, _) |
            InstrKind::SDiv(_, _) | InstrKind::UDiv(_, _) | InstrKind::SRem(_, _) |
            InstrKind::URem(_, _) | InstrKind::Neg(_) |
            InstrKind::FAdd(_, _) | InstrKind::FSub(_, _) | InstrKind::FMul(_, _) |
            InstrKind::FDiv(_, _) | InstrKind::FNeg(_) |
            InstrKind::And(_, _) | InstrKind::Or(_, _) | InstrKind::Xor(_, _) |
            InstrKind::Shl(_, _) | InstrKind::AShr(_, _) | InstrKind::LShr(_, _) |
            InstrKind::Not(_) |
            InstrKind::ICmp(_, _, _) | InstrKind::FCmp(_, _, _) |
            InstrKind::SExt(_, _) | InstrKind::ZExt(_, _) | InstrKind::Trunc(_, _) |
            InstrKind::FPToSI(_, _) | InstrKind::FPToUI(_, _) | InstrKind::SIToFP(_, _) |
            InstrKind::UIToFP(_, _) | InstrKind::FPCast(_, _) | InstrKind::PtrToInt(_, _) |
            InstrKind::IntToPtr(_, _) | InstrKind::Bitcast(_, _) |
            InstrKind::Phi(_) | InstrKind::Select(_, _, _) |
            InstrKind::GlobalRef(_) | InstrKind::FuncRef(_) |
            InstrKind::SizeOf(_) | InstrKind::AlignOf(_) |
            InstrKind::GetFieldPtr(_, _) | InstrKind::GetElementPtr(_, _) |
            InstrKind::GetBytePtr(_, _) |
            InstrKind::GetDataPtr(_) | InstrKind::GetVTablePtr(_) |
            InstrKind::MakeTraitObject { .. } => true,

            // Impure operations (side effects)
            InstrKind::Alloca(_) | // Allocates stack space
            InstrKind::Malloc(_) | InstrKind::MallocArray(_, _) |
            InstrKind::MallocBytes(_) | InstrKind::Calloc(_) |
            InstrKind::Free(_) | InstrKind::Realloc(_, _) |
            InstrKind::RcAlloc { .. } | InstrKind::RcRetain(_) |
            InstrKind::RcRelease(_) | InstrKind::RcGetCount(_) |
            InstrKind::Drop { .. } |
            InstrKind::Memcpy(_, _, _) | InstrKind::Memset(_, _, _) |
            InstrKind::Load(_) | InstrKind::LoadByte(_) |
            InstrKind::VolatileLoad(_) | InstrKind::VolatileStore(_, _) |
            InstrKind::Store(_, _) |
            InstrKind::Call { .. } | InstrKind::CallPtr { .. } |
            InstrKind::VTableCall { .. } |
            InstrKind::InlineAsm { .. } => false,
        }
    }
}

impl Default for DeadCodeEliminator {
    fn default() -> Self {
        Self::new()
    }
}

/// Run all optimization passes on a module
pub fn optimize_module(module: &mut Module) {
    let mut folder = ConstantFolder::new();
    folder.fold_module(module);

    let eliminator = DeadCodeEliminator::new();
    for func in &mut module.functions {
        if !func.is_external {
            eliminator.eliminate_function(func);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding_add() {
        let mut folder = ConstantFolder::new();

        // Create: %0 = const 5, %1 = const 3, %2 = add %0 %1
        let mut instrs = vec![
            Instruction::new(Some(VReg(0)), InstrKind::Const(Constant::Int(5))),
            Instruction::new(Some(VReg(1)), InstrKind::Const(Constant::Int(3))),
            Instruction::new(Some(VReg(2)), InstrKind::Add(VReg(0), VReg(1))),
        ];

        for instr in &mut instrs {
            folder.fold_instruction(instr);
        }

        // %2 should now be const 8
        assert!(matches!(instrs[2].kind, InstrKind::Const(Constant::Int(8))));
    }

    #[test]
    fn test_constant_folding_comparison() {
        let mut folder = ConstantFolder::new();

        let mut instrs = vec![
            Instruction::new(Some(VReg(0)), InstrKind::Const(Constant::Int(5))),
            Instruction::new(Some(VReg(1)), InstrKind::Const(Constant::Int(3))),
            Instruction::new(Some(VReg(2)), InstrKind::ICmp(CmpOp::Sgt, VReg(0), VReg(1))),
        ];

        for instr in &mut instrs {
            folder.fold_instruction(instr);
        }

        // %2 should now be const true (5 > 3)
        assert!(matches!(instrs[2].kind, InstrKind::Const(Constant::Bool(true))));
    }
}
