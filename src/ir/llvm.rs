//! LLVM Backend for Genesis IR
//!
//! Converts Genesis IR to LLVM IR using inkwell.

use std::collections::HashMap;
use std::path::Path;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module as LLVMModule;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue};
use inkwell::OptimizationLevel;
use inkwell::{AddressSpace, IntPredicate, FloatPredicate};

use super::{
    BasicBlock, CmpOp, Constant, Function, InstrKind, Instruction, IrType, Module, Terminator, VReg,
};

/// LLVM code generator
pub struct LLVMCodegen<'ctx> {
    context: &'ctx Context,
    module: LLVMModule<'ctx>,
    builder: Builder<'ctx>,
    /// Map from VReg to LLVM values
    vreg_values: HashMap<u32, BasicValueEnum<'ctx>>,
    /// Map from VReg to type (for alloca/load tracking)
    vreg_types: HashMap<u32, BasicTypeEnum<'ctx>>,
    /// Map from VReg to pointee type (for Ptr types - what the pointer points to)
    pointee_types: HashMap<u32, BasicTypeEnum<'ctx>>,
    /// Map from block IDs to LLVM blocks
    block_map: HashMap<u32, inkwell::basic_block::BasicBlock<'ctx>>,
    /// Current function being generated
    current_fn: Option<FunctionValue<'ctx>>,
    /// Map from global name to its content type (for load operations)
    global_types: HashMap<String, BasicTypeEnum<'ctx>>,
}

impl<'ctx> LLVMCodegen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        Self {
            context,
            module,
            builder,
            vreg_values: HashMap::new(),
            vreg_types: HashMap::new(),
            pointee_types: HashMap::new(),
            block_map: HashMap::new(),
            current_fn: None,
            global_types: HashMap::new(),
        }
    }

    /// Convert Genesis IR module to LLVM IR
    pub fn compile_module(&mut self, ir_module: &Module) {
        // First: create global variables (strings, etc.)
        for global in &ir_module.globals {
            self.declare_global(global);
        }

        // Second pass: declare all functions
        for func in &ir_module.functions {
            self.declare_function(func);
        }

        // Third pass: generate code for all functions
        for func in &ir_module.functions {
            if !func.is_external {
                self.compile_function(func);
            }
        }
    }

    /// Declare a global variable
    fn declare_global(&mut self, global: &super::types::Global) {
        match &global.init {
            Some(Constant::String(s)) => {
                // String constant - create global array with null terminator
                let bytes: Vec<u8> = s.bytes().chain(std::iter::once(0)).collect();
                let i8_type = self.context.i8_type();
                let array_type = i8_type.array_type(bytes.len() as u32);

                let values: Vec<_> = bytes.iter()
                    .map(|b| i8_type.const_int(*b as u64, false))
                    .collect();
                let array_val = i8_type.const_array(&values);

                let global_val = self.module.add_global(array_type, None, &global.name);
                global_val.set_initializer(&array_val);
                global_val.set_constant(global.is_const);
                global_val.set_linkage(inkwell::module::Linkage::Private);
                global_val.set_unnamed_addr(true);
            }
            Some(Constant::Int(n)) => {
                let i64_type = self.context.i64_type();
                let global_val = self.module.add_global(i64_type, None, &global.name);
                global_val.set_initializer(&i64_type.const_int(*n as u64, true));
                global_val.set_constant(global.is_const);
                // Track type for later loads
                self.global_types.insert(global.name.clone(), i64_type.into());
            }
            Some(Constant::Float(f)) => {
                let f64_type = self.context.f64_type();
                let global_val = self.module.add_global(f64_type, None, &global.name);
                global_val.set_initializer(&f64_type.const_float(*f));
                global_val.set_constant(global.is_const);
                // Track type for later loads
                self.global_types.insert(global.name.clone(), f64_type.into());
            }
            Some(Constant::Float32(f)) => {
                let f32_type = self.context.f32_type();
                let global_val = self.module.add_global(f32_type, None, &global.name);
                global_val.set_initializer(&f32_type.const_float(*f as f64));
                global_val.set_constant(global.is_const);
                // Track type for later loads
                self.global_types.insert(global.name.clone(), f32_type.into());
            }
            Some(Constant::Bool(b)) => {
                let bool_type = self.context.bool_type();
                let global_val = self.module.add_global(bool_type, None, &global.name);
                global_val.set_initializer(&bool_type.const_int(if *b { 1 } else { 0 }, false));
                global_val.set_constant(global.is_const);
                // Track type for later loads
                self.global_types.insert(global.name.clone(), bool_type.into());
            }
            _ => {
                // Other globals not yet supported (Array, Struct, Null)
            }
        }
    }

    /// Declare a function (creates the function signature)
    fn declare_function(&mut self, func: &Function) {
        let ret_type = self.convert_type(&func.ret_type);
        let param_types: Vec<BasicMetadataTypeEnum<'ctx>> = func
            .params
            .iter()
            .filter_map(|(_, ty)| self.convert_type(ty).map(|t| t.into()))
            .collect();

        let fn_type = match ret_type {
            Some(ty) => ty.fn_type(&param_types, func.is_vararg),
            None => self.context.void_type().fn_type(&param_types, func.is_vararg),
        };

        self.module.add_function(&func.name, fn_type, None);
    }

    /// Compile a function body
    fn compile_function(&mut self, func: &Function) {
        self.vreg_values.clear();
        self.vreg_types.clear();
        self.block_map.clear();

        eprintln!("DEBUG: Compiling function '{}' with {} blocks", func.name, func.blocks.len());
        for (i, block) in func.blocks.iter().enumerate() {
            eprintln!("  Block {}: id={}, {} instructions, terminator={:?}",
                i, block.id.0, block.instructions.len(), block.terminator);
        }

        let llvm_fn = self.module.get_function(&func.name).unwrap();
        self.current_fn = Some(llvm_fn);

        // Create all basic blocks first
        for block in &func.blocks {
            let default_label = format!("bb{}", block.id.0);
            let label = block.label.as_deref().unwrap_or(&default_label);
            let llvm_block = self.context.append_basic_block(llvm_fn, label);
            self.block_map.insert(block.id.0, llvm_block);
        }

        // Store parameter values
        for (i, (vreg, _)) in func.params.iter().enumerate() {
            let param_val = llvm_fn.get_nth_param(i as u32).unwrap();
            self.vreg_values.insert(vreg.0, param_val);
        }

        // Generate code for each block
        for block in &func.blocks {
            self.compile_block(block);
        }

        self.current_fn = None;
    }

    /// Compile a basic block
    fn compile_block(&mut self, block: &BasicBlock) {
        let llvm_block = self.block_map[&block.id.0];
        self.builder.position_at_end(llvm_block);

        // Compile all instructions
        for instr in &block.instructions {
            self.compile_instruction(instr);
        }

        // Compile terminator
        if let Some(ref term) = block.terminator {
            self.compile_terminator(term);
        }
    }

    /// Compile a single instruction
    fn compile_instruction(&mut self, instr: &Instruction) {
        let result = match &instr.kind {
            InstrKind::Const(c) => Some(self.compile_constant(c)),

            // Arithmetic
            InstrKind::Add(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_add(lhs, rhs, "add").unwrap().into())
            }
            InstrKind::Sub(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_sub(lhs, rhs, "sub").unwrap().into())
            }
            InstrKind::Mul(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_mul(lhs, rhs, "mul").unwrap().into())
            }
            InstrKind::SDiv(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_signed_div(lhs, rhs, "sdiv").unwrap().into())
            }
            InstrKind::UDiv(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_unsigned_div(lhs, rhs, "udiv").unwrap().into())
            }
            InstrKind::SRem(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_signed_rem(lhs, rhs, "srem").unwrap().into())
            }
            InstrKind::URem(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_int_unsigned_rem(lhs, rhs, "urem").unwrap().into())
            }
            InstrKind::Neg(v) => {
                let val = self.get_vreg(*v).into_int_value();
                Some(self.builder.build_int_neg(val, "neg").unwrap().into())
            }

            // Floating point
            InstrKind::FAdd(a, b) => {
                let lhs = self.get_vreg(*a).into_float_value();
                let rhs = self.get_vreg(*b).into_float_value();
                Some(self.builder.build_float_add(lhs, rhs, "fadd").unwrap().into())
            }
            InstrKind::FSub(a, b) => {
                let lhs = self.get_vreg(*a).into_float_value();
                let rhs = self.get_vreg(*b).into_float_value();
                Some(self.builder.build_float_sub(lhs, rhs, "fsub").unwrap().into())
            }
            InstrKind::FMul(a, b) => {
                let lhs = self.get_vreg(*a).into_float_value();
                let rhs = self.get_vreg(*b).into_float_value();
                Some(self.builder.build_float_mul(lhs, rhs, "fmul").unwrap().into())
            }
            InstrKind::FDiv(a, b) => {
                let lhs = self.get_vreg(*a).into_float_value();
                let rhs = self.get_vreg(*b).into_float_value();
                Some(self.builder.build_float_div(lhs, rhs, "fdiv").unwrap().into())
            }
            InstrKind::FNeg(v) => {
                let val = self.get_vreg(*v).into_float_value();
                Some(self.builder.build_float_neg(val, "fneg").unwrap().into())
            }

            // Bitwise
            InstrKind::And(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_and(lhs, rhs, "and").unwrap().into())
            }
            InstrKind::Or(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_or(lhs, rhs, "or").unwrap().into())
            }
            InstrKind::Xor(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_xor(lhs, rhs, "xor").unwrap().into())
            }
            InstrKind::Shl(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_left_shift(lhs, rhs, "shl").unwrap().into())
            }
            InstrKind::AShr(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_right_shift(lhs, rhs, true, "ashr").unwrap().into())
            }
            InstrKind::LShr(a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                Some(self.builder.build_right_shift(lhs, rhs, false, "lshr").unwrap().into())
            }
            InstrKind::Not(v) => {
                let val = self.get_vreg(*v).into_int_value();
                Some(self.builder.build_not(val, "not").unwrap().into())
            }

            // Comparison
            InstrKind::ICmp(op, a, b) => {
                let lhs = self.get_vreg(*a).into_int_value();
                let rhs = self.get_vreg(*b).into_int_value();
                let pred = self.convert_cmp_op(*op);
                Some(self.builder.build_int_compare(pred, lhs, rhs, "icmp").unwrap().into())
            }
            InstrKind::FCmp(op, a, b) => {
                let lhs = self.get_vreg(*a).into_float_value();
                let rhs = self.get_vreg(*b).into_float_value();
                let pred = self.convert_float_cmp_op(*op);
                Some(self.builder.build_float_compare(pred, lhs, rhs, "fcmp").unwrap().into())
            }

            // Conversions
            InstrKind::SExt(v, ty) => {
                let val = self.get_vreg(*v).into_int_value();
                let target = self.convert_type(ty).unwrap().into_int_type();
                Some(self.builder.build_int_s_extend(val, target, "sext").unwrap().into())
            }
            InstrKind::ZExt(v, ty) => {
                let val = self.get_vreg(*v).into_int_value();
                let target = self.convert_type(ty).unwrap().into_int_type();
                Some(self.builder.build_int_z_extend(val, target, "zext").unwrap().into())
            }
            InstrKind::Trunc(v, ty) => {
                let val = self.get_vreg(*v).into_int_value();
                let target = self.convert_type(ty).unwrap().into_int_type();
                Some(self.builder.build_int_truncate(val, target, "trunc").unwrap().into())
            }
            InstrKind::FPToSI(v, ty) => {
                let val = self.get_vreg(*v).into_float_value();
                let target = self.convert_type(ty).unwrap().into_int_type();
                Some(self.builder.build_float_to_signed_int(val, target, "fptosi").unwrap().into())
            }
            InstrKind::FPToUI(v, ty) => {
                let val = self.get_vreg(*v).into_float_value();
                let target = self.convert_type(ty).unwrap().into_int_type();
                Some(self.builder.build_float_to_unsigned_int(val, target, "fptoui").unwrap().into())
            }
            InstrKind::SIToFP(v, ty) => {
                let val = self.get_vreg(*v).into_int_value();
                let target = self.convert_type(ty).unwrap().into_float_type();
                Some(self.builder.build_signed_int_to_float(val, target, "sitofp").unwrap().into())
            }
            InstrKind::UIToFP(v, ty) => {
                let val = self.get_vreg(*v).into_int_value();
                let target = self.convert_type(ty).unwrap().into_float_type();
                Some(self.builder.build_unsigned_int_to_float(val, target, "uitofp").unwrap().into())
            }
            InstrKind::FPCast(v, ty) => {
                let val = self.get_vreg(*v).into_float_value();
                let target = self.convert_type(ty).unwrap().into_float_type();
                Some(self.builder.build_float_cast(val, target, "fpcast").unwrap().into())
            }
            InstrKind::PtrToInt(v, ty) => {
                let val = self.get_vreg(*v).into_pointer_value();
                let target = self.convert_type(ty).unwrap().into_int_type();
                Some(self.builder.build_ptr_to_int(val, target, "ptrtoint").unwrap().into())
            }
            InstrKind::IntToPtr(v, ty) => {
                let val = self.get_vreg(*v);

                // If the value is already a pointer, just use it directly
                // This happens when HashMap::new returns an alloca pointer
                let result = if val.is_pointer_value() {
                    val.into_pointer_value()
                } else {
                    let int_val = val.into_int_value();
                    let target = self.convert_type(ty).unwrap().into_pointer_type();
                    self.builder.build_int_to_ptr(int_val, target, "inttoptr").unwrap()
                };

                // Track the pointee type for subsequent GetFieldPtr operations
                if let Some(vreg) = instr.result {
                    if let IrType::Ptr(inner) = ty {
                        if let Some(inner_ty) = self.convert_type(inner) {
                            self.vreg_types.insert(vreg.0, inner_ty);
                        }
                    }
                }

                Some(result.into())
            }
            InstrKind::Bitcast(v, ty) => {
                let val = self.get_vreg(*v);
                let target = self.convert_type(ty).unwrap();
                // Handle pointer <-> integer conversions specially
                if val.is_pointer_value() && target.is_int_type() {
                    // Pointer to integer: use ptrtoint
                    let ptr_val = val.into_pointer_value();
                    let int_ty = target.into_int_type();
                    Some(self.builder.build_ptr_to_int(ptr_val, int_ty, "ptrtoint").unwrap().into())
                } else if val.is_int_value() && target.is_pointer_type() {
                    // Integer to pointer: use inttoptr
                    let int_val = val.into_int_value();
                    let ptr_ty = target.into_pointer_type();
                    Some(self.builder.build_int_to_ptr(int_val, ptr_ty, "inttoptr").unwrap().into())
                } else {
                    // Regular bitcast
                    Some(self.builder.build_bit_cast(val, target, "bitcast").unwrap())
                }
            }

            // Memory
            InstrKind::Alloca(ty) => {
                let llvm_ty = self.convert_type(ty).unwrap_or(self.context.i64_type().into());
                let alloca = self.builder.build_alloca(llvm_ty, "alloca").unwrap();
                // Track the type for this alloca so Load can use it
                if let Some(vreg) = instr.result {
                    self.vreg_types.insert(vreg.0, llvm_ty);
                    // For Ptr(inner) types, also track the pointee type for GetFieldPtr
                    if let IrType::Ptr(inner) = ty {
                        if let Some(pointee_ty) = self.convert_type(inner) {
                            self.pointee_types.insert(vreg.0, pointee_ty);
                        }
                    }
                }
                Some(alloca.into())
            }
            InstrKind::Malloc(ty) => {
                let llvm_ty = self.convert_type(ty).unwrap_or(self.context.i64_type().into());
                let malloc_ptr = self.builder.build_malloc(llvm_ty, "malloc").unwrap();
                // Track the type for this malloc so Load can use it
                if let Some(vreg) = instr.result {
                    self.vreg_types.insert(vreg.0, llvm_ty);
                }
                Some(malloc_ptr.into())
            }
            InstrKind::MallocArray(ty, count) => {
                let llvm_ty = self.convert_type(ty).unwrap_or(self.context.i64_type().into());
                let count_val = self.get_vreg(*count).into_int_value();
                let malloc_ptr = self.builder.build_array_malloc(llvm_ty, count_val, "malloc_array").unwrap();
                // Track the element type for this malloc so Load/GEP can use it
                if let Some(vreg) = instr.result {
                    self.vreg_types.insert(vreg.0, llvm_ty);
                }
                Some(malloc_ptr.into())
            }
            InstrKind::Free(ptr) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                self.builder.build_free(ptr_val).unwrap();
                None
            }
            InstrKind::Realloc(ptr, size) => {
                // Declare realloc if not already declared
                let realloc_fn = self.module.get_function("realloc").unwrap_or_else(|| {
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i64_ty = self.context.i64_type();
                    let fn_type = i8_ptr_ty.fn_type(&[i8_ptr_ty.into(), i64_ty.into()], false);
                    self.module.add_function("realloc", fn_type, None)
                });
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let size_val = self.get_vreg(*size).into_int_value();
                let result = self.builder.build_call(
                    realloc_fn,
                    &[ptr_val.into(), size_val.into()],
                    "realloc"
                ).unwrap();
                result.try_as_basic_value().left()
            }
            InstrKind::MallocBytes(size) => {
                // Declare malloc if not already declared
                let malloc_fn = self.module.get_function("malloc").unwrap_or_else(|| {
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i64_ty = self.context.i64_type();
                    let fn_type = i8_ptr_ty.fn_type(&[i64_ty.into()], false);
                    self.module.add_function("malloc", fn_type, None)
                });
                let size_val = self.get_vreg(*size).into_int_value();
                let result = self.builder.build_call(
                    malloc_fn,
                    &[size_val.into()],
                    "malloc_bytes"
                ).unwrap();
                result.try_as_basic_value().left()
            }
            InstrKind::Calloc(size) => {
                // Declare calloc if not already declared
                // calloc(size_t nmemb, size_t size) -> void*
                let calloc_fn = self.module.get_function("calloc").unwrap_or_else(|| {
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i64_ty = self.context.i64_type();
                    let fn_type = i8_ptr_ty.fn_type(&[i64_ty.into(), i64_ty.into()], false);
                    self.module.add_function("calloc", fn_type, None)
                });
                let size_val = self.get_vreg(*size).into_int_value();
                let one = self.context.i64_type().const_int(1, false);
                // Call calloc(size, 1) to allocate size bytes zeroed
                let result = self.builder.build_call(
                    calloc_fn,
                    &[size_val.into(), one.into()],
                    "calloc"
                ).unwrap();
                result.try_as_basic_value().left()
            }

            // ============ Reference Counting (HARC) ============
            InstrKind::RcAlloc { ty, type_id } => {
                // Allocate: [refcount: i64 | type_id: i64 | data...]
                // Header size is 16 bytes (2 * i64)
                let header_size = 16u64;
                let llvm_ty = self.convert_type(ty).unwrap_or(self.context.i64_type().into());
                let data_size = llvm_ty.size_of().unwrap();

                // Total size = header + data
                let total_size = self.builder.build_int_add(
                    self.context.i64_type().const_int(header_size, false),
                    data_size,
                    "rc_total_size"
                ).unwrap();

                // Allocate memory
                let malloc_fn = self.module.get_function("malloc").unwrap_or_else(|| {
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i64_ty = self.context.i64_type();
                    let fn_type = i8_ptr_ty.fn_type(&[i64_ty.into()], false);
                    self.module.add_function("malloc", fn_type, None)
                });

                let header_ptr = self.builder.build_call(
                    malloc_fn,
                    &[total_size.into()],
                    "rc_header"
                ).unwrap().try_as_basic_value().left().unwrap().into_pointer_value();

                // Initialize refcount to 1
                let refcount_ptr = header_ptr;
                let i64_ty = self.context.i64_type();
                self.builder.build_store(refcount_ptr, i64_ty.const_int(1, false)).unwrap();

                // Store type_id at offset 8
                let type_id_ptr = unsafe {
                    self.builder.build_gep(
                        i64_ty,
                        header_ptr,
                        &[i64_ty.const_int(1, false)],
                        "type_id_ptr"
                    ).unwrap()
                };
                self.builder.build_store(type_id_ptr, i64_ty.const_int(*type_id, false)).unwrap();

                // Return pointer to data (header + 16 bytes)
                let i8_ty = self.context.i8_type();
                let data_ptr = unsafe {
                    self.builder.build_gep(
                        i8_ty,
                        header_ptr,
                        &[i64_ty.const_int(header_size, false)],
                        "rc_data_ptr"
                    ).unwrap()
                };

                // Track the type for this allocation
                if let Some(vreg) = instr.result {
                    self.vreg_types.insert(vreg.0, llvm_ty);
                }

                Some(data_ptr.into())
            }

            InstrKind::RcRetain(ptr) => {
                // Increment refcount: ptr - 16 bytes gives header
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let i64_ty = self.context.i64_type();
                let i8_ty = self.context.i8_type();

                // Get header pointer (data_ptr - 16)
                let header_ptr = unsafe {
                    self.builder.build_gep(
                        i8_ty,
                        ptr_val,
                        &[i64_ty.const_int((-16i64) as u64, true)],
                        "rc_header"
                    ).unwrap()
                };

                // Load current refcount
                let current = self.builder.build_load(i64_ty, header_ptr, "rc_current").unwrap().into_int_value();

                // Increment
                let new_count = self.builder.build_int_add(
                    current,
                    i64_ty.const_int(1, false),
                    "rc_inc"
                ).unwrap();

                // Store new refcount
                self.builder.build_store(header_ptr, new_count).unwrap();

                None
            }

            InstrKind::RcRelease(ptr) => {
                // Decrement refcount, free if zero
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let i64_ty = self.context.i64_type();
                let i8_ty = self.context.i8_type();

                // Get header pointer (data_ptr - 16)
                let header_ptr = unsafe {
                    self.builder.build_gep(
                        i8_ty,
                        ptr_val,
                        &[i64_ty.const_int((-16i64) as u64, true)],
                        "rc_header"
                    ).unwrap()
                };

                // Load current refcount
                let current = self.builder.build_load(i64_ty, header_ptr, "rc_current").unwrap().into_int_value();

                // Decrement
                let new_count = self.builder.build_int_sub(
                    current,
                    i64_ty.const_int(1, false),
                    "rc_dec"
                ).unwrap();

                // Store new refcount
                self.builder.build_store(header_ptr, new_count).unwrap();

                // Check if zero
                let is_zero = self.builder.build_int_compare(
                    IntPredicate::EQ,
                    new_count,
                    i64_ty.const_int(0, false),
                    "rc_is_zero"
                ).unwrap();

                // Create blocks for conditional free
                let current_fn = self.current_fn.unwrap();
                let free_block = self.context.append_basic_block(current_fn, "rc_free");
                let continue_block = self.context.append_basic_block(current_fn, "rc_continue");

                self.builder.build_conditional_branch(is_zero, free_block, continue_block).unwrap();

                // Free block: call destructor and free memory
                self.builder.position_at_end(free_block);

                // Get type_id for destructor dispatch
                let type_id_ptr = unsafe {
                    self.builder.build_gep(
                        i64_ty,
                        header_ptr,
                        &[i64_ty.const_int(1, false)],
                        "type_id_ptr"
                    ).unwrap()
                };
                let _type_id = self.builder.build_load(i64_ty, type_id_ptr, "type_id").unwrap();

                // TODO: Call type-specific destructor based on type_id
                // For now, just free the memory

                // Free the header (which includes data)
                self.builder.build_free(header_ptr).unwrap();
                self.builder.build_unconditional_branch(continue_block).unwrap();

                // Continue block
                self.builder.position_at_end(continue_block);

                None
            }

            InstrKind::RcGetCount(ptr) => {
                // Return current refcount
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let i64_ty = self.context.i64_type();
                let i8_ty = self.context.i8_type();

                // Get header pointer (data_ptr - 16)
                let header_ptr = unsafe {
                    self.builder.build_gep(
                        i8_ty,
                        ptr_val,
                        &[i64_ty.const_int((-16i64) as u64, true)],
                        "rc_header"
                    ).unwrap()
                };

                // Load and return refcount
                let refcount = self.builder.build_load(i64_ty, header_ptr, "rc_count").unwrap();
                Some(refcount)
            }

            InstrKind::Drop { ptr, type_id } => {
                // Call type-specific destructor
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();

                // Generate destructor call based on type_id
                let drop_fn_name = match *type_id {
                    1 => "__drop_box",
                    2 => "__drop_vec",
                    3 => "__drop_string",
                    4 => "__drop_hashmap",
                    5 => "__drop_hashset",
                    6 => "__drop_closure",
                    7 => "__drop_channel",
                    8 => "__drop_tcpstream",
                    9 => "__drop_tcplistener",
                    10 => "__drop_file",
                    11 => "__drop_future",
                    _ => "__drop_generic",
                };

                // Try to call the drop function if it exists
                if let Some(drop_fn) = self.module.get_function(drop_fn_name) {
                    self.builder.build_call(
                        drop_fn,
                        &[ptr_val.into()],
                        "drop"
                    ).unwrap();
                }
                // If drop function doesn't exist, do nothing (will be freed by RC)

                None
            }

            InstrKind::Memcpy(dst, src, len) => {
                // Use libc memcpy
                let dst_val = self.get_vreg(*dst).into_pointer_value();
                let src_val = self.get_vreg(*src).into_pointer_value();
                let len_val = self.get_vreg(*len).into_int_value();

                // Declare memcpy if not already declared
                let memcpy_fn = self.module.get_function("memcpy").unwrap_or_else(|| {
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i64_ty = self.context.i64_type();
                    let fn_type = i8_ptr_ty.fn_type(&[i8_ptr_ty.into(), i8_ptr_ty.into(), i64_ty.into()], false);
                    self.module.add_function("memcpy", fn_type, None)
                });

                self.builder.build_call(
                    memcpy_fn,
                    &[dst_val.into(), src_val.into(), len_val.into()],
                    "memcpy"
                ).unwrap();
                None // memcpy doesn't return a useful value for us
            }
            InstrKind::Memset(dst, val, len) => {
                // Use libc memset
                let dst_val = self.get_vreg(*dst).into_pointer_value();
                let val_int = self.get_vreg(*val).into_int_value();
                let len_val = self.get_vreg(*len).into_int_value();

                // Declare memset if not already declared
                // memset signature: void *memset(void *s, int c, size_t n)
                let memset_fn = self.module.get_function("memset").unwrap_or_else(|| {
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i32_ty = self.context.i32_type();
                    let i64_ty = self.context.i64_type();
                    let fn_type = i8_ptr_ty.fn_type(&[i8_ptr_ty.into(), i32_ty.into(), i64_ty.into()], false);
                    self.module.add_function("memset", fn_type, None)
                });

                // Truncate val to i32 (memset takes int for the value)
                let val_i32 = self.builder.build_int_truncate(val_int, self.context.i32_type(), "memset_val").unwrap();

                self.builder.build_call(
                    memset_fn,
                    &[dst_val.into(), val_i32.into(), len_val.into()],
                    "memset"
                ).unwrap();
                None // memset doesn't return a useful value for us
            }
            InstrKind::Load(ptr) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                // Use the tracked type for this pointer, or default to i64
                let load_ty = self.vreg_types.get(&ptr.0)
                    .copied()
                    .unwrap_or(self.context.i64_type().into());
                let loaded = self.builder.build_load(load_ty, ptr_val, "load").unwrap();

                // Track types for the loaded value
                if let Some(vreg) = instr.result {
                    if load_ty.is_pointer_type() {
                        // We loaded a pointer - check if we know what it points to
                        // from pointee_types (set by Alloca for Ptr types)
                        if let Some(pointee_ty) = self.pointee_types.get(&ptr.0) {
                            // Propagate the pointee type so GetFieldPtr can use it
                            self.vreg_types.insert(vreg.0, *pointee_ty);
                        }
                    } else if load_ty.is_struct_type() {
                        // Direct struct load - track the struct type
                        self.vreg_types.insert(vreg.0, load_ty);
                    }
                }

                Some(loaded)
            }
            InstrKind::Store(ptr, val) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let value = self.get_vreg(*val);
                self.builder.build_store(ptr_val, value).unwrap();
                None
            }
            InstrKind::GetFieldPtr(ptr, idx) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                // Get the struct type from the tracked type
                let struct_ty = self.vreg_types.get(&ptr.0)
                    .and_then(|ty| {
                        if ty.is_struct_type() {
                            Some(ty.into_struct_type())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| {
                        // Create a struct type with enough fields
                        let i64_ty = self.context.i64_type();
                        let field_count = (*idx as usize) + 1;
                        let fields: Vec<BasicTypeEnum> = (0..field_count.max(2))
                            .map(|_| i64_ty.into())
                            .collect();
                        self.context.struct_type(&fields, false)
                    });

                let gep = unsafe {
                    self.builder.build_in_bounds_gep(
                        struct_ty,
                        ptr_val,
                        &[
                            self.context.i32_type().const_int(0, false),
                            self.context.i32_type().const_int(*idx as u64, false),
                        ],
                        "gep_field",
                    ).unwrap()
                };
                // Track the field type for the result
                if let Some(vreg) = instr.result {
                    // Get the actual field type from the struct
                    let field_ty = struct_ty.get_field_type_at_index(*idx)
                        .unwrap_or(self.context.i64_type().into());
                    self.vreg_types.insert(vreg.0, field_ty);
                }
                Some(gep.into())
            }
            InstrKind::GetElementPtr(ptr, idx) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let idx_val = self.get_vreg(*idx).into_int_value();

                // Get array type from tracked type, extract element type
                let element_ty: BasicTypeEnum = self.vreg_types.get(&ptr.0)
                    .and_then(|ty| {
                        if ty.is_array_type() {
                            Some(ty.into_array_type().get_element_type())
                        } else {
                            None
                        }
                    })
                    .unwrap_or(self.context.i64_type().into());

                let gep = unsafe {
                    self.builder.build_in_bounds_gep(
                        element_ty,
                        ptr_val,
                        &[idx_val],
                        "gep_elem",
                    ).unwrap()
                };
                // Track the element type for the result
                if let Some(vreg) = instr.result {
                    self.vreg_types.insert(vreg.0, element_ty);
                }
                Some(gep.into())
            }
            InstrKind::GetBytePtr(ptr, offset) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let offset_val = self.get_vreg(*offset).into_int_value();

                // Use i8 element type for byte-level addressing
                let i8_ty = self.context.i8_type();

                let gep = unsafe {
                    self.builder.build_in_bounds_gep(
                        i8_ty,
                        ptr_val,
                        &[offset_val],
                        "gep_byte",
                    ).unwrap()
                };
                // Track as i8 type for later load
                if let Some(vreg) = instr.result {
                    self.vreg_types.insert(vreg.0, i8_ty.into());
                }
                Some(gep.into())
            }
            InstrKind::LoadByte(ptr) => {
                let ptr_val = self.get_vreg(*ptr).into_pointer_value();
                let i8_ty = self.context.i8_type();
                let loaded = self.builder.build_load(i8_ty, ptr_val, "loadbyte").unwrap();
                Some(loaded)
            }

            // Function calls
            InstrKind::Call { func, args } => {
                let llvm_fn = if let Some(f) = self.module.get_function(func) {
                    f
                } else {
                    // Try to declare common external functions
                    let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                    let i64_ty = self.context.i64_type();
                    let i32_ty = self.context.i32_type();
                    let void_ty = self.context.void_type();

                    match func.as_str() {
                        "malloc" => {
                            let fn_type = i8_ptr_ty.fn_type(&[i64_ty.into()], false);
                            self.module.add_function("malloc", fn_type, None)
                        }
                        "realloc" => {
                            // realloc takes ptr and i64 size
                            let fn_type = i8_ptr_ty.fn_type(&[i8_ptr_ty.into(), i64_ty.into()], false);
                            self.module.add_function("realloc", fn_type, None)
                        }
                        "free" => {
                            let fn_type = void_ty.fn_type(&[i8_ptr_ty.into()], false);
                            self.module.add_function("free", fn_type, None)
                        }
                        "calloc" => {
                            let fn_type = i8_ptr_ty.fn_type(&[i64_ty.into(), i64_ty.into()], false);
                            self.module.add_function("calloc", fn_type, None)
                        }
                        // === I/O and system calls ===
                        "read" | "write" => {
                            // ssize_t read/write(int fd, void *buf, size_t count)
                            let fn_type = i64_ty.fn_type(&[i32_ty.into(), i8_ptr_ty.into(), i64_ty.into()], false);
                            self.module.add_function(func, fn_type, None)
                        }
                        "close" => {
                            // int close(int fd)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into()], false);
                            self.module.add_function("close", fn_type, None)
                        }
                        "usleep" => {
                            // int usleep(useconds_t usec)
                            let fn_type = i32_ty.fn_type(&[i64_ty.into()], false);
                            self.module.add_function("usleep", fn_type, None)
                        }
                        // === Time functions (Phase 5: Async Timers) ===
                        "clock_gettime" => {
                            // int clock_gettime(clockid_t clockid, struct timespec *tp)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i8_ptr_ty.into()], false);
                            self.module.add_function("clock_gettime", fn_type, None)
                        }
                        // === Epoll functions (Async I/O) ===
                        "epoll_create1" => {
                            // int epoll_create1(int flags)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into()], false);
                            self.module.add_function("epoll_create1", fn_type, None)
                        }
                        "epoll_ctl" => {
                            // int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i32_ty.into(), i32_ty.into(), i8_ptr_ty.into()], false);
                            self.module.add_function("epoll_ctl", fn_type, None)
                        }
                        "epoll_wait" => {
                            // int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i8_ptr_ty.into(), i32_ty.into(), i32_ty.into()], false);
                            self.module.add_function("epoll_wait", fn_type, None)
                        }
                        // === Socket functions ===
                        "socket" => {
                            // int socket(int domain, int type, int protocol)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i32_ty.into(), i32_ty.into()], false);
                            self.module.add_function("socket", fn_type, None)
                        }
                        "bind" => {
                            // int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i8_ptr_ty.into(), i32_ty.into()], false);
                            self.module.add_function("bind", fn_type, None)
                        }
                        "listen" => {
                            // int listen(int sockfd, int backlog)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i32_ty.into()], false);
                            self.module.add_function("listen", fn_type, None)
                        }
                        "accept" => {
                            // int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i8_ptr_ty.into(), i8_ptr_ty.into()], false);
                            self.module.add_function("accept", fn_type, None)
                        }
                        "connect" => {
                            // int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i8_ptr_ty.into(), i32_ty.into()], false);
                            self.module.add_function("connect", fn_type, None)
                        }
                        "fcntl" => {
                            // int fcntl(int fd, int cmd, ...)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i32_ty.into(), i32_ty.into()], false);
                            self.module.add_function("fcntl", fn_type, None)
                        }
                        "setsockopt" => {
                            // int setsockopt(int sockfd, int level, int optname, const void *optval, socklen_t optlen)
                            let fn_type = i32_ty.fn_type(&[i32_ty.into(), i32_ty.into(), i32_ty.into(), i8_ptr_ty.into(), i32_ty.into()], false);
                            self.module.add_function("setsockopt", fn_type, None)
                        }
                        // === String functions ===
                        "strlen" => {
                            let fn_type = i64_ty.fn_type(&[i8_ptr_ty.into()], false);
                            self.module.add_function("strlen", fn_type, None)
                        }
                        "memcpy" => {
                            let fn_type = i8_ptr_ty.fn_type(&[i8_ptr_ty.into(), i8_ptr_ty.into(), i64_ty.into()], false);
                            self.module.add_function("memcpy", fn_type, None)
                        }
                        "memset" => {
                            let fn_type = i8_ptr_ty.fn_type(&[i8_ptr_ty.into(), i32_ty.into(), i64_ty.into()], false);
                            self.module.add_function("memset", fn_type, None)
                        }
                        _ => {
                            // Unknown external function - store dummy value and return
                            if let Some(vreg) = instr.result {
                                self.vreg_values.insert(vreg.0, self.context.i64_type().const_int(0, false).into());
                            }
                            return;
                        }
                    }
                };

                let arg_vals: Vec<BasicMetadataValueEnum<'ctx>> = args
                    .iter()
                    .map(|a| self.get_vreg(*a).into())
                    .collect();
                let call = self.builder.build_call(llvm_fn, &arg_vals, "call").unwrap();
                call.try_as_basic_value().left()
            }
            InstrKind::CallPtr { ptr, args } => {
                let ptr_reg = self.get_vreg(*ptr);
                // The function pointer may be stored as i64, convert to pointer if needed
                let ptr_val = if ptr_reg.is_int_value() {
                    let int_val = ptr_reg.into_int_value();
                    self.builder.build_int_to_ptr(int_val, self.context.ptr_type(inkwell::AddressSpace::default()), "inttoptr").unwrap()
                } else {
                    ptr_reg.into_pointer_value()
                };
                let arg_vals: Vec<BasicMetadataValueEnum<'ctx>> = args
                    .iter()
                    .map(|a| self.get_vreg(*a).into())
                    .collect();
                // Create a function type for the call
                let param_types: Vec<BasicMetadataTypeEnum<'ctx>> = args
                    .iter()
                    .map(|_| self.context.i64_type().into())
                    .collect();
                let fn_type = self.context.i64_type().fn_type(&param_types, false);
                let call = self.builder.build_indirect_call(fn_type, ptr_val, &arg_vals, "callptr").unwrap();
                call.try_as_basic_value().left()
            }

            // Misc
            InstrKind::Phi(preds) => {
                // PHI nodes need the LLVM phi instruction
                // Determine the type from the first predecessor value
                let phi_type = if let Some((first_vreg, _)) = preds.first() {
                    if let Some(val) = self.vreg_values.get(&first_vreg.0) {
                        val.get_type()
                    } else {
                        self.context.i64_type().into()
                    }
                } else {
                    self.context.i64_type().into()
                };

                let phi = self.builder.build_phi(phi_type, "phi").unwrap();

                // Track the pointee type from the first predecessor for pointer PHIs
                let mut first_pointee_type = None;
                for (vreg, block_id) in preds {
                    if let Some(val) = self.vreg_values.get(&vreg.0) {
                        if let Some(block) = self.block_map.get(&block_id.0) {
                            // Track pointee type from first predecessor
                            if first_pointee_type.is_none() {
                                if let Some(ty) = self.vreg_types.get(&vreg.0) {
                                    first_pointee_type = Some(*ty);
                                }
                            }
                            // Coerce value to phi type if needed
                            let coerced_val = if val.get_type() != phi_type {
                                // Type mismatch - try to coerce
                                if phi_type.is_int_type() && val.is_int_value() {
                                    let int_val = val.into_int_value();
                                    let target_int_type = phi_type.into_int_type();
                                    if int_val.get_type().get_bit_width() < target_int_type.get_bit_width() {
                                        self.builder.build_int_s_extend(int_val, target_int_type, "sext").unwrap().into()
                                    } else if int_val.get_type().get_bit_width() > target_int_type.get_bit_width() {
                                        self.builder.build_int_truncate(int_val, target_int_type, "trunc").unwrap().into()
                                    } else {
                                        *val
                                    }
                                } else {
                                    *val
                                }
                            } else {
                                *val
                            };
                            phi.add_incoming(&[(&coerced_val, *block)]);
                        }
                    }
                }

                // Track the pointee type for the PHI result (for get_field_ptr)
                if let Some(vreg) = instr.result {
                    if let Some(pointee_ty) = first_pointee_type {
                        self.vreg_types.insert(vreg.0, pointee_ty);
                    }
                }

                Some(phi.as_basic_value())
            }
            InstrKind::Select(cond, t, f) => {
                let cond_val = self.get_vreg(*cond).into_int_value();
                let true_val = self.get_vreg(*t);
                let false_val = self.get_vreg(*f);
                Some(self.builder.build_select(cond_val, true_val, false_val, "select").unwrap())
            }

            InstrKind::GlobalRef(name) => {
                // stderr/stdin are special cases - they're external global pointers that need to be loaded
                if name == "stderr" || name == "stdin" {
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let global = if let Some(g) = self.module.get_global(name) {
                        g
                    } else {
                        let g = self.module.add_global(ptr_type, Some(AddressSpace::default()), name);
                        g.set_linkage(inkwell::module::Linkage::External);
                        g
                    };
                    // Always load the pointer value from the stdio global
                    let load_name = format!("{}_val", name);
                    let stdio_ptr = self.builder.build_load(ptr_type, global.as_pointer_value(), &load_name).unwrap();
                    Some(stdio_ptr)
                } else if let Some(global) = self.module.get_global(name) {
                    // Get pointer to a global variable
                    // Track the content type for later loads
                    if let Some(vreg) = instr.result {
                        if let Some(content_type) = self.global_types.get(name).copied() {
                            self.vreg_types.insert(vreg.0, content_type);
                        }
                    }
                    Some(global.as_pointer_value().into())
                } else {
                    // Global not found - return null pointer
                    Some(self.context.ptr_type(AddressSpace::default()).const_null().into())
                }
            }

            InstrKind::FuncRef(name) => {
                // Get pointer to a function
                if let Some(func) = self.module.get_function(name) {
                    Some(func.as_global_value().as_pointer_value().into())
                } else {
                    // Function not found - return null pointer
                    Some(self.context.ptr_type(AddressSpace::default()).const_null().into())
                }
            }
        };

        // Store result if instruction has one
        if let (Some(vreg), Some(value)) = (instr.result, result) {
            self.vreg_values.insert(vreg.0, value);
        }
    }

    /// Compile a terminator instruction
    fn compile_terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Ret(None) => {
                self.builder.build_return(None).unwrap();
            }
            Terminator::Ret(Some(v)) => {
                let val = self.get_vreg(*v);
                // Check if we need to convert the return value to match function return type
                if let Some(llvm_fn) = self.current_fn {
                    let ret_type = llvm_fn.get_type().get_return_type();
                    if let Some(expected_type) = ret_type {
                        let actual_type = val.get_type();
                        // If types don't match, try to convert
                        if actual_type != expected_type {
                            if let (BasicTypeEnum::IntType(expected_int), BasicValueEnum::IntValue(actual_int)) =
                                (expected_type, val) {
                                // Truncate or extend as needed
                                let expected_bits = expected_int.get_bit_width();
                                let actual_bits = actual_int.get_type().get_bit_width();
                                let converted = if actual_bits > expected_bits {
                                    self.builder.build_int_truncate(actual_int, expected_int, "ret_trunc").unwrap()
                                } else if actual_bits < expected_bits {
                                    self.builder.build_int_s_extend(actual_int, expected_int, "ret_ext").unwrap()
                                } else {
                                    actual_int
                                };
                                self.builder.build_return(Some(&converted)).unwrap();
                                return;
                            }
                        }
                    }
                }
                self.builder.build_return(Some(&val)).unwrap();
            }
            Terminator::Br(block) => {
                let target = self.block_map[&block.0];
                self.builder.build_unconditional_branch(target).unwrap();
            }
            Terminator::CondBr { cond, then_block, else_block } => {
                let cond_val = self.get_vreg(*cond).into_int_value();
                let then_bb = self.block_map[&then_block.0];
                let else_bb = self.block_map[&else_block.0];
                self.builder.build_conditional_branch(cond_val, then_bb, else_bb).unwrap();
            }
            Terminator::Switch { value, default, cases } => {
                let val = self.get_vreg(*value).into_int_value();
                let default_bb = self.block_map[&default.0];
                let cases: Vec<_> = cases
                    .iter()
                    .filter_map(|(c, b)| {
                        if let Constant::Int(n) = c {
                            let case_val = self.context.i64_type().const_int(*n as u64, false);
                            Some((case_val, self.block_map[&b.0]))
                        } else {
                            None
                        }
                    })
                    .collect();
                self.builder.build_switch(val, default_bb, &cases).unwrap();
            }
            Terminator::Unreachable => {
                self.builder.build_unreachable().unwrap();
            }
        }
    }

    /// Convert a constant to LLVM value
    fn compile_constant(&self, constant: &Constant) -> BasicValueEnum<'ctx> {
        match constant {
            Constant::Int(n) => self.context.i64_type().const_int(*n as u64, true).into(),
            Constant::Float(f) => self.context.f64_type().const_float(*f).into(),
            Constant::Float32(f) => self.context.f32_type().const_float(*f as f64).into(),
            Constant::Bool(b) => self.context.bool_type().const_int(*b as u64, false).into(),
            Constant::Null => self.context.ptr_type(AddressSpace::default()).const_null().into(),
            Constant::String(s) => {
                let global = self.builder.build_global_string_ptr(s, "str").unwrap();
                global.as_pointer_value().into()
            }
            Constant::Array(_) => {
                // TODO: Array constants
                self.context.i64_type().const_int(0, false).into()
            }
            Constant::Struct(_) => {
                // TODO: Struct constants
                self.context.i64_type().const_int(0, false).into()
            }
        }
    }

    /// Convert IR type to LLVM type
    fn convert_type(&self, ty: &IrType) -> Option<BasicTypeEnum<'ctx>> {
        match ty {
            IrType::Void => None,
            IrType::Bool => Some(self.context.bool_type().into()),
            IrType::I8 => Some(self.context.i8_type().into()),
            IrType::I16 => Some(self.context.i16_type().into()),
            IrType::I32 => Some(self.context.i32_type().into()),
            IrType::I64 => Some(self.context.i64_type().into()),
            IrType::F32 => Some(self.context.f32_type().into()),
            IrType::F64 => Some(self.context.f64_type().into()),
            IrType::Ptr(_) => Some(self.context.ptr_type(AddressSpace::default()).into()),
            IrType::Array(elem, size) => {
                let elem_ty = self.convert_type(elem)?;
                Some(elem_ty.array_type(*size as u32).into())
            }
            IrType::Struct(fields) => {
                let field_types: Vec<BasicTypeEnum<'ctx>> = fields
                    .iter()
                    .filter_map(|f| self.convert_type(f))
                    .collect();
                Some(self.context.struct_type(&field_types, false).into())
            }
            IrType::Fn { params: _, ret: _ } => {
                // Return pointer to function
                Some(self.context.ptr_type(AddressSpace::default()).into())
            }
        }
    }

    /// Convert comparison operator to LLVM int predicate
    fn convert_cmp_op(&self, op: CmpOp) -> IntPredicate {
        match op {
            CmpOp::Eq => IntPredicate::EQ,
            CmpOp::Ne => IntPredicate::NE,
            CmpOp::Slt => IntPredicate::SLT,
            CmpOp::Sle => IntPredicate::SLE,
            CmpOp::Sgt => IntPredicate::SGT,
            CmpOp::Sge => IntPredicate::SGE,
            CmpOp::Ult => IntPredicate::ULT,
            CmpOp::Ule => IntPredicate::ULE,
            CmpOp::Ugt => IntPredicate::UGT,
            CmpOp::Uge => IntPredicate::UGE,
            CmpOp::Uno => IntPredicate::NE, // Uno doesn't apply to integers, fallback to NE
        }
    }

    /// Convert comparison operator to LLVM float predicate
    fn convert_float_cmp_op(&self, op: CmpOp) -> FloatPredicate {
        match op {
            CmpOp::Eq => FloatPredicate::OEQ,
            CmpOp::Ne => FloatPredicate::ONE,
            CmpOp::Slt | CmpOp::Ult => FloatPredicate::OLT,
            CmpOp::Sle | CmpOp::Ule => FloatPredicate::OLE,
            CmpOp::Sgt | CmpOp::Ugt => FloatPredicate::OGT,
            CmpOp::Sge | CmpOp::Uge => FloatPredicate::OGE,
            CmpOp::Uno => FloatPredicate::UNO,
        }
    }

    /// Get value for a VReg
    fn get_vreg(&self, vreg: VReg) -> BasicValueEnum<'ctx> {
        self.vreg_values
            .get(&vreg.0)
            .copied()
            .unwrap_or_else(|| self.context.i64_type().const_int(0, false).into())
    }

    /// Get the LLVM IR as a string
    pub fn get_llvm_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    /// Verify the module
    pub fn verify(&self) -> Result<(), String> {
        self.module
            .verify()
            .map_err(|e| e.to_string())
    }

    /// Optimize the module
    pub fn optimize(&self, level: OptLevel) {
        let opt_level = match level {
            OptLevel::None => OptimizationLevel::None,
            OptLevel::Less => OptimizationLevel::Less,
            OptLevel::Default => OptimizationLevel::Default,
            OptLevel::Aggressive => OptimizationLevel::Aggressive,
        };

        // Initialize target
        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                opt_level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        // Run optimization passes
        let passes = match level {
            OptLevel::None => "default<O0>",
            OptLevel::Less => "default<O1>",
            OptLevel::Default => "default<O2>",
            OptLevel::Aggressive => "default<O3>",
        };

        self.module
            .run_passes(passes, &target_machine, PassBuilderOptions::create())
            .unwrap();
    }

    /// Write object file
    pub fn write_object_file(&self, path: &Path) -> Result<(), String> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| e.to_string())?;

        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple)
            .map_err(|e| e.to_string())?;

        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                OptimizationLevel::Default,
                RelocMode::PIC,  // Use PIC for PIE executables
                CodeModel::Default,
            )
            .ok_or("Could not create target machine")?;

        target_machine
            .write_to_file(&self.module, FileType::Object, path)
            .map_err(|e| e.to_string())
    }

    /// Write LLVM IR to file
    pub fn write_llvm_ir(&self, path: &Path) -> Result<(), String> {
        self.module
            .print_to_file(path)
            .map_err(|e| e.to_string())
    }

    /// Write bitcode to file
    pub fn write_bitcode(&self, path: &Path) -> bool {
        self.module.write_bitcode_to_path(path)
    }
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::Default
    }
}

/// Compile Genesis IR module to native executable
pub fn compile_to_executable(
    ir_module: &Module,
    output_path: &Path,
    opt_level: OptLevel,
) -> Result<(), String> {
    let context = Context::create();
    let mut codegen = LLVMCodegen::new(&context, &ir_module.name);

    // Compile IR to LLVM
    codegen.compile_module(ir_module);

    // Verify
    codegen.verify()?;

    // Optimize
    codegen.optimize(opt_level);

    // Write object file
    let obj_path = output_path.with_extension("o");
    codegen.write_object_file(&obj_path)?;

    // Link with system linker (cc)
    let status = std::process::Command::new("cc")
        .arg(&obj_path)
        .arg("-o")
        .arg(output_path)
        .arg("-lc")
        .arg("-lm") // Link with libm for math functions
        .status()
        .map_err(|e| format!("Failed to run linker: {}", e))?;

    if !status.success() {
        return Err("Linking failed".to_string());
    }

    // Clean up object file
    let _ = std::fs::remove_file(&obj_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llvm_basic() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test");

        // Just verify we can create a codegen
        assert!(codegen.verify().is_ok());
    }
}
