//! AST to IR Lowering
//!
//! Converts type-checked AST to Genesis IR.
//!
//! # Memory Management (HARC)
//!
//! This module integrates with the HARC (Hybrid Automatic Reference Counting)
//! system to automatically insert retain/release calls for RC types.

use std::collections::{HashMap, HashSet};

use crate::ast::{
    self, BinaryOp, Block, Expr, ExprKind, FnDef, Item, Literal, Pattern, PatternKind,
    Program, Stmt, StmtKind, UnaryOp,
};
use crate::typeck::{Ty, TyKind, MonomorphCollector};
use crate::memory::{DropTracker, RcTypeInfo};
use crate::macro_expand::MacroExpander;

use super::builder::IrBuilder;
use super::instr::CmpOp;
use super::types::{IrType, Module, VReg};

/// Enum variant info for lowering
#[derive(Debug, Clone)]
pub struct EnumVariantInfo {
    /// Discriminant value for this variant
    pub discriminant: u32,
    /// Type of payload (None for unit variants)
    pub payload_type: Option<IrType>,
}

/// Lowers AST to IR
pub struct Lowerer {
    builder: IrBuilder,
    /// Map from variable names to their stack slots (alloca results)
    locals: HashMap<String, VReg>,
    /// Map from variable names to their types
    local_types: HashMap<String, IrType>,
    /// Map from VReg to its type (for coercion)
    vreg_types: HashMap<VReg, IrType>,
    /// Map from function names to their parameter types
    fn_signatures: HashMap<String, Vec<IrType>>,
    /// Map from function names to their return types
    fn_return_types: HashMap<String, IrType>,
    /// Map from "EnumName::VariantName" to variant info
    enum_variants: HashMap<String, EnumVariantInfo>,
    /// Map from enum name to its IR type (tagged union)
    enum_types: HashMap<String, IrType>,
    /// Map from struct name to its IR type (for user-defined structs)
    struct_types: HashMap<String, IrType>,
    /// Expression types from type checking (if available)
    expr_types: HashMap<crate::span::Span, Ty>,
    /// Current module prefix (for resolving unqualified function calls)
    current_module: Option<String>,
    /// Counter for generating unique closure names
    closure_counter: u32,
    /// Pending closures to be generated after current function
    pending_closures: Vec<PendingClosure>,
    /// Set of local variable names that are closures (for call_ptr dispatch)
    closure_locals: HashSet<String>,
    /// Set of local variable names that are direct pointers (Box, etc.) - should not be loaded
    ptr_locals: HashSet<String>,
    /// Set of function wrappers already generated (to avoid duplicates)
    generated_wrappers: HashSet<String>,
    /// Pending function wrappers to be generated
    pending_wrappers: Vec<String>,
    /// Stack of loop contexts for break/continue
    loop_stack: Vec<LoopContext>,

    // ============ HARC Memory Management ============
    /// Tracks variables for automatic drop insertion
    drop_tracker: DropTracker,
    /// Information about which types need RC
    rc_type_info: RcTypeInfo,
    /// Map from variable names to their source type names (for drop dispatch)
    var_type_names: HashMap<String, String>,

    // ============ Monomorphization ============
    /// Monomorphization collector with generic instantiation info
    monomorph: Option<MonomorphCollector>,
    /// Specialized struct definitions: "Wrapper$i64" -> [(field_name, IrType)]
    specialized_structs: HashMap<String, Vec<(String, IrType)>>,
    /// Generic struct definitions from AST: "Wrapper" -> [(field_name, ast::Type)]
    generic_struct_defs: HashMap<String, Vec<(String, ast::Type)>>,
    /// Generic function definitions from AST: "identity" -> FnDef
    generic_fn_defs: HashMap<String, ast::FnDef>,
    /// Type substitution for current specialized function: "T" -> Ty
    current_type_subst: HashMap<String, Ty>,
    /// Generic function calls: call_span -> (fn_name, type_args)
    generic_fn_calls: HashMap<crate::span::Span, (String, Vec<Ty>)>,
    /// Macro expander for user-defined macros
    macro_expander: MacroExpander,
    /// Current impl type name (for resolving Self)
    current_impl_type: Option<String>,
    /// Global constants: const_name -> (global_ir_name, type)
    global_consts: HashMap<String, (String, IrType)>,
}

/// Information about a closure that needs to be generated
#[derive(Debug, Clone)]
struct PendingClosure {
    name: String,
    params: Vec<(String, IrType)>,
    captures: Vec<(String, VReg, IrType)>,
    /// Names of captured variables that are themselves closures
    captured_closures: HashSet<String>,
    body: ast::Expr,
}

/// Context for the current loop (used for break/continue)
#[derive(Debug, Clone)]
struct LoopContext {
    /// Block to jump to on `break`
    exit_block: super::types::BlockId,
    /// Block to jump to on `continue` (condition check or next iteration)
    continue_block: super::types::BlockId,
    /// Optional label for this loop
    label: Option<String>,
}

/// Format specifier parsed from format string placeholders
#[derive(Debug, Clone, Default)]
struct FormatSpec {
    /// Minimum width for the output
    width: Option<usize>,
    /// Precision for floating point numbers
    precision: Option<usize>,
    /// Type specifier: 'x' for hex, 'b' for binary, 'o' for octal, etc.
    type_spec: Option<char>,
    /// Whether to pad with zeros
    zero_pad: bool,
    /// Alignment: '<' left, '>' right, '^' center
    align: Option<char>,
}

impl Lowerer {
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            builder: IrBuilder::new(module_name),
            locals: HashMap::new(),
            local_types: HashMap::new(),
            vreg_types: HashMap::new(),
            fn_signatures: HashMap::new(),
            fn_return_types: HashMap::new(),
            enum_variants: HashMap::new(),
            enum_types: HashMap::new(),
            struct_types: HashMap::new(),
            expr_types: HashMap::new(),
            current_module: None,
            closure_counter: 0,
            pending_closures: Vec::new(),
            closure_locals: HashSet::new(),
            ptr_locals: HashSet::new(),
            generated_wrappers: HashSet::new(),
            pending_wrappers: Vec::new(),
            loop_stack: Vec::new(),
            // HARC Memory Management
            drop_tracker: DropTracker::new(),
            rc_type_info: RcTypeInfo::new(),
            var_type_names: HashMap::new(),
            // Monomorphization
            monomorph: None,
            specialized_structs: HashMap::new(),
            generic_struct_defs: HashMap::new(),
            generic_fn_defs: HashMap::new(),
            current_type_subst: HashMap::new(),
            generic_fn_calls: HashMap::new(),
            // Macros
            macro_expander: MacroExpander::new(),
            // Impl type context (for resolving Self)
            current_impl_type: None,
            // Global constants
            global_consts: HashMap::new(),
        }
    }

    /// Set expression types for better IR generation
    pub fn with_expr_types(mut self, types: HashMap<crate::span::Span, Ty>) -> Self {
        self.expr_types = types;
        self
    }

    /// Set monomorphization collector for generic type specialization
    pub fn with_monomorph(mut self, monomorph: MonomorphCollector) -> Self {
        self.monomorph = Some(monomorph);
        self
    }

    /// Set generic function call info for monomorphization
    pub fn with_generic_fn_calls(mut self, calls: HashMap<crate::span::Span, (String, Vec<Ty>)>) -> Self {
        self.generic_fn_calls = calls;
        self
    }

    // ============ Monomorphization Helpers ============

    /// Convert AST type to IR type (for struct field collection)
    fn ast_type_to_ir_type_simple(&self, ty: &ast::Type) -> IrType {
        use crate::ast::TypeKind;
        match &ty.kind {
            TypeKind::Path(path) => {
                let name = &path.segments[0].ident.name;
                match name.as_str() {
                    "i8" => IrType::I8,
                    "i16" => IrType::I16,
                    "i32" => IrType::I32,
                    "i64" | "isize" => IrType::I64,
                    "u8" => IrType::I8,
                    "u16" => IrType::I16,
                    "u32" => IrType::I32,
                    "u64" | "usize" => IrType::I64,
                    "f32" => IrType::F32,
                    "f64" => IrType::F64,
                    "bool" => IrType::Bool,
                    _ => IrType::I64, // Default for unknown types
                }
            }
            TypeKind::Reference { inner, .. } => {
                IrType::Ptr(Box::new(self.ast_type_to_ir_type_simple(inner)))
            }
            TypeKind::Array { element, size: _ } => {
                // For struct field collection, just use pointer for arrays
                // (actual size will be computed when the struct is lowered)
                let elem_ty = self.ast_type_to_ir_type_simple(element);
                IrType::Ptr(Box::new(elem_ty))
            }
            TypeKind::Tuple(elems) => {
                let elem_tys: Vec<IrType> = elems.iter()
                    .map(|e| self.ast_type_to_ir_type_simple(e))
                    .collect();
                IrType::Struct(elem_tys)
            }
            _ => IrType::I64,
        }
    }

    /// Collect non-generic struct types from the AST
    fn collect_struct_types(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    // Only collect non-generic structs (generics are handled separately)
                    let is_generic = s.generics.as_ref()
                        .map(|g| !g.params.is_empty())
                        .unwrap_or(false);

                    if !is_generic {
                        // Convert field types to IR types
                        let field_types: Vec<IrType> = s.fields
                            .iter()
                            .map(|f| self.ast_type_to_ir_type_simple(&f.ty))
                            .collect();

                        // Store as pointer to struct (matches how struct literals work)
                        let struct_ty = IrType::Struct(field_types);
                        self.struct_types.insert(s.name.name.clone(), IrType::Ptr(Box::new(struct_ty)));
                    }
                }
                Item::Mod(m) => {
                    if let Some(ref items) = m.items {
                        let sub_program = Program {
                            items: items.clone(),
                            span: m.span,
                        };
                        self.collect_struct_types(&sub_program);
                    }
                }
                _ => {}
            }
        }
    }

    /// Collect generic struct definitions from the AST
    fn collect_generic_struct_defs(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    if let Some(ref generics) = s.generics {
                        if !generics.params.is_empty() {
                            let fields: Vec<(String, ast::Type)> = s.fields
                                .iter()
                                .map(|f| (f.name.name.clone(), f.ty.clone()))
                                .collect();
                            self.generic_struct_defs.insert(s.name.name.clone(), fields);
                        }
                    }
                }
                Item::Mod(m) => {
                    if let Some(ref items) = m.items {
                        let sub_program = Program {
                            items: items.clone(),
                            span: m.span,
                        };
                        self.collect_generic_struct_defs(&sub_program);
                    }
                }
                _ => {}
            }
        }
    }

    /// Generate specialized struct definitions from monomorphization info
    fn generate_specialized_structs(&mut self) {
        if let Some(ref mono) = self.monomorph.clone() {
            for (struct_name, instances) in &mono.struct_instances {
                if let Some(field_defs) = self.generic_struct_defs.get(struct_name).cloned() {
                    // Get type parameters for this struct
                    if let Some(type_params) = mono.get_struct_type_params(struct_name) {
                        for type_args in instances {
                            // Build substitution map: T -> concrete_type
                            let subst: HashMap<String, &Ty> = type_params
                                .iter()
                                .zip(type_args.iter())
                                .map(|(name, ty)| (name.clone(), ty))
                                .collect();

                            // Specialize each field
                            let specialized_fields: Vec<(String, IrType)> = field_defs
                                .iter()
                                .map(|(name, ast_ty)| {
                                    let ir_ty = self.specialize_ast_type(ast_ty, &subst);
                                    (name.clone(), ir_ty)
                                })
                                .collect();

                            // Register specialized struct
                            let mangled_name = MonomorphCollector::mangle(struct_name, type_args);
                            self.specialized_structs.insert(mangled_name, specialized_fields);
                        }
                    }
                }
            }
        }
    }

    /// Specialize an AST type by substituting generic parameters
    fn specialize_ast_type(&self, ast_ty: &ast::Type, subst: &HashMap<String, &Ty>) -> IrType {
        match &ast_ty.kind {
            ast::TypeKind::Path(path) => {
                let name = &path.segments[0].ident.name;
                // Check if this is a generic parameter being substituted
                if let Some(concrete_ty) = subst.get(name) {
                    return self.ty_to_ir_type(concrete_ty);
                }
                // Otherwise, it's a concrete type path
                self.ty_to_ir_type(&self.ast_type_to_ty(ast_ty))
            }
            ast::TypeKind::Array { element, size } => {
                let elem_ir = self.specialize_ast_type(element, subst);
                // Handle size expression
                let sz = match &size.kind {
                    ast::ExprKind::Literal(ast::Literal::Int(n)) => *n as usize,
                    _ => 0,
                };
                IrType::Array(Box::new(elem_ir), sz)
            }
            ast::TypeKind::Slice { element } => {
                let elem_ir = self.specialize_ast_type(element, subst);
                IrType::Ptr(Box::new(elem_ir))
            }
            ast::TypeKind::Tuple(elements) => {
                let elem_irs: Vec<IrType> = elements
                    .iter()
                    .map(|e| self.specialize_ast_type(e, subst))
                    .collect();
                IrType::Struct(elem_irs)
            }
            ast::TypeKind::Reference { inner, .. } => {
                let inner_ir = self.specialize_ast_type(inner, subst);
                IrType::Ptr(Box::new(inner_ir))
            }
            _ => self.ty_to_ir_type(&self.ast_type_to_ty(ast_ty)),
        }
    }

    /// Get specialized struct field types by mangled name
    fn get_specialized_struct_fields(&self, mangled_name: &str) -> Option<&[(String, IrType)]> {
        self.specialized_structs.get(mangled_name).map(|v| v.as_slice())
    }

    /// Get field index for a specialized struct
    fn get_specialized_field_index(&self, mangled_name: &str, field_name: &str) -> Option<u32> {
        self.specialized_structs.get(mangled_name).and_then(|fields| {
            fields.iter().position(|(name, _)| name == field_name).map(|i| i as u32)
        })
    }

    /// Get the mangled struct name from an expression's resolved type
    fn get_struct_mangled_name(&self, expr: &Expr) -> Option<String> {
        self.expr_types.get(&expr.span).and_then(|ty| {
            match &ty.kind {
                TyKind::Named { name, generics } if !generics.is_empty() => {
                    // Check if this is a user-defined generic struct
                    if self.generic_struct_defs.contains_key(name) {
                        Some(MonomorphCollector::mangle(name, generics))
                    } else {
                        None
                    }
                }
                TyKind::Named { name, .. } => {
                    // Non-generic struct
                    if self.specialized_structs.contains_key(name) {
                        Some(name.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
    }

    // ============ Generic Function Monomorphization ============

    /// Check if a function definition has generic parameters
    fn is_generic_fn(&self, f: &ast::FnDef) -> bool {
        f.generics.as_ref().map_or(false, |g| !g.params.is_empty())
    }

    /// Collect generic function definitions from the program
    fn collect_generic_fn_defs(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Function(f) => {
                    if self.is_generic_fn(f) {
                        self.generic_fn_defs.insert(f.name.name.clone(), f.clone());
                    }
                }
                Item::Mod(m) => {
                    if let Some(ref items) = m.items {
                        let prefix = m.name.name.clone();
                        for item in items {
                            if let Item::Function(f) = item {
                                if self.is_generic_fn(f) {
                                    let full_name = format!("{}::{}", prefix, f.name.name);
                                    self.generic_fn_defs.insert(full_name, f.clone());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Collect macro definitions and register them with the expander
    fn collect_macro_defs(&mut self, program: &Program) {
        for item in &program.items {
            if let Item::Macro(m) = item {
                self.macro_expander.register(m.clone());
            }
        }
    }

    /// Generate function signatures for specialized generic functions
    fn generate_specialized_fn_signatures(&mut self) {
        if let Some(ref mono) = self.monomorph.clone() {
            for (fn_name, instances) in &mono.fn_instances {
                if let Some(fn_def) = self.generic_fn_defs.get(fn_name).cloned() {
                    if let Some(ref generics) = fn_def.generics {
                        let type_params: Vec<String> = generics.params
                            .iter()
                            .map(|p| p.name.name.clone())
                            .collect();

                        for type_args in instances {
                            // Build substitution map
                            let subst: HashMap<String, &Ty> = type_params
                                .iter()
                                .zip(type_args.iter())
                                .map(|(name, ty)| (name.clone(), ty))
                                .collect();

                            // Generate mangled name
                            let mangled_name = MonomorphCollector::mangle(fn_name, type_args);

                            // Specialize parameter types
                            let param_types: Vec<IrType> = fn_def.params
                                .iter()
                                .map(|p| self.specialize_ast_type(&p.ty, &subst))
                                .collect();

                            // Specialize return type
                            let ret_type = fn_def.return_type.as_ref()
                                .map(|t| self.specialize_ast_type(t, &subst))
                                .unwrap_or(IrType::Void);

                            // Register signatures
                            self.fn_signatures.insert(mangled_name.clone(), param_types);
                            self.fn_return_types.insert(mangled_name, ret_type);
                        }
                    }
                }
            }
        }
    }

    /// Generate specialized function implementations
    fn generate_specialized_functions(&mut self) {
        if let Some(ref mono) = self.monomorph.clone() {
            for (fn_name, instances) in &mono.fn_instances {
                if let Some(fn_def) = self.generic_fn_defs.get(fn_name).cloned() {
                    if let Some(ref generics) = fn_def.generics {
                        let type_params: Vec<String> = generics.params
                            .iter()
                            .map(|p| p.name.name.clone())
                            .collect();

                        for type_args in instances {
                            // Build substitution map
                            let subst: HashMap<String, Ty> = type_params
                                .iter()
                                .zip(type_args.iter())
                                .map(|(name, ty)| (name.clone(), ty.clone()))
                                .collect();

                            // Generate mangled name
                            let mangled_name = MonomorphCollector::mangle(fn_name, type_args);

                            // Set current substitution for use in lower_specialized_function
                            self.current_type_subst = subst.clone();

                            // Lower the specialized function
                            self.lower_specialized_function(&fn_def, &mangled_name, &subst);

                            // Clear substitution
                            self.current_type_subst.clear();
                        }
                    }
                }
            }
        }
    }

    /// Lower a specialized version of a generic function
    fn lower_specialized_function(
        &mut self,
        f: &ast::FnDef,
        mangled_name: &str,
        subst: &HashMap<String, Ty>,
    ) {
        self.locals.clear();
        self.local_types.clear();
        self.vreg_types.clear();

        // Build substitution for specialize_ast_type
        let subst_ref: HashMap<String, &Ty> = subst
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect();

        // Convert parameter types with substitution
        let param_types: Vec<IrType> = f.params
            .iter()
            .map(|p| self.specialize_ast_type(&p.ty, &subst_ref))
            .collect();

        // Convert return type with substitution
        let ret_type = f.return_type.as_ref()
            .map(|t| self.specialize_ast_type(t, &subst_ref))
            .unwrap_or(IrType::Void);

        // Start function with mangled name
        let param_vregs = self.builder.start_function(mangled_name, param_types, ret_type.clone());

        // Create stack slots for parameters and bind them
        for (param, vreg) in f.params.iter().zip(param_vregs.iter()) {
            let param_ty = self.specialize_ast_type(&param.ty, &subst_ref);
            let slot = self.builder.alloca(param_ty.clone());
            self.builder.store(slot, *vreg);
            self.locals.insert(param.name.name.clone(), slot);
            self.local_types.insert(param.name.name.clone(), param_ty);
        }

        // Lower function body
        let result = self.lower_block(&f.body);

        // Add return if not already present
        if ret_type == IrType::Void {
            self.builder.ret(None);
        } else if let Some(result) = result {
            self.builder.ret(Some(result));
        }

        self.builder.finish_function();

        // Generate any pending closures and wrappers
        self.generate_pending_closures();
        self.generate_pending_wrappers();
    }

    // ============ HARC Memory Management Helpers ============

    /// Get type ID for a type name (for destructor dispatch)
    fn get_type_id(&self, type_name: &str) -> u64 {
        // Extract base type name
        let base = type_name.split('<').next().unwrap_or(type_name);
        match base {
            "Box" => 1,
            "Vec" => 2,
            "String" => 3,
            "HashMap" => 4,
            "HashSet" => 5,
            "Closure" => 6,
            "Channel" => 7,
            "TcpStream" => 8,
            "TcpListener" => 9,
            "File" => 10,
            "Future" => 11,
            _ => 100, // Custom type
        }
    }

    /// Check if a type name represents an RC type
    fn is_rc_type_name(&self, type_name: &str) -> bool {
        self.rc_type_info.is_rc_type(type_name)
    }

    /// Register a local variable for drop tracking
    fn register_local_for_drop(&mut self, name: &str, type_name: &str, span: crate::span::Span) {
        let is_rc = self.is_rc_type_name(type_name);
        self.drop_tracker.register(name.to_string(), type_name.to_string(), span, is_rc);
        self.var_type_names.insert(name.to_string(), type_name.to_string());
    }

    /// Emit release calls for variables leaving scope
    fn emit_scope_drops(&mut self) {
        let drops = self.drop_tracker.leave_scope();
        for entry in drops {
            if entry.is_rc {
                // Get the variable's slot
                if let Some(&slot) = self.locals.get(&entry.name) {
                    // For stack-allocated types (String, Vec, HashMap, etc.),
                    // call the type-specific destructor instead of rc_release
                    let base_type = entry.type_name.split('<').next().unwrap_or(&entry.type_name);

                    // Check if local is stored as pointer or i64
                    let local_is_ptr = self.local_types.get(&entry.name).map_or(false, |ty| {
                        matches!(ty, IrType::Ptr(_))
                    });

                    match base_type {
                        "String" => {
                            // Check if this is a pointer local (struct allocated directly)
                            // or a value local (pointer to struct stored in slot)
                            let ptr_val = if self.ptr_locals.contains(&entry.name) {
                                // Direct struct pointer - convert to i64
                                self.builder.ptrtoint(slot, IrType::I64)
                            } else {
                                // Pointer stored in slot - load it
                                let loaded = self.builder.load(slot);
                                // Convert to i64 if it's a pointer, otherwise use as is
                                if local_is_ptr {
                                    self.builder.ptrtoint(loaded, IrType::I64)
                                } else {
                                    loaded // Already i64
                                }
                            };
                            self.builder.call("__drop_string", vec![ptr_val]);
                        }
                        "Vec" => {
                            // Check if this is a pointer local (struct allocated directly)
                            // or a value local (pointer to struct stored in slot)
                            let ptr_val = if self.ptr_locals.contains(&entry.name) {
                                // Direct struct pointer - convert to i64
                                self.builder.ptrtoint(slot, IrType::I64)
                            } else {
                                // Pointer stored in slot - load it
                                let loaded = self.builder.load(slot);
                                // Convert to i64 if it's a pointer, otherwise use as is
                                if local_is_ptr {
                                    self.builder.ptrtoint(loaded, IrType::I64)
                                } else {
                                    loaded // Already i64
                                }
                            };
                            self.builder.call("__drop_vec", vec![ptr_val]);
                        }
                        "HashMap" => {
                            let ptr_val = if self.ptr_locals.contains(&entry.name) {
                                self.builder.ptrtoint(slot, IrType::I64)
                            } else {
                                let loaded = self.builder.load(slot);
                                if local_is_ptr {
                                    self.builder.ptrtoint(loaded, IrType::I64)
                                } else {
                                    loaded
                                }
                            };
                            self.builder.call("__drop_hashmap", vec![ptr_val]);
                        }
                        "HashSet" => {
                            let ptr_val = if self.ptr_locals.contains(&entry.name) {
                                self.builder.ptrtoint(slot, IrType::I64)
                            } else {
                                let loaded = self.builder.load(slot);
                                if local_is_ptr {
                                    self.builder.ptrtoint(loaded, IrType::I64)
                                } else {
                                    loaded
                                }
                            };
                            self.builder.call("__drop_hashset", vec![ptr_val]);
                        }
                        "File" => {
                            let ptr_val = if self.ptr_locals.contains(&entry.name) {
                                self.builder.ptrtoint(slot, IrType::I64)
                            } else {
                                let loaded = self.builder.load(slot);
                                if local_is_ptr {
                                    self.builder.ptrtoint(loaded, IrType::I64)
                                } else {
                                    loaded
                                }
                            };
                            self.builder.call("__drop_file", vec![ptr_val]);
                        }
                        "Box" => {
                            // Box is heap-allocated, call __drop_box
                            let ptr = if self.ptr_locals.contains(&entry.name) {
                                slot
                            } else {
                                self.builder.load(slot)
                            };
                            let ptr_val = if local_is_ptr {
                                self.builder.ptrtoint(ptr, IrType::I64)
                            } else {
                                ptr
                            };
                            self.builder.call("__drop_box", vec![ptr_val]);
                        }
                        _ => {
                            // For other RC types, use rc_release
                            let ptr = if self.ptr_locals.contains(&entry.name) {
                                slot // Direct pointer
                            } else {
                                self.builder.load(slot) // Load from alloca
                            };
                            self.builder.rc_release(ptr);
                        }
                    }
                }
            }
        }
    }

    /// Enter a new scope for drop tracking
    fn enter_drop_scope(&mut self) {
        self.drop_tracker.enter_scope();
    }

    /// Leave scope and emit drops
    fn leave_drop_scope(&mut self) {
        self.emit_scope_drops();
    }

    /// Get the source type name for an expression (from type checker)
    fn get_expr_type_name(&self, expr: &Expr) -> String {
        // Special case: Vec::get returns element type, not i64
        if let ExprKind::Call { func, args } = &expr.kind {
            if let ExprKind::Path(path) = &func.kind {
                let func_name = path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::");
                if func_name == "Vec::get" && !args.is_empty() {
                    // Get element type from Vec<T>
                    if let Some(ty) = self.expr_types.get(&args[0].span) {
                        if let crate::typeck::TyKind::Named { name, generics, .. } = &ty.kind {
                            if name == "Vec" && !generics.is_empty() {
                                return self.ty_to_type_name(&generics[0]);
                            }
                        }
                    }
                }
            }
        }

        if let Some(ty) = self.expr_types.get(&expr.span) {
            self.ty_to_type_name(ty)
        } else {
            "unknown".to_string()
        }
    }

    /// Convert Ty to a type name string
    fn ty_to_type_name(&self, ty: &Ty) -> String {
        match &ty.kind {
            crate::typeck::TyKind::Int(_) => "i64".to_string(),
            crate::typeck::TyKind::Uint(_) => "u64".to_string(),
            crate::typeck::TyKind::Float(_) => "f64".to_string(),
            crate::typeck::TyKind::Bool => "bool".to_string(),
            crate::typeck::TyKind::Str => "str".to_string(),
            crate::typeck::TyKind::Unit => "()".to_string(),
            crate::typeck::TyKind::Named { name, generics } => {
                if generics.is_empty() {
                    name.clone()
                } else {
                    let gen_strs: Vec<String> = generics.iter()
                        .map(|g| self.ty_to_type_name(g))
                        .collect();
                    format!("{}<{}>", name, gen_strs.join(", "))
                }
            }
            crate::typeck::TyKind::Ref { inner, .. } => format!("&{}", self.ty_to_type_name(inner)),
            crate::typeck::TyKind::Tuple(elems) => {
                let elem_strs: Vec<String> = elems.iter()
                    .map(|e| self.ty_to_type_name(e))
                    .collect();
                format!("({})", elem_strs.join(", "))
            }
            _ => "unknown".to_string(),
        }
    }

    /// Convert AST type to a type name string (for HARC)
    fn ast_type_to_type_name(&self, ty: &ast::Type) -> String {
        match &ty.kind {
            ast::TypeKind::Path(path) => {
                let name = path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::");
                if let Some(args) = &path.segments.last().and_then(|s| s.generics.as_ref()) {
                    let gen_strs: Vec<String> = args.iter()
                        .map(|g| self.ast_type_to_type_name(g))
                        .collect();
                    format!("{}<{}>", name, gen_strs.join(", "))
                } else {
                    name
                }
            }
            ast::TypeKind::Array { element, .. } => {
                format!("[{}]", self.ast_type_to_type_name(element))
            }
            ast::TypeKind::Slice { element } => {
                format!("[{}]", self.ast_type_to_type_name(element))
            }
            ast::TypeKind::Tuple(elems) => {
                let elem_strs: Vec<String> = elems.iter()
                    .map(|e| self.ast_type_to_type_name(e))
                    .collect();
                format!("({})", elem_strs.join(", "))
            }
            ast::TypeKind::Reference { inner, mutable } => {
                if *mutable {
                    format!("&mut {}", self.ast_type_to_type_name(inner))
                } else {
                    format!("&{}", self.ast_type_to_type_name(inner))
                }
            }
            ast::TypeKind::FnPtr { params, return_type } => {
                let param_strs: Vec<String> = params.iter()
                    .map(|p| self.ast_type_to_type_name(p))
                    .collect();
                if let Some(ret) = return_type {
                    format!("fn({}) -> {}", param_strs.join(", "), self.ast_type_to_type_name(ret))
                } else {
                    format!("fn({})", param_strs.join(", "))
                }
            }
            ast::TypeKind::Infer => "_".to_string(),
            ast::TypeKind::Never => "!".to_string(),
            ast::TypeKind::Option(inner) => format!("Option<{}>", self.ast_type_to_type_name(inner)),
            ast::TypeKind::Result { ok, err } => {
                format!("Result<{}, {}>", self.ast_type_to_type_name(ok), self.ast_type_to_type_name(err))
            }
            ast::TypeKind::SelfType => "Self".to_string(),
            ast::TypeKind::Projection { base, assoc_name } => {
                format!("{}::{}", self.ast_type_to_type_name(base), assoc_name)
            }
        }
    }

    /// Get the IR type for an expression using the inferred type from type checking
    fn get_expr_ir_type(&self, expr: &Expr) -> IrType {
        if let Some(ty) = self.expr_types.get(&expr.span) {
            self.ty_to_ir_type(ty)
        } else {
            // Fallback to infer_expr_type which handles literals and simple expressions
            self.infer_expr_type(expr)
        }
    }

    /// Lower a complete program to IR
    pub fn lower_program(mut self, program: &Program) -> Module {
        // Register built-in enum variants (Option, Result)
        self.register_builtin_enums();

        // Generate async runtime support functions
        self.generate_async_runtime();

        // Generate channel runtime support functions (Phase 6)
        self.generate_channel_runtime();

        // Generate HARC drop functions for RC types
        self.generate_drop_functions();

        // Collect user-defined struct types (non-generic)
        self.collect_struct_types(program);

        // Collect generic struct definitions
        self.collect_generic_struct_defs(program);

        // Generate specialized structs from monomorphization info
        self.generate_specialized_structs();

        // Collect generic function definitions
        self.collect_generic_fn_defs(program);

        // Collect macro definitions for expansion
        self.collect_macro_defs(program);

        // Generate specialized function signatures
        self.generate_specialized_fn_signatures();

        // First pass: collect all function signatures and enum definitions
        for item in &program.items {
            match item {
                Item::Function(f) => {
                    // Skip generic functions - they're handled separately
                    if self.is_generic_fn(f) {
                        continue;
                    }
                    let param_types: Vec<IrType> = f.params
                        .iter()
                        .map(|p| self.ty_to_ir_type(&self.ast_type_to_ty(&p.ty)))
                        .collect();
                    self.fn_signatures.insert(f.name.name.clone(), param_types);
                    let ret_type = f.return_type.as_ref()
                        .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                        .unwrap_or(IrType::Void);
                    self.fn_return_types.insert(f.name.name.clone(), ret_type);
                }
                Item::Enum(e) => {
                    self.collect_enum_info(e);
                }
                Item::Mod(m) => {
                    if let Some(ref items) = m.items {
                        let prefix = m.name.name.clone();
                        for item in items {
                            if let Item::Function(f) = item {
                                // Skip generic functions in modules
                                if self.is_generic_fn(f) {
                                    continue;
                                }
                                let param_types: Vec<IrType> = f.params
                                    .iter()
                                    .map(|p| self.ty_to_ir_type(&self.ast_type_to_ty(&p.ty)))
                                    .collect();
                                let full_name = format!("{}::{}", prefix, f.name.name);
                                self.fn_signatures.insert(full_name.clone(), param_types);
                                let ret_type = f.return_type.as_ref()
                                    .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                                    .unwrap_or(IrType::Void);
                                self.fn_return_types.insert(full_name, ret_type);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Generate specialized generic functions
        self.generate_specialized_functions();

        // Second pass: generate code for all non-generic functions
        for item in &program.items {
            match item {
                Item::Function(f) => {
                    // Skip generic functions - they've been specialized
                    if self.is_generic_fn(f) {
                        continue;
                    }
                    self.lower_function(f);
                }
                Item::Struct(_) => {
                    // Struct definitions don't generate code directly
                }
                Item::Enum(_) => {
                    // Enum definitions don't generate code directly
                }
                Item::Const(c) => {
                    // Lower global constants
                    self.lower_const_def(c);
                }
                Item::Impl(i) => {
                    // Get the impl type name for resolving Self
                    let impl_type_name = self.ast_type_to_type_name(&i.self_type);
                    self.current_impl_type = Some(impl_type_name.clone());

                    // Lower methods with qualified names (Type::method)
                    for item in &i.items {
                        if let ast::ImplItem::Function(f) = item {
                            if !self.is_generic_fn(f) {
                                // Use qualified name: "Counter::create" instead of just "create"
                                self.lower_function_with_prefix(f, &impl_type_name);
                            }
                        }
                    }

                    // Clear impl type context
                    self.current_impl_type = None;
                }
                Item::Mod(m) => {
                    // Lower functions inside inline modules
                    if let Some(ref items) = m.items {
                        let prefix = m.name.name.clone();
                        for item in items {
                            if let Item::Function(f) = item {
                                if !self.is_generic_fn(f) {
                                    self.lower_function_with_prefix(f, &prefix);
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Other items not yet supported
                }
            }
        }

        self.builder.finish()
    }

    /// Lower a function definition
    fn lower_function(&mut self, f: &FnDef) {
        self.locals.clear();
        self.local_types.clear();
        self.vreg_types.clear();

        // Convert parameter types
        let param_types: Vec<IrType> = f
            .params
            .iter()
            .map(|p| self.ty_to_ir_type(&self.ast_type_to_ty(&p.ty)))
            .collect();

        let base_ret_type = f
            .return_type
            .as_ref()
            .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
            .unwrap_or(IrType::Void);

        // For async functions, wrap return type in Future<T>
        // Future is represented as a pointer to { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
        // state: 0 = NotStarted (lazy), 1 = Ready
        let ret_type = if f.is_async {
            IrType::ptr(IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]))
        } else {
            base_ret_type.clone()
        };

        // For async functions, we generate a LAZY future:
        // - Create a closure that captures all parameters
        // - Return Future with state=0 and closure stored
        // The actual body execution happens when .await is called
        if f.is_async {
            // Start function
            let param_vregs = self.builder.start_function(&f.name.name, param_types.clone(), ret_type.clone());

            // Create stack slots for parameters and bind them
            for (param, vreg) in f.params.iter().zip(param_vregs.iter()) {
                let param_ty = self.ty_to_ir_type(&self.ast_type_to_ty(&param.ty));
                let slot = self.builder.alloca(param_ty.clone());
                self.builder.store(slot, *vreg);
                self.locals.insert(param.name.name.clone(), slot);
                self.local_types.insert(param.name.name.clone(), param_ty);
            }

            // Generate async body as closure
            let async_closure_name = format!("__async_body_{}", f.name.name);

            // Collect parameters as captures for the closure
            let param_names: std::collections::HashSet<String> = f.params.iter()
                .map(|p| p.name.name.clone())
                .collect();

            // Find free variables in body (for nested captures)
            let free_vars = self.find_free_variables_in_block(&f.body, &param_names);

            // Build captures list: all parameters + any other free vars
            let mut captures: Vec<(String, VReg, IrType)> = Vec::new();
            for param in &f.params {
                let name = &param.name.name;
                if let Some(&slot) = self.locals.get(name) {
                    let ty = self.local_types.get(name).cloned().unwrap_or(IrType::I64);
                    captures.push((name.clone(), slot, ty));
                }
            }
            // Add other free variables
            for (name, slot, ty) in free_vars {
                if !captures.iter().any(|(n, _, _)| n == &name) {
                    captures.push((name, slot, ty));
                }
            }

            // Create environment struct type
            let env_field_types: Vec<IrType> = captures.iter()
                .map(|(name, _, ty)| {
                    if self.closure_locals.contains(name) {
                        IrType::I64
                    } else {
                        ty.clone()
                    }
                })
                .collect();
            let env_ty = if env_field_types.is_empty() {
                IrType::I64
            } else {
                IrType::Struct(env_field_types)
            };

            // Allocate and populate environment on HEAP (must survive function return)
            let env_ptr = self.builder.malloc(env_ty.clone());
            for (i, (name, slot, _ty)) in captures.iter().enumerate() {
                if !captures.is_empty() {
                    let field_ptr = self.builder.get_field_ptr(env_ptr, i as u32);
                    if self.closure_locals.contains(name) {
                        let ptr_as_i64 = self.builder.bitcast(*slot, IrType::I64);
                        self.builder.store(field_ptr, ptr_as_i64);
                    } else {
                        let val = self.builder.load(*slot);
                        self.builder.store(field_ptr, val);
                    }
                }
            }

            // Get function pointer for the async body
            let fn_ptr = self.builder.func_ref(&async_closure_name);
            let fn_as_i64 = self.builder.bitcast(fn_ptr, IrType::I64);
            let env_as_i64 = self.builder.bitcast(env_ptr, IrType::I64);

            // Allocate Future struct: { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
            let future_ty = IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]);
            let future_ptr = self.builder.malloc(future_ty.clone());

            // Set state to NotStarted (0) - LAZY execution
            let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
            let not_started = self.builder.const_int(0);
            self.builder.store(state_ptr, not_started);

            // Store fn_ptr
            let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
            self.builder.store(fn_ptr_field, fn_as_i64);

            // Store env_ptr
            let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
            self.builder.store(env_ptr_field, env_as_i64);

            // Initialize value to 0
            let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
            let zero = self.builder.const_int(0);
            self.builder.store(value_ptr, zero);

            // Return the Future pointer
            self.builder.ret(Some(future_ptr));

            self.builder.finish_function();

            // Now generate the async body closure function
            // Save state
            let saved_locals = std::mem::take(&mut self.locals);
            let saved_local_types = std::mem::take(&mut self.local_types);
            let saved_vreg_types = std::mem::take(&mut self.vreg_types);
            let saved_closure_locals = std::mem::take(&mut self.closure_locals);

            // Async body takes env_ptr as i64 and returns the result
            // Using i64 for env_ptr to be consistent with closure calling convention
            let body_ret_type = base_ret_type.clone();
            let body_param_types = vec![IrType::I64]; // env_ptr as i64
            let body_params = self.builder.start_function(&async_closure_name, body_param_types, body_ret_type.clone());
            let env_param_i64 = body_params[0];
            // Convert i64 to ptr
            let env_param = self.builder.inttoptr(env_param_i64, IrType::ptr(IrType::I64));

            // Extract captures from environment
            for (i, (name, _, ty)) in captures.iter().enumerate() {
                let field_ptr = self.builder.get_field_ptr(env_param, i as u32);
                let slot = self.builder.alloca(ty.clone());
                if self.closure_locals.contains(name) || saved_closure_locals.contains(name) {
                    // Closure capture - store as pointer
                    let val = self.builder.load(field_ptr);
                    let ptr = self.builder.inttoptr(val, IrType::ptr(IrType::I64));
                    self.builder.store(slot, ptr);
                    self.closure_locals.insert(name.clone());
                } else {
                    let val = self.builder.load(field_ptr);
                    self.builder.store(slot, val);
                }
                self.locals.insert(name.clone(), slot);
                self.local_types.insert(name.clone(), ty.clone());
            }

            // Lower the actual function body
            let result = self.lower_block(&f.body);

            // Return result
            if body_ret_type == IrType::Void {
                self.builder.ret(None);
            } else if let Some(result) = result {
                self.builder.ret(Some(result));
            } else {
                let zero = self.builder.const_int(0);
                self.builder.ret(Some(zero));
            }

            self.builder.finish_function();

            // Restore state
            self.locals = saved_locals;
            self.local_types = saved_local_types;
            self.vreg_types = saved_vreg_types;
            self.closure_locals = saved_closure_locals;

            // Generate any pending closures from the async body
            self.generate_pending_closures();
            self.generate_pending_wrappers();

            return; // Early return - async fn is complete
        }

        // Non-async function handling
        // Start function
        let param_vregs = self.builder.start_function(&f.name.name, param_types, ret_type.clone());

        // For main function, initialize the async runtime at the start
        // This ensures timers and reactor are ready before any async code runs
        if f.name.name == "main" {
            self.builder.call("__rt_init", vec![]);
        }

        // Create stack slots for parameters and bind them
        for (param, vreg) in f.params.iter().zip(param_vregs.iter()) {
            let param_ty = self.ty_to_ir_type(&self.ast_type_to_ty(&param.ty));
            let slot = self.builder.alloca(param_ty.clone());
            self.builder.store(slot, *vreg);
            self.locals.insert(param.name.name.clone(), slot);
            self.local_types.insert(param.name.name.clone(), param_ty);
        }

        // Lower function body
        let result = self.lower_block(&f.body);

        // Non-async: just return result
        {
            // Add return if not already present
            if base_ret_type == IrType::Void {
                self.builder.ret(None);
            } else if let Some(result) = result {
                self.builder.ret(Some(result));
            }
        }

        self.builder.finish_function();

        // Generate any pending closures and wrappers from this function
        self.generate_pending_closures();
        self.generate_pending_wrappers();
    }

    /// Generate all pending closure functions
    fn generate_pending_closures(&mut self) {
        // Take pending closures to avoid borrow issues
        let closures = std::mem::take(&mut self.pending_closures);

        for closure in closures {
            // Save current state (we'll create new ones for the closure)
            let saved_locals = std::mem::take(&mut self.locals);
            let saved_local_types = std::mem::take(&mut self.local_types);
            let saved_vreg_types = std::mem::take(&mut self.vreg_types);
            let saved_closure_locals = std::mem::take(&mut self.closure_locals);

            // Build parameter list: env_ptr first, then closure params
            let mut all_params = vec![IrType::ptr(IrType::I64)]; // env pointer
            let mut param_names = vec!["__env".to_string()];
            for (name, ty) in &closure.params {
                all_params.push(ty.clone());
                param_names.push(name.clone());
            }

            // Infer return type from body (simplified: assume i64)
            let ret_type = IrType::I64;

            // Start the closure function
            let param_vregs = self.builder.start_function(&closure.name, all_params, ret_type);

            // First param is environment pointer
            let env_ptr = param_vregs[0];

            // Bind captured variables from environment
            for (i, (name, _slot, ty)) in closure.captures.iter().enumerate() {
                // Check if this capture is a closure
                let is_captured_closure = closure.captured_closures.contains(name);

                if is_captured_closure {
                    // For captured closures, the env contains a pointer (as i64)
                    // Load the i64, convert to pointer, and use that as the slot
                    let field_ptr = self.builder.get_field_ptr(env_ptr, i as u32);
                    let ptr_as_i64 = self.builder.load(field_ptr);
                    // Convert back to pointer - this becomes the slot for the closure
                    let closure_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
                    let slot = self.builder.bitcast(ptr_as_i64, IrType::ptr(closure_ty.clone()));

                    self.locals.insert(name.clone(), slot);
                    self.local_types.insert(name.clone(), closure_ty);
                    // Mark this as a closure local so calls work
                    self.closure_locals.insert(name.clone());
                } else {
                    // Regular variable capture
                    let field_ptr = self.builder.get_field_ptr(env_ptr, i as u32);
                    let slot = self.builder.alloca(ty.clone());
                    let val = self.builder.load(field_ptr);
                    self.builder.store(slot, val);
                    self.locals.insert(name.clone(), slot);
                    self.local_types.insert(name.clone(), ty.clone());
                }
            }

            // Bind regular parameters
            for (i, (name, ty)) in closure.params.iter().enumerate() {
                let vreg = param_vregs[i + 1]; // +1 because env is first
                let slot = self.builder.alloca(ty.clone());
                self.builder.store(slot, vreg);
                self.locals.insert(name.clone(), slot);
                self.local_types.insert(name.clone(), ty.clone());
            }

            // Lower the closure body
            let result = self.lower_expr(&closure.body);
            self.builder.ret(Some(result));

            self.builder.finish_function();

            // Restore saved state
            self.locals = saved_locals;
            self.local_types = saved_local_types;
            self.vreg_types = saved_vreg_types;
            self.closure_locals = saved_closure_locals;
        }
    }

    /// Schedule a wrapper function for generation (returns wrapper name)
    /// The wrapper takes (env_ptr, args...) and calls original_fn(args...)
    fn schedule_function_wrapper(&mut self, fn_name: &str) -> String {
        let wrapper_name = format!("__closure_wrapper_{}", fn_name);

        // Check if already scheduled or generated
        if !self.generated_wrappers.contains(&wrapper_name) {
            self.generated_wrappers.insert(wrapper_name.clone());
            self.pending_wrappers.push(fn_name.to_string());
        }

        wrapper_name
    }

    /// Generate all pending function wrappers
    fn generate_pending_wrappers(&mut self) {
        let wrappers = std::mem::take(&mut self.pending_wrappers);

        for fn_name in wrappers {
            let wrapper_name = format!("__closure_wrapper_{}", fn_name);

            // Get the original function's parameter types
            let param_types = self.fn_signatures.get(&fn_name).cloned().unwrap_or_default();

            // Build wrapper parameter list: env_ptr first, then original params
            let mut wrapper_params = vec![IrType::I64]; // env pointer (ignored)
            wrapper_params.extend(param_types.clone());

            // Get the actual return type from the original function
            let ret_type = self.fn_return_types.get(&fn_name).cloned().unwrap_or(IrType::I64);

            // Start the wrapper function
            let param_vregs = self.builder.start_function(&wrapper_name, wrapper_params, ret_type.clone());

            // Skip the first param (env_ptr) and call the original function with the rest
            let call_args: Vec<VReg> = param_vregs.iter().skip(1).copied().collect();
            let result = self.builder.call(&fn_name, call_args);
            self.builder.ret(Some(result));

            self.builder.finish_function();

            // Register the wrapper in fn_signatures and fn_return_types
            let mut wrapper_sig = vec![IrType::I64]; // env_ptr
            wrapper_sig.extend(param_types);
            self.fn_signatures.insert(wrapper_name.clone(), wrapper_sig);
            self.fn_return_types.insert(wrapper_name, ret_type);
        }
    }

    /// Lower a constant definition to a global variable
    fn lower_const_def(&mut self, c: &ast::ConstDef) {
        let const_name = c.name.name.clone();
        let global_name = format!("__const_{}", const_name);

        // Evaluate the constant expression to get initial value
        if let Some((init_val, ir_type)) = self.eval_const_expr(&c.value) {
            // Add global constant to module
            self.builder.add_global(&global_name, ir_type.clone(), Some(init_val), true);

            // Register the constant for lookup
            self.global_consts.insert(const_name, (global_name, ir_type));
        }
    }

    /// Evaluate a constant expression at compile time
    /// Returns (Constant value, IrType) if successful
    fn eval_const_expr(&self, expr: &Expr) -> Option<(super::types::Constant, IrType)> {
        match &expr.kind {
            ExprKind::Literal(lit) => {
                match lit {
                    Literal::Int(n) => Some((super::types::Constant::Int(*n as i64), IrType::I64)),
                    Literal::Float(f) => Some((super::types::Constant::Float(*f), IrType::F64)),
                    Literal::Bool(b) => Some((super::types::Constant::Bool(*b), IrType::Bool)),
                    Literal::String(s) => Some((super::types::Constant::String(s.clone()), IrType::ptr(IrType::I8))),
                    Literal::Char(c) => Some((super::types::Constant::Int(*c as i64), IrType::I64)),
                }
            }
            // Support negative numbers: -42, -3.14
            ExprKind::Unary { op: UnaryOp::Neg, operand } => {
                match &operand.kind {
                    ExprKind::Literal(Literal::Int(n)) => Some((super::types::Constant::Int(-(*n as i64)), IrType::I64)),
                    ExprKind::Literal(Literal::Float(f)) => Some((super::types::Constant::Float(-*f), IrType::F64)),
                    _ => None,
                }
            }
            _ => None, // Other expressions not supported as constants yet
        }
    }

    /// Lower a function definition with a module prefix
    fn lower_function_with_prefix(&mut self, f: &FnDef, prefix: &str) {
        self.locals.clear();
        self.local_types.clear();
        self.vreg_types.clear();

        // Set current module context for resolving unqualified calls
        self.current_module = Some(prefix.to_string());

        // Convert parameter types
        let param_types: Vec<IrType> = f
            .params
            .iter()
            .map(|p| self.ty_to_ir_type(&self.ast_type_to_ty(&p.ty)))
            .collect();

        let base_ret_type = f
            .return_type
            .as_ref()
            .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
            .unwrap_or(IrType::Void);

        // For async functions, wrap return type in Future<T>
        // Future is represented as a pointer to { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
        let ret_type = if f.is_async {
            IrType::ptr(IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]))
        } else {
            base_ret_type.clone()
        };

        let full_name = format!("{}::{}", prefix, f.name.name);

        // For async functions, we generate a LAZY future (same as lower_function)
        if f.is_async {
            let param_vregs = self.builder.start_function(&full_name, param_types.clone(), ret_type.clone());

            for (param, vreg) in f.params.iter().zip(param_vregs.iter()) {
                let param_ty = self.ty_to_ir_type(&self.ast_type_to_ty(&param.ty));
                let slot = self.builder.alloca(param_ty.clone());
                self.builder.store(slot, *vreg);
                self.locals.insert(param.name.name.clone(), slot);
                self.local_types.insert(param.name.name.clone(), param_ty);
            }

            // Generate async body as closure
            let async_closure_name = format!("__async_body_{}", full_name.replace("::", "_"));

            let param_names: std::collections::HashSet<String> = f.params.iter()
                .map(|p| p.name.name.clone())
                .collect();

            let free_vars = self.find_free_variables_in_block(&f.body, &param_names);

            let mut captures: Vec<(String, VReg, IrType)> = Vec::new();
            for param in &f.params {
                let name = &param.name.name;
                if let Some(&slot) = self.locals.get(name) {
                    let ty = self.local_types.get(name).cloned().unwrap_or(IrType::I64);
                    captures.push((name.clone(), slot, ty));
                }
            }
            for (name, slot, ty) in free_vars {
                if !captures.iter().any(|(n, _, _)| n == &name) {
                    captures.push((name, slot, ty));
                }
            }

            let env_field_types: Vec<IrType> = captures.iter()
                .map(|(name, _, ty)| {
                    if self.closure_locals.contains(name) {
                        IrType::I64
                    } else {
                        ty.clone()
                    }
                })
                .collect();
            let env_ty = if env_field_types.is_empty() {
                IrType::I64
            } else {
                IrType::Struct(env_field_types)
            };

            let env_ptr = self.builder.malloc(env_ty.clone());
            for (i, (name, slot, _ty)) in captures.iter().enumerate() {
                if !captures.is_empty() {
                    let field_ptr = self.builder.get_field_ptr(env_ptr, i as u32);
                    if self.closure_locals.contains(name) {
                        let ptr_as_i64 = self.builder.bitcast(*slot, IrType::I64);
                        self.builder.store(field_ptr, ptr_as_i64);
                    } else {
                        let val = self.builder.load(*slot);
                        self.builder.store(field_ptr, val);
                    }
                }
            }

            let fn_ptr = self.builder.func_ref(&async_closure_name);
            let fn_as_i64 = self.builder.bitcast(fn_ptr, IrType::I64);
            let env_as_i64 = self.builder.bitcast(env_ptr, IrType::I64);

            let future_ty = IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]);
            let future_ptr = self.builder.malloc(future_ty.clone());

            let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
            let not_started = self.builder.const_int(0);
            self.builder.store(state_ptr, not_started);

            let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
            self.builder.store(fn_ptr_field, fn_as_i64);

            let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
            self.builder.store(env_ptr_field, env_as_i64);

            let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
            let zero = self.builder.const_int(0);
            self.builder.store(value_ptr, zero);

            self.builder.ret(Some(future_ptr));
            self.builder.finish_function();

            // Generate the async body closure function
            let saved_locals = std::mem::take(&mut self.locals);
            let saved_local_types = std::mem::take(&mut self.local_types);
            let saved_vreg_types = std::mem::take(&mut self.vreg_types);
            let saved_closure_locals = std::mem::take(&mut self.closure_locals);

            // Using i64 for env_ptr to be consistent with closure calling convention
            let body_ret_type = base_ret_type.clone();
            let body_param_types = vec![IrType::I64]; // env_ptr as i64
            let body_params = self.builder.start_function(&async_closure_name, body_param_types, body_ret_type.clone());
            let env_param_i64 = body_params[0];
            // Convert i64 to ptr
            let env_param = self.builder.inttoptr(env_param_i64, IrType::ptr(IrType::I64));

            for (i, (name, _, ty)) in captures.iter().enumerate() {
                let field_ptr = self.builder.get_field_ptr(env_param, i as u32);
                let slot = self.builder.alloca(ty.clone());
                if self.closure_locals.contains(name) || saved_closure_locals.contains(name) {
                    let val = self.builder.load(field_ptr);
                    let ptr = self.builder.inttoptr(val, IrType::ptr(IrType::I64));
                    self.builder.store(slot, ptr);
                    self.closure_locals.insert(name.clone());
                } else {
                    let val = self.builder.load(field_ptr);
                    self.builder.store(slot, val);
                }
                self.locals.insert(name.clone(), slot);
                self.local_types.insert(name.clone(), ty.clone());
            }

            let result = self.lower_block(&f.body);

            if body_ret_type == IrType::Void {
                self.builder.ret(None);
            } else if let Some(result) = result {
                self.builder.ret(Some(result));
            } else {
                let zero = self.builder.const_int(0);
                self.builder.ret(Some(zero));
            }

            self.builder.finish_function();

            self.locals = saved_locals;
            self.local_types = saved_local_types;
            self.vreg_types = saved_vreg_types;
            self.closure_locals = saved_closure_locals;

            self.generate_pending_closures();
            self.generate_pending_wrappers();
            self.current_module = None;
            return;
        }

        // Non-async function handling
        let param_vregs = self.builder.start_function(&full_name, param_types, ret_type.clone());

        // Create stack slots for parameters and bind them
        for (param, vreg) in f.params.iter().zip(param_vregs.iter()) {
            let param_ty = self.ty_to_ir_type(&self.ast_type_to_ty(&param.ty));
            let slot = self.builder.alloca(param_ty.clone());
            self.builder.store(slot, *vreg);
            self.locals.insert(param.name.name.clone(), slot);
            self.local_types.insert(param.name.name.clone(), param_ty);
        }

        // Lower function body
        let result = self.lower_block(&f.body);

        // Non-async: just return result
        if base_ret_type == IrType::Void {
            self.builder.ret(None);
        } else if let Some(result) = result {
            self.builder.ret(Some(result));
        }

        self.builder.finish_function();

        // Generate any pending closures and wrappers from this function
        self.generate_pending_closures();
        self.generate_pending_wrappers();

        // Clear module context
        self.current_module = None;
    }

    /// Lower a block
    fn lower_block(&mut self, block: &Block) -> Option<VReg> {
        // Enter a new scope for drop tracking (HARC)
        self.enter_drop_scope();

        // Lower all statements
        for stmt in &block.stmts {
            self.lower_stmt(stmt);
        }

        // Lower final expression if present
        let result = block.expr.as_ref().map(|e| self.lower_expr(e));

        // Leave scope and emit drops for RC variables (HARC)
        self.leave_drop_scope();

        result
    }

    /// Lower a statement
    fn lower_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { pattern, ty, value } => {
                // Check if the value is a struct or array literal
                // These return pointers directly, so we don't need a separate alloca
                // Track if this is a closure binding
                let mut is_closure = false;

                let slot = if let Some(val) = value {
                    match &val.kind {
                        ExprKind::Struct { .. } => {
                            // Struct literal - returns a pointer to heap-allocated struct
                            // We need to store this pointer in a stack slot
                            let ptr = self.lower_expr(val);
                            let slot = self.builder.alloca(IrType::ptr(IrType::I64));
                            self.builder.store(slot, ptr);
                            slot
                        }
                        ExprKind::Array(_) => {
                            // Array literal - use its result directly (stack allocated)
                            self.lower_expr(val)
                        }
                        ExprKind::Closure { .. } => {
                            // Closure - returns pointer to closure struct directly
                            is_closure = true;
                            self.lower_expr(val)
                        }
                        ExprKind::SpawnTask { .. } => {
                            // spawn(future) returns Future pointer directly - mark as ptr_local
                            let var_name = if let PatternKind::Ident { name, .. } = &pattern.kind {
                                Some(name.name.clone())
                            } else {
                                None
                            };
                            if let Some(name) = var_name {
                                self.ptr_locals.insert(name);
                            }
                            self.lower_expr(val)
                        }
                        ExprKind::Call { func, args: _ } => {
                            // Check if this is an enum variant constructor or Box::new
                            let (is_enum, is_box, is_vec, is_string, is_hashmap, is_option_result_method, is_future, is_iter) = if let ExprKind::Path(path) = &func.kind {
                                let full_path = path.segments.iter()
                                    .map(|s| s.ident.name.clone())
                                    .collect::<Vec<_>>()
                                    .join("::");
                                let is_opt_res = matches!(full_path.as_str(),
                                    "Option::map" | "Option::and_then" |
                                    "Result::map" | "Result::map_err" | "Result::and_then" |
                                    "Vec::pop" | "Vec::first" | "Vec::last" |
                                    "HashMap::get" | "HashMap::remove"
                                );
                                let is_hashmap_or_hashset = full_path == "HashMap::new" || full_path == "HashMap::with_capacity" ||
                                    full_path == "HashSet::new" || full_path == "HashSet::with_capacity";
                                // Functions that return iterator structs (MapIter, FilterIter)
                                let is_iter_returning = matches!(full_path.as_str(),
                                    "VecIter::map" | "VecIter::filter" | "VecIter::enumerate"
                                );
                                // Check if this is an async runtime function that returns Future
                                let is_async_runtime = matches!(full_path.as_str(),
                                    "spawn" | "yield_now" | "sleep_ms"
                                );
                                // Check if the function returns Future (from type checker)
                                let returns_future = self.expr_types.get(&val.span).map_or(false, |ty| {
                                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                                        name == "Future"
                                    } else {
                                        false
                                    }
                                });
                                // Functions that return Vec
                                let is_vec_returning = matches!(full_path.as_str(),
                                    "Vec::new" | "Vec::with_capacity" |
                                    "String::split" | "String::lines" | "String::chars" | "String::bytes" |
                                    "String::split_whitespace" |
                                    // VecIter methods that return Vec
                                    "VecIter::collect" | "VecIter::take" | "VecIter::skip" |
                                    "MapIter::collect" | "FilterIter::collect"
                                );
                                // Functions that return String
                                let is_string_returning = matches!(full_path.as_str(),
                                    "String::new" | "String::from" | "read_line" |
                                    "String::concat" | "String::substring" |
                                    "String::trim" | "String::trim_start" | "String::trim_end" |
                                    "String::to_uppercase" | "String::to_lowercase" |
                                    "String::replace" | "String::repeat" | "String::concat_with"
                                );
                                (
                                    self.enum_variants.contains_key(&full_path),
                                    full_path == "Box::new",
                                    is_vec_returning,
                                    is_string_returning,
                                    is_hashmap_or_hashset,
                                    is_opt_res,
                                    is_async_runtime || returns_future,
                                    is_iter_returning
                                )
                            } else {
                                (false, false, false, false, false, false, false, false)
                            };
                            if is_enum {
                                // Enum variant constructor returns pointer directly - mark as ptr_local
                                let var_name = if let PatternKind::Ident { name, .. } = &pattern.kind {
                                    Some(name.name.clone())
                                } else {
                                    None
                                };
                                if let Some(name) = var_name {
                                    self.ptr_locals.insert(name);
                                }
                                self.lower_expr(val)
                            } else if is_box || is_vec || is_string || is_hashmap || is_option_result_method || is_future || is_iter {
                                // Box::new / Vec::new / String::new / HashMap::new / Option::map / Future returns pointer directly - mark as ptr_local
                                let var_name = if let PatternKind::Ident { name, .. } = &pattern.kind {
                                    Some(name.name.clone())
                                } else {
                                    None
                                };
                                if let Some(name) = var_name {
                                    self.ptr_locals.insert(name);
                                }
                                self.lower_expr(val)
                            } else {
                                // Normal function call - allocate and store result
                                // Use inferred type from type checker if no annotation
                                let ir_ty = ty
                                    .as_ref()
                                    .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                                    .unwrap_or_else(|| self.get_expr_ir_type(val));
                                let slot = self.builder.alloca(ir_ty);
                                let val_vreg = self.lower_expr(val);
                                self.builder.store(slot, val_vreg);
                                slot
                            }
                        }
                        ExprKind::Path(path) => {
                            // Check if this is a unit enum variant
                            let full_path = path.segments.iter()
                                .map(|s| s.ident.name.clone())
                                .collect::<Vec<_>>()
                                .join("::");
                            if self.enum_variants.contains_key(&full_path) {
                                // Unit enum variant returns pointer directly - mark as ptr_local
                                let var_name = if let PatternKind::Ident { name, .. } = &pattern.kind {
                                    Some(name.name.clone())
                                } else {
                                    None
                                };
                                if let Some(name) = var_name {
                                    self.ptr_locals.insert(name);
                                }
                                self.lower_expr(val)
                            } else {
                                // Normal variable - allocate and store
                                // Use inferred type from type checker if no annotation
                                let ir_ty = ty
                                    .as_ref()
                                    .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                                    .unwrap_or_else(|| self.get_expr_ir_type(val));
                                let slot = self.builder.alloca(ir_ty);
                                let val_vreg = self.lower_expr(val);
                                self.builder.store(slot, val_vreg);
                                slot
                            }
                        }
                        ExprKind::MacroCall(invocation) => {
                            // Macros like format! and vec! return pointers to stack-allocated structs
                            let macro_name = &invocation.name.name;
                            let is_struct_returning = matches!(macro_name.as_str(),
                                "format" | "vec"
                            );
                            if is_struct_returning {
                                // Mark as ptr_local and return pointer directly
                                let var_name = if let PatternKind::Ident { name, .. } = &pattern.kind {
                                    Some(name.name.clone())
                                } else {
                                    None
                                };
                                if let Some(name) = var_name {
                                    self.ptr_locals.insert(name);
                                }
                                self.lower_expr(val)
                            } else {
                                // Other macros - allocate and store result
                                let ir_ty = ty
                                    .as_ref()
                                    .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                                    .unwrap_or_else(|| self.get_expr_ir_type(val));
                                let slot = self.builder.alloca(ir_ty);
                                let val_vreg = self.lower_expr(val);
                                self.builder.store(slot, val_vreg);
                                slot
                            }
                        }
                        _ => {
                            // Normal value - allocate and store
                            // Use inferred type from type checker if no annotation
                            let ir_ty = ty
                                .as_ref()
                                .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                                .unwrap_or_else(|| self.get_expr_ir_type(val));
                            let slot = self.builder.alloca(ir_ty);
                            let val_vreg = self.lower_expr(val);
                            self.builder.store(slot, val_vreg);
                            slot
                        }
                    }
                } else {
                    // No initializer - just allocate
                    let ir_ty = ty
                        .as_ref()
                        .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                        .unwrap_or(IrType::I64);
                    self.builder.alloca(ir_ty)
                };

                // Bind pattern and track local type
                self.bind_pattern(pattern, slot);

                // Track local type for non-closure bindings
                if let PatternKind::Ident { name, .. } = &pattern.kind {
                    // Get the IR type for this binding
                    let ir_ty = if is_closure {
                        // Store closure type as {i64, i64} (fn_ptr, env_ptr)
                        self.closure_locals.insert(name.name.clone());
                        IrType::Struct(vec![IrType::I64, IrType::I64])
                    } else if let Some(t) = ty {
                        self.ty_to_ir_type(&self.ast_type_to_ty(t))
                    } else if let Some(val) = value {
                        self.get_expr_ir_type(val)
                    } else {
                        IrType::I64
                    };
                    self.local_types.insert(name.name.clone(), ir_ty);
                }

                // HARC: Register variable for automatic drop if it's an RC type
                if let PatternKind::Ident { name, .. } = &pattern.kind {
                    // Get the type name from the type annotation or inferred type
                    let type_name = if let Some(t) = ty {
                        self.ast_type_to_type_name(t)
                    } else if let Some(val) = value {
                        self.get_expr_type_name(val)
                    } else {
                        "unknown".to_string()
                    };

                    // Register for drop tracking
                    self.register_local_for_drop(&name.name, &type_name, stmt.span);
                }
            }
            StmtKind::Expr(expr) => {
                self.lower_expr(expr);
            }
            StmtKind::Item(_) => {
                // Items in blocks not yet handled
            }
        }
    }

    /// Lower an expression
    fn lower_expr(&mut self, expr: &Expr) -> VReg {
        match &expr.kind {
            ExprKind::Literal(lit) => self.lower_literal(lit),

            ExprKind::Path(path) => {
                // Check if this is an enum variant (qualified path like Enum::Variant)
                let full_path = path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::");

                // Check for unit enum variant
                if let Some(variant_info) = self.enum_variants.get(&full_path).cloned() {
                    if variant_info.payload_type.is_none() {
                        // Unit variant - just return discriminant
                        return self.lower_enum_variant_constructor(&full_path, &variant_info, &[]);
                    }
                }

                // Variable reference
                let name = &path.segments[0].ident.name;
                if let Some(&slot) = self.locals.get(name) {
                    // For ptr_locals (Box, closures), return the pointer directly, don't load
                    if self.ptr_locals.contains(name) || self.closure_locals.contains(name) {
                        slot
                    } else {
                        let vreg = self.builder.load(slot);
                        // Track the type of this vreg
                        if let Some(ty) = self.local_types.get(name).cloned() {
                            self.vreg_types.insert(vreg, ty);
                        }
                        vreg
                    }
                } else {
                    // Check if this is a global constant
                    if let Some((global_name, ir_type)) = self.global_consts.get(name).cloned() {
                        // Load value from global constant
                        let global_ptr = self.builder.global_ref(&global_name);
                        let vreg = self.builder.load(global_ptr);
                        self.vreg_types.insert(vreg, ir_type);
                        vreg
                    } else if self.fn_signatures.contains_key(name) {
                        // Check if this is a function reference being used as a value (for closure-like usage)
                        // Schedule a wrapper function that accepts (env_ptr, args...)
                        let wrapper_name = self.schedule_function_wrapper(name);

                        // Create a closure struct wrapping the wrapper function
                        // Closure struct: { fn_ptr (i64), env_ptr (i64) }
                        let closure_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
                        let closure_ptr = self.builder.alloca(closure_ty);

                        // Store wrapper function pointer
                        let fn_ptr_slot = self.builder.get_field_ptr(closure_ptr, 0);
                        let fn_ptr = self.builder.func_ref(&wrapper_name);
                        let fn_as_i64 = self.builder.bitcast(fn_ptr, IrType::I64);
                        self.builder.store(fn_ptr_slot, fn_as_i64);

                        // Store null environment pointer
                        let env_ptr_slot = self.builder.get_field_ptr(closure_ptr, 1);
                        let null_env = self.builder.const_int(0);
                        self.builder.store(env_ptr_slot, null_env);

                        closure_ptr
                    } else {
                        // Unknown variable - emit 0 as placeholder
                        self.builder.const_int(0)
                    }
                }
            }

            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
                // Coerce types if needed for comparison/arithmetic
                let (l, r) = self.coerce_to_common_type(l, r);
                self.lower_binary_op(*op, l, r)
            }

            ExprKind::Unary { op, operand } => {
                let v = self.lower_expr(operand);
                let result = self.lower_unary_op(*op, v);
                // Propagate type from operand to result
                if let Some(ty) = self.vreg_types.get(&v).cloned() {
                    self.vreg_types.insert(result, ty);
                }
                result
            }

            ExprKind::Call { func, args } => {
                // Get function name (use full qualified path)
                let func_name = if let ExprKind::Path(path) = &func.kind {
                    if path.segments.len() == 1 {
                        path.segments[0].ident.name.clone()
                    } else {
                        // Qualified path like Module::func or Enum::Variant
                        path.segments.iter()
                            .map(|s| s.ident.name.clone())
                            .collect::<Vec<_>>()
                            .join("::")
                    }
                } else {
                    "unknown".to_string()
                };

                // Check if this is an enum variant constructor
                if let Some(variant_info) = self.enum_variants.get(&func_name).cloned() {
                    return self.lower_enum_variant_constructor(&func_name, &variant_info, args);
                }

                // Check if this is a closure call (variable in closure_locals)
                if let ExprKind::Path(path) = &func.kind {
                    if path.segments.len() == 1 && self.closure_locals.contains(&func_name) {
                        // Get closure struct pointer from local (closure_slot is already a pointer to {fn_ptr, env_ptr})
                        let closure_slot = self.locals.get(&func_name).copied().unwrap();

                        // Load fn_ptr from field 0 (use closure_slot directly since it's a pointer)
                        let fn_ptr_field = self.builder.get_field_ptr(closure_slot, 0);
                        let fn_ptr = self.builder.load(fn_ptr_field);

                        // Load env_ptr from field 1
                        let env_ptr_field = self.builder.get_field_ptr(closure_slot, 1);
                        let env_ptr = self.builder.load(env_ptr_field);

                        // Lower arguments
                        let arg_vregs: Vec<VReg> = args.iter().map(|a| self.lower_expr(a)).collect();

                        // Call with env_ptr as first argument
                        let mut call_args = vec![env_ptr];
                        call_args.extend(arg_vregs);
                        return self.builder.call_ptr(fn_ptr, call_args);
                    }
                }

                // Handle built-in print functions
                if func_name == "print" || func_name == "println" {
                    return self.lower_print_call(args, func_name == "println");
                }

                // Handle built-in eprint functions (stderr)
                if func_name == "eprint" || func_name == "eprintln" {
                    return self.lower_eprint_call(args, func_name == "eprintln");
                }

                // Handle built-in read_line function (stdin)
                if func_name == "read_line" {
                    return self.lower_read_line();
                }

                // Handle Box::new - heap allocation
                if func_name == "Box::new" {
                    return self.lower_box_new(args);
                }

                // Handle Box::drop / drop - explicit free (normally called by ownership system)
                if func_name == "drop" || func_name == "Box::drop" {
                    if let Some(arg) = args.first() {
                        let ptr = self.lower_expr(arg);
                        self.builder.free(ptr);
                    }
                    return self.builder.const_int(0);
                }

                // Handle Vec functions
                if func_name == "Vec::new" {
                    return self.lower_vec_new();
                }
                if func_name == "Vec::with_capacity" {
                    return self.lower_vec_with_capacity(args);
                }
                if func_name == "Vec::push" {
                    return self.lower_vec_push(args);
                }
                if func_name == "Vec::get" {
                    return self.lower_vec_get(args);
                }
                if func_name == "Vec::len" {
                    return self.lower_vec_len(args);
                }
                if func_name == "Vec::pop" {
                    return self.lower_vec_pop(args);
                }
                if func_name == "Vec::set" {
                    return self.lower_vec_set(args);
                }
                if func_name == "Vec::is_empty" {
                    return self.lower_vec_is_empty(args);
                }
                if func_name == "Vec::capacity" {
                    return self.lower_vec_capacity(args);
                }
                if func_name == "Vec::clear" {
                    return self.lower_vec_clear(args);
                }
                if func_name == "Vec::first" {
                    return self.lower_vec_first(args);
                }
                if func_name == "Vec::last" {
                    return self.lower_vec_last(args);
                }
                if func_name == "Vec::iter" {
                    return self.lower_vec_iter(args);
                }

                // Handle VecIter methods
                if func_name == "VecIter::next" {
                    return self.lower_veciter_next(args);
                }
                if func_name == "VecIter::map" {
                    return self.lower_veciter_map(args);
                }
                if func_name == "VecIter::filter" {
                    return self.lower_veciter_filter(args);
                }
                if func_name == "VecIter::fold" {
                    return self.lower_veciter_fold(args);
                }
                if func_name == "VecIter::collect" {
                    return self.lower_veciter_collect(args);
                }
                if func_name == "VecIter::find" {
                    return self.lower_veciter_find(args);
                }
                if func_name == "VecIter::any" {
                    return self.lower_veciter_any(args);
                }
                if func_name == "VecIter::all" {
                    return self.lower_veciter_all(args);
                }
                if func_name == "VecIter::count" {
                    return self.lower_veciter_count(args);
                }
                if func_name == "VecIter::sum" {
                    return self.lower_veciter_sum(args);
                }
                if func_name == "VecIter::enumerate" {
                    return self.lower_veciter_enumerate(args);
                }
                if func_name == "VecIter::take" {
                    return self.lower_veciter_take(args);
                }
                if func_name == "VecIter::skip" {
                    return self.lower_veciter_skip(args);
                }
                if func_name == "VecIter::for_each" {
                    return self.lower_veciter_for_each(args);
                }

                // Handle MapIter methods
                if func_name == "MapIter::next" {
                    return self.lower_mapiter_next(args);
                }
                if func_name == "MapIter::collect" {
                    return self.lower_mapiter_collect(args);
                }

                // Handle FilterIter methods
                if func_name == "FilterIter::next" {
                    return self.lower_filteriter_next(args);
                }
                if func_name == "FilterIter::collect" {
                    return self.lower_filteriter_collect(args);
                }

                // Handle String functions
                if func_name == "String::new" {
                    return self.lower_string_new();
                }
                if func_name == "String::from" {
                    return self.lower_string_from(args);
                }
                if func_name == "String::len" {
                    return self.lower_string_len(args);
                }
                if func_name == "String::push" {
                    return self.lower_string_push(args);
                }
                if func_name == "String::is_empty" {
                    return self.lower_string_is_empty(args);
                }
                if func_name == "String::clear" {
                    return self.lower_string_clear(args);
                }
                if func_name == "String::char_at" {
                    return self.lower_string_char_at(args);
                }
                if func_name == "String::push_str" {
                    return self.lower_string_push_str(args);
                }
                if func_name == "String::concat" {
                    return self.lower_string_concat(args);
                }
                if func_name == "String::substring" {
                    return self.lower_string_substring(args);
                }
                if func_name == "String::capacity" {
                    return self.lower_string_capacity(args);
                }
                if func_name == "String::contains" {
                    return self.lower_string_contains(args);
                }
                if func_name == "String::starts_with" {
                    return self.lower_string_starts_with(args);
                }
                if func_name == "String::ends_with" {
                    return self.lower_string_ends_with(args);
                }
                if func_name == "String::find" {
                    return self.lower_string_find(args);
                }
                if func_name == "String::rfind" {
                    return self.lower_string_rfind(args);
                }
                if func_name == "String::to_uppercase" {
                    return self.lower_string_to_uppercase(args);
                }
                if func_name == "String::to_lowercase" {
                    return self.lower_string_to_lowercase(args);
                }
                if func_name == "String::trim" {
                    return self.lower_string_trim(args);
                }
                if func_name == "String::trim_start" {
                    return self.lower_string_trim_start(args);
                }
                if func_name == "String::trim_end" {
                    return self.lower_string_trim_end(args);
                }
                if func_name == "String::split" {
                    return self.lower_string_split(args);
                }
                if func_name == "String::lines" {
                    return self.lower_string_lines(args);
                }
                if func_name == "String::replace" {
                    return self.lower_string_replace(args);
                }
                if func_name == "String::repeat" {
                    return self.lower_string_repeat(args);
                }
                if func_name == "String::chars" {
                    return self.lower_string_chars(args);
                }
                if func_name == "String::bytes" {
                    return self.lower_string_bytes(args);
                }
                if func_name == "String::split_whitespace" {
                    return self.lower_string_split_whitespace(args);
                }
                if func_name == "String::concat_with" {
                    return self.lower_string_concat_with(args);
                }

                // Handle Option functions
                if func_name == "Option::is_some" {
                    return self.lower_option_is_some(args);
                }
                if func_name == "Option::is_none" {
                    return self.lower_option_is_none(args);
                }
                if func_name == "Option::unwrap" {
                    return self.lower_option_unwrap(args);
                }
                if func_name == "Option::unwrap_or" {
                    return self.lower_option_unwrap_or(args);
                }
                if func_name == "Option::expect" {
                    return self.lower_option_expect(args);
                }
                if func_name == "Option::map" {
                    return self.lower_option_map(args);
                }
                if func_name == "Option::and_then" {
                    return self.lower_option_and_then(args);
                }

                // Handle Result functions
                if func_name == "Result::is_ok" {
                    return self.lower_result_is_ok(args);
                }
                if func_name == "Result::is_err" {
                    return self.lower_result_is_err(args);
                }
                if func_name == "Result::unwrap" {
                    return self.lower_result_unwrap(args);
                }
                if func_name == "Result::unwrap_err" {
                    return self.lower_result_unwrap_err(args);
                }
                if func_name == "Result::unwrap_or" {
                    return self.lower_result_unwrap_or(args);
                }
                if func_name == "Result::expect" {
                    return self.lower_result_expect(args);
                }
                if func_name == "Result::map" {
                    return self.lower_result_map(args);
                }
                if func_name == "Result::map_err" {
                    return self.lower_result_map_err(args);
                }
                if func_name == "Result::and_then" {
                    return self.lower_result_and_then(args);
                }

                // Handle integer/bool to_string functions
                if func_name == "i64::to_string" {
                    return self.lower_int_to_string(args, "i64");
                }
                if func_name == "i32::to_string" {
                    return self.lower_int_to_string(args, "i32");
                }
                if func_name == "bool::to_string" {
                    return self.lower_bool_to_string(args);
                }

                // Handle parse functions (String -> integer)
                if func_name == "i64::parse" {
                    return self.lower_parse_int(args, "i64");
                }
                if func_name == "i32::parse" {
                    return self.lower_parse_int(args, "i32");
                }
                if func_name == "f64::parse" {
                    return self.lower_parse_float(args, "f64");
                }
                if func_name == "f32::parse" {
                    return self.lower_parse_float(args, "f32");
                }
                if func_name == "bool::parse" {
                    return self.lower_parse_bool(args);
                }

                // ============ Math Functions ============
                // Genesis Math Module - comprehensive math library via libm
                // See docs/math.md for complete API documentation
                //
                // Categories:
                // - Constants: PI, E, INFINITY, NEG_INFINITY, NAN
                // - Basic: abs, min, max
                // - Powers: sqrt, cbrt, pow, hypot
                // - Trig: sin, cos, tan, asin, acos, atan, atan2
                // - Hyperbolic: sinh, cosh, tanh
                // - Exp/Log: exp, exp2, ln, log2, log10, log
                // - Rounding: floor, ceil, round, trunc
                // - Checks: is_nan, is_infinite, is_finite
                //
                // Implementation: Functions call libm via declare_math() in builder.rs

                // === f64 Basic Functions ===
                if func_name == "f64::abs" {
                    return self.lower_math_f64_unary(args, "fabs");
                }
                if func_name == "f64::sqrt" {
                    return self.lower_math_f64_unary(args, "sqrt");
                }
                if func_name == "f64::cbrt" {
                    return self.lower_math_f64_unary(args, "cbrt");
                }
                if func_name == "f64::sin" {
                    return self.lower_math_f64_unary(args, "sin");
                }
                if func_name == "f64::cos" {
                    return self.lower_math_f64_unary(args, "cos");
                }
                if func_name == "f64::tan" {
                    return self.lower_math_f64_unary(args, "tan");
                }
                if func_name == "f64::asin" {
                    return self.lower_math_f64_unary(args, "asin");
                }
                if func_name == "f64::acos" {
                    return self.lower_math_f64_unary(args, "acos");
                }
                if func_name == "f64::atan" {
                    return self.lower_math_f64_unary(args, "atan");
                }
                if func_name == "f64::sinh" {
                    return self.lower_math_f64_unary(args, "sinh");
                }
                if func_name == "f64::cosh" {
                    return self.lower_math_f64_unary(args, "cosh");
                }
                if func_name == "f64::tanh" {
                    return self.lower_math_f64_unary(args, "tanh");
                }
                if func_name == "f64::exp" {
                    return self.lower_math_f64_unary(args, "exp");
                }
                if func_name == "f64::exp2" {
                    return self.lower_math_f64_unary(args, "exp2");
                }
                if func_name == "f64::ln" {
                    return self.lower_math_f64_unary(args, "log");
                }
                if func_name == "f64::log2" {
                    return self.lower_math_f64_unary(args, "log2");
                }
                if func_name == "f64::log10" {
                    return self.lower_math_f64_unary(args, "log10");
                }
                if func_name == "f64::floor" {
                    return self.lower_math_f64_unary(args, "floor");
                }
                if func_name == "f64::ceil" {
                    return self.lower_math_f64_unary(args, "ceil");
                }
                if func_name == "f64::round" {
                    return self.lower_math_f64_unary(args, "round");
                }
                if func_name == "f64::trunc" {
                    return self.lower_math_f64_unary(args, "trunc");
                }
                if func_name == "f64::pow" {
                    return self.lower_math_f64_binary(args, "pow");
                }
                if func_name == "f64::atan2" {
                    return self.lower_math_f64_binary(args, "atan2");
                }
                if func_name == "f64::hypot" {
                    return self.lower_math_f64_binary(args, "hypot");
                }
                if func_name == "f64::min" {
                    return self.lower_math_f64_binary(args, "fmin");
                }
                if func_name == "f64::max" {
                    return self.lower_math_f64_binary(args, "fmax");
                }
                if func_name == "f64::fmod" {
                    return self.lower_math_f64_binary(args, "fmod");
                }
                if func_name == "f64::copysign" {
                    return self.lower_math_f64_binary(args, "copysign");
                }
                if func_name == "f64::log" {
                    return self.lower_math_log_base(args);
                }
                if func_name == "f64::is_nan" {
                    return self.lower_math_is_nan(args);
                }
                if func_name == "f64::is_infinite" {
                    return self.lower_math_is_infinite(args);
                }
                if func_name == "f64::is_finite" {
                    return self.lower_math_is_finite(args);
                }
                if func_name == "f64::PI" {
                    return self.lower_math_const_pi();
                }
                if func_name == "f64::E" {
                    return self.lower_math_const_e();
                }
                if func_name == "f64::INFINITY" {
                    return self.lower_math_const_infinity();
                }
                if func_name == "f64::NEG_INFINITY" {
                    return self.lower_math_const_neg_infinity();
                }
                if func_name == "f64::NAN" {
                    return self.lower_math_const_nan();
                }

                // ============ Math Functions (f32) ============
                if func_name == "f32::abs" {
                    return self.lower_math_f32_unary(args, "fabsf");
                }
                if func_name == "f32::sqrt" {
                    return self.lower_math_f32_unary(args, "sqrtf");
                }
                if func_name == "f32::sin" {
                    return self.lower_math_f32_unary(args, "sinf");
                }
                if func_name == "f32::cos" {
                    return self.lower_math_f32_unary(args, "cosf");
                }
                if func_name == "f32::tan" {
                    return self.lower_math_f32_unary(args, "tanf");
                }
                if func_name == "f32::exp" {
                    return self.lower_math_f32_unary(args, "expf");
                }
                if func_name == "f32::ln" {
                    return self.lower_math_f32_unary(args, "logf");
                }
                if func_name == "f32::floor" {
                    return self.lower_math_f32_unary(args, "floorf");
                }
                if func_name == "f32::ceil" {
                    return self.lower_math_f32_unary(args, "ceilf");
                }
                if func_name == "f32::round" {
                    return self.lower_math_f32_unary(args, "roundf");
                }
                if func_name == "f32::trunc" {
                    return self.lower_math_f32_unary(args, "truncf");
                }
                if func_name == "f32::pow" {
                    return self.lower_math_f32_binary(args, "powf");
                }
                if func_name == "f32::min" {
                    return self.lower_math_f32_binary(args, "fminf");
                }
                if func_name == "f32::max" {
                    return self.lower_math_f32_binary(args, "fmaxf");
                }

                // ============ Math Functions (i64) ============
                if func_name == "i64::abs" {
                    return self.lower_math_i64_abs(args);
                }
                if func_name == "i64::min" {
                    return self.lower_math_i64_min(args);
                }
                if func_name == "i64::max" {
                    return self.lower_math_i64_max(args);
                }
                if func_name == "i64::pow" {
                    return self.lower_math_i64_pow(args);
                }

                // ============ Math Functions (i32) ============
                if func_name == "i32::abs" {
                    return self.lower_math_i32_abs(args);
                }
                if func_name == "i32::min" {
                    return self.lower_math_i32_min(args);
                }
                if func_name == "i32::max" {
                    return self.lower_math_i32_max(args);
                }

                // Handle File I/O methods
                if func_name == "File::open" {
                    return self.lower_file_open(args, "r");
                }
                if func_name == "File::create" {
                    return self.lower_file_open(args, "w");
                }
                if func_name == "File::read_to_string" {
                    return self.lower_file_read_to_string(args);
                }
                if func_name == "File::write_string" {
                    return self.lower_file_write_string(args);
                }
                if func_name == "File::close" {
                    return self.lower_file_close(args);
                }

                // Handle extended filesystem operations
                if func_name == "File::exists" {
                    return self.lower_file_exists(args);
                }
                if func_name == "File::size" {
                    return self.lower_file_size(args);
                }
                if func_name == "File::is_file" {
                    return self.lower_file_is_file(args);
                }
                if func_name == "File::is_dir" {
                    return self.lower_file_is_dir(args);
                }
                if func_name == "File::remove" {
                    return self.lower_file_remove(args);
                }

                // Handle directory operations
                if func_name == "Dir::create" {
                    return self.lower_dir_create(args);
                }
                if func_name == "Dir::create_all" {
                    return self.lower_dir_create_all(args);
                }
                if func_name == "Dir::remove" {
                    return self.lower_dir_remove(args);
                }
                if func_name == "Dir::list" {
                    return self.lower_dir_list(args);
                }

                // Handle path operations (Fs:: prefix to avoid conflict with macro $x:path)
                if func_name == "Fs::path_join" {
                    return self.lower_path_join(args);
                }
                if func_name == "Fs::path_parent" {
                    return self.lower_path_parent(args);
                }
                if func_name == "Fs::path_filename" {
                    return self.lower_path_filename(args);
                }
                if func_name == "Fs::path_extension" {
                    return self.lower_path_extension(args);
                }

                // Handle HashMap methods
                if func_name == "HashMap::new" {
                    return self.lower_hashmap_new();
                }
                if func_name == "HashMap::with_capacity" {
                    return self.lower_hashmap_with_capacity(args);
                }
                if func_name == "HashMap::insert" {
                    return self.lower_hashmap_insert(args);
                }
                if func_name == "HashMap::get" {
                    return self.lower_hashmap_get(args);
                }
                if func_name == "HashMap::contains_key" {
                    return self.lower_hashmap_contains_key(args);
                }
                if func_name == "HashMap::remove" {
                    return self.lower_hashmap_remove(args);
                }
                if func_name == "HashMap::len" {
                    return self.lower_hashmap_len(args);
                }
                if func_name == "HashMap::is_empty" {
                    return self.lower_hashmap_is_empty(args);
                }
                if func_name == "HashMap::clear" {
                    return self.lower_hashmap_clear(args);
                }
                if func_name == "HashMap::capacity" {
                    return self.lower_hashmap_capacity(args);
                }

                // Handle HashSet functions
                if func_name == "HashSet::new" {
                    return self.lower_hashset_new();
                }
                if func_name == "HashSet::with_capacity" {
                    return self.lower_hashset_with_capacity(args);
                }
                if func_name == "HashSet::insert" {
                    return self.lower_hashset_insert(args);
                }
                if func_name == "HashSet::contains" {
                    return self.lower_hashset_contains(args);
                }
                if func_name == "HashSet::remove" {
                    return self.lower_hashset_remove(args);
                }
                if func_name == "HashSet::len" {
                    return self.lower_hashset_len(args);
                }
                if func_name == "HashSet::is_empty" {
                    return self.lower_hashset_is_empty(args);
                }
                if func_name == "HashSet::clear" {
                    return self.lower_hashset_clear(args);
                }
                if func_name == "HashSet::capacity" {
                    return self.lower_hashset_capacity(args);
                }

                // Handle Future functions
                if func_name == "Future::poll" {
                    return self.lower_future_poll(args);
                }
                if func_name == "Future::is_ready" {
                    return self.lower_future_is_ready(args);
                }
                if func_name == "Future::get" {
                    return self.lower_future_get(args);
                }

                // Handle async runtime functions
                if func_name == "block_on" {
                    return self.lower_block_on(args);
                }
                if func_name == "spawn" {
                    return self.lower_spawn(args);
                }
                if func_name == "yield_now" {
                    return self.lower_yield_now();
                }
                if func_name == "sleep_ms" {
                    return self.lower_sleep_ms(args);
                }
                if func_name == "pending" {
                    return self.lower_pending();
                }
                if func_name == "wake" {
                    return self.lower_wake(args);
                }

                // ============ Time Module Functions ============
                if func_name == "time::now_ms" {
                    return self.lower_time_now_ms();
                }
                if func_name == "time::now_us" {
                    return self.lower_time_now_us();
                }
                if func_name == "time::now_ns" {
                    return self.lower_time_now_ns();
                }
                if func_name == "time::elapsed_ms" {
                    return self.lower_time_elapsed_ms(args);
                }
                if func_name == "time::elapsed_us" {
                    return self.lower_time_elapsed_us(args);
                }
                if func_name == "time::elapsed_ns" {
                    return self.lower_time_elapsed_ns(args);
                }

                // ============ Duration Functions ============
                if func_name == "Duration::from_secs" {
                    return self.lower_duration_from_secs(args);
                }
                if func_name == "Duration::from_millis" {
                    return self.lower_duration_from_millis(args);
                }
                if func_name == "Duration::from_micros" {
                    return self.lower_duration_from_micros(args);
                }
                if func_name == "Duration::from_nanos" {
                    return self.lower_duration_from_nanos(args);
                }
                if func_name == "Duration::as_secs" {
                    return self.lower_duration_as_secs(args);
                }
                if func_name == "Duration::as_millis" {
                    return self.lower_duration_as_millis(args);
                }
                if func_name == "Duration::as_micros" {
                    return self.lower_duration_as_micros(args);
                }
                if func_name == "Duration::as_nanos" {
                    return self.lower_duration_as_nanos(args);
                }
                if func_name == "Duration::add" {
                    return self.lower_duration_add(args);
                }
                if func_name == "Duration::sub" {
                    return self.lower_duration_sub(args);
                }

                // ============ Random Module Functions ============
                if func_name == "random::seed" {
                    return self.lower_random_seed(args);
                }
                if func_name == "random::next_i64" {
                    return self.lower_random_next_i64();
                }
                if func_name == "random::next_f64" {
                    return self.lower_random_next_f64();
                }
                if func_name == "random::range" {
                    return self.lower_random_range(args);
                }
                if func_name == "random::coin" {
                    return self.lower_random_coin();
                }

                // Handle channel functions (Phase 6)
                if func_name == "channel" {
                    return self.lower_channel_create(args);
                }
                if func_name == "Channel::sender" {
                    return self.lower_channel_get_sender(args);
                }
                if func_name == "Channel::receiver" {
                    return self.lower_channel_get_receiver(args);
                }
                if func_name == "Sender::send" {
                    return self.lower_channel_send(args);
                }
                if func_name == "Receiver::recv" {
                    return self.lower_channel_recv(args);
                }
                if func_name == "Sender::try_send" {
                    return self.lower_channel_try_send(args);
                }
                if func_name == "Receiver::try_recv" {
                    return self.lower_channel_try_recv(args);
                }
                if func_name == "Sender::is_closed" {
                    return self.lower_sender_is_closed(args);
                }
                if func_name == "Receiver::is_closed" {
                    return self.lower_receiver_is_closed(args);
                }

                // Handle TCP I/O functions (Phase 4.1)
                if func_name == "TcpListener::bind" {
                    return self.lower_tcp_listener_bind(args);
                }
                if func_name == "TcpListener::accept" {
                    return self.lower_tcp_listener_accept(args);
                }
                if func_name == "TcpListener::close" {
                    return self.lower_tcp_listener_close(args);
                }
                if func_name == "TcpStream::read_string" {
                    return self.lower_tcp_stream_read_string(args);
                }
                if func_name == "TcpStream::write_string" {
                    return self.lower_tcp_stream_write_string(args);
                }
                if func_name == "TcpStream::close" {
                    return self.lower_tcp_stream_close(args);
                }

                // Qualify unqualified function calls for module context
                let qualified_name = if !func_name.contains("::") {
                    if let Some(ref module) = self.current_module {
                        format!("{}::{}", module, func_name)
                    } else {
                        func_name
                    }
                } else {
                    func_name
                };

                // Check if this is a generic function call - use mangled name
                let final_name = if let Some((fn_name, type_args)) = self.generic_fn_calls.get(&expr.span).cloned() {
                    // This is a call to a generic function - use mangled name
                    MonomorphCollector::mangle(&fn_name, &type_args)
                } else {
                    qualified_name.clone()
                };

                // Lower arguments
                let arg_vregs: Vec<VReg> = args.iter().map(|a| self.lower_expr(a)).collect();

                // Coerce arguments to match function signature
                let arg_vregs = if let Some(param_types) = self.fn_signatures.get(&final_name).cloned() {
                    arg_vregs.into_iter().zip(param_types.iter())
                        .map(|(arg, expected_ty)| {
                            if let Some(arg_ty) = self.vreg_types.get(&arg).cloned() {
                                if arg_ty != *expected_ty {
                                    self.coerce_arg(arg, &arg_ty, expected_ty)
                                } else {
                                    arg
                                }
                            } else {
                                arg
                            }
                        })
                        .collect()
                } else {
                    arg_vregs
                };

                self.builder.call(&final_name, arg_vregs)
            }

            ExprKind::If { condition, then_branch, else_branch } => {
                self.lower_if(condition, then_branch, else_branch.as_deref())
            }

            ExprKind::Block(block) => {
                self.lower_block(block).unwrap_or_else(|| self.builder.const_int(0))
            }

            ExprKind::Return { value } => {
                if let Some(val) = value {
                    let v = self.lower_expr(val);
                    self.builder.ret(Some(v));
                } else {
                    self.builder.ret(None);
                }
                self.builder.const_int(0) // Unreachable
            }

            ExprKind::Assign { target, value } => {
                let val = self.lower_expr(value);
                if let ExprKind::Path(path) = &target.kind {
                    let name = &path.segments[0].ident.name;
                    if let Some(&slot) = self.locals.get(name) {
                        self.builder.store(slot, val);
                    }
                }
                self.builder.const_int(0) // Unit
            }

            ExprKind::While { condition, body, label } => {
                self.lower_while(condition, body, label.as_ref().map(|l| l.name.clone()))
            }

            ExprKind::Loop { body, label } => {
                self.lower_loop(body, label.as_ref().map(|l| l.name.clone()))
            }

            ExprKind::Break { label, value } => {
                // Find the target loop context
                let target_ctx = if let Some(lbl) = label {
                    // Find loop with matching label
                    self.loop_stack.iter().rev()
                        .find(|ctx| ctx.label.as_ref() == Some(&lbl.name))
                        .cloned()
                } else {
                    // Use innermost loop
                    self.loop_stack.last().cloned()
                };

                if let Some(ctx) = target_ctx {
                    // If break has a value, lower it (though we don't use it yet)
                    if let Some(val_expr) = value {
                        let _ = self.lower_expr(val_expr);
                    }
                    // Jump to exit block
                    self.builder.br(ctx.exit_block);
                    // Create a new block for any unreachable code after break
                    let unreachable_block = self.builder.create_block();
                    self.builder.start_block(unreachable_block);
                }
                self.builder.const_int(0)
            }

            ExprKind::Continue { label } => {
                // Find the target loop context
                let target_ctx = if let Some(lbl) = label {
                    // Find loop with matching label
                    self.loop_stack.iter().rev()
                        .find(|ctx| ctx.label.as_ref() == Some(&lbl.name))
                        .cloned()
                } else {
                    // Use innermost loop
                    self.loop_stack.last().cloned()
                };

                if let Some(ctx) = target_ctx {
                    // Jump to continue block (condition/next iteration)
                    self.builder.br(ctx.continue_block);
                    // Create a new block for any unreachable code after continue
                    let unreachable_block = self.builder.create_block();
                    self.builder.start_block(unreachable_block);
                }
                self.builder.const_int(0)
            }

            ExprKind::Struct { path: _, fields } => {
                // Allocate space for the struct on the heap
                // (heap allocation ensures the struct survives function returns)
                // Infer field types from the initializer expressions
                let field_types: Vec<IrType> = fields.iter()
                    .map(|(_, value)| self.infer_expr_type(value))
                    .collect();
                let struct_ty = IrType::Struct(field_types);
                let struct_ptr = self.builder.malloc(struct_ty);

                // Initialize each field
                for (i, (_name, value)) in fields.iter().enumerate() {
                    let val = self.lower_expr(value);
                    let field_ptr = self.builder.get_field_ptr(struct_ptr, i as u32);
                    self.builder.store(field_ptr, val);
                }

                struct_ptr
            }

            ExprKind::MethodCall { receiver, method, args } => {
                // Get the receiver type to construct qualified method name
                let type_name = self.expr_types.get(&receiver.span).map(|ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        // Resolve "Self" to the current impl type
                        if name == "Self" {
                            self.current_impl_type.clone().unwrap_or_else(|| name.clone())
                        } else {
                            name.clone()
                        }
                    } else {
                        "unknown".to_string()
                    }
                }).unwrap_or_else(|| "unknown".to_string());

                // Construct qualified method name: "Counter::get_value"
                let qualified_method = format!("{}::{}", type_name, method.name);

                // Lower the receiver - for user structs this gives us the struct pointer
                let receiver_val = self.lower_expr(receiver);

                // Lower all other arguments
                let arg_vregs: Vec<VReg> = args.iter().map(|a| self.lower_expr(a)).collect();

                // Prepend receiver to arguments
                let mut all_args = vec![receiver_val];
                all_args.extend(arg_vregs);

                // Call the method function with qualified name
                self.builder.call(&qualified_method, all_args)
            }

            ExprKind::Field { object, field } => {
                // Check if object is a heap-allocated user struct
                let is_heap_struct = self.expr_types.get(&object.span).map_or(false, |ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        // Resolve "Self" to the current impl type
                        let resolved_name = if name == "Self" {
                            self.current_impl_type.as_ref().unwrap_or(name)
                        } else {
                            name
                        };
                        // User-defined structs and built-in heap types
                        self.struct_types.contains_key(resolved_name) ||
                        resolved_name == "Option" || resolved_name == "Result" || resolved_name == "String" ||
                        resolved_name == "Vec" || resolved_name == "HashMap" || resolved_name == "HashSet"
                    } else {
                        false
                    }
                });

                let struct_ptr = if is_heap_struct {
                    // For heap-allocated structs, load the pointer from the slot
                    // and convert i64 to pointer (opaque pointer handling)
                    let slot = self.lower_expr_place(object);
                    let ptr_val = self.builder.load(slot);
                    self.builder.inttoptr(ptr_val, IrType::Ptr(Box::new(IrType::I64)))
                } else {
                    // For stack-allocated structs, use the slot directly
                    self.lower_expr_place(object)
                };

                // Find field index
                let field_idx = self.get_field_index(object, &field.name);
                let field_ptr = self.builder.get_field_ptr(struct_ptr, field_idx);
                self.builder.load(field_ptr)
            }

            ExprKind::Array(elements) => {
                // Allocate array
                let len = elements.len();
                let arr_ty = IrType::Array(Box::new(IrType::I64), len);
                let arr_ptr = self.builder.alloca(arr_ty);

                // Initialize each element
                for (i, elem) in elements.iter().enumerate() {
                    let val = self.lower_expr(elem);
                    let idx = self.builder.const_int(i as i64);
                    let elem_ptr = self.builder.get_element_ptr(arr_ptr, idx);
                    self.builder.store(elem_ptr, val);
                }

                arr_ptr
            }

            ExprKind::Index { object, index } => {
                // Check if object is Vec or String type
                let is_vec = self.expr_types.get(&object.span).map_or(false, |ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        name == "Vec"
                    } else {
                        false
                    }
                });
                let is_string = self.expr_types.get(&object.span).map_or(false, |ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        name == "String"
                    } else {
                        false
                    }
                });

                let idx = self.lower_expr(index);

                if is_vec || is_string {
                    // Vec/String: struct { *T ptr, i64 len, i64 cap }
                    // Get the struct pointer, then load the data pointer from field 0
                    let struct_ptr = self.lower_expr(object);
                    let data_ptr_ptr = self.builder.get_field_ptr(struct_ptr, 0);
                    let data_ptr = self.builder.load(data_ptr_ptr);
                    
                    let data_ptr = if is_string {
                        self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I8)))
                    } else {
                        self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)))
                    };

                    if is_string {
                        // String data is i8, use byte operations
                        let elem_ptr = self.builder.get_byte_ptr(data_ptr, idx);
                        let elem = self.builder.load_byte(elem_ptr);
                        // Zero-extend to i64 for consistency
                        self.builder.zext(elem, IrType::I64)
                    } else {
                        // Vec: use regular element pointer
                        let elem_ptr = self.builder.get_element_ptr(data_ptr, idx);
                        self.builder.load(elem_ptr)
                    }
                } else {
                    // Regular array: get pointer to array and index directly
                    let arr_ptr = self.lower_expr_place(object);
                    let elem_ptr = self.builder.get_element_ptr(arr_ptr, idx);
                    self.builder.load(elem_ptr)
                }
            }

            ExprKind::For { pattern, iterable, body, label } => {
                // For now, only support range expressions: for i in start..end
                self.lower_for_range(pattern, iterable, body, label.as_ref().map(|l| l.name.clone()))
            }

            ExprKind::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms)
            }

            ExprKind::Closure { params, body } => {
                self.lower_closure(params, body)
            }

            ExprKind::Deref { operand } => {
                // Dereference a pointer (like *boxed for Box<T>)
                let ptr = self.lower_expr(operand);
                self.builder.load(ptr)
            }

            ExprKind::Await { operand } => {
                // Await a Future<T> - handles NotStarted, Ready, and Pending states
                // Future struct: { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
                // state: 0 = NotStarted, 1 = Ready, 2 = Pending
                let future_ptr = self.lower_expr(operand);

                // Create blocks for control flow
                let check_state_block = self.builder.create_block();
                let not_started_block = self.builder.create_block();
                let ready_block = self.builder.create_block();
                let pending_block = self.builder.create_block();
                let done_block = self.builder.create_block();

                // Alloca for result (needed for phi-like behavior)
                let result_slot = self.builder.alloca(IrType::I64);

                // Jump to state check
                self.builder.br(check_state_block);

                // Check state
                self.builder.start_block(check_state_block);
                let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
                let state = self.builder.load(state_ptr);

                // Check if ready (state == 1)
                let one = self.builder.const_int(1);
                let is_ready = self.builder.icmp(CmpOp::Eq, state, one);

                let check_pending_block = self.builder.create_block();
                self.builder.cond_br(is_ready, ready_block, check_pending_block);

                // Check if pending (state == 2)
                self.builder.start_block(check_pending_block);
                let two = self.builder.const_int(2);
                let is_pending = self.builder.icmp(CmpOp::Eq, state, two);
                self.builder.cond_br(is_pending, pending_block, not_started_block);

                // Pending block: the inner future is still pending
                // Return pending marker to suspend the outer async function
                self.builder.start_block(pending_block);
                let pending_marker = self.builder.const_int(i64::MIN);
                self.builder.store(result_slot, pending_marker);
                self.builder.br(done_block);

                // Not started block: execute the closure and store result
                self.builder.start_block(not_started_block);

                // Load fn_ptr and env_ptr from Future (as i64 values)
                let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
                let fn_ptr_i64 = self.builder.load(fn_ptr_field);

                // Check if fn_ptr == -1 (sleep marker - no closure to call)
                let neg_one = self.builder.const_int(-1);
                let is_sleep = self.builder.icmp(CmpOp::Eq, fn_ptr_i64, neg_one);

                let call_closure_block = self.builder.create_block();
                let sleep_not_started_block = self.builder.create_block();
                self.builder.cond_br(is_sleep, sleep_not_started_block, call_closure_block);

                // Sleep future in NotStarted - should not happen normally, return pending
                self.builder.start_block(sleep_not_started_block);
                let pending_marker2 = self.builder.const_int(i64::MIN);
                self.builder.store(result_slot, pending_marker2);
                self.builder.br(done_block);

                // Call the closure
                self.builder.start_block(call_closure_block);
                let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
                let env_ptr_i64 = self.builder.load(env_ptr_field);

                // Call the closure: fn_ptr(env_ptr) -> result
                let call_result = self.builder.call_ptr(fn_ptr_i64, vec![env_ptr_i64]);

                // Check if call result is pending marker
                let call_is_pending = self.builder.icmp(CmpOp::Eq, call_result, pending_marker);
                let store_result_block = self.builder.create_block();
                let set_pending_block = self.builder.create_block();
                self.builder.cond_br(call_is_pending, set_pending_block, store_result_block);

                // Call returned pending - set state to Pending and return pending
                self.builder.start_block(set_pending_block);
                self.builder.store(state_ptr, two);
                self.builder.store(result_slot, call_result);
                self.builder.br(done_block);

                // Call succeeded - store result and set ready
                self.builder.start_block(store_result_block);
                let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
                self.builder.store(value_ptr, call_result);
                self.builder.store(state_ptr, one);
                self.builder.store(result_slot, call_result);
                self.builder.br(done_block);

                // Ready block: value is already computed
                self.builder.start_block(ready_block);
                let value_ptr2 = self.builder.get_field_ptr(future_ptr, 3);
                let existing_value = self.builder.load(value_ptr2);
                self.builder.store(result_slot, existing_value);
                self.builder.br(done_block);

                // Done block: return the result
                self.builder.start_block(done_block);
                self.builder.load(result_slot)
            }

            ExprKind::SpawnTask { future } => {
                // spawn(future_expr) - spawn a task for concurrent execution
                // This just calls lower_spawn with the future expression
                self.lower_spawn(&[(**future).clone()])
            }

            ExprKind::Select { arms } => {
                // select! { a = f1 => body1, b = f2 => body2, ... }
                // Poll all futures, execute body of first to complete
                self.lower_select(arms)
            }

            ExprKind::Join { futures } => {
                // join!(f1, f2, f3, ...)
                // Poll all futures, wait for all to complete, return tuple
                self.lower_join(futures)
            }

            ExprKind::MacroCall(invocation) => {
                // Handle built-in macros
                let macro_name = &invocation.name.name;
                match macro_name.as_str() {
                    "vec" => {
                        // vec![] or vec![elem; count] or vec![e1, e2, ...]
                        // Create a new Vec
                        self.lower_builtin_vec_macro(&invocation.tokens)
                    }
                    "println" | "print" => {
                        // println!("...") or println!("{}", arg)
                        self.lower_builtin_print_macro(&invocation.tokens, macro_name == "println")
                    }
                    "eprintln" | "eprint" => {
                        // Same as println but to stderr
                        self.lower_builtin_eprint_macro(&invocation.tokens, macro_name == "eprintln")
                    }
                    "format" => {
                        // format!("...") - returns String
                        self.lower_builtin_format_macro(&invocation.tokens)
                    }
                    "panic" => {
                        // panic!("message") - aborts the program
                        self.lower_builtin_panic_macro(&invocation.tokens)
                    }
                    "todo" | "unreachable" => {
                        // todo!() / unreachable!() - placeholder that aborts
                        self.lower_builtin_panic_macro(&invocation.tokens)
                    }
                    "assert" => {
                        // assert!(condition) or assert!(condition, "message")
                        self.lower_builtin_assert_macro(&invocation.tokens, false)
                    }
                    "assert_eq" => {
                        // assert_eq!(a, b)
                        self.lower_builtin_assert_eq_macro(&invocation.tokens, true)
                    }
                    "assert_ne" => {
                        // assert_ne!(a, b)
                        self.lower_builtin_assert_eq_macro(&invocation.tokens, false)
                    }
                    "debug_assert" => {
                        // Like assert but only in debug builds (for now, same as assert)
                        self.lower_builtin_assert_macro(&invocation.tokens, false)
                    }
                    _ => {
                        // User-defined macro - try to expand it (with recursive expansion)
                        if self.macro_expander.has_macro(macro_name) {
                            match self.macro_expander.expand_recursive(invocation) {
                                Ok(expanded_tokens) => {
                                    // Try to convert tokens directly to an expression
                                    if let Some(expanded_expr) = crate::macro_expand::tokens_to_expr(&expanded_tokens, expr.span) {
                                        self.lower_expr(&expanded_expr)
                                    } else {
                                        // Tokens couldn't be converted - try re-parsing
                                        let source = crate::macro_expand::tokens_to_source(&expanded_tokens);
                                        if let Some(parsed_expr) = self.parse_expr_from_source(&source) {
                                            self.lower_expr(&parsed_expr)
                                        } else {
                                            // Parsing failed - emit error and return 0
                                            eprintln!("error: failed to parse macro expansion for `{}!`", macro_name);
                                            eprintln!("  --> expanded to: {}", source.chars().take(100).collect::<String>());
                                            self.builder.const_int(0)
                                        }
                                    }
                                }
                                Err(e) => {
                                    // Expansion failed - emit error and return 0
                                    eprintln!("error: macro expansion failed for `{}!`: {}", macro_name, e.message());
                                    self.builder.const_int(0)
                                }
                            }
                        } else {
                            // Unknown macro - emit error and return 0
                            eprintln!("error: unknown macro `{}!`", macro_name);
                            self.builder.const_int(0)
                        }
                    }
                }
            }

            // Other expression kinds
            _ => {
                // Placeholder for unimplemented expressions
                self.builder.const_int(0)
            }
        }
    }

    /// Parse an expression from source string (used for macro expansion)
    fn parse_expr_from_source(&self, source: &str) -> Option<Expr> {
        use crate::parser::Parser;
        let mut parser = Parser::new(source);
        parser.parse_expr().ok()
    }

    /// Lower an expression to a place (pointer/address)
    /// Used when we need the address of something, not its value
    fn lower_expr_place(&mut self, expr: &Expr) -> VReg {
        match &expr.kind {
            ExprKind::Path(path) => {
                // Variable reference - return the slot (pointer) directly
                let name = &path.segments[0].ident.name;
                if let Some(&slot) = self.locals.get(name) {
                    slot // Return the alloca'd pointer, don't load
                } else {
                    // Unknown variable - allocate a dummy slot
                    self.builder.alloca(IrType::I64)
                }
            }
            ExprKind::Field { object, field } => {
                // Check if object is a heap-allocated user struct
                let is_heap_struct = self.expr_types.get(&object.span).map_or(false, |ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        // Resolve "Self" to the current impl type
                        let resolved_name = if name == "Self" {
                            self.current_impl_type.as_ref().unwrap_or(name)
                        } else {
                            name
                        };
                        // User-defined structs and built-in heap types
                        self.struct_types.contains_key(resolved_name) ||
                        resolved_name == "Option" || resolved_name == "Result" || resolved_name == "String" ||
                        resolved_name == "Vec" || resolved_name == "HashMap" || resolved_name == "HashSet"
                    } else {
                        false
                    }
                });

                let struct_ptr = if is_heap_struct {
                    // For heap-allocated structs, load the pointer from the slot
                    // and convert i64 to pointer (opaque pointer handling)
                    let slot = self.lower_expr_place(object);
                    let ptr_val = self.builder.load(slot);
                    self.builder.inttoptr(ptr_val, IrType::Ptr(Box::new(IrType::I64)))
                } else {
                    // For stack-allocated structs, use the slot directly
                    self.lower_expr_place(object)
                };

                let field_idx = self.get_field_index(object, &field.name);
                self.builder.get_field_ptr(struct_ptr, field_idx)
            }
            ExprKind::Index { object, index } => {
                // Check if object is Vec or String type
                let is_vec = self.expr_types.get(&object.span).map_or(false, |ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        name == "Vec"
                    } else {
                        false
                    }
                });
                let is_string = self.expr_types.get(&object.span).map_or(false, |ty| {
                    if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                        name == "String"
                    } else {
                        false
                    }
                });

                let idx = self.lower_expr(index);

                if is_vec || is_string {
                    // Vec/String: struct { *T ptr, i64 len, i64 cap }
                    // Get the struct pointer, then load the data pointer from field 0
                    let struct_ptr = self.lower_expr(object);
                    let data_ptr_ptr = self.builder.get_field_ptr(struct_ptr, 0);
                    let data_ptr = self.builder.load(data_ptr_ptr);
                    
                    let data_ptr = if is_string {
                        self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I8)))
                    } else {
                        self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)))
                    };

                    if is_string {
                        // String data is i8, use byte pointer
                        self.builder.get_byte_ptr(data_ptr, idx)
                    } else {
                        // Vec: use regular element pointer
                        self.builder.get_element_ptr(data_ptr, idx)
                    }
                } else {
                    // Regular array: get pointer to array and index directly
                    let arr_ptr = self.lower_expr_place(object);
                    self.builder.get_element_ptr(arr_ptr, idx)
                }
            }
            _ => {
                // For other expressions, evaluate and store in temp
                let val = self.lower_expr(expr);
                let slot = self.builder.alloca(IrType::I64);
                self.builder.store(slot, val);
                slot
            }
        }
    }

    /// Lower a literal
    fn lower_literal(&mut self, lit: &Literal) -> VReg {
        match lit {
            Literal::Int(n) => {
                let vreg = self.builder.const_int(*n as i64);
                self.vreg_types.insert(vreg, IrType::I64);
                vreg
            }
            Literal::Float(f) => {
                let vreg = self.builder.const_float(*f);
                self.vreg_types.insert(vreg, IrType::F64);
                vreg
            }
            Literal::Bool(b) => {
                let vreg = self.builder.const_bool(*b);
                self.vreg_types.insert(vreg, IrType::Bool);
                vreg
            }
            Literal::Char(c) => {
                let vreg = self.builder.const_int(*c as i64);
                self.vreg_types.insert(vreg, IrType::I64);
                vreg
            }
            Literal::String(s) => {
                // Create a global string constant and return pointer to it
                let global_name = self.builder.add_string_constant(s);
                self.builder.global_string_ptr(&global_name)
            }
        }
    }

    /// Coerce a single vreg from one type to another
    fn coerce_arg(&mut self, vreg: VReg, from: &IrType, to: &IrType) -> VReg {
        use IrType::*;
        match (from, to) {
            // Same type - no coercion needed
            (a, b) if a == b => vreg,
            // Integer widening (sign extend)
            (I8, I16) | (I8, I32) | (I8, I64) |
            (I16, I32) | (I16, I64) |
            (I32, I64) => {
                let coerced = self.builder.sext(vreg, to.clone());
                self.vreg_types.insert(coerced, to.clone());
                coerced
            }
            // Integer narrowing (truncate)
            (I64, I32) | (I64, I16) | (I64, I8) |
            (I32, I16) | (I32, I8) |
            (I16, I8) => {
                let coerced = self.builder.trunc(vreg, to.clone());
                self.vreg_types.insert(coerced, to.clone());
                coerced
            }
            // Float widening
            (F32, F64) => {
                // Note: would need FPExt instruction, for now just return as-is
                vreg
            }
            // Float narrowing
            (F64, F32) => {
                // Note: would need FPTrunc instruction, for now just return as-is
                vreg
            }
            _ => vreg, // Unknown coercion, return as-is
        }
    }

    /// Coerce two vregs to a common type (for binary operations)
    /// Returns (coerced_left, coerced_right)
    fn coerce_to_common_type(&mut self, left: VReg, right: VReg) -> (VReg, VReg) {
        let left_ty = self.vreg_types.get(&left).cloned();
        let right_ty = self.vreg_types.get(&right).cloned();

        match (&left_ty, &right_ty) {
            (Some(l_ty), Some(r_ty)) if l_ty == r_ty => {
                // Same type, no coercion needed
                (left, right)
            }
            (Some(IrType::I32), Some(IrType::I64)) => {
                // Extend i32 to i64
                let coerced = self.builder.sext(left, IrType::I64);
                self.vreg_types.insert(coerced, IrType::I64);
                (coerced, right)
            }
            (Some(IrType::I64), Some(IrType::I32)) => {
                // Extend i32 to i64
                let coerced = self.builder.sext(right, IrType::I64);
                self.vreg_types.insert(coerced, IrType::I64);
                (left, coerced)
            }
            (Some(IrType::I16), Some(IrType::I32)) | (Some(IrType::I16), Some(IrType::I64)) => {
                let target = right_ty.clone().unwrap();
                let coerced = self.builder.sext(left, target.clone());
                self.vreg_types.insert(coerced, target);
                (coerced, right)
            }
            (Some(IrType::I32), Some(IrType::I16)) | (Some(IrType::I64), Some(IrType::I16)) => {
                let target = left_ty.clone().unwrap();
                let coerced = self.builder.sext(right, target.clone());
                self.vreg_types.insert(coerced, target);
                (left, coerced)
            }
            (Some(IrType::I8), Some(ty)) if ty.is_int() => {
                let coerced = self.builder.sext(left, ty.clone());
                self.vreg_types.insert(coerced, ty.clone());
                (coerced, right)
            }
            (Some(ty), Some(IrType::I8)) if ty.is_int() => {
                let coerced = self.builder.sext(right, ty.clone());
                self.vreg_types.insert(coerced, ty.clone());
                (left, coerced)
            }
            _ => {
                // Unknown types or no coercion needed
                (left, right)
            }
        }
    }

    /// Infer the IR type of an expression without lowering it
    fn infer_expr_type(&self, expr: &Expr) -> IrType {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int(_) => IrType::I64,
                Literal::Float(_) => IrType::F64,
                Literal::Bool(_) => IrType::Bool,
                Literal::Char(_) => IrType::I64,
                Literal::String(_) => IrType::Ptr(Box::new(IrType::I8)),
            },
            ExprKind::Path(path) => {
                // Look up variable type
                let name = &path.segments[0].ident.name;
                self.local_types.get(name).cloned().unwrap_or(IrType::I64)
            }
            ExprKind::Unary { op, operand } => {
                match op {
                    UnaryOp::Neg => self.infer_expr_type(operand),
                    UnaryOp::Not => IrType::Bool,
                    UnaryOp::Deref => IrType::I64, // Simplified
                    UnaryOp::Ref | UnaryOp::RefMut => IrType::Ptr(Box::new(self.infer_expr_type(operand))),
                }
            }
            ExprKind::Binary { left, .. } => {
                // Use left operand's type for now
                self.infer_expr_type(left)
            }
            _ => IrType::I64, // Default
        }
    }

    /// Lower a Box::new call - allocates on heap and stores value
    fn lower_box_new(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            // Box::new() with no args - return null pointer
            return self.builder.const_int(0);
        }

        let arg = &args[0];

        // Infer the type of the value being boxed
        let value_type = self.infer_expr_type(arg);

        // Allocate memory on the heap for this type
        let heap_ptr = self.builder.malloc(value_type.clone());

        // Lower the value expression
        let value = self.lower_expr(arg);

        // Store the value at the heap location
        self.builder.store(heap_ptr, value);

        // Track the type for this pointer
        self.vreg_types.insert(heap_ptr, value_type);

        // Return the heap pointer
        heap_ptr
    }

    // ============ Vec<T> Implementation ============
    // Vec is represented as a struct: { ptr: *T, len: i64, cap: i64 }
    // Stored on stack, data buffer on heap

    /// Create the Vec struct type
    fn vec_struct_type(&self) -> IrType {
        IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I64)), // ptr to data
            IrType::I64,                         // len
            IrType::I64,                         // cap
        ])
    }

    /// Lower Vec::new() - creates empty vector
    fn lower_vec_new(&mut self) -> VReg {
        // Allocate Vec struct on stack
        let vec_ty = self.vec_struct_type();
        let vec_ptr = self.builder.alloca(vec_ty.clone());

        // Initialize fields: ptr=null, len=0, cap=0
        let null_ptr = self.builder.const_null();
        let zero = self.builder.const_int(0);

        // Store ptr (field 0)
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        self.builder.store(ptr_field, null_ptr);

        // Store len (field 1)
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        self.builder.store(len_field, zero);

        // Store cap (field 2)
        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        self.builder.store(cap_field, zero);

        // Track type
        self.vreg_types.insert(vec_ptr, vec_ty);

        vec_ptr
    }

    /// Lower Vec::with_capacity(cap) - creates vector with initial capacity
    fn lower_vec_with_capacity(&mut self, args: &[Expr]) -> VReg {
        let cap = if let Some(arg) = args.first() {
            self.lower_expr(arg)
        } else {
            self.builder.const_int(0)
        };

        // Allocate Vec struct on stack
        let vec_ty = self.vec_struct_type();
        let vec_ptr = self.builder.alloca(vec_ty.clone());

        // Allocate data buffer on heap (cap * 8 bytes for i64 elements)
        let eight = self.builder.const_int(8);
        let size = self.builder.mul(cap, eight);

        // Use realloc with null ptr to allocate initial buffer
        let null_ptr = self.builder.const_null();
        let data_ptr = self.builder.realloc(null_ptr, size);

        // Initialize fields
        // Store ptr (field 0)
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        self.builder.store(ptr_field, data_ptr);

        // Store len=0 (field 1)
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let zero = self.builder.const_int(0);
        self.builder.store(len_field, zero);

        // Store cap (field 2)
        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        self.builder.store(cap_field, cap);

        // Track type
        self.vreg_types.insert(vec_ptr, vec_ty);

        vec_ptr
    }

    /// Lower Vec::push(vec, elem) - adds element to vector
    fn lower_vec_push(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let vec_ptr = self.lower_expr(&args[0]);
        let elem = self.lower_expr(&args[1]);

        // Load current len and cap
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);

        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        let cap = self.builder.load(cap_field);

        // Load data pointer
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Check if we need to grow (len >= cap)
        // For now, simplified: always grow if cap is 0 or len == cap
        // TODO: Add proper conditional growth with if/else blocks

        // Calculate new capacity: if cap == 0 then 4 else cap * 2
        let zero_const = self.builder.const_int(0);
        let is_zero = self.builder.icmp(super::instr::CmpOp::Eq, cap, zero_const);
        let four = self.builder.const_int(4);
        let two = self.builder.const_int(2);
        let doubled = self.builder.mul(cap, two);
        let new_cap = self.builder.select(is_zero, four, doubled);

        // Calculate new size in bytes
        let eight = self.builder.const_int(8);
        let new_size = self.builder.mul(new_cap, eight);

        // Reallocate (realloc handles null ptr as malloc)
        let new_data_ptr = self.builder.realloc(data_ptr, new_size);

        // Update ptr and cap (simplified: always update, even if no grow)
        // In a real implementation, we'd use conditional blocks
        self.builder.store(ptr_field, new_data_ptr);
        self.builder.store(cap_field, new_cap);

        // Store element at data[len]
        let elem_ptr = self.builder.get_element_ptr(new_data_ptr, len);
        self.builder.store(elem_ptr, elem);

        // Increment len
        let one = self.builder.const_int(1);
        let new_len = self.builder.add(len, one);
        self.builder.store(len_field, new_len);

        self.builder.const_int(0) // Return unit
    }

    /// Lower Vec::get(vec, index) - gets element at index
    fn lower_vec_get(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        // Get element type info from Vec<T>
        let elem_type_info = self.expr_types.get(&args[0].span).and_then(|ty| {
            if let crate::typeck::TyKind::Named { name, generics, .. } = &ty.kind {
                if name == "Vec" && !generics.is_empty() {
                    if let crate::typeck::TyKind::Named { name: elem_name, .. } = &generics[0].kind {
                        return Some(elem_name.clone());
                    }
                }
            }
            None
        });

        let elem_is_ptr = elem_type_info.as_ref().map_or(false, |name| {
            name == "String" || name == "Vec" || name == "HashMap" ||
            name == "HashSet" || name == "Box"
        });

        // Vec may be stored as i64 (from Result::unwrap), convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let index = self.lower_expr(&args[1]);

        // Load data pointer
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Get element pointer and load
        let elem_ptr = self.builder.get_element_ptr(data_ptr, index);
        let loaded = self.builder.load(elem_ptr);

        // If element type is a pointer, convert i64 back to TYPED pointer
        if elem_is_ptr {
            let elem_ir_type = match elem_type_info.as_deref() {
                Some("String") => IrType::Ptr(Box::new(self.string_struct_type())),
                Some("Vec") => IrType::Ptr(Box::new(self.vec_struct_type())),
                Some("HashMap") => IrType::Ptr(Box::new(self.hashmap_struct_type())),
                Some("HashSet") => IrType::Ptr(Box::new(self.hashset_struct_type())),
                Some("Box") => IrType::Ptr(Box::new(IrType::I64)),
                _ => IrType::Ptr(Box::new(IrType::I8)),
            };
            let result = self.builder.inttoptr(loaded, elem_ir_type.clone());
            // Register the type for later use (println, drop, etc.)
            self.vreg_types.insert(result, elem_ir_type);
            result
        } else {
            loaded
        }
    }

    /// Lower Vec::len(vec) - gets length
    fn lower_vec_len(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        // Vec may be stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));

        // Load len field
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        self.builder.load(len_field)
    }

    /// Lower Vec::pop(vec) - removes and returns last element as Option<T>
    fn lower_vec_pop(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            // Return None
            return self.create_option_none();
        }

        let vec_ptr = self.lower_expr(&args[0]);

        // Load current len
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);

        // Check if len == 0
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, len, zero);

        // Create blocks for branching
        let empty_block = self.builder.create_block();
        let non_empty_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_empty, empty_block, non_empty_block);

        // Empty case: return None
        self.builder.start_block(empty_block);
        let none_result = self.create_option_none();
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Non-empty case: decrement len, get element, return Some
        self.builder.start_block(non_empty_block);

        // Decrement len
        let one = self.builder.const_int(1);
        let new_len = self.builder.sub(len, one);
        self.builder.store(len_field, new_len);

        // Load data pointer and get element at new_len (which is the last index)
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let elem_ptr = self.builder.get_element_ptr(data_ptr, new_len);
        let elem = self.builder.load(elem_ptr);

        // Create Some(elem)
        let some_result = self.create_option_some(elem);
        let non_empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block with phi
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(none_result, empty_exit), (some_result, non_empty_exit)])
    }

    /// Lower Vec::set(vec, index, value) - sets element at index
    fn lower_vec_set(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 3 {
            return self.builder.const_int(0);
        }

        let vec_ptr = self.lower_expr(&args[0]);
        let index = self.lower_expr(&args[1]);
        let value = self.lower_expr(&args[2]);

        // Load data pointer
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Get element pointer and store
        let elem_ptr = self.builder.get_element_ptr(data_ptr, index);
        self.builder.store(elem_ptr, value);

        self.builder.const_int(0) // Return unit
    }

    /// Lower Vec::is_empty(vec) - checks if vector is empty
    fn lower_vec_is_empty(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(true);
        }

        let vec_ptr = self.lower_expr(&args[0]);

        // Load len field
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);

        // Check if len == 0
        let zero = self.builder.const_int(0);
        self.builder.icmp(CmpOp::Eq, len, zero)
    }

    /// Lower Vec::capacity(vec) - gets capacity
    fn lower_vec_capacity(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let vec_ptr = self.lower_expr(&args[0]);

        // Load cap field (field 2)
        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        self.builder.load(cap_field)
    }

    /// Lower Vec::clear(vec) - removes all elements (sets len to 0)
    fn lower_vec_clear(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let vec_ptr = self.lower_expr(&args[0]);

        // Set len to 0
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let zero = self.builder.const_int(0);
        self.builder.store(len_field, zero);

        self.builder.const_int(0) // Return unit
    }

    /// Lower Vec::first(vec) - gets first element as Option<T>
    fn lower_vec_first(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.create_option_none();
        }

        let vec_ptr = self.lower_expr(&args[0]);

        // Load len
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);

        // Check if len == 0
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, len, zero);

        // Create blocks for branching
        let empty_block = self.builder.create_block();
        let non_empty_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_empty, empty_block, non_empty_block);

        // Empty case: return None
        self.builder.start_block(empty_block);
        let none_result = self.create_option_none();
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Non-empty case: return Some(first element)
        self.builder.start_block(non_empty_block);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let elem_ptr = self.builder.get_element_ptr(data_ptr, zero);
        let elem = self.builder.load(elem_ptr);
        let some_result = self.create_option_some(elem);
        let non_empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block with phi
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(none_result, empty_exit), (some_result, non_empty_exit)])
    }

    /// Lower Vec::last(vec) - gets last element as Option<T>
    fn lower_vec_last(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.create_option_none();
        }

        let vec_ptr = self.lower_expr(&args[0]);

        // Load len
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);

        // Check if len == 0
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, len, zero);

        // Create blocks for branching
        let empty_block = self.builder.create_block();
        let non_empty_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_empty, empty_block, non_empty_block);

        // Empty case: return None
        self.builder.start_block(empty_block);
        let none_result = self.create_option_none();
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Non-empty case: return Some(last element)
        self.builder.start_block(non_empty_block);
        let one = self.builder.const_int(1);
        let last_idx = self.builder.sub(len, one);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let elem_ptr = self.builder.get_element_ptr(data_ptr, last_idx);
        let elem = self.builder.load(elem_ptr);
        let some_result = self.create_option_some(elem);
        let non_empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block with phi
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(none_result, empty_exit), (some_result, non_empty_exit)])
    }

    /// Lower Vec::iter(vec) - returns the Vec itself (lazy iteration)
    /// VecIter is just an alias for Vec, iteration state is managed per-call
    fn lower_vec_iter(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }
        // Simply return the Vec pointer - iteration state is managed by each method
        self.lower_expr(&args[0])
    }

    // ============ VecIter Methods ============
    // VecIter is the same as Vec - methods iterate from index 0

    /// Lower VecIter::next(iter) -> Option<T>
    /// Returns the first element as Option, since VecIter is stateless
    fn lower_veciter_next(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.create_option_none();
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);

        // Check if vec is empty
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, len, zero);

        let some_block = self.builder.create_block();
        let none_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_empty, none_block, some_block);

        // Some: return first element
        self.builder.start_block(some_block);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let first_elem = self.builder.load(data_ptr);
        let some_result = self.create_option_some(first_elem);
        let some_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // None
        self.builder.start_block(none_block);
        let none_result = self.create_option_none();
        let none_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        self.builder.start_block(merge_block);
        self.builder.phi(vec![(some_result, some_exit), (none_result, none_exit)])
    }

    /// Lower VecIter::count(iter) -> i64 - returns the number of elements
    fn lower_veciter_count(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        // VecIter is stored as i64 in local vars, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));

        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        self.builder.load(len_field)
    }

    /// Lower VecIter::sum(iter) -> T - sums all elements
    fn lower_veciter_sum(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        let sum_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(sum_slot, zero);

        let i_slot = self.builder.alloca(IrType::I64);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let sum = self.builder.load(sum_slot);
        let new_sum = self.builder.add(sum, elem);
        self.builder.store(sum_slot, new_sum);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        self.builder.load(sum_slot)
    }

    /// Lower VecIter::collect(iter) -> Vec<T> - creates a new Vec with copied elements
    fn lower_veciter_collect(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_vec_new();
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Create new Vec
        let new_vec = self.lower_vec_new();

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        self.lower_vec_push_i64_raw(new_vec, elem);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        new_vec
    }

    /// Lower VecIter::fold(iter, init, closure) -> U
    fn lower_veciter_fold(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 3 {
            return self.builder.const_int(0);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let init_val = self.lower_expr(&args[1]);
        let closure_ptr = self.lower_expr(&args[2]);

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Extract fn_ptr and env_ptr from closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Accumulator slot
        let acc_slot = self.builder.alloca(IrType::I64);
        self.builder.store(acc_slot, init_val);

        // Index slot
        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        // Loop blocks
        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop header: check i < len
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        // Body: acc = f(acc, elem)
        self.builder.start_block(body_block);
        let acc = self.builder.load(acc_slot);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let new_acc = self.builder.call_ptr(fn_ptr, vec![env_ptr, acc, elem]);
        self.builder.store(acc_slot, new_acc);

        // Increment i
        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        // Done: return acc
        self.builder.start_block(done_block);
        self.builder.load(acc_slot)
    }

    /// Lower VecIter::map(iter, closure) -> MapIter
    /// MapIter is a struct: { vec_ptr, closure_ptr }
    fn lower_veciter_map(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.lower_expr(&args[1]);

        // Create MapIter struct: { vec_ptr, closure_ptr }
        let map_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let map_ptr = self.builder.malloc(map_ty);

        // Store vec_ptr (field 0) - convert ptr to i64
        let vec_field = self.builder.get_field_ptr(map_ptr, 0);
        let vec_as_i64 = self.builder.ptrtoint(vec_ptr, IrType::I64);
        self.builder.store(vec_field, vec_as_i64);

        // Store closure (field 1) - convert ptr to i64
        let closure_field = self.builder.get_field_ptr(map_ptr, 1);
        let closure_as_i64 = self.builder.ptrtoint(closure_ptr, IrType::I64);
        self.builder.store(closure_field, closure_as_i64);

        map_ptr
    }

    /// Lower VecIter::filter(iter, predicate) -> FilterIter
    fn lower_veciter_filter(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.lower_expr(&args[1]);

        // Create FilterIter struct: { vec_ptr, closure_ptr }
        let filter_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let filter_ptr = self.builder.malloc(filter_ty);

        // Store vec_ptr (field 0)
        let vec_field = self.builder.get_field_ptr(filter_ptr, 0);
        let vec_as_i64 = self.builder.ptrtoint(vec_ptr, IrType::I64);
        self.builder.store(vec_field, vec_as_i64);

        // Store closure (field 1)
        let closure_field = self.builder.get_field_ptr(filter_ptr, 1);
        let closure_as_i64 = self.builder.ptrtoint(closure_ptr, IrType::I64);
        self.builder.store(closure_field, closure_as_i64);

        filter_ptr
    }

    /// Lower VecIter::find(iter, predicate) -> Option<T>
    /// VecIter is just a Vec pointer - we iterate from 0
    fn lower_veciter_find(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.create_option_none();
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.lower_expr(&args[1]);

        // Get Vec info directly
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Extract closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let check_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, not_found_block, check_block);

        self.builder.start_block(check_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let matches = self.builder.call_ptr(fn_ptr, vec![env_ptr, elem]);
        let zero_check = self.builder.const_int(0);
        let matches_bool = self.builder.icmp(CmpOp::Ne, matches, zero_check);

        // Increment index
        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);

        self.builder.cond_br(matches_bool, found_block, loop_block);

        self.builder.start_block(found_block);
        let result_some = self.create_option_some(elem);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        self.builder.start_block(not_found_block);
        let result_none = self.create_option_none();
        let not_found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.phi(vec![(result_some, found_exit), (result_none, not_found_exit)])
    }

    /// Lower VecIter::any(iter, predicate) -> bool
    fn lower_veciter_any(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.lower_expr(&args[1]);

        // Get Vec info directly
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Extract closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let matches = self.builder.call_ptr(fn_ptr, vec![env_ptr, elem]);
        let zero_check = self.builder.const_int(0);
        let matches_bool = self.builder.icmp(CmpOp::Ne, matches, zero_check);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);

        self.builder.cond_br(matches_bool, found_block, loop_block);

        self.builder.start_block(found_block);
        let true_val = self.builder.const_bool(true);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        let false_val = self.builder.const_bool(false);
        let loop_exit = loop_block;
        self.builder.phi(vec![(true_val, found_exit), (false_val, loop_exit)])
    }

    /// Lower VecIter::all(iter, predicate) -> bool
    fn lower_veciter_all(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(true);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.lower_expr(&args[1]);

        // Get Vec info directly
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Extract closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let fail_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let matches = self.builder.call_ptr(fn_ptr, vec![env_ptr, elem]);
        let zero_check = self.builder.const_int(0);
        let matches_bool = self.builder.icmp(CmpOp::Ne, matches, zero_check);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);

        self.builder.cond_br(matches_bool, loop_block, fail_block);

        self.builder.start_block(fail_block);
        let false_val = self.builder.const_bool(false);
        let fail_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        let true_val = self.builder.const_bool(true);
        let loop_exit = loop_block;
        self.builder.phi(vec![(false_val, fail_exit), (true_val, loop_exit)])
    }

    /// Lower VecIter::enumerate(iter) -> VecIter (just returns the iterator)
    fn lower_veciter_enumerate(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }
        // EnumerateIter just wraps the base iterator
        // The actual enumeration happens during iteration
        self.lower_expr(&args[0])
    }

    /// Lower VecIter::take(iter, n) -> collects first n elements into a new Vec
    fn lower_veciter_take(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.lower_vec_new();
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let n = self.lower_expr(&args[1]);

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Create new Vec
        let new_vec = self.lower_vec_new();

        // Copy min(n, len) elements
        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        // Check i >= n OR i >= len (either condition means we're done)
        let cmp_n = self.builder.icmp(CmpOp::Sge, i, n);
        let cmp_len = self.builder.icmp(CmpOp::Sge, i, len);
        // Use OR on booleans - result is already a boolean (i1)
        let at_end = self.builder.or(cmp_n, cmp_len);
        // at_end is already a boolean (i1), use directly in cond_br
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        self.lower_vec_push_i64_raw(new_vec, elem);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        new_vec
    }

    /// Lower VecIter::skip(iter, n) -> collects elements after skipping n into a new Vec
    fn lower_veciter_skip(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.lower_vec_new();
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let n = self.lower_expr(&args[1]);

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Create new Vec
        let new_vec = self.lower_vec_new();

        // Copy elements starting from index n
        let i_slot = self.builder.alloca(IrType::I64);
        self.builder.store(i_slot, n);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        self.lower_vec_push_i64_raw(new_vec, elem);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        new_vec
    }

    /// Lower VecIter::for_each(iter, closure) -> ()
    fn lower_veciter_for_each(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        // VecIter is stored as i64, convert to pointer
        let vec_reg = self.lower_expr(&args[0]);
        let vec_ptr = self.builder.inttoptr(vec_reg, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.lower_expr(&args[1]);

        // Get Vec info directly
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Extract closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        self.builder.call_ptr(fn_ptr, vec![env_ptr, elem]);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        self.builder.const_int(0)
    }

    // ============ MapIter Methods ============
    // MapIter: { vec_ptr (as i64), closure_ptr (as i64) }
    // base_iter is directly a Vec pointer (stored as i64)

    /// Lower MapIter::next(iter) -> Option<U>
    /// Note: This is stateless - returns first mapped element each time
    /// For proper stateful iteration, use MapIter::collect instead
    fn lower_mapiter_next(&mut self, _args: &[Expr]) -> VReg {
        // Stateless next is not useful for MapIter - return None
        // Use MapIter::collect for proper iteration
        self.create_option_none()
    }

    /// Lower MapIter::collect(iter) -> Vec<U>
    fn lower_mapiter_collect(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_vec_new();
        }

        let map_ptr = self.lower_expr(&args[0]);

        // Load vec_ptr and closure_ptr from MapIter
        let base_field = self.builder.get_field_ptr(map_ptr, 0);
        let vec_as_i64 = self.builder.load(base_field);
        let closure_field = self.builder.get_field_ptr(map_ptr, 1);
        let closure_as_i64 = self.builder.load(closure_field);

        // Convert back to pointers
        let vec_ptr = self.builder.inttoptr(vec_as_i64, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.builder.inttoptr(closure_as_i64, IrType::Ptr(Box::new(IrType::Struct(vec![IrType::I64, IrType::I64]))));

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Get closure parts
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Create new Vec
        let new_vec = self.lower_vec_new();

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, body_block);

        self.builder.start_block(body_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let mapped = self.builder.call_ptr(fn_ptr, vec![env_ptr, elem]);
        self.lower_vec_push_i64_raw(new_vec, mapped);

        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        new_vec
    }

    // ============ FilterIter Methods ============
    // FilterIter: { vec_ptr (as i64), closure_ptr (as i64) }

    /// Lower FilterIter::next(iter) -> Option<T>
    /// Note: This is stateless - returns first matching element each time
    fn lower_filteriter_next(&mut self, _args: &[Expr]) -> VReg {
        // Stateless next is not useful for FilterIter - return None
        // Use FilterIter::collect for proper iteration
        self.create_option_none()
    }

    /// Lower FilterIter::collect(iter) -> Vec<T>
    fn lower_filteriter_collect(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_vec_new();
        }

        let filter_ptr = self.lower_expr(&args[0]);

        // Load vec_ptr and closure_ptr from FilterIter
        let base_field = self.builder.get_field_ptr(filter_ptr, 0);
        let vec_as_i64 = self.builder.load(base_field);
        let closure_field = self.builder.get_field_ptr(filter_ptr, 1);
        let closure_as_i64 = self.builder.load(closure_field);

        // Convert back to pointers
        let vec_ptr = self.builder.inttoptr(vec_as_i64, IrType::Ptr(Box::new(self.vec_struct_type())));
        let closure_ptr = self.builder.inttoptr(closure_as_i64, IrType::Ptr(Box::new(IrType::Struct(vec![IrType::I64, IrType::I64]))));

        // Get Vec info
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);
        let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

        // Get closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Create new Vec
        let new_vec = self.lower_vec_new();

        let i_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_slot, zero);

        let loop_block = self.builder.create_block();
        let check_block = self.builder.create_block();
        let push_block = self.builder.create_block();
        let next_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_slot);
        let at_end = self.builder.icmp(CmpOp::Sge, i, len);
        self.builder.cond_br(at_end, done_block, check_block);

        self.builder.start_block(check_block);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, i);
        let elem = self.builder.load(elem_ptr);
        let matches = self.builder.call_ptr(fn_ptr, vec![env_ptr, elem]);
        let zero_check = self.builder.const_int(0);
        let matches_bool = self.builder.icmp(CmpOp::Ne, matches, zero_check);
        self.builder.cond_br(matches_bool, push_block, next_block);

        self.builder.start_block(push_block);
        self.lower_vec_push_i64_raw(new_vec, elem);
        self.builder.br(next_block);

        self.builder.start_block(next_block);
        let one = self.builder.const_int(1);
        let new_i = self.builder.add(i, one);
        self.builder.store(i_slot, new_i);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        new_vec
    }

    /// Helper: Create an Option::None value
    fn create_option_none(&mut self) -> VReg {
        let option_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let opt_ptr = self.builder.malloc(option_ty);

        // Store discriminant = 0 (None)
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        self.builder.store(discrim_ptr, zero_i32);

        opt_ptr
    }

    /// Helper: Create an Option::Some(value)
    fn create_option_some(&mut self, value: VReg) -> VReg {
        let option_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let opt_ptr = self.builder.malloc(option_ty);

        // Store discriminant = 1 (Some)
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        self.builder.store(discrim_ptr, one_i32);

        // Store payload
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        self.builder.store(payload_ptr, value);

        opt_ptr
    }

    // ============ String Implementation ============
    // String is represented as a struct: { ptr: *u8, len: i64, cap: i64 }
    // Similar to Vec<u8>

    /// Create the String struct type
    fn string_struct_type(&self) -> IrType {
        IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)), // ptr to data
            IrType::I64,                        // len
            IrType::I64,                        // cap
        ])
    }

    /// Lower String::new() - creates empty string
    fn lower_string_new(&mut self) -> VReg {
        // Allocate String struct on stack
        let str_ty = self.string_struct_type();
        let str_ptr = self.builder.alloca(str_ty.clone());

        // Initialize fields: ptr=null, len=0, cap=0
        let null_ptr = self.builder.const_null();
        let zero = self.builder.const_int(0);

        // Store ptr (field 0)
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        self.builder.store(ptr_field, null_ptr);

        // Store len (field 1)
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        self.builder.store(len_field, zero);

        // Store cap (field 2)
        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        self.builder.store(cap_field, zero);

        // Track type
        self.vreg_types.insert(str_ptr, str_ty);

        str_ptr
    }

    /// Lower String::from(literal) - creates string from literal
    fn lower_string_from(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_string_new();
        }

        let arg = &args[0];

        // Get the string literal
        let (str_name, str_len) = if let ExprKind::Literal(Literal::String(s)) = &arg.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            // Not a literal, just create empty string
            return self.lower_string_new();
        };

        // Allocate String struct on stack
        let str_ty = self.string_struct_type();
        let str_ptr = self.builder.alloca(str_ty.clone());

        // Track type BEFORE accessing fields so GetFieldPtr knows the struct layout
        self.vreg_types.insert(str_ptr, str_ty);

        // Get pointer to the global string constant
        let data_ptr = self.builder.global_string_ptr(&str_name);

        // Store ptr (field 0) - note: this points to read-only memory
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        self.builder.store(ptr_field, data_ptr);

        // Store len (field 1)
        let len = self.builder.const_int(str_len);
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        self.builder.store(len_field, len);

        // Store cap (field 2) - set to 0 for literals to indicate no allocated capacity
        // This tells __drop_string not to free the pointer (it's a global constant)
        let zero = self.builder.const_int(0);
        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        self.builder.store(cap_field, zero);

        str_ptr
    }

    /// Lower String::len(s) - gets length
    fn lower_string_len(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load len field
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        self.builder.load(len_field)
    }

    /// Lower String::push(s, char) - appends character
    fn lower_string_push(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let str_ptr = self.lower_expr(&args[0]);
        let char_val = self.lower_expr(&args[1]);

        // Load current len and cap
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);

        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        let cap = self.builder.load(cap_field);

        // Load data pointer (field 0 is already a pointer type)
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Calculate new capacity: if cap == 0 then 16 else cap * 2
        let zero_const = self.builder.const_int(0);
        let is_zero = self.builder.icmp(super::instr::CmpOp::Eq, cap, zero_const);
        let sixteen = self.builder.const_int(16);
        let two = self.builder.const_int(2);
        let doubled = self.builder.mul(cap, two);
        let new_cap = self.builder.select(is_zero, sixteen, doubled);

        // Calculate new size in bytes (1 byte per char for u8)
        let new_size = new_cap;

        // Reallocate
        let new_data_ptr = self.builder.realloc(data_ptr, new_size);

        // Update ptr and cap
        self.builder.store(ptr_field, new_data_ptr);
        self.builder.store(cap_field, new_cap);

        // Store character at data[len]
        let char_ptr = self.builder.get_element_ptr(new_data_ptr, len);
        // Truncate i64 to i8 for storage
        let char_i8 = self.builder.trunc(char_val, IrType::I8);
        self.builder.store(char_ptr, char_i8);

        // Increment len
        let one = self.builder.const_int(1);
        let new_len = self.builder.add(len, one);
        self.builder.store(len_field, new_len);

        self.builder.const_int(0) // Return unit
    }

    /// Lower String::is_empty(s) - checks if string is empty
    fn lower_string_is_empty(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(true);
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load len field
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);

        // Check if len == 0
        let zero = self.builder.const_int(0);
        self.builder.icmp(CmpOp::Eq, len, zero)
    }

    /// Lower String::clear(s) - clears string (sets len to 0)
    fn lower_string_clear(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Set len to 0
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let zero = self.builder.const_int(0);
        self.builder.store(len_field, zero);

        self.builder.const_int(0) // Return unit
    }

    /// Lower String::capacity(s) - gets capacity
    fn lower_string_capacity(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load cap field (field 2)
        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        self.builder.load(cap_field)
    }

    /// Lower String::char_at(s, index) - gets character at index
    fn lower_string_char_at(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let str_ptr = self.lower_expr(&args[0]);
        let index = self.lower_expr(&args[1]);

        // Load data pointer (field 0 is already a pointer type)
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Use byte-level pointer arithmetic: ptr + index (for i8 elements)
        let char_ptr = self.builder.get_byte_ptr(data_ptr, index);

        // Load single byte
        let char_i8 = self.builder.load_byte(char_ptr);

        // Extend i8 to i64
        self.builder.zext(char_i8, IrType::I64)
    }

    /// Lower String::push_str(s, str) - appends string literal
    fn lower_string_push_str(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Get the string literal to append
        let (literal_name, literal_len) = if let ExprKind::Literal(Literal::String(s)) = &args[1].kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            // Not a literal - can't handle dynamic strings yet
            return self.builder.const_int(0);
        };

        // Load current len and cap
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);

        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        let cap = self.builder.load(cap_field);

        // Load data pointer (field 0 is already a pointer type)
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Calculate new length
        let add_len = self.builder.const_int(literal_len);
        let new_len = self.builder.add(len, add_len);

        // Calculate required capacity (new_len + some padding)
        let padding = self.builder.const_int(16);
        let required_cap = self.builder.add(new_len, padding);

        // Check if we need to grow: required_cap > cap
        let needs_grow = self.builder.icmp(CmpOp::Sgt, required_cap, cap);

        // Calculate new capacity: use required_cap if we need to grow, else keep cap
        let new_cap = self.builder.select(needs_grow, required_cap, cap);

        // Reallocate if needed
        let new_data_ptr = self.builder.realloc(data_ptr, new_cap);

        // Update ptr and cap
        self.builder.store(ptr_field, new_data_ptr);
        self.builder.store(cap_field, new_cap);

        // Get pointer to the literal
        let literal_ptr = self.builder.global_string_ptr(&literal_name);

        // Copy the literal bytes using memcpy
        // dst = new_data_ptr + len, src = literal_ptr, size = literal_len
        let dst_ptr = self.builder.get_element_ptr(new_data_ptr, len);
        self.builder.memcpy(dst_ptr, literal_ptr, add_len);

        // Update len
        self.builder.store(len_field, new_len);

        self.builder.const_int(0) // Return unit
    }

    /// Lower String::concat(s1, s2) - concatenates two strings, returns new string
    fn lower_string_concat(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.lower_string_new();
        }

        let s1_ptr = self.lower_expr(&args[0]);
        let s2_ptr = self.lower_expr(&args[1]);

        // Load s1 len and data
        let s1_len_field = self.builder.get_field_ptr(s1_ptr, 1);
        let s1_len = self.builder.load(s1_len_field);
        let s1_ptr_field = self.builder.get_field_ptr(s1_ptr, 0);
        let s1_data = self.builder.load(s1_ptr_field);
        
        let s1_data = self.builder.inttoptr(s1_data, IrType::Ptr(Box::new(IrType::I8)));

        // Load s2 len and data
        let s2_len_field = self.builder.get_field_ptr(s2_ptr, 1);
        let s2_len = self.builder.load(s2_len_field);
        let s2_ptr_field = self.builder.get_field_ptr(s2_ptr, 0);
        let s2_data = self.builder.load(s2_ptr_field);
        
        let s2_data = self.builder.inttoptr(s2_data, IrType::Ptr(Box::new(IrType::I8)));

        // Calculate new length
        let new_len = self.builder.add(s1_len, s2_len);

        // Allocate new String struct on stack
        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());

        // Allocate buffer on heap for combined data
        let new_data = self.builder.malloc_bytes(new_len);

        // Copy s1 data
        self.builder.memcpy(new_data, s1_data, s1_len);

        // Copy s2 data after s1
        let s2_dst = self.builder.get_element_ptr(new_data, s1_len);
        self.builder.memcpy(s2_dst, s2_data, s2_len);

        // Store ptr (field 0)
        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);

        // Store len (field 1)
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, new_len);

        // Store cap (field 2) - same as len
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, new_len);

        // Track type
        self.vreg_types.insert(result_ptr, str_ty);

        result_ptr
    }

    /// Lower String::substring(s, start, end) - gets substring
    fn lower_string_substring(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 3 {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);
        let start = self.lower_expr(&args[1]);
        let end = self.lower_expr(&args[2]);

        // Load source data
        let src_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(src_ptr_field);
        
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Calculate new length
        let new_len = self.builder.sub(end, start);

        // Allocate new String struct on stack
        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());

        // Allocate buffer on heap
        let new_data = self.builder.malloc_bytes(new_len);

        // Copy substring
        let src_start = self.builder.get_element_ptr(src_data, start);
        self.builder.memcpy(new_data, src_start, new_len);

        // Store ptr (field 0)
        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);

        // Store len (field 1)
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, new_len);

        // Store cap (field 2)
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, new_len);

        // Track type
        self.vreg_types.insert(result_ptr, str_ty);

        result_ptr
    }

    /// Lower String::contains(s, pattern) - check if string contains substring
    fn lower_string_contains(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        let str_ptr = self.lower_expr(&args[0]);
        let pattern = &args[1];

        // Get pattern string and length
        let (pattern_name, pattern_len) = if let ExprKind::Literal(Literal::String(s)) = &pattern.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return self.builder.const_bool(false);
        };

        // Empty pattern always matches
        if pattern_len == 0 {
            return self.builder.const_bool(true);
        }

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let pattern_ptr = self.builder.global_string_ptr(&pattern_name);
        let pattern_len_val = self.builder.const_int(pattern_len);

        // Simple linear search: check each position
        let check_block = self.builder.create_block();
        let match_block = self.builder.create_block();
        let no_match_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Allocate loop counter and result
        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(i_ptr, zero);

        // Calculate max position to check
        let max_pos = self.builder.sub(str_len, pattern_len_val);
        let one = self.builder.const_int(1);
        let max_pos_plus_one = self.builder.add(max_pos, one);

        self.builder.br(check_block);

        // Check block: compare pattern at current position
        self.builder.start_block(check_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, max_pos_plus_one);

        let inner_check = self.builder.create_block();
        self.builder.cond_br(cond, inner_check, no_match_block);

        // Inner check: compare bytes
        self.builder.start_block(inner_check);
        let j_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(j_ptr, zero);

        let byte_check = self.builder.create_block();
        let bytes_equal = self.builder.create_block();
        let next_pos = self.builder.create_block();

        self.builder.br(byte_check);

        // Byte check block
        self.builder.start_block(byte_check);
        let j = self.builder.load(j_ptr);
        let j_done = self.builder.icmp(super::instr::CmpOp::Sge, j, pattern_len_val);
        self.builder.cond_br(j_done, match_block, bytes_equal);

        // Compare bytes
        self.builder.start_block(bytes_equal);
        let str_idx = self.builder.add(i, j);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, str_idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let pat_byte_ptr = self.builder.get_byte_ptr(pattern_ptr, j);
        let pat_byte = self.builder.load_byte(pat_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, pat_byte);

        let inc_j = self.builder.create_block();
        self.builder.cond_br(bytes_match, inc_j, next_pos);

        // Increment j
        self.builder.start_block(inc_j);
        let j_new = self.builder.add(j, one);
        self.builder.store(j_ptr, j_new);
        self.builder.br(byte_check);

        // Next position: increment i
        self.builder.start_block(next_pos);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(check_block);

        // Match found
        self.builder.start_block(match_block);
        let true_val = self.builder.const_bool(true);
        self.builder.br(done_block);

        // No match
        self.builder.start_block(no_match_block);
        let false_val = self.builder.const_bool(false);
        self.builder.br(done_block);

        // Done: phi result
        self.builder.start_block(done_block);
        self.builder.phi(vec![(true_val, match_block), (false_val, no_match_block)])
    }

    /// Lower String::starts_with(s, prefix) - check if string starts with prefix
    fn lower_string_starts_with(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        let str_ptr = self.lower_expr(&args[0]);
        let pattern = &args[1];

        // Get pattern string and length
        let (pattern_name, pattern_len) = if let ExprKind::Literal(Literal::String(s)) = &pattern.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return self.builder.const_bool(false);
        };

        // Empty prefix always matches
        if pattern_len == 0 {
            return self.builder.const_bool(true);
        }

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let pattern_ptr = self.builder.global_string_ptr(&pattern_name);
        let pattern_len_val = self.builder.const_int(pattern_len);

        // Check if string is long enough
        let len_ok = self.builder.icmp(super::instr::CmpOp::Sge, str_len, pattern_len_val);

        let check_block = self.builder.create_block();
        let fail_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(len_ok, check_block, fail_block);

        // Check prefix bytes
        self.builder.start_block(check_block);
        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let compare_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop check
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let loop_done = self.builder.icmp(super::instr::CmpOp::Sge, i, pattern_len_val);
        self.builder.cond_br(loop_done, success_block, compare_block);

        // Compare bytes
        self.builder.start_block(compare_block);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, i);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let pat_byte_ptr = self.builder.get_byte_ptr(pattern_ptr, i);
        let pat_byte = self.builder.load_byte(pat_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, pat_byte);

        let inc_block = self.builder.create_block();
        self.builder.cond_br(bytes_match, inc_block, fail_block);

        // Increment i
        self.builder.start_block(inc_block);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        // Success
        self.builder.start_block(success_block);
        let true_val = self.builder.const_bool(true);
        self.builder.br(done_block);

        // Fail
        self.builder.start_block(fail_block);
        let false_val = self.builder.const_bool(false);
        self.builder.br(done_block);

        // Done: phi result
        self.builder.start_block(done_block);
        self.builder.phi(vec![(true_val, success_block), (false_val, fail_block)])
    }

    /// Lower String::ends_with(s, suffix) - check if string ends with suffix
    fn lower_string_ends_with(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        let str_ptr = self.lower_expr(&args[0]);
        let pattern = &args[1];

        // Get pattern string and length
        let (pattern_name, pattern_len) = if let ExprKind::Literal(Literal::String(s)) = &pattern.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return self.builder.const_bool(false);
        };

        // Empty suffix always matches
        if pattern_len == 0 {
            return self.builder.const_bool(true);
        }

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let pattern_ptr = self.builder.global_string_ptr(&pattern_name);
        let pattern_len_val = self.builder.const_int(pattern_len);

        // Check if string is long enough
        let len_ok = self.builder.icmp(super::instr::CmpOp::Sge, str_len, pattern_len_val);

        let check_block = self.builder.create_block();
        let fail_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(len_ok, check_block, fail_block);

        // Check suffix bytes (start from end of string)
        self.builder.start_block(check_block);
        let start_offset = self.builder.sub(str_len, pattern_len_val);

        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let compare_block = self.builder.create_block();
        let success_block = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop check
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let loop_done = self.builder.icmp(super::instr::CmpOp::Sge, i, pattern_len_val);
        self.builder.cond_br(loop_done, success_block, compare_block);

        // Compare bytes
        self.builder.start_block(compare_block);
        let str_idx = self.builder.add(start_offset, i);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, str_idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let pat_byte_ptr = self.builder.get_byte_ptr(pattern_ptr, i);
        let pat_byte = self.builder.load_byte(pat_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, pat_byte);

        let inc_block = self.builder.create_block();
        self.builder.cond_br(bytes_match, inc_block, fail_block);

        // Increment i
        self.builder.start_block(inc_block);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        // Success
        self.builder.start_block(success_block);
        let true_val = self.builder.const_bool(true);
        self.builder.br(done_block);

        // Fail
        self.builder.start_block(fail_block);
        let false_val = self.builder.const_bool(false);
        self.builder.br(done_block);

        // Done: phi result
        self.builder.start_block(done_block);
        self.builder.phi(vec![(true_val, success_block), (false_val, fail_block)])
    }

    /// Lower String::find(s, pattern) -> Option<i64>
    fn lower_string_find(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.create_option_none();
        }

        let str_ptr = self.lower_expr(&args[0]);
        let pattern = &args[1];

        // Get pattern string and length
        let (pattern_name, pattern_len) = if let ExprKind::Literal(Literal::String(s)) = &pattern.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return self.create_option_none();
        };

        // Empty pattern found at position 0
        if pattern_len == 0 {
            return self.create_option_some_i64(0);
        }

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let pattern_ptr = self.builder.global_string_ptr(&pattern_name);
        let pattern_len_val = self.builder.const_int(pattern_len);

        // Allocate result Option
        let opt_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let result_ptr = self.builder.alloca(opt_ty.clone());

        // Simple linear search
        let check_block = self.builder.create_block();
        let inner_check = self.builder.create_block();
        let byte_check = self.builder.create_block();
        let bytes_equal = self.builder.create_block();
        let inc_j = self.builder.create_block();
        let next_pos = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let max_pos = self.builder.sub(str_len, pattern_len_val);
        let max_pos_plus_one = self.builder.add(max_pos, one);

        self.builder.br(check_block);

        // Check block
        self.builder.start_block(check_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, max_pos_plus_one);
        self.builder.cond_br(cond, inner_check, not_found_block);

        // Inner check
        self.builder.start_block(inner_check);
        let j_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(j_ptr, zero);
        self.builder.br(byte_check);

        // Byte check
        self.builder.start_block(byte_check);
        let j = self.builder.load(j_ptr);
        let j_done = self.builder.icmp(super::instr::CmpOp::Sge, j, pattern_len_val);
        self.builder.cond_br(j_done, found_block, bytes_equal);

        // Compare bytes
        self.builder.start_block(bytes_equal);
        let i_val = self.builder.load(i_ptr);
        let str_idx = self.builder.add(i_val, j);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, str_idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let pat_byte_ptr = self.builder.get_byte_ptr(pattern_ptr, j);
        let pat_byte = self.builder.load_byte(pat_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, pat_byte);
        self.builder.cond_br(bytes_match, inc_j, next_pos);

        // Increment j
        self.builder.start_block(inc_j);
        let j_new = self.builder.add(j, one);
        self.builder.store(j_ptr, j_new);
        self.builder.br(byte_check);

        // Next position
        self.builder.start_block(next_pos);
        let i_val = self.builder.load(i_ptr);
        let i_new = self.builder.add(i_val, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(check_block);

        // Found: create Some(i)
        self.builder.start_block(found_block);
        let found_i = self.builder.load(i_ptr);
        let one_i32 = self.builder.const_i32(1);
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(discrim_ptr, one_i32);
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr, found_i);
        self.builder.br(done_block);

        // Not found: create None
        self.builder.start_block(not_found_block);
        let zero_i32 = self.builder.const_i32(0);
        let discrim_ptr2 = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(discrim_ptr2, zero_i32);
        let payload_ptr2 = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr2, zero);
        self.builder.br(done_block);

        // Done
        self.builder.start_block(done_block);
        self.vreg_types.insert(result_ptr, opt_ty);
        result_ptr
    }

    /// Lower String::rfind(s, pattern) -> Option<i64>
    fn lower_string_rfind(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.create_option_none();
        }

        let str_ptr = self.lower_expr(&args[0]);
        let pattern = &args[1];

        // Get pattern string and length
        let (pattern_name, pattern_len) = if let ExprKind::Literal(Literal::String(s)) = &pattern.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return self.create_option_none();
        };

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let pattern_ptr = self.builder.global_string_ptr(&pattern_name);
        let pattern_len_val = self.builder.const_int(pattern_len);

        // Allocate result Option
        let opt_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let result_ptr = self.builder.alloca(opt_ty.clone());

        // Empty pattern found at end
        if pattern_len == 0 {
            let one_i32 = self.builder.const_i32(1);
            let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
            self.builder.store(discrim_ptr, one_i32);
            let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
            self.builder.store(payload_ptr, str_len);
            self.vreg_types.insert(result_ptr, opt_ty);
            return result_ptr;
        }

        // Search from end (start at len - pattern_len)
        let check_block = self.builder.create_block();
        let inner_check = self.builder.create_block();
        let byte_check = self.builder.create_block();
        let bytes_equal = self.builder.create_block();
        let inc_j = self.builder.create_block();
        let next_pos = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        let start_pos = self.builder.sub(str_len, pattern_len_val);
        self.builder.store(i_ptr, start_pos);

        self.builder.br(check_block);

        // Check block (i >= 0)
        self.builder.start_block(check_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Sge, i, zero);
        self.builder.cond_br(cond, inner_check, not_found_block);

        // Inner check
        self.builder.start_block(inner_check);
        let j_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(j_ptr, zero);
        self.builder.br(byte_check);

        // Byte check
        self.builder.start_block(byte_check);
        let j = self.builder.load(j_ptr);
        let j_done = self.builder.icmp(super::instr::CmpOp::Sge, j, pattern_len_val);
        self.builder.cond_br(j_done, found_block, bytes_equal);

        // Compare bytes
        self.builder.start_block(bytes_equal);
        let i_val = self.builder.load(i_ptr);
        let str_idx = self.builder.add(i_val, j);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, str_idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let pat_byte_ptr = self.builder.get_byte_ptr(pattern_ptr, j);
        let pat_byte = self.builder.load_byte(pat_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, pat_byte);
        self.builder.cond_br(bytes_match, inc_j, next_pos);

        // Increment j
        self.builder.start_block(inc_j);
        let j_new = self.builder.add(j, one);
        self.builder.store(j_ptr, j_new);
        self.builder.br(byte_check);

        // Next position (decrement i)
        self.builder.start_block(next_pos);
        let i_val = self.builder.load(i_ptr);
        let i_new = self.builder.sub(i_val, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(check_block);

        // Found: create Some(i)
        self.builder.start_block(found_block);
        let found_i = self.builder.load(i_ptr);
        let one_i32 = self.builder.const_i32(1);
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(discrim_ptr, one_i32);
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr, found_i);
        self.builder.br(done_block);

        // Not found: create None
        self.builder.start_block(not_found_block);
        let zero_i32 = self.builder.const_i32(0);
        let discrim_ptr2 = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(discrim_ptr2, zero_i32);
        let payload_ptr2 = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr2, zero);
        self.builder.br(done_block);

        // Done
        self.builder.start_block(done_block);
        self.vreg_types.insert(result_ptr, opt_ty);
        result_ptr
    }

    /// Lower String::to_uppercase(s) -> String
    fn lower_string_to_uppercase(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load source data and length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(ptr_field);
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Allocate new string
        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let new_data = self.builder.malloc_bytes(len);

        // Loop through and convert each byte
        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop check
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, len);
        self.builder.cond_br(cond, body_block, done_block);

        // Body: convert lowercase to uppercase (ASCII)
        self.builder.start_block(body_block);
        let src_byte_ptr = self.builder.get_byte_ptr(src_data, i);
        let byte_val = self.builder.load_byte(src_byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        // Check if lowercase (97-122 = 'a'-'z')
        let lower_a = self.builder.const_int(97);
        let lower_z = self.builder.const_int(122);
        let is_lower_start = self.builder.icmp(super::instr::CmpOp::Sge, byte_i64, lower_a);
        let is_lower_end = self.builder.icmp(super::instr::CmpOp::Sle, byte_i64, lower_z);
        let is_lower = self.builder.and(is_lower_start, is_lower_end);

        // Convert: subtract 32 if lowercase
        let diff = self.builder.const_int(32);
        let upper_byte = self.builder.sub(byte_i64, diff);
        let result_byte = self.builder.select(is_lower, upper_byte, byte_i64);
        let result_byte_i8 = self.builder.trunc(result_byte, IrType::I8);

        let dst_byte_ptr = self.builder.get_byte_ptr(new_data, i);
        self.builder.store(dst_byte_ptr, result_byte_i8);

        // Increment i
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        // Done: store result
        self.builder.start_block(done_block);
        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, len);
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::to_lowercase(s) -> String
    fn lower_string_to_lowercase(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load source data and length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(ptr_field);
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Allocate new string
        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let new_data = self.builder.malloc_bytes(len);

        // Loop through and convert each byte
        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop check
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, len);
        self.builder.cond_br(cond, body_block, done_block);

        // Body: convert uppercase to lowercase (ASCII)
        self.builder.start_block(body_block);
        let src_byte_ptr = self.builder.get_byte_ptr(src_data, i);
        let byte_val = self.builder.load_byte(src_byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        // Check if uppercase (65-90 = 'A'-'Z')
        let upper_a = self.builder.const_int(65);
        let upper_z = self.builder.const_int(90);
        let is_upper_start = self.builder.icmp(super::instr::CmpOp::Sge, byte_i64, upper_a);
        let is_upper_end = self.builder.icmp(super::instr::CmpOp::Sle, byte_i64, upper_z);
        let is_upper = self.builder.and(is_upper_start, is_upper_end);

        // Convert: add 32 if uppercase
        let diff = self.builder.const_int(32);
        let lower_byte = self.builder.add(byte_i64, diff);
        let result_byte = self.builder.select(is_upper, lower_byte, byte_i64);
        let result_byte_i8 = self.builder.trunc(result_byte, IrType::I8);

        let dst_byte_ptr = self.builder.get_byte_ptr(new_data, i);
        self.builder.store(dst_byte_ptr, result_byte_i8);

        // Increment i
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        // Done: store result
        self.builder.start_block(done_block);
        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, len);
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::trim(s) -> String (remove leading and trailing whitespace)
    fn lower_string_trim(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load source data and length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(ptr_field);
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Find start (skip leading whitespace)
        let start_ptr = self.builder.alloca(IrType::I64);
        let end_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(start_ptr, zero);
        self.builder.store(end_ptr, len);

        // Loop to find start
        let start_loop = self.builder.create_block();
        let start_check = self.builder.create_block();
        let start_done = self.builder.create_block();

        self.builder.br(start_loop);

        self.builder.start_block(start_loop);
        let start = self.builder.load(start_ptr);
        let in_bounds = self.builder.icmp(super::instr::CmpOp::Slt, start, len);
        self.builder.cond_br(in_bounds, start_check, start_done);

        self.builder.start_block(start_check);
        let byte_ptr = self.builder.get_byte_ptr(src_data, start);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        // Check whitespace: space (32), tab (9), newline (10), carriage return (13)
        let const_32 = self.builder.const_int(32);
        let const_9 = self.builder.const_int(9);
        let const_10 = self.builder.const_int(10);
        let const_13 = self.builder.const_int(13);
        let is_space = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_32);
        let is_tab = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_9);
        let is_nl = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_10);
        let is_cr = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_13);
        let ws1 = self.builder.or(is_space, is_tab);
        let ws2 = self.builder.or(is_nl, is_cr);
        let is_ws = self.builder.or(ws1, ws2);

        let inc_start = self.builder.create_block();
        self.builder.cond_br(is_ws, inc_start, start_done);

        self.builder.start_block(inc_start);
        let new_start = self.builder.add(start, one);
        self.builder.store(start_ptr, new_start);
        self.builder.br(start_loop);

        self.builder.start_block(start_done);
        let final_start = self.builder.load(start_ptr);

        // Loop to find end (from the back)
        let end_loop = self.builder.create_block();
        let end_check = self.builder.create_block();
        let end_done = self.builder.create_block();

        self.builder.br(end_loop);

        self.builder.start_block(end_loop);
        let end = self.builder.load(end_ptr);
        let end_gt_start = self.builder.icmp(super::instr::CmpOp::Sgt, end, final_start);
        self.builder.cond_br(end_gt_start, end_check, end_done);

        self.builder.start_block(end_check);
        let end_minus_one = self.builder.sub(end, one);
        let byte_ptr = self.builder.get_byte_ptr(src_data, end_minus_one);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        let const_32 = self.builder.const_int(32);
        let const_9 = self.builder.const_int(9);
        let const_10 = self.builder.const_int(10);
        let const_13 = self.builder.const_int(13);
        let is_space = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_32);
        let is_tab = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_9);
        let is_nl = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_10);
        let is_cr = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_13);
        let ws1 = self.builder.or(is_space, is_tab);
        let ws2 = self.builder.or(is_nl, is_cr);
        let is_ws = self.builder.or(ws1, ws2);

        let dec_end = self.builder.create_block();
        self.builder.cond_br(is_ws, dec_end, end_done);

        self.builder.start_block(dec_end);
        self.builder.store(end_ptr, end_minus_one);
        self.builder.br(end_loop);

        self.builder.start_block(end_done);
        let final_end = self.builder.load(end_ptr);

        // Create substring
        let new_len = self.builder.sub(final_end, final_start);

        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let new_data = self.builder.malloc_bytes(new_len);

        let src_start = self.builder.get_byte_ptr(src_data, final_start);
        self.builder.memcpy(new_data, src_start, new_len);

        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, new_len);
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, new_len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::trim_start(s) -> String (remove leading whitespace)
    fn lower_string_trim_start(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load source data and length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(ptr_field);
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Find start
        let start_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(start_ptr, zero);

        let start_loop = self.builder.create_block();
        let start_check = self.builder.create_block();
        let start_done = self.builder.create_block();

        self.builder.br(start_loop);

        self.builder.start_block(start_loop);
        let start = self.builder.load(start_ptr);
        let in_bounds = self.builder.icmp(super::instr::CmpOp::Slt, start, len);
        self.builder.cond_br(in_bounds, start_check, start_done);

        self.builder.start_block(start_check);
        let byte_ptr = self.builder.get_byte_ptr(src_data, start);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        let const_32 = self.builder.const_int(32);
        let const_9 = self.builder.const_int(9);
        let const_10 = self.builder.const_int(10);
        let const_13 = self.builder.const_int(13);
        let is_space = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_32);
        let is_tab = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_9);
        let is_nl = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_10);
        let is_cr = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_13);
        let ws1 = self.builder.or(is_space, is_tab);
        let ws2 = self.builder.or(is_nl, is_cr);
        let is_ws = self.builder.or(ws1, ws2);

        let inc_start = self.builder.create_block();
        self.builder.cond_br(is_ws, inc_start, start_done);

        self.builder.start_block(inc_start);
        let new_start = self.builder.add(start, one);
        self.builder.store(start_ptr, new_start);
        self.builder.br(start_loop);

        self.builder.start_block(start_done);
        let final_start = self.builder.load(start_ptr);
        let new_len = self.builder.sub(len, final_start);

        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let new_data = self.builder.malloc_bytes(new_len);

        let src_start = self.builder.get_byte_ptr(src_data, final_start);
        self.builder.memcpy(new_data, src_start, new_len);

        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, new_len);
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, new_len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::trim_end(s) -> String (remove trailing whitespace)
    fn lower_string_trim_end(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load source data and length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(ptr_field);
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Find end
        let end_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(end_ptr, len);

        let end_loop = self.builder.create_block();
        let end_check = self.builder.create_block();
        let end_done = self.builder.create_block();

        self.builder.br(end_loop);

        self.builder.start_block(end_loop);
        let end = self.builder.load(end_ptr);
        let end_gt_zero = self.builder.icmp(super::instr::CmpOp::Sgt, end, zero);
        self.builder.cond_br(end_gt_zero, end_check, end_done);

        self.builder.start_block(end_check);
        let end_minus_one = self.builder.sub(end, one);
        let byte_ptr = self.builder.get_byte_ptr(src_data, end_minus_one);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        let const_32 = self.builder.const_int(32);
        let const_9 = self.builder.const_int(9);
        let const_10 = self.builder.const_int(10);
        let const_13 = self.builder.const_int(13);
        let is_space = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_32);
        let is_tab = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_9);
        let is_nl = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_10);
        let is_cr = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, const_13);
        let ws1 = self.builder.or(is_space, is_tab);
        let ws2 = self.builder.or(is_nl, is_cr);
        let is_ws = self.builder.or(ws1, ws2);

        let dec_end = self.builder.create_block();
        self.builder.cond_br(is_ws, dec_end, end_done);

        self.builder.start_block(dec_end);
        self.builder.store(end_ptr, end_minus_one);
        self.builder.br(end_loop);

        self.builder.start_block(end_done);
        let new_len = self.builder.load(end_ptr);

        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let new_data = self.builder.malloc_bytes(new_len);

        self.builder.memcpy(new_data, src_data, new_len);

        let ptr_field = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_field, new_data);
        let len_field = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_field, new_len);
        let cap_field = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_field, new_len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::split(s, delimiter) -> Vec<String>
    fn lower_string_split(&mut self, args: &[Expr]) -> VReg {
        // Create empty Vec first
        let vec_ptr = self.lower_vec_new_for_type(&IrType::Ptr(Box::new(self.string_struct_type())));

        if args.len() < 2 {
            return vec_ptr;
        }

        let str_ptr = self.lower_expr(&args[0]);
        let delim = &args[1];

        // Get delimiter
        let (delim_name, delim_len) = if let ExprKind::Literal(Literal::String(s)) = &delim.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return vec_ptr;
        };

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let delim_ptr = self.builder.global_string_ptr(&delim_name);
        let delim_len_val = self.builder.const_int(delim_len);

        // Split by scanning for delimiter
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        let start_ptr = self.builder.alloca(IrType::I64);
        let i_ptr = self.builder.alloca(IrType::I64);
        let j_ptr = self.builder.alloca(IrType::I64);  // Allocate j_ptr outside loop
        self.builder.store(start_ptr, zero);
        self.builder.store(i_ptr, zero);

        let outer_loop = self.builder.create_block();
        let check_delim = self.builder.create_block();
        let delim_match = self.builder.create_block();
        let delim_byte_check = self.builder.create_block();
        let delim_bytes_equal = self.builder.create_block();
        let delim_inc_j = self.builder.create_block();
        let found_delim = self.builder.create_block();
        let no_delim = self.builder.create_block();
        let done = self.builder.create_block();

        self.builder.br(outer_loop);

        // Outer loop: i < str_len
        self.builder.start_block(outer_loop);
        let i = self.builder.load(i_ptr);
        let max_i = self.builder.sub(str_len, delim_len_val);
        let max_i_plus_one = self.builder.add(max_i, one);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, max_i_plus_one);
        self.builder.cond_br(cond, check_delim, done);

        // Check if delimiter matches at position i
        self.builder.start_block(check_delim);
        self.builder.store(j_ptr, zero);  // Just reset j to 0, don't allocate
        self.builder.br(delim_match);

        self.builder.start_block(delim_match);
        let j = self.builder.load(j_ptr);
        let j_done = self.builder.icmp(super::instr::CmpOp::Sge, j, delim_len_val);
        self.builder.cond_br(j_done, found_delim, delim_byte_check);

        self.builder.start_block(delim_byte_check);
        let i_cur = self.builder.load(i_ptr);
        let str_idx = self.builder.add(i_cur, j);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, str_idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let delim_byte_ptr = self.builder.get_byte_ptr(delim_ptr, j);
        let delim_byte = self.builder.load_byte(delim_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, delim_byte);
        self.builder.cond_br(bytes_match, delim_inc_j, no_delim);

        self.builder.start_block(delim_inc_j);
        let j_new = self.builder.add(j, one);
        self.builder.store(j_ptr, j_new);
        self.builder.br(delim_match);

        // Found delimiter: add substring from start to i
        self.builder.start_block(found_delim);
        let start = self.builder.load(start_ptr);
        let i_cur = self.builder.load(i_ptr);
        let part_len = self.builder.sub(i_cur, start);

        // Create substring - allocate on heap, not stack
        let str_ty = self.string_struct_type();
        let part_ptr = self.builder.malloc(str_ty.clone());
        let part_data = self.builder.malloc_bytes(part_len);
        let src_start = self.builder.get_byte_ptr(str_data, start);
        self.builder.memcpy(part_data, src_start, part_len);

        let ptr_f = self.builder.get_field_ptr(part_ptr, 0);
        self.builder.store(ptr_f, part_data);
        let len_f = self.builder.get_field_ptr(part_ptr, 1);
        self.builder.store(len_f, part_len);
        let cap_f = self.builder.get_field_ptr(part_ptr, 2);
        self.builder.store(cap_f, part_len);
        self.vreg_types.insert(part_ptr, str_ty);

        // Push to vec
        self.lower_vec_push_raw(vec_ptr, part_ptr);

        // Update start to i + delim_len
        let new_start = self.builder.add(i_cur, delim_len_val);
        self.builder.store(start_ptr, new_start);
        self.builder.store(i_ptr, new_start);
        self.builder.br(outer_loop);

        // No delimiter at this position, increment i
        self.builder.start_block(no_delim);
        let i_cur = self.builder.load(i_ptr);
        let i_new = self.builder.add(i_cur, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(outer_loop);

        // Done: add final part from start to end
        self.builder.start_block(done);
        let start = self.builder.load(start_ptr);
        let final_len = self.builder.sub(str_len, start);

        let str_ty = self.string_struct_type();
        let final_ptr = self.builder.malloc(str_ty.clone());
        let final_data = self.builder.malloc_bytes(final_len);
        let src_final = self.builder.get_byte_ptr(str_data, start);
        self.builder.memcpy(final_data, src_final, final_len);

        let ptr_f = self.builder.get_field_ptr(final_ptr, 0);
        self.builder.store(ptr_f, final_data);
        let len_f = self.builder.get_field_ptr(final_ptr, 1);
        self.builder.store(len_f, final_len);
        let cap_f = self.builder.get_field_ptr(final_ptr, 2);
        self.builder.store(cap_f, final_len);
        self.vreg_types.insert(final_ptr, str_ty);

        self.lower_vec_push_raw(vec_ptr, final_ptr);

        vec_ptr
    }

    /// Lower String::lines(s) -> Vec<String>
    fn lower_string_lines(&mut self, args: &[Expr]) -> VReg {
        // Delegate to split with "\n"
        if args.is_empty() {
            return self.lower_vec_new_for_type(&IrType::Ptr(Box::new(self.string_struct_type())));
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Create a fake args array with "\n" as delimiter
        // For simplicity, we'll implement line splitting directly
        let vec_ptr = self.lower_vec_new_for_type(&IrType::Ptr(Box::new(self.string_struct_type())));

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        let newline = self.builder.const_int(10); // '\n'

        let start_ptr = self.builder.alloca(IrType::I64);
        let i_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(start_ptr, zero);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let check_block = self.builder.create_block();
        let found_nl = self.builder.create_block();
        let not_nl = self.builder.create_block();
        let done = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, str_len);
        self.builder.cond_br(cond, check_block, done);

        // Check for newline
        self.builder.start_block(check_block);
        let byte_ptr = self.builder.get_byte_ptr(str_data, i);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);
        let is_nl = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, newline);
        self.builder.cond_br(is_nl, found_nl, not_nl);

        // Found newline
        self.builder.start_block(found_nl);
        let start = self.builder.load(start_ptr);
        let line_len = self.builder.sub(i, start);

        let str_ty = self.string_struct_type();
        let line_ptr = self.builder.malloc(str_ty.clone());
        let line_data = self.builder.malloc_bytes(line_len);
        let src_start = self.builder.get_byte_ptr(str_data, start);
        self.builder.memcpy(line_data, src_start, line_len);

        let ptr_f = self.builder.get_field_ptr(line_ptr, 0);
        self.builder.store(ptr_f, line_data);
        let len_f = self.builder.get_field_ptr(line_ptr, 1);
        self.builder.store(len_f, line_len);
        let cap_f = self.builder.get_field_ptr(line_ptr, 2);
        self.builder.store(cap_f, line_len);
        self.vreg_types.insert(line_ptr, str_ty);

        self.lower_vec_push_raw(vec_ptr, line_ptr);

        let i_plus_one = self.builder.add(i, one);
        self.builder.store(start_ptr, i_plus_one);
        self.builder.store(i_ptr, i_plus_one);
        self.builder.br(loop_block);

        // Not newline
        self.builder.start_block(not_nl);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        // Done: add final line
        self.builder.start_block(done);
        let start = self.builder.load(start_ptr);
        let final_len = self.builder.sub(str_len, start);

        // Only add if non-empty
        let len_gt_zero = self.builder.icmp(super::instr::CmpOp::Sgt, final_len, zero);
        let add_final = self.builder.create_block();
        let finish = self.builder.create_block();
        self.builder.cond_br(len_gt_zero, add_final, finish);

        self.builder.start_block(add_final);
        let str_ty = self.string_struct_type();
        let final_ptr = self.builder.malloc(str_ty.clone());
        let final_data = self.builder.malloc_bytes(final_len);
        let src_final = self.builder.get_byte_ptr(str_data, start);
        self.builder.memcpy(final_data, src_final, final_len);

        let ptr_f = self.builder.get_field_ptr(final_ptr, 0);
        self.builder.store(ptr_f, final_data);
        let len_f = self.builder.get_field_ptr(final_ptr, 1);
        self.builder.store(len_f, final_len);
        let cap_f = self.builder.get_field_ptr(final_ptr, 2);
        self.builder.store(cap_f, final_len);
        self.vreg_types.insert(final_ptr, str_ty);

        self.lower_vec_push_raw(vec_ptr, final_ptr);
        self.builder.br(finish);

        self.builder.start_block(finish);
        vec_ptr
    }

    /// Lower String::replace(s, from, to) -> String
    fn lower_string_replace(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 3 {
            if !args.is_empty() {
                // Return copy of original string
                return self.lower_expr(&args[0]);
            }
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);
        let from_arg = &args[1];
        let to_arg = &args[2];

        // Get from and to strings
        let (from_name, from_len) = if let ExprKind::Literal(Literal::String(s)) = &from_arg.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return str_ptr;
        };

        let (to_name, to_len) = if let ExprKind::Literal(Literal::String(s)) = &to_arg.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return str_ptr;
        };

        // Empty from pattern: return original
        if from_len == 0 {
            return str_ptr;
        }

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let from_ptr = self.builder.global_string_ptr(&from_name);
        let from_len_val = self.builder.const_int(from_len);
        let to_ptr = self.builder.global_string_ptr(&to_name);
        let to_len_val = self.builder.const_int(to_len);

        // First pass: count occurrences to calculate result size
        let count_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(count_ptr, zero);

        let i_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(i_ptr, zero);

        let count_loop = self.builder.create_block();
        let count_check = self.builder.create_block();
        let count_match = self.builder.create_block();
        let count_byte_check = self.builder.create_block();
        let count_bytes_eq = self.builder.create_block();
        let count_inc_j = self.builder.create_block();
        let count_found = self.builder.create_block();
        let count_not_found = self.builder.create_block();
        let count_done = self.builder.create_block();

        self.builder.br(count_loop);

        self.builder.start_block(count_loop);
        let i = self.builder.load(i_ptr);
        let max_i = self.builder.sub(str_len, from_len_val);
        let max_i_plus_one = self.builder.add(max_i, one);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, max_i_plus_one);
        self.builder.cond_br(cond, count_check, count_done);

        self.builder.start_block(count_check);
        let j_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(j_ptr, zero);
        self.builder.br(count_match);

        self.builder.start_block(count_match);
        let j = self.builder.load(j_ptr);
        let j_done = self.builder.icmp(super::instr::CmpOp::Sge, j, from_len_val);
        self.builder.cond_br(j_done, count_found, count_byte_check);

        self.builder.start_block(count_byte_check);
        let i_cur = self.builder.load(i_ptr);
        let idx = self.builder.add(i_cur, j);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let from_byte_ptr = self.builder.get_byte_ptr(from_ptr, j);
        let from_byte = self.builder.load_byte(from_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, from_byte);
        self.builder.cond_br(bytes_match, count_inc_j, count_not_found);

        self.builder.start_block(count_inc_j);
        let j_new = self.builder.add(j, one);
        self.builder.store(j_ptr, j_new);
        self.builder.br(count_match);

        self.builder.start_block(count_found);
        let count = self.builder.load(count_ptr);
        let new_count = self.builder.add(count, one);
        self.builder.store(count_ptr, new_count);
        let i_cur = self.builder.load(i_ptr);
        let new_i = self.builder.add(i_cur, from_len_val);
        self.builder.store(i_ptr, new_i);
        self.builder.br(count_loop);

        self.builder.start_block(count_not_found);
        let i_cur = self.builder.load(i_ptr);
        let new_i = self.builder.add(i_cur, one);
        self.builder.store(i_ptr, new_i);
        self.builder.br(count_loop);

        self.builder.start_block(count_done);
        let count = self.builder.load(count_ptr);

        // Calculate result size: str_len + count * (to_len - from_len)
        let diff = self.builder.sub(to_len_val, from_len_val);
        let size_change = self.builder.mul(count, diff);
        let result_len = self.builder.add(str_len, size_change);

        // Allocate result
        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let result_data = self.builder.malloc_bytes(result_len);

        // Second pass: build result
        let src_ptr = self.builder.alloca(IrType::I64);
        let dst_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(src_ptr, zero);
        self.builder.store(dst_ptr, zero);

        let build_loop = self.builder.create_block();
        let build_check = self.builder.create_block();
        let build_match = self.builder.create_block();
        let build_byte_check = self.builder.create_block();
        let build_bytes_eq = self.builder.create_block();
        let build_inc_j = self.builder.create_block();
        let build_found = self.builder.create_block();
        let build_not_found = self.builder.create_block();
        let build_done = self.builder.create_block();

        self.builder.br(build_loop);

        self.builder.start_block(build_loop);
        let src = self.builder.load(src_ptr);
        let max_src = self.builder.sub(str_len, from_len_val);
        let max_src_plus_one = self.builder.add(max_src, one);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, src, max_src_plus_one);
        self.builder.cond_br(cond, build_check, build_done);

        self.builder.start_block(build_check);
        let j_ptr2 = self.builder.alloca(IrType::I64);
        self.builder.store(j_ptr2, zero);
        self.builder.br(build_match);

        self.builder.start_block(build_match);
        let j = self.builder.load(j_ptr2);
        let j_done = self.builder.icmp(super::instr::CmpOp::Sge, j, from_len_val);
        self.builder.cond_br(j_done, build_found, build_byte_check);

        self.builder.start_block(build_byte_check);
        let src_cur = self.builder.load(src_ptr);
        let idx = self.builder.add(src_cur, j);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, idx);
        let str_byte = self.builder.load_byte(str_byte_ptr);
        let from_byte_ptr = self.builder.get_byte_ptr(from_ptr, j);
        let from_byte = self.builder.load_byte(from_byte_ptr);
        let bytes_match = self.builder.icmp(super::instr::CmpOp::Eq, str_byte, from_byte);
        self.builder.cond_br(bytes_match, build_inc_j, build_not_found);

        self.builder.start_block(build_inc_j);
        let j_new = self.builder.add(j, one);
        self.builder.store(j_ptr2, j_new);
        self.builder.br(build_match);

        // Found: copy 'to' string
        self.builder.start_block(build_found);
        let dst = self.builder.load(dst_ptr);
        let dst_loc = self.builder.get_byte_ptr(result_data, dst);
        self.builder.memcpy(dst_loc, to_ptr, to_len_val);
        let new_dst = self.builder.add(dst, to_len_val);
        self.builder.store(dst_ptr, new_dst);
        let src_cur = self.builder.load(src_ptr);
        let new_src = self.builder.add(src_cur, from_len_val);
        self.builder.store(src_ptr, new_src);
        self.builder.br(build_loop);

        // Not found: copy one byte
        self.builder.start_block(build_not_found);
        let src_cur = self.builder.load(src_ptr);
        let dst_cur = self.builder.load(dst_ptr);
        let str_byte_ptr = self.builder.get_byte_ptr(str_data, src_cur);
        let byte = self.builder.load_byte(str_byte_ptr);
        let dst_byte_ptr = self.builder.get_byte_ptr(result_data, dst_cur);
        self.builder.store(dst_byte_ptr, byte);
        let new_src = self.builder.add(src_cur, one);
        let new_dst = self.builder.add(dst_cur, one);
        self.builder.store(src_ptr, new_src);
        self.builder.store(dst_ptr, new_dst);
        self.builder.br(build_loop);

        // Done: copy remaining bytes
        self.builder.start_block(build_done);
        let src = self.builder.load(src_ptr);
        let dst = self.builder.load(dst_ptr);
        let remaining = self.builder.sub(str_len, src);
        let src_loc = self.builder.get_byte_ptr(str_data, src);
        let dst_loc = self.builder.get_byte_ptr(result_data, dst);
        self.builder.memcpy(dst_loc, src_loc, remaining);

        // Store result fields
        let ptr_f = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_f, result_data);
        let len_f = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_f, result_len);
        let cap_f = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_f, result_len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::repeat(s, n) -> String
    fn lower_string_repeat(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.lower_string_new();
        }

        let str_ptr = self.lower_expr(&args[0]);
        let count = self.lower_expr(&args[1]);

        // Load source length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let src_len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let src_data = self.builder.load(ptr_field);
        let src_data = self.builder.inttoptr(src_data, IrType::Ptr(Box::new(IrType::I8)));

        // Calculate result length
        let result_len = self.builder.mul(src_len, count);

        // Allocate result
        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let result_data = self.builder.malloc_bytes(result_len);

        // Copy string n times
        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, count);
        self.builder.cond_br(cond, body_block, done_block);

        self.builder.start_block(body_block);
        let offset = self.builder.mul(i, src_len);
        let dst = self.builder.get_byte_ptr(result_data, offset);
        self.builder.memcpy(dst, src_data, src_len);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        let ptr_f = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_f, result_data);
        let len_f = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_f, result_len);
        let cap_f = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_f, result_len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    /// Lower String::chars(s) -> Vec<i64>
    fn lower_string_chars(&mut self, args: &[Expr]) -> VReg {
        let vec_ptr = self.lower_vec_new_for_type(&IrType::I64);

        if args.is_empty() {
            return vec_ptr;
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load string data and length
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(ptr_field);

        // Loop through and add each byte as i64
        let i_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        self.builder.store(i_ptr, zero);

        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.br(loop_block);

        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, len);
        self.builder.cond_br(cond, body_block, done_block);

        self.builder.start_block(body_block);
        let byte_ptr = self.builder.get_byte_ptr(str_data, i);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        // Push to vec
        self.lower_vec_push_i64_raw(vec_ptr, byte_i64);

        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        self.builder.start_block(done_block);
        vec_ptr
    }

    /// Lower String::bytes(s) -> Vec<i64> (same as chars for ASCII)
    fn lower_string_bytes(&mut self, args: &[Expr]) -> VReg {
        self.lower_string_chars(args)
    }

    /// Lower String::split_whitespace(s) -> Vec<String>
    /// Splits on whitespace characters: space (32), tab (9), newline (10), carriage return (13)
    fn lower_string_split_whitespace(&mut self, args: &[Expr]) -> VReg {
        let vec_ptr = self.lower_vec_new_for_type(&IrType::Ptr(Box::new(self.string_struct_type())));

        if args.is_empty() {
            return vec_ptr;
        }

        let str_ptr = self.lower_expr(&args[0]);

        // Load string data and length
        let str_len_field = self.builder.get_field_ptr(str_ptr, 1);
        let str_len = self.builder.load(str_len_field);
        let str_ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let str_data = self.builder.load(str_ptr_field);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        let space = self.builder.const_int(32);   // ' '
        let tab = self.builder.const_int(9);      // '\t'
        let newline = self.builder.const_int(10); // '\n'
        let cr = self.builder.const_int(13);      // '\r'

        let start_ptr = self.builder.alloca(IrType::I64);
        let i_ptr = self.builder.alloca(IrType::I64);
        let in_word_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(start_ptr, zero);
        self.builder.store(i_ptr, zero);
        self.builder.store(in_word_ptr, zero);

        let loop_block = self.builder.create_block();
        let check_char = self.builder.create_block();
        let is_ws = self.builder.create_block();
        let not_ws = self.builder.create_block();
        let end_word = self.builder.create_block();
        let inc_i = self.builder.create_block();
        let done = self.builder.create_block();
        let add_final = self.builder.create_block();
        let finish = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop: i < str_len
        self.builder.start_block(loop_block);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, str_len);
        self.builder.cond_br(cond, check_char, done);

        // Check if character is whitespace
        self.builder.start_block(check_char);
        let byte_ptr = self.builder.get_byte_ptr(str_data, i);
        let byte_val = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte_val, IrType::I64);

        let is_space = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, space);
        let is_tab = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, tab);
        let is_nl = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, newline);
        let is_cr = self.builder.icmp(super::instr::CmpOp::Eq, byte_i64, cr);

        let ws1 = self.builder.or(is_space, is_tab);
        let ws2 = self.builder.or(is_nl, is_cr);
        let is_whitespace = self.builder.or(ws1, ws2);

        self.builder.cond_br(is_whitespace, is_ws, not_ws);

        // Is whitespace
        self.builder.start_block(is_ws);
        let in_word = self.builder.load(in_word_ptr);
        let was_in_word = self.builder.icmp(super::instr::CmpOp::Ne, in_word, zero);
        self.builder.cond_br(was_in_word, end_word, inc_i);

        // End of word: extract substring
        self.builder.start_block(end_word);
        let start = self.builder.load(start_ptr);
        let i_cur = self.builder.load(i_ptr);
        let word_len = self.builder.sub(i_cur, start);

        // Create substring on heap
        let str_ty = self.string_struct_type();
        let word_ptr = self.builder.malloc(str_ty.clone());
        let word_data = self.builder.malloc_bytes(word_len);
        let src_start = self.builder.get_byte_ptr(str_data, start);
        self.builder.memcpy(word_data, src_start, word_len);

        let ptr_f = self.builder.get_field_ptr(word_ptr, 0);
        self.builder.store(ptr_f, word_data);
        let len_f = self.builder.get_field_ptr(word_ptr, 1);
        self.builder.store(len_f, word_len);
        let cap_f = self.builder.get_field_ptr(word_ptr, 2);
        self.builder.store(cap_f, word_len);
        self.vreg_types.insert(word_ptr, str_ty);

        self.lower_vec_push_raw(vec_ptr, word_ptr);

        self.builder.store(in_word_ptr, zero);
        self.builder.br(inc_i);

        // Not whitespace: start or continue word
        self.builder.start_block(not_ws);
        let in_word = self.builder.load(in_word_ptr);
        let not_in_word = self.builder.icmp(super::instr::CmpOp::Eq, in_word, zero);

        let start_word = self.builder.create_block();
        let continue_word = self.builder.create_block();
        self.builder.cond_br(not_in_word, start_word, continue_word);

        // Start new word
        self.builder.start_block(start_word);
        let i_cur = self.builder.load(i_ptr);
        self.builder.store(start_ptr, i_cur);
        self.builder.store(in_word_ptr, one);
        self.builder.br(inc_i);

        // Continue in word
        self.builder.start_block(continue_word);
        self.builder.br(inc_i);

        // Increment i
        self.builder.start_block(inc_i);
        let i = self.builder.load(i_ptr);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(loop_block);

        // Done: check if we have a final word
        self.builder.start_block(done);
        let in_word = self.builder.load(in_word_ptr);
        let has_final = self.builder.icmp(super::instr::CmpOp::Ne, in_word, zero);
        self.builder.cond_br(has_final, add_final, finish);

        // Add final word
        self.builder.start_block(add_final);
        let start = self.builder.load(start_ptr);
        let final_len = self.builder.sub(str_len, start);

        let str_ty = self.string_struct_type();
        let final_ptr = self.builder.malloc(str_ty.clone());
        let final_data = self.builder.malloc_bytes(final_len);
        let src_final = self.builder.get_byte_ptr(str_data, start);
        self.builder.memcpy(final_data, src_final, final_len);

        let ptr_f = self.builder.get_field_ptr(final_ptr, 0);
        self.builder.store(ptr_f, final_data);
        let len_f = self.builder.get_field_ptr(final_ptr, 1);
        self.builder.store(len_f, final_len);
        let cap_f = self.builder.get_field_ptr(final_ptr, 2);
        self.builder.store(cap_f, final_len);
        self.vreg_types.insert(final_ptr, str_ty);

        self.lower_vec_push_raw(vec_ptr, final_ptr);
        self.builder.br(finish);

        self.builder.start_block(finish);
        vec_ptr
    }

    /// Lower String::concat_with(vec, separator) -> String
    /// Joins Vec<String> with separator
    fn lower_string_concat_with(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.lower_string_new();
        }

        let vec_ptr = self.lower_expr(&args[0]);
        let sep_arg = &args[1];

        // Get separator
        let (sep_name, sep_len) = if let ExprKind::Literal(Literal::String(s)) = &sep_arg.kind {
            let name = self.builder.add_string_constant(s);
            (name, s.len() as i64)
        } else {
            return self.lower_string_new();
        };

        let sep_ptr = self.builder.global_string_ptr(&sep_name);
        let sep_len_val = self.builder.const_int(sep_len);

        // Load vec data
        let vec_len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let vec_len = self.builder.load(vec_len_field);
        let vec_ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let vec_data = self.builder.load(vec_ptr_field);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        // First pass: calculate total length
        let total_len_ptr = self.builder.alloca(IrType::I64);
        let i_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(total_len_ptr, zero);
        self.builder.store(i_ptr, zero);

        let calc_loop = self.builder.create_block();
        let calc_body = self.builder.create_block();
        let calc_done = self.builder.create_block();

        self.builder.br(calc_loop);

        // Calculate total length
        self.builder.start_block(calc_loop);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, vec_len);
        self.builder.cond_br(cond, calc_body, calc_done);

        self.builder.start_block(calc_body);
        // Get element using get_element_ptr (treats vec_data as array of pointers)
        let elem_ptr_ptr = self.builder.get_element_ptr(vec_data, i);
        let elem_ptr_i64 = self.builder.load(elem_ptr_ptr);
        // Convert i64 to pointer for opaque pointer mode
        let elem_ptr = self.builder.inttoptr(elem_ptr_i64, IrType::Ptr(Box::new(self.string_struct_type())));

        let elem_len_field = self.builder.get_field_ptr(elem_ptr, 1);
        let elem_len = self.builder.load(elem_len_field);

        let total = self.builder.load(total_len_ptr);
        let new_total = self.builder.add(total, elem_len);

        // Add separator length if not first element
        let is_not_first = self.builder.icmp(super::instr::CmpOp::Sgt, i, zero);
        let add_sep = self.builder.create_block();
        let no_sep = self.builder.create_block();
        let next_calc = self.builder.create_block();
        self.builder.cond_br(is_not_first, add_sep, no_sep);

        self.builder.start_block(add_sep);
        let with_sep = self.builder.add(new_total, sep_len_val);
        self.builder.store(total_len_ptr, with_sep);
        self.builder.br(next_calc);

        self.builder.start_block(no_sep);
        self.builder.store(total_len_ptr, new_total);
        self.builder.br(next_calc);

        self.builder.start_block(next_calc);
        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(calc_loop);

        // Allocate result and copy
        self.builder.start_block(calc_done);
        let total_len = self.builder.load(total_len_ptr);

        let str_ty = self.string_struct_type();
        let result_ptr = self.builder.alloca(str_ty.clone());
        let result_data = self.builder.malloc_bytes(total_len);

        let pos_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(i_ptr, zero);
        self.builder.store(pos_ptr, zero);

        let copy_loop = self.builder.create_block();
        let copy_body = self.builder.create_block();
        let copy_done = self.builder.create_block();

        self.builder.br(copy_loop);

        // Copy loop
        self.builder.start_block(copy_loop);
        let i = self.builder.load(i_ptr);
        let cond = self.builder.icmp(super::instr::CmpOp::Slt, i, vec_len);
        self.builder.cond_br(cond, copy_body, copy_done);

        self.builder.start_block(copy_body);
        let pos = self.builder.load(pos_ptr);

        // Add separator if not first
        let is_not_first = self.builder.icmp(super::instr::CmpOp::Sgt, i, zero);
        let add_sep_block = self.builder.create_block();
        let copy_elem_block = self.builder.create_block();
        self.builder.cond_br(is_not_first, add_sep_block, copy_elem_block);

        // Add separator
        self.builder.start_block(add_sep_block);
        let sep_dst = self.builder.get_byte_ptr(result_data, pos);
        self.builder.memcpy(sep_dst, sep_ptr, sep_len_val);
        let pos_after_sep = self.builder.add(pos, sep_len_val);
        self.builder.store(pos_ptr, pos_after_sep);
        self.builder.br(copy_elem_block);

        // Copy element
        self.builder.start_block(copy_elem_block);
        let pos = self.builder.load(pos_ptr);
        // Get element using get_element_ptr
        let elem_ptr_ptr = self.builder.get_element_ptr(vec_data, i);
        let elem_ptr_i64 = self.builder.load(elem_ptr_ptr);
        let elem_ptr = self.builder.inttoptr(elem_ptr_i64, IrType::Ptr(Box::new(self.string_struct_type())));

        let elem_data_field = self.builder.get_field_ptr(elem_ptr, 0);
        let elem_data = self.builder.load(elem_data_field);
        let elem_len_field = self.builder.get_field_ptr(elem_ptr, 1);
        let elem_len = self.builder.load(elem_len_field);

        let elem_dst = self.builder.get_byte_ptr(result_data, pos);
        self.builder.memcpy(elem_dst, elem_data, elem_len);

        let new_pos = self.builder.add(pos, elem_len);
        self.builder.store(pos_ptr, new_pos);

        let i_new = self.builder.add(i, one);
        self.builder.store(i_ptr, i_new);
        self.builder.br(copy_loop);

        // Done: set up result string
        self.builder.start_block(copy_done);
        let ptr_f = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(ptr_f, result_data);
        let len_f = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(len_f, total_len);
        let cap_f = self.builder.get_field_ptr(result_ptr, 2);
        self.builder.store(cap_f, total_len);

        self.vreg_types.insert(result_ptr, str_ty);
        result_ptr
    }

    // Helper: Create Option::Some with a constant i64 value
    fn create_option_some_i64(&mut self, value: i64) -> VReg {
        let val = self.builder.const_int(value);
        self.create_option_some(val)
    }

    /// Helper: Create new Vec (uses existing lower_vec_new)
    fn lower_vec_new_for_type(&mut self, _elem_ty: &IrType) -> VReg {
        self.lower_vec_new()
    }

    /// Helper: Push pointer value to Vec<String>
    fn lower_vec_push_raw(&mut self, vec_ptr: VReg, value: VReg) {
        // Load current len and cap
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        let cap = self.builder.load(cap_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Check if we need to grow
        let needs_grow = self.builder.icmp(super::instr::CmpOp::Sge, len, cap);

        let grow_block = self.builder.create_block();
        let push_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(needs_grow, grow_block, push_block);

        // Grow
        self.builder.start_block(grow_block);
        let zero = self.builder.const_int(0);
        let eight = self.builder.const_int(8);
        let sixteen = self.builder.const_int(16);
        let two = self.builder.const_int(2);

        let cap_is_zero = self.builder.icmp(super::instr::CmpOp::Eq, cap, zero);
        let doubled = self.builder.mul(cap, two);
        let new_cap = self.builder.select(cap_is_zero, sixteen, doubled);
        let new_size = self.builder.mul(new_cap, eight);
        let new_data = self.builder.realloc(data_ptr, new_size);

        self.builder.store(ptr_field, new_data);
        self.builder.store(cap_field, new_cap);
        self.builder.br(push_block);

        // Push
        self.builder.start_block(push_block);
        let data_now = self.builder.load(ptr_field);
        let slot = self.builder.get_element_ptr(data_now, len);
        self.builder.store(slot, value);

        let one = self.builder.const_int(1);
        let new_len = self.builder.add(len, one);
        self.builder.store(len_field, new_len);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
    }

    /// Helper: Push i64 value to Vec<i64>
    fn lower_vec_push_i64_raw(&mut self, vec_ptr: VReg, value: VReg) {
        // Load current len and cap
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        let len = self.builder.load(len_field);
        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        let cap = self.builder.load(cap_field);
        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Check if we need to grow
        let needs_grow = self.builder.icmp(super::instr::CmpOp::Sge, len, cap);

        let grow_block = self.builder.create_block();
        let push_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(needs_grow, grow_block, push_block);

        // Grow
        self.builder.start_block(grow_block);
        let zero = self.builder.const_int(0);
        let eight = self.builder.const_int(8);
        let sixteen = self.builder.const_int(16);
        let two = self.builder.const_int(2);

        let cap_is_zero = self.builder.icmp(super::instr::CmpOp::Eq, cap, zero);
        let doubled = self.builder.mul(cap, two);
        let new_cap = self.builder.select(cap_is_zero, sixteen, doubled);
        let new_size = self.builder.mul(new_cap, eight);
        let new_data = self.builder.realloc(data_ptr, new_size);

        self.builder.store(ptr_field, new_data);
        self.builder.store(cap_field, new_cap);
        self.builder.br(push_block);

        // Push
        self.builder.start_block(push_block);
        let data_now = self.builder.load(ptr_field);
        let slot = self.builder.get_element_ptr(data_now, len);
        self.builder.store(slot, value);

        let one = self.builder.const_int(1);
        let new_len = self.builder.add(len, one);
        self.builder.store(len_field, new_len);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
    }

    // ============ Option<T> Methods ============
    // Option layout: { i32 discriminant, i64 payload }
    // None = discriminant 0
    // Some(T) = discriminant 1, payload = value

    /// Lower Option::is_some(opt) -> bool
    /// Returns true if the Option contains a value (discriminant == 1)
    fn lower_option_is_some(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(false);
        }

        let opt_ptr = self.lower_expr(&args[0]);

        // Get discriminant (field 0)
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Check if discriminant == 1 (Some)
        let one = self.builder.const_int(1);
        // Truncate to i32 for comparison
        let one_i32 = self.builder.trunc(one, IrType::I32);
        self.builder.icmp(super::instr::CmpOp::Eq, discrim, one_i32)
    }

    /// Lower Option::is_none(opt) -> bool
    /// Returns true if the Option is empty (discriminant == 0)
    fn lower_option_is_none(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(true);
        }

        let opt_ptr = self.lower_expr(&args[0]);

        // Get discriminant (field 0)
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Check if discriminant == 0 (None)
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        self.builder.icmp(super::instr::CmpOp::Eq, discrim, zero_i32)
    }

    /// Lower Option::unwrap(opt) -> T
    /// Extracts the value from Some. Undefined behavior if None.
    fn lower_option_unwrap(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let opt_ptr = self.lower_expr(&args[0]);

        // Get payload (field 1)
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        self.builder.load(payload_ptr)
    }

    /// Lower Option::unwrap_or(opt, default) -> T
    /// Returns the value if Some, otherwise returns default.
    fn lower_option_unwrap_or(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let opt_ptr = self.lower_expr(&args[0]);
        let default_val = self.lower_expr(&args[1]);

        // Get discriminant
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if is_some (discriminant == 1)
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        let is_some = self.builder.icmp(super::instr::CmpOp::Eq, discrim, one_i32);

        // Select payload if Some, otherwise default
        self.builder.select(is_some, payload, default_val)
    }

    /// Lower Option::expect(opt, msg) -> T
    /// Like unwrap, but with an error message. For now, same as unwrap.
    fn lower_option_expect(&mut self, args: &[Expr]) -> VReg {
        // For now, expect behaves like unwrap
        // In the future, we could add a check and print the message before aborting
        self.lower_option_unwrap(args)
    }

    // ============ Result<T, E> Methods ============
    // Result layout: { i32 discriminant, i64 payload }
    // Ok(T) = discriminant 0, payload = value
    // Err(E) = discriminant 1, payload = error

    /// Lower Result::is_ok(result) -> bool
    /// Returns true if the Result is Ok (discriminant == 0)
    fn lower_result_is_ok(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(false);
        }

        let result_ptr = self.lower_expr(&args[0]);

        // Get discriminant (field 0)
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Check if discriminant == 0 (Ok)
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        self.builder.icmp(super::instr::CmpOp::Eq, discrim, zero_i32)
    }

    /// Lower Result::is_err(result) -> bool
    /// Returns true if the Result is Err (discriminant == 1)
    fn lower_result_is_err(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(false);
        }

        let result_ptr = self.lower_expr(&args[0]);

        // Get discriminant (field 0)
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Check if discriminant == 1 (Err)
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        self.builder.icmp(super::instr::CmpOp::Eq, discrim, one_i32)
    }

    /// Lower Result::unwrap(result) -> T
    /// Extracts the Ok value. Undefined behavior if Err.
    fn lower_result_unwrap(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let result_ptr = self.lower_expr(&args[0]);

        // Get payload (field 1)
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.load(payload_ptr)
    }

    /// Lower Result::unwrap_err(result) -> E
    /// Extracts the Err value. Undefined behavior if Ok.
    fn lower_result_unwrap_err(&mut self, args: &[Expr]) -> VReg {
        // Same as unwrap - just get the payload
        self.lower_result_unwrap(args)
    }

    /// Lower Result::unwrap_or(result, default) -> T
    /// Returns the Ok value if Ok, otherwise returns default.
    fn lower_result_unwrap_or(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let result_ptr = self.lower_expr(&args[0]);
        let default_val = self.lower_expr(&args[1]);

        // Get discriminant
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if is_ok (discriminant == 0)
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        let is_ok = self.builder.icmp(super::instr::CmpOp::Eq, discrim, zero_i32);

        // Select payload if Ok, otherwise default
        self.builder.select(is_ok, payload, default_val)
    }

    /// Lower Result::expect(result, msg) -> T
    /// Like unwrap, but with an error message. For now, same as unwrap.
    fn lower_result_expect(&mut self, args: &[Expr]) -> VReg {
        self.lower_result_unwrap(args)
    }

    /// Lower Option::map(opt, f) -> Option<U>
    /// If Some, applies f to the value and returns Some(result).
    /// If None, returns None.
    fn lower_option_map(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let opt_ptr = self.lower_expr(&args[0]);
        let closure_ptr = self.lower_expr(&args[1]);

        // Get discriminant from source
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload from source
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if Some (discriminant == 1)
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        let is_some = self.builder.icmp(super::instr::CmpOp::Eq, discrim, one_i32);

        // Create blocks for branching
        let some_block = self.builder.create_block();
        let none_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_some, some_block, none_block);

        // Some branch: call closure with payload, wrap in new Some
        self.builder.start_block(some_block);

        // Extract fn_ptr and env_ptr from closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Call closure with env_ptr and payload
        let mapped_val = self.builder.call_ptr(fn_ptr, vec![env_ptr, payload]);

        // Create Some(mapped_val) - use malloc for heap allocation (survives block scope)
        let some_result = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));
        let some_discrim_ptr = self.builder.get_field_ptr(some_result, 0);
        self.builder.store(some_discrim_ptr, one_i32);
        let some_payload_ptr = self.builder.get_field_ptr(some_result, 1);
        self.builder.store(some_payload_ptr, mapped_val);

        let some_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // None branch: create None - use malloc for heap allocation
        self.builder.start_block(none_block);
        let none_result = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));
        let none_discrim_ptr = self.builder.get_field_ptr(none_result, 0);
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        self.builder.store(none_discrim_ptr, zero_i32);

        let none_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(some_result, some_exit), (none_result, none_exit)])
    }

    /// Lower Option::and_then(opt, f) -> Option<U>
    /// If Some, applies f to the value (f returns Option<U>).
    /// If None, returns None.
    fn lower_option_and_then(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let opt_ptr = self.lower_expr(&args[0]);
        let closure_ptr = self.lower_expr(&args[1]);

        // Get discriminant from source
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload from source
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if Some (discriminant == 1)
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        let is_some = self.builder.icmp(super::instr::CmpOp::Eq, discrim, one_i32);

        // Create blocks for branching
        let some_block = self.builder.create_block();
        let none_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_some, some_block, none_block);

        // Some branch: call closure with payload (returns Option<U>)
        self.builder.start_block(some_block);

        // Extract fn_ptr and env_ptr from closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Call closure with env_ptr and payload - returns Option<U> pointer as i64
        // (LLVM codegen treats all CallPtr results as i64)
        let call_result = self.builder.call_ptr(fn_ptr, vec![env_ptr, payload]);
        // Convert i64 to pointer
        let option_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let some_result = self.builder.inttoptr(call_result, IrType::Ptr(Box::new(option_ty.clone())));

        let some_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // None branch: create None (must use heap allocation like enum constructors)
        self.builder.start_block(none_block);
        let none_result = self.builder.malloc(option_ty);
        let none_discrim_ptr = self.builder.get_field_ptr(none_result, 0);
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        self.builder.store(none_discrim_ptr, zero_i32);

        let none_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(some_result, some_exit), (none_result, none_exit)])
    }

    /// Lower Result::map(result, f) -> Result<U, E>
    /// If Ok, applies f to the value and returns Ok(result).
    /// If Err, returns Err unchanged.
    fn lower_result_map(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let result_ptr = self.lower_expr(&args[0]);
        let closure_ptr = self.lower_expr(&args[1]);

        // Get discriminant from source
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload from source
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if Ok (discriminant == 0)
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        let is_ok = self.builder.icmp(super::instr::CmpOp::Eq, discrim, zero_i32);

        // Create blocks for branching
        let ok_block = self.builder.create_block();
        let err_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_ok, ok_block, err_block);

        // Ok branch: call closure with payload, wrap in new Ok
        self.builder.start_block(ok_block);

        // Extract fn_ptr and env_ptr from closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Call closure with env_ptr and payload
        let mapped_val = self.builder.call_ptr(fn_ptr, vec![env_ptr, payload]);

        // Create Ok(mapped_val) - use malloc for heap allocation
        let ok_result = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));
        let ok_discrim_ptr = self.builder.get_field_ptr(ok_result, 0);
        self.builder.store(ok_discrim_ptr, zero_i32);
        let ok_payload_ptr = self.builder.get_field_ptr(ok_result, 1);
        self.builder.store(ok_payload_ptr, mapped_val);

        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Err branch: preserve Err unchanged - use malloc for heap allocation
        self.builder.start_block(err_block);
        let err_result = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));
        let err_discrim_ptr = self.builder.get_field_ptr(err_result, 0);
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        self.builder.store(err_discrim_ptr, one_i32);
        let err_payload_ptr = self.builder.get_field_ptr(err_result, 1);
        self.builder.store(err_payload_ptr, payload);

        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(ok_result, ok_exit), (err_result, err_exit)])
    }

    /// Lower Result::map_err(result, f) -> Result<T, F>
    /// If Err, applies f to the error and returns Err(result).
    /// If Ok, returns Ok unchanged.
    fn lower_result_map_err(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let result_ptr = self.lower_expr(&args[0]);
        let closure_ptr = self.lower_expr(&args[1]);

        // Get discriminant from source
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload from source
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if Err (discriminant == 1)
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        let is_err = self.builder.icmp(super::instr::CmpOp::Eq, discrim, one_i32);

        // Create blocks for branching
        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_err, err_block, ok_block);

        // Err branch: call closure with payload, wrap in new Err
        self.builder.start_block(err_block);

        // Extract fn_ptr and env_ptr from closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Call closure with env_ptr and payload
        let mapped_val = self.builder.call_ptr(fn_ptr, vec![env_ptr, payload]);

        // Create Err(mapped_val) - use malloc for heap allocation
        let err_result = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));
        let err_discrim_ptr = self.builder.get_field_ptr(err_result, 0);
        self.builder.store(err_discrim_ptr, one_i32);
        let err_payload_ptr = self.builder.get_field_ptr(err_result, 1);
        self.builder.store(err_payload_ptr, mapped_val);

        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Ok branch: preserve Ok unchanged - use malloc for heap allocation
        self.builder.start_block(ok_block);
        let ok_result = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));
        let ok_discrim_ptr = self.builder.get_field_ptr(ok_result, 0);
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        self.builder.store(ok_discrim_ptr, zero_i32);
        let ok_payload_ptr = self.builder.get_field_ptr(ok_result, 1);
        self.builder.store(ok_payload_ptr, payload);

        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(err_result, err_exit), (ok_result, ok_exit)])
    }

    /// Lower Result::and_then(result, f) -> Result<U, E>
    /// If Ok, applies f to the value (f returns Result<U, E>).
    /// If Err, returns Err unchanged.
    fn lower_result_and_then(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let result_ptr = self.lower_expr(&args[0]);
        let closure_ptr = self.lower_expr(&args[1]);

        // Get discriminant from source
        let discrim_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let discrim = self.builder.load(discrim_ptr);

        // Get payload from source
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        let payload = self.builder.load(payload_ptr);

        // Check if Ok (discriminant == 0)
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.trunc(zero, IrType::I32);
        let is_ok = self.builder.icmp(super::instr::CmpOp::Eq, discrim, zero_i32);

        // Create blocks for branching
        let ok_block = self.builder.create_block();
        let err_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_ok, ok_block, err_block);

        // Ok branch: call closure with payload (returns Result<U, E>)
        self.builder.start_block(ok_block);

        // Extract fn_ptr and env_ptr from closure
        let fn_ptr_field = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let env_ptr_field = self.builder.get_field_ptr(closure_ptr, 1);
        let env_ptr = self.builder.load(env_ptr_field);

        // Call closure with env_ptr and payload - returns Result<U, E> pointer as i64
        let call_result = self.builder.call_ptr(fn_ptr, vec![env_ptr, payload]);
        // Convert i64 to pointer
        let result_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let ok_result = self.builder.inttoptr(call_result, IrType::Ptr(Box::new(result_ty.clone())));

        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Err branch: preserve Err unchanged (must use heap allocation)
        self.builder.start_block(err_block);
        let err_result = self.builder.malloc(result_ty);
        let err_discrim_ptr = self.builder.get_field_ptr(err_result, 0);
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        self.builder.store(err_discrim_ptr, one_i32);
        let err_payload_ptr = self.builder.get_field_ptr(err_result, 1);
        self.builder.store(err_payload_ptr, payload);

        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(ok_result, ok_exit), (err_result, err_exit)])
    }

    /// Lower a print/println call
    fn lower_print_call(&mut self, args: &[Expr], with_newline: bool) -> VReg {
        // Ensure stdio is declared
        self.builder.declare_stdio();

        if args.is_empty() {
            // Empty print - just print newline if println
            if with_newline {
                let nl_str = self.builder.add_string_constant("");
                let nl_ptr = self.builder.global_string_ptr(&nl_str);
                self.builder.call("puts", vec![nl_ptr])
            } else {
                self.builder.const_int(0)
            }
        } else {
            // For now, only handle first argument
            let arg = &args[0];

            match &arg.kind {
                ExprKind::Literal(Literal::String(s)) => {
                    // String literal - use puts (adds newline) or printf
                    if with_newline {
                        let str_name = self.builder.add_string_constant(s);
                        let str_ptr = self.builder.global_string_ptr(&str_name);
                        self.builder.call("puts", vec![str_ptr])
                    } else {
                        // printf without newline
                        let str_name = self.builder.add_string_constant(s);
                        let str_ptr = self.builder.global_string_ptr(&str_name);
                        self.builder.call("printf", vec![str_ptr])
                    }
                }
                _ => {
                    // Check if this is a String type (from expr_types)
                    let is_string_from_expr = self.expr_types.get(&arg.span).map_or(false, |ty| {
                        if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                            name == "String"
                        } else {
                            false
                        }
                    });

                    // Lower the expression first so we can check vreg_types
                    let val = self.lower_expr(arg);

                    // Also check vreg_types for propagated String pointers (from Vec::get, etc.)
                    let is_string_from_vreg = self.vreg_types.get(&val).map_or(false, |ty| {
                        if let IrType::Ptr(inner) = ty {
                            // Check if it's a pointer to String struct (3 fields: ptr, len, cap)
                            if let IrType::Struct(fields) = inner.as_ref() {
                                fields.len() == 3 &&
                                matches!(&fields[0], IrType::Ptr(_)) &&
                                matches!(&fields[1], IrType::I64) &&
                                matches!(&fields[2], IrType::I64)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    });

                    let is_string = is_string_from_expr || is_string_from_vreg;

                    if is_string {
                        // Use the already-lowered value as String pointer
                        let string_ptr = val;
                        // Get pointer to data field (field 0 of String struct)
                        let data_ptr_ptr = self.builder.get_field_ptr(string_ptr, 0);
                        // Load the actual char pointer
                        let data_ptr = self.builder.load(data_ptr_ptr);
                        // Get the length field (field 1 of String struct)
                        let len_ptr = self.builder.get_field_ptr(string_ptr, 1);
                        let len_i64 = self.builder.load(len_ptr);
                        // Truncate len to i32 for printf precision (required by %.*s)
                        let len = self.builder.trunc(len_i64, IrType::I32);

                        // Use printf with %.*s to print exact length (no null terminator needed)
                        let format = if with_newline { "%.*s\n" } else { "%.*s" };
                        let fmt_name = self.builder.add_string_constant(format);
                        let fmt_ptr = self.builder.global_string_ptr(&fmt_name);
                        // printf("%.*s", len, data_ptr) - note: len comes before data_ptr
                        self.builder.call("printf", vec![fmt_ptr, len, data_ptr])
                    } else {
                        // For other types, use printf with format string
                        // (val already lowered above)

                        // Determine format string based on type
                        // First check expr_types (type checker info), then fallback to vreg_types (IR info)
                        let is_float = self.expr_types.get(&arg.span).map_or(false, |ty| {
                            matches!(ty.kind, crate::typeck::TyKind::Float(_))
                        }) || self.vreg_types.get(&val).map_or(false, |ty| {
                            matches!(ty, IrType::F32 | IrType::F64)
                        });
                        let is_bool = self.expr_types.get(&arg.span).map_or(false, |ty| {
                            matches!(ty.kind, crate::typeck::TyKind::Bool)
                        }) || self.vreg_types.get(&val).map_or(false, |ty| {
                            matches!(ty, IrType::Bool)
                        });

                        if is_float {
                            let format = if with_newline { "%g\n" } else { "%g" };
                            let fmt_name = self.builder.add_string_constant(format);
                            let fmt_ptr = self.builder.global_string_ptr(&fmt_name);
                            self.builder.call("printf", vec![fmt_ptr, val])
                        } else if is_bool {
                            // For booleans, print "true" or "false"
                            let true_str = self.builder.add_string_constant(if with_newline { "true\n" } else { "true" });
                            let false_str = self.builder.add_string_constant(if with_newline { "false\n" } else { "false" });
                            let true_ptr = self.builder.global_string_ptr(&true_str);
                            let false_ptr = self.builder.global_string_ptr(&false_str);
                            let str_ptr = self.builder.select(val, true_ptr, false_ptr);
                            self.builder.call("printf", vec![str_ptr])
                        } else {
                            // Default: integers
                            let format = if with_newline { "%lld\n" } else { "%lld" };
                            let fmt_name = self.builder.add_string_constant(format);
                            let fmt_ptr = self.builder.global_string_ptr(&fmt_name);
                            self.builder.call("printf", vec![fmt_ptr, val])
                        }
                    }
                }
            }
        }
    }

    /// Lower an eprint/eprintln call (prints to stderr)
    fn lower_eprint_call(&mut self, args: &[Expr], with_newline: bool) -> VReg {
        // Ensure stdio is declared (includes stderr)
        self.builder.declare_stdio();

        // Get stderr file handle
        let stderr = self.builder.get_stderr();

        if args.is_empty() {
            // Empty eprint - just print newline if eprintln
            if with_newline {
                let nl_str = self.builder.add_string_constant("\n");
                let nl_ptr = self.builder.global_string_ptr(&nl_str);
                self.builder.call("fputs", vec![nl_ptr, stderr])
            } else {
                self.builder.const_int(0)
            }
        } else {
            // For now, only handle first argument
            let arg = &args[0];

            match &arg.kind {
                ExprKind::Literal(Literal::String(s)) => {
                    // String literal - use fputs or fprintf
                    let content = if with_newline {
                        format!("{}\n", s)
                    } else {
                        s.clone()
                    };
                    let str_name = self.builder.add_string_constant(&content);
                    let str_ptr = self.builder.global_string_ptr(&str_name);
                    self.builder.call("fputs", vec![str_ptr, stderr])
                }
                _ => {
                    // For other types (integers), use fprintf with format string
                    let val = self.lower_expr(arg);

                    // Determine format string based on type
                    let format = if with_newline { "%lld\n" } else { "%lld" };
                    let fmt_name = self.builder.add_string_constant(format);
                    let fmt_ptr = self.builder.global_string_ptr(&fmt_name);

                    self.builder.call("fprintf", vec![stderr, fmt_ptr, val])
                }
            }
        }
    }

    /// Lower a read_line call (reads a line from stdin)
    fn lower_read_line(&mut self) -> VReg {
        // Ensure stdio is declared (includes stdin and fgets)
        self.builder.declare_stdio();

        // Get stdin file handle
        let stdin = self.builder.get_stdin();

        // Allocate buffer on heap (1024 bytes should be enough for most lines)
        let buffer_size = 1024i64;
        let size_vreg = self.builder.const_int(buffer_size);
        let buffer = self.builder.malloc_array(IrType::I8, size_vreg);

        // Call fgets(buffer, size, stdin) to read a line
        let size_i32 = self.builder.trunc(size_vreg, IrType::I32);
        let _result = self.builder.call("fgets", vec![buffer, size_i32, stdin]);

        // Create a String struct to hold the result
        // String = { *i8 data, i64 len, i64 cap }
        let string_ptr = self.builder.alloca(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        // Store data pointer
        let data_ptr = self.builder.get_field_ptr(string_ptr, 0);
        self.builder.store(data_ptr, buffer);

        // Calculate length using strlen
        let len = self.builder.call("strlen", vec![buffer]);

        // Store length
        let len_ptr = self.builder.get_field_ptr(string_ptr, 1);
        self.builder.store(len_ptr, len);

        // Store capacity
        let cap_ptr = self.builder.get_field_ptr(string_ptr, 2);
        self.builder.store(cap_ptr, size_vreg);

        string_ptr
    }

    /// Lower an integer to_string call (i64::to_string or i32::to_string)
    fn lower_int_to_string(&mut self, args: &[Expr], int_type: &str) -> VReg {
        self.builder.declare_stdio();

        // Get the integer value
        let int_val = if !args.is_empty() {
            self.lower_expr(&args[0])
        } else {
            self.builder.const_int(0)
        };

        // Allocate buffer on heap (32 bytes is enough for any 64-bit integer)
        let buffer_size = 32i64;
        let size_vreg = self.builder.const_int(buffer_size);
        let buffer = self.builder.malloc_array(IrType::I8, size_vreg);

        // Use sprintf to format the integer into the buffer
        // For i64: "%lld", for i32: "%d"
        let format_str = if int_type == "i64" { "%lld" } else { "%d" };
        let fmt_name = self.builder.add_string_constant(format_str);
        let fmt_ptr = self.builder.global_string_ptr(&fmt_name);

        // Call sprintf(buffer, format, value)
        let _result = self.builder.call("sprintf", vec![buffer, fmt_ptr, int_val]);

        // Create a String struct to hold the result
        // String = { *i8 data, i64 len, i64 cap }
        let string_ptr = self.builder.alloca(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        // Store data pointer
        let data_ptr = self.builder.get_field_ptr(string_ptr, 0);
        self.builder.store(data_ptr, buffer);

        // Calculate length using strlen
        let len = self.builder.call("strlen", vec![buffer]);

        // Store length
        let len_ptr = self.builder.get_field_ptr(string_ptr, 1);
        self.builder.store(len_ptr, len);

        // Store capacity
        let cap_ptr = self.builder.get_field_ptr(string_ptr, 2);
        self.builder.store(cap_ptr, size_vreg);

        string_ptr
    }

    /// Lower a bool to_string call
    fn lower_bool_to_string(&mut self, args: &[Expr]) -> VReg {
        self.builder.declare_stdio();

        // Get the bool value
        let bool_val = if !args.is_empty() {
            self.lower_expr(&args[0])
        } else {
            self.builder.const_int(0)
        };

        // Create blocks for true and false branches
        let true_block = self.builder.fresh_block();
        let false_block = self.builder.fresh_block();
        let merge_block = self.builder.fresh_block();

        // Branch based on bool value
        self.builder.cond_br(bool_val, true_block, false_block);

        // True branch: return "true"
        self.builder.start_block(true_block);
        let true_str = self.builder.add_string_constant("true");
        let true_ptr = self.builder.global_string_ptr(&true_str);

        // Allocate String struct for "true"
        let true_string = self.builder.alloca(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));
        let true_data = self.builder.get_field_ptr(true_string, 0);
        self.builder.store(true_data, true_ptr);
        let true_len = self.builder.const_int(4); // "true" has 4 chars
        let true_len_ptr = self.builder.get_field_ptr(true_string, 1);
        self.builder.store(true_len_ptr, true_len);
        let true_cap = self.builder.const_int(5);
        let true_cap_ptr = self.builder.get_field_ptr(true_string, 2);
        self.builder.store(true_cap_ptr, true_cap);
        let true_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // False branch: return "false"
        self.builder.start_block(false_block);
        let false_str = self.builder.add_string_constant("false");
        let false_ptr = self.builder.global_string_ptr(&false_str);

        // Allocate String struct for "false"
        let false_string = self.builder.alloca(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));
        let false_data = self.builder.get_field_ptr(false_string, 0);
        self.builder.store(false_data, false_ptr);
        let false_len = self.builder.const_int(5); // "false" has 5 chars
        let false_len_ptr = self.builder.get_field_ptr(false_string, 1);
        self.builder.store(false_len_ptr, false_len);
        let false_cap = self.builder.const_int(6);
        let false_cap_ptr = self.builder.get_field_ptr(false_string, 2);
        self.builder.store(false_cap_ptr, false_cap);
        let false_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block: phi between the two strings
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(true_string, true_exit), (false_string, false_exit)])
    }

    /// Lower a parse call (i64::parse or i32::parse)
    /// Uses strtol from libc. Returns Result<i64/i32, i64>
    /// Error codes: 1 = empty/whitespace only, 2 = invalid characters
    fn lower_parse_int(&mut self, args: &[Expr], int_type: &str) -> VReg {
        self.builder.declare_stdio();

        // Get the String argument
        let string_ptr = if !args.is_empty() {
            self.lower_expr(&args[0])
        } else {
            // Return Err(1) for no argument
            let err_code = self.builder.const_int(1);
            return self.create_result_err(err_code);
        };

        // Load the data pointer from String struct (field 0)
        // (comes as i64 with opaque pointers, convert back to ptr)
        let data_field_ptr = self.builder.get_field_ptr(string_ptr, 0);
        let str_ptr = self.builder.load(data_field_ptr);
        let str_ptr = self.builder.inttoptr(str_ptr, IrType::Ptr(Box::new(IrType::I8)));

        // Allocate space for endptr (char**)
        let endptr = self.builder.alloca(IrType::Ptr(Box::new(IrType::I8)));

        // Call strtol(str, &endptr, 10)
        // strtol returns long (i64 on 64-bit systems)
        // base must be i32
        let base = self.builder.const_i32(10);
        let result = self.builder.call("strtol", vec![str_ptr, endptr, base]);

        // Load endptr value (comes as i64 with opaque pointers, convert back to ptr)
        let endptr_val = self.builder.load(endptr);
        let endptr_val = self.builder.inttoptr(endptr_val, IrType::Ptr(Box::new(IrType::I8)));

        // Check if endptr == str_ptr (nothing was parsed - empty or whitespace)
        // Convert pointers to integers for comparison (LLVM opaque pointers)
        let endptr_int = self.builder.ptrtoint(endptr_val, IrType::I64);
        let str_int = self.builder.ptrtoint(str_ptr, IrType::I64);
        let ptrs_equal = self.builder.icmp(CmpOp::Eq, endptr_int, str_int);

        // Create blocks for error checking
        let empty_err_block = self.builder.create_block();
        let check_trailing_block = self.builder.create_block();
        let invalid_err_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If endptr == str_ptr, error (empty/whitespace)
        self.builder.cond_br(ptrs_equal, empty_err_block, check_trailing_block);

        // Empty error block: return Err(1)
        self.builder.start_block(empty_err_block);
        let err1_code = self.builder.const_int(1);
        let err1_result = self.create_result_err(err1_code);
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Check for trailing characters block
        self.builder.start_block(check_trailing_block);
        // Load the byte at *endptr using load_byte (returns i8, then extend to i64)
        let trailing_char_i8 = self.builder.load_byte(endptr_val);
        let trailing_char = self.builder.zext(trailing_char_i8, IrType::I64);
        // Check if it's null terminator (0) or newline (10, from read_line)
        let zero = self.builder.const_int(0);
        let is_null = self.builder.icmp(CmpOp::Eq, trailing_char, zero);
        let ten = self.builder.const_int(10);
        let is_newline = self.builder.icmp(CmpOp::Eq, trailing_char, ten);
        let is_valid_end = self.builder.or(is_null, is_newline);
        self.builder.cond_br(is_valid_end, success_block, invalid_err_block);

        // Invalid error block: return Err(2)
        self.builder.start_block(invalid_err_block);
        let err2_code = self.builder.const_int(2);
        let err2_result = self.create_result_err(err2_code);
        let invalid_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success block: return Ok(result)
        self.builder.start_block(success_block);
        // Convert to i32 if needed
        let final_val = if int_type == "i32" {
            self.builder.trunc(result, IrType::I32)
        } else {
            result
        };
        let ok_result = self.create_result_ok(final_val);
        let success_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block: phi between results
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err1_result, empty_exit),
            (err2_result, invalid_exit),
            (ok_result, success_exit),
        ])
    }

    /// Lower f64::parse or f32::parse - parse String to float
    /// Uses strtod from libc for both (f32 gets truncated precision).
    /// Returns Result<f64/f32, i64>
    /// Error codes: 1 = empty/whitespace only, 2 = invalid characters
    fn lower_parse_float(&mut self, args: &[Expr], _float_type: &str) -> VReg {
        self.builder.declare_stdio();

        // Get the String argument
        let string_ptr = if !args.is_empty() {
            self.lower_expr(&args[0])
        } else {
            // Return Err(1) for no argument
            let err_code = self.builder.const_int(1);
            return self.create_result_err(err_code);
        };

        // Load the data pointer from String struct (field 0)
        let data_field_ptr = self.builder.get_field_ptr(string_ptr, 0);
        let str_ptr = self.builder.load(data_field_ptr);
        let str_ptr = self.builder.inttoptr(str_ptr, IrType::Ptr(Box::new(IrType::I8)));

        // Allocate space for endptr (char**)
        let endptr = self.builder.alloca(IrType::Ptr(Box::new(IrType::I8)));

        // Always use strtod (returns f64/double) for simplicity
        let result = self.builder.call("strtod", vec![str_ptr, endptr]);

        // Load endptr value
        let endptr_val = self.builder.load(endptr);
        let endptr_val = self.builder.inttoptr(endptr_val, IrType::Ptr(Box::new(IrType::I8)));

        // Check if endptr == str_ptr (nothing was parsed - empty or whitespace)
        let endptr_int = self.builder.ptrtoint(endptr_val, IrType::I64);
        let str_int = self.builder.ptrtoint(str_ptr, IrType::I64);
        let ptrs_equal = self.builder.icmp(CmpOp::Eq, endptr_int, str_int);

        // Create blocks for error checking
        let empty_err_block = self.builder.create_block();
        let check_trailing_block = self.builder.create_block();
        let invalid_err_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If endptr == str_ptr, error (empty/whitespace)
        self.builder.cond_br(ptrs_equal, empty_err_block, check_trailing_block);

        // Empty error block: return Err(1)
        self.builder.start_block(empty_err_block);
        let err1_code = self.builder.const_int(1);
        let err1_result = self.create_result_err(err1_code);
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Check for trailing characters block
        self.builder.start_block(check_trailing_block);
        let trailing_char_i8 = self.builder.load_byte(endptr_val);
        let trailing_char = self.builder.zext(trailing_char_i8, IrType::I64);
        // Check if it's null terminator (0) or newline (10)
        let zero = self.builder.const_int(0);
        let is_null = self.builder.icmp(CmpOp::Eq, trailing_char, zero);
        let ten = self.builder.const_int(10);
        let is_newline = self.builder.icmp(CmpOp::Eq, trailing_char, ten);
        let is_valid_end = self.builder.or(is_null, is_newline);
        self.builder.cond_br(is_valid_end, success_block, invalid_err_block);

        // Invalid error block: return Err(2)
        self.builder.start_block(invalid_err_block);
        let err2_code = self.builder.const_int(2);
        let err2_result = self.create_result_err(err2_code);
        let invalid_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success block: return Ok(result)
        self.builder.start_block(success_block);
        // Convert f64 to i64 bits for storage in Result payload using bitcast
        let float_bits = self.builder.bitcast(result, IrType::I64);
        let ok_result = self.create_result_ok(float_bits);
        let success_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block: phi between results
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err1_result, empty_exit),
            (err2_result, invalid_exit),
            (ok_result, success_exit),
        ])
    }

    /// Lower bool::parse - parse String to bool
    /// Accepts "true"/"false" (case insensitive)
    /// Returns Result<bool, i64>
    /// Error codes: 1 = empty string, 2 = invalid value
    fn lower_parse_bool(&mut self, args: &[Expr]) -> VReg {
        // Get the String argument
        let string_ptr = if !args.is_empty() {
            self.lower_expr(&args[0])
        } else {
            // Return Err(1) for no argument
            let err_code = self.builder.const_int(1);
            return self.create_result_err(err_code);
        };

        // Load the data pointer and length from String struct
        let data_field_ptr = self.builder.get_field_ptr(string_ptr, 0);
        let str_data = self.builder.load(data_field_ptr);
        let len_field_ptr = self.builder.get_field_ptr(string_ptr, 1);
        let str_len = self.builder.load(len_field_ptr);

        let zero = self.builder.const_int(0);
        let four = self.builder.const_int(4);
        let five = self.builder.const_int(5);

        // Check if empty
        let is_empty = self.builder.icmp(CmpOp::Eq, str_len, zero);
        let empty_err_block = self.builder.create_block();
        let check_len_block = self.builder.create_block();
        self.builder.cond_br(is_empty, empty_err_block, check_len_block);

        // Empty error
        self.builder.start_block(empty_err_block);
        let err1_code = self.builder.const_int(1);
        let err1_result = self.create_result_err(err1_code);
        let empty_exit = self.builder.current_block_id().unwrap();

        let merge_block = self.builder.create_block();
        self.builder.br(merge_block);

        // Check length (4 for true, 5 for false)
        self.builder.start_block(check_len_block);
        let is_len_4 = self.builder.icmp(CmpOp::Eq, str_len, four);

        let check_true_block = self.builder.create_block();
        let check_false_block = self.builder.create_block();
        let invalid_err_block = self.builder.create_block();

        self.builder.cond_br(is_len_4, check_true_block, check_false_block);

        // Check for "true" (case insensitive)
        self.builder.start_block(check_true_block);
        // Check bytes: t/T(116/84), r/R(114/82), u/U(117/85), e/E(101/69)
        let byte0_ptr = self.builder.get_byte_ptr(str_data, zero);
        let byte0 = self.builder.load_byte(byte0_ptr);
        let byte0 = self.builder.zext(byte0, IrType::I64);
        let one_i = self.builder.const_int(1);
        let byte1_ptr = self.builder.get_byte_ptr(str_data, one_i);
        let byte1 = self.builder.load_byte(byte1_ptr);
        let byte1 = self.builder.zext(byte1, IrType::I64);
        let two_i = self.builder.const_int(2);
        let byte2_ptr = self.builder.get_byte_ptr(str_data, two_i);
        let byte2 = self.builder.load_byte(byte2_ptr);
        let byte2 = self.builder.zext(byte2, IrType::I64);
        let three_i = self.builder.const_int(3);
        let byte3_ptr = self.builder.get_byte_ptr(str_data, three_i);
        let byte3 = self.builder.load_byte(byte3_ptr);
        let byte3 = self.builder.zext(byte3, IrType::I64);

        // Check 't' or 'T'
        let t_lower = self.builder.const_int(116);
        let t_upper = self.builder.const_int(84);
        let is_t_lower = self.builder.icmp(CmpOp::Eq, byte0, t_lower);
        let is_t_upper = self.builder.icmp(CmpOp::Eq, byte0, t_upper);
        let is_t = self.builder.or(is_t_lower, is_t_upper);

        // Check 'r' or 'R'
        let r_lower = self.builder.const_int(114);
        let r_upper = self.builder.const_int(82);
        let is_r_lower = self.builder.icmp(CmpOp::Eq, byte1, r_lower);
        let is_r_upper = self.builder.icmp(CmpOp::Eq, byte1, r_upper);
        let is_r = self.builder.or(is_r_lower, is_r_upper);

        // Check 'u' or 'U'
        let u_lower = self.builder.const_int(117);
        let u_upper = self.builder.const_int(85);
        let is_u_lower = self.builder.icmp(CmpOp::Eq, byte2, u_lower);
        let is_u_upper = self.builder.icmp(CmpOp::Eq, byte2, u_upper);
        let is_u = self.builder.or(is_u_lower, is_u_upper);

        // Check 'e' or 'E'
        let e_lower = self.builder.const_int(101);
        let e_upper = self.builder.const_int(69);
        let is_e_lower = self.builder.icmp(CmpOp::Eq, byte3, e_lower);
        let is_e_upper = self.builder.icmp(CmpOp::Eq, byte3, e_upper);
        let is_e = self.builder.or(is_e_lower, is_e_upper);

        let tr = self.builder.and(is_t, is_r);
        let ue = self.builder.and(is_u, is_e);
        let is_true = self.builder.and(tr, ue);

        let return_true_block = self.builder.create_block();
        self.builder.cond_br(is_true, return_true_block, invalid_err_block);

        // Return Ok(true)
        self.builder.start_block(return_true_block);
        let one = self.builder.const_int(1);
        let ok_true = self.create_result_ok(one);
        let true_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Check for "false" (case insensitive) - only if length is 5
        self.builder.start_block(check_false_block);
        let is_len_5 = self.builder.icmp(CmpOp::Eq, str_len, five);
        let check_false_bytes = self.builder.create_block();
        self.builder.cond_br(is_len_5, check_false_bytes, invalid_err_block);

        self.builder.start_block(check_false_bytes);
        // Check bytes: f/F(102/70), a/A(97/65), l/L(108/76), s/S(115/83), e/E(101/69)
        let f_byte0_ptr = self.builder.get_byte_ptr(str_data, zero);
        let f_byte0 = self.builder.load_byte(f_byte0_ptr);
        let f_byte0 = self.builder.zext(f_byte0, IrType::I64);
        let f_byte1_ptr = self.builder.get_byte_ptr(str_data, one_i);
        let f_byte1 = self.builder.load_byte(f_byte1_ptr);
        let f_byte1 = self.builder.zext(f_byte1, IrType::I64);
        let f_byte2_ptr = self.builder.get_byte_ptr(str_data, two_i);
        let f_byte2 = self.builder.load_byte(f_byte2_ptr);
        let f_byte2 = self.builder.zext(f_byte2, IrType::I64);
        let f_byte3_ptr = self.builder.get_byte_ptr(str_data, three_i);
        let f_byte3 = self.builder.load_byte(f_byte3_ptr);
        let f_byte3 = self.builder.zext(f_byte3, IrType::I64);
        let four_i = self.builder.const_int(4);
        let f_byte4_ptr = self.builder.get_byte_ptr(str_data, four_i);
        let f_byte4 = self.builder.load_byte(f_byte4_ptr);
        let f_byte4 = self.builder.zext(f_byte4, IrType::I64);

        // Check 'f' or 'F'
        let f_lower = self.builder.const_int(102);
        let f_upper = self.builder.const_int(70);
        let is_f_lower = self.builder.icmp(CmpOp::Eq, f_byte0, f_lower);
        let is_f_upper = self.builder.icmp(CmpOp::Eq, f_byte0, f_upper);
        let is_f = self.builder.or(is_f_lower, is_f_upper);

        // Check 'a' or 'A'
        let a_lower = self.builder.const_int(97);
        let a_upper = self.builder.const_int(65);
        let is_a_lower = self.builder.icmp(CmpOp::Eq, f_byte1, a_lower);
        let is_a_upper = self.builder.icmp(CmpOp::Eq, f_byte1, a_upper);
        let is_a = self.builder.or(is_a_lower, is_a_upper);

        // Check 'l' or 'L'
        let l_lower = self.builder.const_int(108);
        let l_upper = self.builder.const_int(76);
        let is_l_lower = self.builder.icmp(CmpOp::Eq, f_byte2, l_lower);
        let is_l_upper = self.builder.icmp(CmpOp::Eq, f_byte2, l_upper);
        let is_l = self.builder.or(is_l_lower, is_l_upper);

        // Check 's' or 'S'
        let s_lower = self.builder.const_int(115);
        let s_upper = self.builder.const_int(83);
        let is_s_lower = self.builder.icmp(CmpOp::Eq, f_byte3, s_lower);
        let is_s_upper = self.builder.icmp(CmpOp::Eq, f_byte3, s_upper);
        let is_s = self.builder.or(is_s_lower, is_s_upper);

        // Check 'e' or 'E' (reuse constants from above)
        let is_e2_lower = self.builder.icmp(CmpOp::Eq, f_byte4, e_lower);
        let is_e2_upper = self.builder.icmp(CmpOp::Eq, f_byte4, e_upper);
        let is_e2 = self.builder.or(is_e2_lower, is_e2_upper);

        let fa = self.builder.and(is_f, is_a);
        let ls = self.builder.and(is_l, is_s);
        let fals = self.builder.and(fa, ls);
        let is_false = self.builder.and(fals, is_e2);

        let return_false_block = self.builder.create_block();
        self.builder.cond_br(is_false, return_false_block, invalid_err_block);

        // Return Ok(false)
        self.builder.start_block(return_false_block);
        let ok_false = self.create_result_ok(zero);
        let false_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Invalid error block: return Err(2)
        self.builder.start_block(invalid_err_block);
        let err2_code = self.builder.const_int(2);
        let err2_result = self.create_result_err(err2_code);
        let invalid_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block: phi between results
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err1_result, empty_exit),
            (ok_true, true_exit),
            (ok_false, false_exit),
            (err2_result, invalid_exit),
        ])
    }

    /// Helper: Create Result::Ok(value)
    fn create_result_ok(&mut self, value: VReg) -> VReg {
        // Result = { i32 discriminant, i64 payload }
        // Ok has discriminant = 0
        let result_ptr = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));

        let disc_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let disc_val = self.builder.const_i32(0);
        self.builder.store(disc_ptr, disc_val);

        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr, value);

        result_ptr
    }

    /// Helper: Create Result::Err(error_code)
    fn create_result_err(&mut self, error_code: VReg) -> VReg {
        // Result = { i32 discriminant, i64 payload }
        // Err has discriminant = 1
        let result_ptr = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));

        let disc_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let disc_val = self.builder.const_i32(1);
        self.builder.store(disc_ptr, disc_val);

        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr, error_code);

        result_ptr
    }

    /// Helper: Create Result::Ok with a pointer payload (for File, String)
    fn create_result_ok_ptr(&mut self, ptr_value: VReg) -> VReg {
        // Result = { i32 discriminant, i64 payload }
        // Ok has discriminant = 0
        // Store pointer as i64 for payload
        let result_ptr = self.builder.malloc(IrType::Struct(vec![IrType::I32, IrType::I64]));

        let disc_ptr = self.builder.get_field_ptr(result_ptr, 0);
        let disc_val = self.builder.const_i32(0);
        self.builder.store(disc_ptr, disc_val);

        // Convert pointer to i64 for storage
        let ptr_as_i64 = self.builder.ptrtoint(ptr_value, IrType::I64);
        let payload_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(payload_ptr, ptr_as_i64);

        result_ptr
    }

    /// Lower File::open or File::create
    /// mode: "r" for open (read), "w" for create (write)
    fn lower_file_open(&mut self, args: &[Expr], mode: &str) -> VReg {
        // Get path argument (str literal)
        let path_ptr = self.lower_expr(&args[0]);

        // Create mode string constant
        let mode_name = self.builder.add_string_constant(mode);
        let mode_ptr = self.builder.global_string_ptr(&mode_name);

        // Call fopen(path, mode)
        let file_handle = self.builder.call("fopen", vec![path_ptr, mode_ptr]);

        // Check if fopen returned NULL
        let null_ptr = self.builder.const_null();
        let null_as_int = self.builder.ptrtoint(null_ptr, IrType::I64);
        let handle_as_int = self.builder.ptrtoint(file_handle, IrType::I64);
        let is_null = self.builder.icmp(CmpOp::Eq, handle_as_int, null_as_int);

        // Create blocks
        let error_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_null, error_block, success_block);

        // Error block: return Err(1) for file not found / permission denied
        self.builder.start_block(error_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let error_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success block: wrap FILE* in File struct and return Ok(File)
        self.builder.start_block(success_block);
        // Create File struct { *i8 handle }
        let file_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
        ]));
        let handle_field = self.builder.get_field_ptr(file_struct, 0);
        self.builder.store(handle_field, file_handle);

        let ok_result = self.create_result_ok_ptr(file_struct);
        let success_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err_result, error_exit),
            (ok_result, success_exit),
        ])
    }

    /// Lower File::read_to_string(file)
    fn lower_file_read_to_string(&mut self, args: &[Expr]) -> VReg {
        // Get File argument - may be i64 from Result::unwrap, need to convert to pointer
        let file_arg = self.lower_expr(&args[0]);
        // File struct type: { *i8 handle }
        let file_struct_ty = IrType::Struct(vec![IrType::Ptr(Box::new(IrType::I8))]);
        let file_struct = self.builder.inttoptr(file_arg, IrType::Ptr(Box::new(file_struct_ty)));

        // Get FILE* handle from File struct
        let handle_field = self.builder.get_field_ptr(file_struct, 0);
        let file_handle = self.builder.load(handle_field);

        // Get file size using fseek/ftell
        // fseek(file, 0, SEEK_END) where SEEK_END = 2
        let zero = self.builder.const_int(0);
        let seek_end = self.builder.const_i32(2);
        self.builder.call("fseek", vec![file_handle, zero, seek_end]);

        // ftell(file) -> size
        let file_size = self.builder.call("ftell", vec![file_handle]);

        // fseek(file, 0, SEEK_SET) where SEEK_SET = 0
        let seek_set = self.builder.const_i32(0);
        self.builder.call("fseek", vec![file_handle, zero, seek_set]);

        // Allocate buffer: malloc(size + 1) for null terminator
        let one = self.builder.const_int(1);
        let buf_size = self.builder.add(file_size, one);
        // Use malloc_array with I8 type - this uses inkwell's build_array_malloc which handles size correctly
        let buffer = self.builder.malloc_array(IrType::I8, buf_size);

        // fread(buffer, 1, size, file)
        let size_one = self.builder.const_int(1);
        let bytes_read = self.builder.call("fread", vec![buffer, size_one, file_size, file_handle]);

        // Null-terminate the buffer
        let end_ptr = self.builder.get_element_ptr(buffer, bytes_read);
        let null_byte = self.builder.const_i8(0);
        self.builder.store(end_ptr, null_byte);

        // Create String struct { *i8 data, i64 len, i64 cap }
        let string_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        let data_field = self.builder.get_field_ptr(string_struct, 0);
        self.builder.store(data_field, buffer);

        let len_field = self.builder.get_field_ptr(string_struct, 1);
        self.builder.store(len_field, bytes_read);

        let cap_field = self.builder.get_field_ptr(string_struct, 2);
        self.builder.store(cap_field, buf_size);

        // Return Ok(String)
        self.create_result_ok_ptr(string_struct)
    }

    /// Lower File::write_string(file, data)
    fn lower_file_write_string(&mut self, args: &[Expr]) -> VReg {
        // Get File argument - may be i64 from Result::unwrap, need to convert to pointer
        let file_arg = self.lower_expr(&args[0]);
        // File struct type: { *i8 handle }
        let file_struct_ty = IrType::Struct(vec![IrType::Ptr(Box::new(IrType::I8))]);
        let file_struct = self.builder.inttoptr(file_arg, IrType::Ptr(Box::new(file_struct_ty)));
        // Get string data (could be &str literal or String)
        let data_ptr = self.lower_expr(&args[1]);

        // Get FILE* handle from File struct
        let handle_field = self.builder.get_field_ptr(file_struct, 0);
        let file_handle = self.builder.load(handle_field);

        // Get string length using strlen
        let str_len = self.builder.call("strlen", vec![data_ptr]);

        // fwrite(data, 1, len, file)
        let size_one = self.builder.const_int(1);
        let bytes_written = self.builder.call("fwrite", vec![data_ptr, size_one, str_len, file_handle]);

        // Return Ok(bytes_written)
        self.create_result_ok(bytes_written)
    }

    /// Lower File::close(file)
    fn lower_file_close(&mut self, args: &[Expr]) -> VReg {
        // Get File argument - may be i64 from Result::unwrap, need to convert to pointer
        let file_arg = self.lower_expr(&args[0]);
        // File struct type: { *i8 handle }
        let file_struct_ty = IrType::Struct(vec![IrType::Ptr(Box::new(IrType::I8))]);
        let file_struct = self.builder.inttoptr(file_arg, IrType::Ptr(Box::new(file_struct_ty)));

        // Get FILE* handle from File struct
        let handle_field = self.builder.get_field_ptr(file_struct, 0);
        let file_handle = self.builder.load(handle_field);

        // fclose(file)
        self.builder.call("fclose", vec![file_handle]);

        // Return unit (void)
        self.builder.const_int(0)
    }

    // ============ Extended Filesystem Operations ============

    /// Lower File::exists(path) -> bool
    /// Uses access() with F_OK (0) to check if file exists
    fn lower_file_exists(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Call access(path, F_OK) where F_OK = 0
        let f_ok = self.builder.const_i32(0);
        let result = self.builder.call("access", vec![path_ptr, f_ok]);

        // access() returns 0 on success, -1 on failure
        let zero = self.builder.const_i32(0);
        self.builder.icmp(CmpOp::Eq, result, zero)
    }

    /// Lower File::size(path) -> Result<i64, i64>
    /// Uses stat() to get file size from st_size field
    fn lower_file_size(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Allocate stat buffer (144 bytes for Linux x86_64)
        let stat_buf_size = self.builder.const_int(144);
        let stat_buf = self.builder.malloc_array(IrType::I8, stat_buf_size);

        // Call stat(path, &stat_buf)
        let result = self.builder.call("stat", vec![path_ptr, stat_buf]);

        // Check if stat succeeded (result == 0)
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_success, ok_block, err_block);

        // Error block: return Err(1)
        self.builder.start_block(err_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // OK block: extract st_size
        // On Linux x86_64, st_size is at offset 48 in struct stat
        self.builder.start_block(ok_block);
        let offset = self.builder.const_int(48);
        let stat_buf_as_int = self.builder.ptrtoint(stat_buf, IrType::I64);
        let size_ptr_int = self.builder.add(stat_buf_as_int, offset);
        let size_ptr = self.builder.inttoptr(size_ptr_int, IrType::Ptr(Box::new(IrType::I64)));
        let file_size = self.builder.load(size_ptr);
        let ok_result = self.create_result_ok(file_size);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err_result, err_exit),
            (ok_result, ok_exit),
        ])
    }

    /// Lower File::is_file(path) -> bool
    /// Uses stat() and checks S_ISREG: (st_mode & 0170000) == 0100000
    fn lower_file_is_file(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Allocate stat buffer
        let stat_buf_size = self.builder.const_int(144);
        let stat_buf = self.builder.malloc_array(IrType::I8, stat_buf_size);

        // Call stat(path, &stat_buf)
        let result = self.builder.call("stat", vec![path_ptr, stat_buf]);

        // Check if stat succeeded
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_success, ok_block, err_block);

        // Error: return false
        self.builder.start_block(err_block);
        let false_val = self.builder.const_int(0);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // OK: check st_mode
        // On Linux x86_64, st_mode is at offset 24 (after st_dev=8, st_ino=8, st_nlink=8)
        self.builder.start_block(ok_block);
        let offset = self.builder.const_int(24);
        let stat_buf_as_int = self.builder.ptrtoint(stat_buf, IrType::I64);
        let mode_ptr_int = self.builder.add(stat_buf_as_int, offset);
        let mode_ptr = self.builder.inttoptr(mode_ptr_int, IrType::Ptr(Box::new(IrType::I32)));
        let mode = self.builder.load(mode_ptr);
        let mode_i64 = self.builder.sext(mode, IrType::I64);

        // S_ISREG: (mode & 0170000) == 0100000
        let mask = self.builder.const_int(0o170000);
        let mode_masked = self.builder.and(mode_i64, mask);
        let regular_file_bit = self.builder.const_int(0o100000);
        let is_reg = self.builder.icmp(CmpOp::Eq, mode_masked, regular_file_bit);
        let is_reg_i64 = self.builder.zext(is_reg, IrType::I64);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        let result = self.builder.phi(vec![
            (false_val, err_exit),
            (is_reg_i64, ok_exit),
        ]);
        // Convert to bool (1 bit)
        let one = self.builder.const_int(1);
        self.builder.icmp(CmpOp::Eq, result, one)
    }

    /// Lower File::is_dir(path) -> bool
    /// Uses stat() and checks S_ISDIR: (st_mode & 0170000) == 0040000
    fn lower_file_is_dir(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Allocate stat buffer
        let stat_buf_size = self.builder.const_int(144);
        let stat_buf = self.builder.malloc_array(IrType::I8, stat_buf_size);

        // Call stat(path, &stat_buf)
        let result = self.builder.call("stat", vec![path_ptr, stat_buf]);

        // Check if stat succeeded
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_success, ok_block, err_block);

        // Error: return false
        self.builder.start_block(err_block);
        let false_val = self.builder.const_int(0);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // OK: check st_mode for directory
        self.builder.start_block(ok_block);
        let offset = self.builder.const_int(24);
        let stat_buf_as_int = self.builder.ptrtoint(stat_buf, IrType::I64);
        let mode_ptr_int = self.builder.add(stat_buf_as_int, offset);
        let mode_ptr = self.builder.inttoptr(mode_ptr_int, IrType::Ptr(Box::new(IrType::I32)));
        let mode = self.builder.load(mode_ptr);
        let mode_i64 = self.builder.sext(mode, IrType::I64);

        // S_ISDIR: (mode & 0170000) == 0040000
        let mask = self.builder.const_int(0o170000);
        let mode_masked = self.builder.and(mode_i64, mask);
        let dir_bit = self.builder.const_int(0o40000);
        let is_dir = self.builder.icmp(CmpOp::Eq, mode_masked, dir_bit);
        let is_dir_i64 = self.builder.zext(is_dir, IrType::I64);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        let result = self.builder.phi(vec![
            (false_val, err_exit),
            (is_dir_i64, ok_exit),
        ]);
        let one = self.builder.const_int(1);
        self.builder.icmp(CmpOp::Eq, result, one)
    }

    /// Lower File::remove(path) -> Result<(), i64>
    /// Uses remove() C function
    fn lower_file_remove(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Call remove(path)
        let result = self.builder.call("remove", vec![path_ptr]);

        // Check if remove succeeded (result == 0)
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_success, ok_block, err_block);

        // Error block
        self.builder.start_block(err_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success block
        self.builder.start_block(ok_block);
        let unit_val = self.builder.const_int(0);
        let ok_result = self.create_result_ok(unit_val);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err_result, err_exit),
            (ok_result, ok_exit),
        ])
    }

    // ============ Directory Operations ============

    /// Lower Dir::create(path) -> Result<(), i64>
    /// Uses mkdir() with mode 0755
    fn lower_dir_create(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Call mkdir(path, 0755)
        let mode = self.builder.const_i32(0o755);
        let result = self.builder.call("mkdir", vec![path_ptr, mode]);

        // Check if mkdir succeeded (result == 0)
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_success, ok_block, err_block);

        // Error block
        self.builder.start_block(err_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success block
        self.builder.start_block(ok_block);
        let unit_val = self.builder.const_int(0);
        let ok_result = self.create_result_ok(unit_val);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err_result, err_exit),
            (ok_result, ok_exit),
        ])
    }

    /// Lower Dir::create_all(path) -> Result<(), i64>
    /// Creates directory and all parent directories
    /// Simplified implementation: just calls mkdir on the full path
    /// and checks if it succeeded or already exists
    fn lower_dir_create_all(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Get path length
        let path_len = self.builder.call("strlen", vec![path_ptr]);

        // Allocate buffer for path manipulation (path_len + 1 for null terminator)
        let one = self.builder.const_int(1);
        let buf_size = self.builder.add(path_len, one);
        let path_buf = self.builder.malloc_array(IrType::I8, buf_size);

        // Copy path to buffer
        self.builder.call("memcpy", vec![path_buf, path_ptr, buf_size]);

        // We'll iterate through the path creating directories at each '/' position
        // Use a simple counting loop with stack-allocated index
        let idx_ptr = self.builder.alloca(IrType::I64);
        let start = self.builder.const_int(1);
        self.builder.store(idx_ptr, start);

        // Loop header
        let loop_header = self.builder.create_block();
        let loop_body = self.builder.create_block();
        let loop_end = self.builder.create_block();
        let final_create = self.builder.create_block();

        self.builder.br(loop_header);

        // Loop header: while idx < path_len
        self.builder.start_block(loop_header);
        let idx = self.builder.load(idx_ptr);
        let cond = self.builder.icmp(CmpOp::Slt, idx, path_len);
        self.builder.cond_br(cond, loop_body, loop_end);

        // Loop body
        self.builder.start_block(loop_body);
        let char_ptr = self.builder.get_element_ptr(path_buf, idx);
        let current_char = self.builder.load(char_ptr);
        let current_char_i64 = self.builder.sext(current_char, IrType::I64);
        let slash = self.builder.const_int(47); // '/'

        let is_slash = self.builder.icmp(CmpOp::Eq, current_char_i64, slash);

        let create_block = self.builder.create_block();
        let skip_block = self.builder.create_block();
        let continue_block = self.builder.create_block();

        self.builder.cond_br(is_slash, create_block, skip_block);

        // Create directory at this point
        self.builder.start_block(create_block);
        // Temporarily null-terminate
        let null_byte = self.builder.const_i8(0);
        self.builder.store(char_ptr, null_byte);
        // mkdir - ignore result
        let mode = self.builder.const_i32(0o755);
        self.builder.call("mkdir", vec![path_buf, mode]);
        // Restore '/'
        let slash_byte = self.builder.const_i8(47);
        self.builder.store(char_ptr, slash_byte);
        self.builder.br(continue_block);

        // Skip
        self.builder.start_block(skip_block);
        self.builder.br(continue_block);

        // Continue: increment index
        self.builder.start_block(continue_block);
        let next_idx = self.builder.add(idx, one);
        self.builder.store(idx_ptr, next_idx);
        self.builder.br(loop_header);

        // End of loop - create final directory
        self.builder.start_block(loop_end);
        self.builder.br(final_create);

        self.builder.start_block(final_create);
        let mode2 = self.builder.const_i32(0o755);
        let result = self.builder.call("mkdir", vec![path_buf, mode2]);

        // Check result - success if 0, or if directory already exists
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        // Also check if directory already exists by calling stat
        let stat_buf_size = self.builder.const_int(144);
        let stat_buf = self.builder.malloc_array(IrType::I8, stat_buf_size);
        let stat_result = self.builder.call("stat", vec![path_buf, stat_buf]);
        let stat_zero = self.builder.const_i32(0);
        let stat_ok = self.builder.icmp(CmpOp::Eq, stat_result, stat_zero);

        // Success if mkdir succeeded OR path already exists
        let ok = self.builder.or(is_success, stat_ok);

        let success_block = self.builder.create_block();
        let error_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(ok, success_block, error_block);

        // Success
        self.builder.start_block(success_block);
        let unit_val = self.builder.const_int(0);
        let ok_result = self.create_result_ok(unit_val);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Error
        self.builder.start_block(error_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (ok_result, ok_exit),
            (err_result, err_exit),
        ])
    }

    /// Lower Dir::remove(path) -> Result<(), i64>
    /// Uses rmdir() for empty directories
    fn lower_dir_remove(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Call rmdir(path)
        let result = self.builder.call("rmdir", vec![path_ptr]);

        // Check if rmdir succeeded (result == 0)
        let zero = self.builder.const_i32(0);
        let is_success = self.builder.icmp(CmpOp::Eq, result, zero);

        let err_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_success, ok_block, err_block);

        // Error block
        self.builder.start_block(err_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success block
        self.builder.start_block(ok_block);
        let unit_val = self.builder.const_int(0);
        let ok_result = self.create_result_ok(unit_val);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err_result, err_exit),
            (ok_result, ok_exit),
        ])
    }

    /// Lower Dir::list(path) -> Result<Vec<String>, i64>
    /// Uses opendir/readdir/closedir
    /// Returns a Vec<String> containing all entries except "." and ".."
    fn lower_dir_list(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Call opendir(path)
        let dir_ptr = self.builder.call("opendir", vec![path_ptr]);

        // Check if opendir succeeded (dir_ptr != NULL)
        let null_ptr = self.builder.const_null();
        let dir_as_int = self.builder.ptrtoint(dir_ptr, IrType::I64);
        let null_as_int = self.builder.ptrtoint(null_ptr, IrType::I64);
        let is_null = self.builder.icmp(CmpOp::Eq, dir_as_int, null_as_int);

        let err_block = self.builder.create_block();
        let read_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(is_null, err_block, read_block);

        // Error block - opendir failed
        self.builder.start_block(err_block);
        let err_code = self.builder.const_int(1);
        let err_result = self.create_result_err(err_code);
        let err_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Read block - create Vec and loop through entries
        self.builder.start_block(read_block);

        // Create empty Vec<String>
        let vec_ty = self.vec_struct_type();
        let vec_ptr = self.builder.alloca(vec_ty.clone());

        // Initialize Vec fields: ptr=null, len=0, cap=0
        let null_data = self.builder.const_null();
        let zero = self.builder.const_int(0);

        let ptr_field = self.builder.get_field_ptr(vec_ptr, 0);
        self.builder.store(ptr_field, null_data);
        let len_field = self.builder.get_field_ptr(vec_ptr, 1);
        self.builder.store(len_field, zero);
        let cap_field = self.builder.get_field_ptr(vec_ptr, 2);
        self.builder.store(cap_field, zero);

        // Loop: read directory entries
        let loop_block = self.builder.create_block();
        let check_entry_block = self.builder.create_block();
        let skip_dot_block = self.builder.create_block();
        let add_entry_block = self.builder.create_block();
        let loop_continue_block = self.builder.create_block();
        let loop_end_block = self.builder.create_block();

        self.builder.br(loop_block);

        // Loop header: call readdir
        self.builder.start_block(loop_block);
        let entry_ptr = self.builder.call("readdir", vec![dir_ptr]);

        // Check if entry is NULL (end of directory)
        let entry_as_int = self.builder.ptrtoint(entry_ptr, IrType::I64);
        let entry_is_null = self.builder.icmp(CmpOp::Eq, entry_as_int, null_as_int);
        self.builder.cond_br(entry_is_null, loop_end_block, check_entry_block);

        // Check entry block: get d_name and check for "." and ".."
        self.builder.start_block(check_entry_block);

        // In Linux x86_64, struct dirent has d_name at offset 19:
        // - d_ino: i64 (0-7)
        // - d_off: i64 (8-15)
        // - d_reclen: u16 (16-17)
        // - d_type: u8 (18)
        // - d_name[256]: char (19+)
        let d_name_offset = self.builder.const_int(19);
        let d_name_ptr = self.builder.get_byte_ptr(entry_ptr, d_name_offset);

        // Load first character to check for "."
        let first_char = self.builder.load_byte(d_name_ptr);
        let first_char_i64 = self.builder.zext(first_char, IrType::I64);
        let dot_char = self.builder.const_int(46); // '.'
        let is_dot_first = self.builder.icmp(CmpOp::Eq, first_char_i64, dot_char);

        self.builder.cond_br(is_dot_first, skip_dot_block, add_entry_block);

        // Skip dot block: check if it's "." or ".."
        self.builder.start_block(skip_dot_block);

        // Load second character
        let one = self.builder.const_int(1);
        let second_char_ptr = self.builder.get_byte_ptr(d_name_ptr, one);
        let second_char = self.builder.load_byte(second_char_ptr);
        let second_char_i64 = self.builder.zext(second_char, IrType::I64);
        let null_char = self.builder.const_int(0);

        // Check if second char is '\0' (just ".")
        let is_single_dot = self.builder.icmp(CmpOp::Eq, second_char_i64, null_char);

        let check_double_dot = self.builder.create_block();
        self.builder.cond_br(is_single_dot, loop_continue_block, check_double_dot);

        // Check for ".."
        self.builder.start_block(check_double_dot);
        let is_second_dot = self.builder.icmp(CmpOp::Eq, second_char_i64, dot_char);

        let check_triple_block = self.builder.create_block();
        self.builder.cond_br(is_second_dot, check_triple_block, add_entry_block);

        // Check if third char is '\0' (just "..")
        self.builder.start_block(check_triple_block);
        let two = self.builder.const_int(2);
        let third_char_ptr = self.builder.get_byte_ptr(d_name_ptr, two);
        let third_char = self.builder.load_byte(third_char_ptr);
        let third_char_i64 = self.builder.zext(third_char, IrType::I64);
        let is_double_dot = self.builder.icmp(CmpOp::Eq, third_char_i64, null_char);
        self.builder.cond_br(is_double_dot, loop_continue_block, add_entry_block);

        // Add entry block: create String from d_name and push to Vec
        self.builder.start_block(add_entry_block);

        // Get string length using strlen
        let name_len = self.builder.call("strlen", vec![d_name_ptr]);

        // Allocate buffer for string data (len + 1 for null terminator)
        let one_i64 = self.builder.const_int(1);
        let buf_size = self.builder.add(name_len, one_i64);
        let buffer = self.builder.malloc_array(IrType::I8, buf_size);

        // Copy string data
        self.builder.call("memcpy", vec![buffer, d_name_ptr, buf_size]);

        // Create String struct
        let string_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        let str_data_field = self.builder.get_field_ptr(string_struct, 0);
        self.builder.store(str_data_field, buffer);
        let str_len_field = self.builder.get_field_ptr(string_struct, 1);
        self.builder.store(str_len_field, name_len);
        let str_cap_field = self.builder.get_field_ptr(string_struct, 2);
        self.builder.store(str_cap_field, buf_size);

        // Push String pointer to Vec (converted to i64 for storage)
        let string_as_i64 = self.builder.ptrtoint(string_struct, IrType::I64);
        self.lower_vec_push_i64_raw(vec_ptr, string_as_i64);

        self.builder.br(loop_continue_block);

        // Loop continue: go back to loop header
        self.builder.start_block(loop_continue_block);
        self.builder.br(loop_block);

        // Loop end: close directory and return Ok
        self.builder.start_block(loop_end_block);

        // Call closedir
        self.builder.call("closedir", vec![dir_ptr]);

        // Create Result::Ok with Vec pointer
        let vec_as_i64 = self.builder.ptrtoint(vec_ptr, IrType::I64);
        let ok_result = self.create_result_ok(vec_as_i64);
        let ok_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (err_result, err_exit),
            (ok_result, ok_exit),
        ])
    }

    // ============ Path Operations ============

    /// Lower Path::join(base, path) -> String
    /// Joins two paths with a '/' separator
    fn lower_path_join(&mut self, args: &[Expr]) -> VReg {
        let base_ptr = self.lower_expr(&args[0]);
        let path_ptr = self.lower_expr(&args[1]);

        // Get lengths
        let base_len = self.builder.call("strlen", vec![base_ptr]);
        let path_len = self.builder.call("strlen", vec![path_ptr]);

        // Calculate result length: base_len + 1 + path_len + 1 (for '/' and null terminator)
        let one = self.builder.const_int(1);
        let two = self.builder.const_int(2);
        let total_len = self.builder.add(base_len, path_len);
        let total_len = self.builder.add(total_len, two);

        // Allocate buffer
        let buffer = self.builder.malloc_array(IrType::I8, total_len);

        // Copy base to buffer
        self.builder.call("memcpy", vec![buffer, base_ptr, base_len]);

        // Add '/' separator
        let separator_ptr = self.builder.get_element_ptr(buffer, base_len);
        let slash = self.builder.const_i8(47); // '/'
        self.builder.store(separator_ptr, slash);

        // Copy path after separator
        let path_dest_offset = self.builder.add(base_len, one);
        let path_dest = self.builder.get_element_ptr(buffer, path_dest_offset);
        let path_copy_len = self.builder.add(path_len, one); // Include null terminator
        self.builder.call("memcpy", vec![path_dest, path_ptr, path_copy_len]);

        // Create String struct
        let string_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        let data_field = self.builder.get_field_ptr(string_struct, 0);
        self.builder.store(data_field, buffer);

        // String length is base_len + 1 + path_len
        let str_len = self.builder.add(base_len, path_len);
        let str_len = self.builder.add(str_len, one);
        let len_field = self.builder.get_field_ptr(string_struct, 1);
        self.builder.store(len_field, str_len);

        let cap_field = self.builder.get_field_ptr(string_struct, 2);
        self.builder.store(cap_field, total_len);

        string_struct
    }

    /// Lower Path::parent(path) -> Option<String>
    /// Returns everything before the last '/'
    fn lower_path_parent(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Find last '/' using strrchr
        let slash = self.builder.const_i32(47); // '/'
        let last_slash = self.builder.call("strrchr", vec![path_ptr, slash]);

        // Check if '/' was found
        let null_ptr = self.builder.const_null();
        let slash_as_int = self.builder.ptrtoint(last_slash, IrType::I64);
        let null_as_int = self.builder.ptrtoint(null_ptr, IrType::I64);
        let not_found = self.builder.icmp(CmpOp::Eq, slash_as_int, null_as_int);

        let none_block = self.builder.create_block();
        let some_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(not_found, none_block, some_block);

        // None block: no '/' found
        self.builder.start_block(none_block);
        let none_result = self.create_option_none();
        let none_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Some block: extract parent
        self.builder.start_block(some_block);

        // Check if slash is at the beginning (root directory case)
        let path_as_int = self.builder.ptrtoint(path_ptr, IrType::I64);
        let is_root = self.builder.icmp(CmpOp::Eq, slash_as_int, path_as_int);

        let root_block = self.builder.create_block();
        let normal_block = self.builder.create_block();
        let merge_some = self.builder.create_block();

        self.builder.cond_br(is_root, root_block, normal_block);

        // Root case: return "/" as parent
        self.builder.start_block(root_block);
        let root_str = self.create_string_from_literal("/");
        let root_some = self.create_option_some_ptr(root_str);
        let root_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_some);

        // Normal case: extract substring before last '/'
        self.builder.start_block(normal_block);
        // Calculate length: slash_ptr - path_ptr
        let parent_len = self.builder.sub(slash_as_int, path_as_int);

        // If parent_len == 0, return "/"
        let zero = self.builder.const_int(0);
        let is_zero = self.builder.icmp(CmpOp::Eq, parent_len, zero);

        let empty_parent_block = self.builder.create_block();
        let normal_parent_block = self.builder.create_block();

        self.builder.cond_br(is_zero, empty_parent_block, normal_parent_block);

        // Empty parent: return "/"
        self.builder.start_block(empty_parent_block);
        let root_str2 = self.create_string_from_literal("/");
        let empty_some = self.create_option_some_ptr(root_str2);
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_some);

        // Normal parent: create substring
        self.builder.start_block(normal_parent_block);
        let one = self.builder.const_int(1);
        let buf_size = self.builder.add(parent_len, one);
        let buffer = self.builder.malloc_array(IrType::I8, buf_size);
        self.builder.call("memcpy", vec![buffer, path_ptr, parent_len]);

        // Null terminate
        let end_ptr = self.builder.get_element_ptr(buffer, parent_len);
        let null_byte = self.builder.const_i8(0);
        self.builder.store(end_ptr, null_byte);

        // Create String struct
        let string_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        let data_field = self.builder.get_field_ptr(string_struct, 0);
        self.builder.store(data_field, buffer);
        let len_field = self.builder.get_field_ptr(string_struct, 1);
        self.builder.store(len_field, parent_len);
        let cap_field = self.builder.get_field_ptr(string_struct, 2);
        self.builder.store(cap_field, buf_size);

        let normal_some = self.create_option_some_ptr(string_struct);
        let normal_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_some);

        // Merge some variants
        self.builder.start_block(merge_some);
        let some_result = self.builder.phi(vec![
            (root_some, root_exit),
            (empty_some, empty_exit),
            (normal_some, normal_exit),
        ]);
        let some_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Final merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (none_result, none_exit),
            (some_result, some_exit),
        ])
    }

    /// Lower Path::filename(path) -> Option<String>
    /// Returns everything after the last '/'
    fn lower_path_filename(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // Find last '/' using strrchr
        let slash = self.builder.const_i32(47); // '/'
        let last_slash = self.builder.call("strrchr", vec![path_ptr, slash]);

        // Check if '/' was found
        let null_ptr = self.builder.const_null();
        let slash_as_int = self.builder.ptrtoint(last_slash, IrType::I64);
        let null_as_int = self.builder.ptrtoint(null_ptr, IrType::I64);
        let not_found = self.builder.icmp(CmpOp::Eq, slash_as_int, null_as_int);

        let no_slash_block = self.builder.create_block();
        let has_slash_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(not_found, no_slash_block, has_slash_block);

        // No slash: entire path is the filename
        self.builder.start_block(no_slash_block);
        let path_len = self.builder.call("strlen", vec![path_ptr]);
        let one = self.builder.const_int(1);
        let buf_size = self.builder.add(path_len, one);
        let buffer1 = self.builder.malloc_array(IrType::I8, buf_size);
        self.builder.call("memcpy", vec![buffer1, path_ptr, buf_size]);

        let string_struct1 = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));
        let data_field1 = self.builder.get_field_ptr(string_struct1, 0);
        self.builder.store(data_field1, buffer1);
        let len_field1 = self.builder.get_field_ptr(string_struct1, 1);
        self.builder.store(len_field1, path_len);
        let cap_field1 = self.builder.get_field_ptr(string_struct1, 2);
        self.builder.store(cap_field1, buf_size);

        let no_slash_result = self.create_option_some_ptr(string_struct1);
        let no_slash_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Has slash: extract filename (part after last '/')
        self.builder.start_block(has_slash_block);
        // Move pointer past the '/'
        let one2 = self.builder.const_int(1);
        let filename_start = self.builder.add(slash_as_int, one2);
        let filename_ptr = self.builder.inttoptr(filename_start, IrType::Ptr(Box::new(IrType::I8)));

        let filename_len = self.builder.call("strlen", vec![filename_ptr]);

        // Check if filename is empty (path ends with '/')
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, filename_len, zero);

        let empty_block = self.builder.create_block();
        let normal_block = self.builder.create_block();

        self.builder.cond_br(is_empty, empty_block, normal_block);

        // Empty filename: return None
        self.builder.start_block(empty_block);
        let empty_result = self.create_option_none();
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Normal filename
        self.builder.start_block(normal_block);
        let one3 = self.builder.const_int(1);
        let buf_size2 = self.builder.add(filename_len, one3);
        let buffer2 = self.builder.malloc_array(IrType::I8, buf_size2);
        self.builder.call("memcpy", vec![buffer2, filename_ptr, buf_size2]);

        let string_struct2 = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));
        let data_field2 = self.builder.get_field_ptr(string_struct2, 0);
        self.builder.store(data_field2, buffer2);
        let len_field2 = self.builder.get_field_ptr(string_struct2, 1);
        self.builder.store(len_field2, filename_len);
        let cap_field2 = self.builder.get_field_ptr(string_struct2, 2);
        self.builder.store(cap_field2, buf_size2);

        let normal_result = self.create_option_some_ptr(string_struct2);
        let normal_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (no_slash_result, no_slash_exit),
            (empty_result, empty_exit),
            (normal_result, normal_exit),
        ])
    }

    /// Lower Path::extension(path) -> Option<String>
    /// Returns everything after the last '.' in the filename
    fn lower_path_extension(&mut self, args: &[Expr]) -> VReg {
        let path_ptr = self.lower_expr(&args[0]);

        // First, find the filename (part after last '/')
        let slash = self.builder.const_i32(47); // '/'
        let last_slash = self.builder.call("strrchr", vec![path_ptr, slash]);

        // Determine where to start looking for '.'
        let null_ptr = self.builder.const_null();
        let slash_as_int = self.builder.ptrtoint(last_slash, IrType::I64);
        let null_as_int = self.builder.ptrtoint(null_ptr, IrType::I64);
        let has_slash = self.builder.icmp(CmpOp::Ne, slash_as_int, null_as_int);

        let with_slash_block = self.builder.create_block();
        let no_slash_block = self.builder.create_block();
        let find_dot_block = self.builder.create_block();

        self.builder.cond_br(has_slash, with_slash_block, no_slash_block);

        // With slash: start from after '/'
        self.builder.start_block(with_slash_block);
        let one = self.builder.const_int(1);
        let filename_start1 = self.builder.add(slash_as_int, one);
        let filename_ptr1 = self.builder.inttoptr(filename_start1, IrType::Ptr(Box::new(IrType::I8)));
        let ws_exit = self.builder.current_block_id().unwrap();
        self.builder.br(find_dot_block);

        // No slash: use entire path
        self.builder.start_block(no_slash_block);
        let ns_exit = self.builder.current_block_id().unwrap();
        self.builder.br(find_dot_block);

        // Find dot in filename
        self.builder.start_block(find_dot_block);
        let filename_ptr = self.builder.phi(vec![
            (filename_ptr1, ws_exit),
            (path_ptr, ns_exit),
        ]);

        // Find last '.' in filename
        let dot = self.builder.const_i32(46); // '.'
        let last_dot = self.builder.call("strrchr", vec![filename_ptr, dot]);

        // Check if '.' was found
        let dot_as_int = self.builder.ptrtoint(last_dot, IrType::I64);
        let dot_not_found = self.builder.icmp(CmpOp::Eq, dot_as_int, null_as_int);

        let no_dot_block = self.builder.create_block();
        let has_dot_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(dot_not_found, no_dot_block, has_dot_block);

        // No dot: return None
        self.builder.start_block(no_dot_block);
        let none_result = self.create_option_none();
        let none_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Has dot: extract extension (part after '.')
        self.builder.start_block(has_dot_block);
        // Move pointer past the '.'
        let one2 = self.builder.const_int(1);
        let ext_start = self.builder.add(dot_as_int, one2);
        let ext_ptr = self.builder.inttoptr(ext_start, IrType::Ptr(Box::new(IrType::I8)));

        let ext_len = self.builder.call("strlen", vec![ext_ptr]);

        // Check if extension is empty (path ends with '.')
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, ext_len, zero);

        let empty_ext_block = self.builder.create_block();
        let normal_ext_block = self.builder.create_block();

        self.builder.cond_br(is_empty, empty_ext_block, normal_ext_block);

        // Empty extension: return None
        self.builder.start_block(empty_ext_block);
        let empty_result = self.create_option_none();
        let empty_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Normal extension
        self.builder.start_block(normal_ext_block);
        let one3 = self.builder.const_int(1);
        let buf_size = self.builder.add(ext_len, one3);
        let buffer = self.builder.malloc_array(IrType::I8, buf_size);
        self.builder.call("memcpy", vec![buffer, ext_ptr, buf_size]);

        let string_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));
        let data_field = self.builder.get_field_ptr(string_struct, 0);
        self.builder.store(data_field, buffer);
        let len_field = self.builder.get_field_ptr(string_struct, 1);
        self.builder.store(len_field, ext_len);
        let cap_field = self.builder.get_field_ptr(string_struct, 2);
        self.builder.store(cap_field, buf_size);

        let ext_result = self.create_option_some_ptr(string_struct);
        let ext_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (none_result, none_exit),
            (empty_result, empty_exit),
            (ext_result, ext_exit),
        ])
    }

    /// Helper: Create Option<String> Some with pointer value
    fn create_option_some_ptr(&mut self, ptr_value: VReg) -> VReg {
        let option_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let opt_ptr = self.builder.malloc(option_ty);

        // Store discriminant = 1 (Some)
        let discrim_ptr = self.builder.get_field_ptr(opt_ptr, 0);
        let one = self.builder.const_int(1);
        let one_i32 = self.builder.trunc(one, IrType::I32);
        self.builder.store(discrim_ptr, one_i32);

        // Store pointer as i64
        let ptr_as_i64 = self.builder.ptrtoint(ptr_value, IrType::I64);
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        self.builder.store(payload_ptr, ptr_as_i64);

        opt_ptr
    }

    /// Helper: Create a String from a string literal
    fn create_string_from_literal(&mut self, s: &str) -> VReg {
        let str_name = self.builder.add_string_constant(s);
        let str_ptr = self.builder.global_string_ptr(&str_name);
        let str_len = self.builder.const_int(s.len() as i64);
        let one = self.builder.const_int(1);
        let buf_size = self.builder.add(str_len, one);

        // Allocate and copy
        let buffer = self.builder.malloc_array(IrType::I8, buf_size);
        self.builder.call("memcpy", vec![buffer, str_ptr, buf_size]);

        // Create String struct
        let string_struct = self.builder.malloc(IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I8)),
            IrType::I64,
            IrType::I64,
        ]));

        let data_field = self.builder.get_field_ptr(string_struct, 0);
        self.builder.store(data_field, buffer);
        let len_field = self.builder.get_field_ptr(string_struct, 1);
        self.builder.store(len_field, str_len);
        let cap_field = self.builder.get_field_ptr(string_struct, 2);
        self.builder.store(cap_field, buf_size);

        string_struct
    }

    // ============ HashMap<K, V> Implementation ============
    // HashMap uses open addressing with linear probing
    // HashMap struct: { *Entry entries, i64 count, i64 capacity }
    // Entry struct: { i64 key, i64 value, i64 occupied } where 0=empty, 1=occupied, 2=deleted

    /// Create the HashMap struct type
    fn hashmap_struct_type(&self) -> IrType {
        IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I64)), // entries pointer (each entry is 3 i64s)
            IrType::I64,                         // count - number of occupied entries
            IrType::I64,                         // capacity - total slots
        ])
    }

    /// Entry size in i64 units (key, value, occupied = 3)
    fn hashmap_entry_size(&self) -> i64 {
        3
    }

    /// FNV-1a hash function for i64 keys
    /// Much better distribution than simple modulo
    /// FNV-1a: hash = ((hash XOR byte) * prime) for each byte
    fn lower_hashmap_hash(&mut self, key: VReg, capacity: VReg) -> VReg {
        // FNV-1a 64-bit constants
        // offset_basis = 14695981039346656037 = 0xcbf29ce484222325
        // prime = 1099511628211 = 0x100000001b3
        let offset_basis = self.builder.const_int(0xcbf29ce484222325u64 as i64);
        let prime = self.builder.const_int(0x100000001b3u64 as i64);
        let mask = self.builder.const_int(0xFF);

        // Start with offset basis
        let mut hash = offset_basis;

        // Process each byte of the 64-bit key (unrolled loop)
        // Byte 0 (least significant)
        let byte0 = self.builder.and(key, mask);
        hash = self.builder.xor(hash, byte0);
        hash = self.builder.mul(hash, prime);

        // Byte 1
        let eight = self.builder.const_int(8);
        let shifted1 = self.builder.lshr(key, eight);
        let byte1 = self.builder.and(shifted1, mask);
        hash = self.builder.xor(hash, byte1);
        hash = self.builder.mul(hash, prime);

        // Byte 2
        let sixteen = self.builder.const_int(16);
        let shifted2 = self.builder.lshr(key, sixteen);
        let byte2 = self.builder.and(shifted2, mask);
        hash = self.builder.xor(hash, byte2);
        hash = self.builder.mul(hash, prime);

        // Byte 3
        let twenty_four = self.builder.const_int(24);
        let shifted3 = self.builder.lshr(key, twenty_four);
        let byte3 = self.builder.and(shifted3, mask);
        hash = self.builder.xor(hash, byte3);
        hash = self.builder.mul(hash, prime);

        // Byte 4
        let thirty_two = self.builder.const_int(32);
        let shifted4 = self.builder.lshr(key, thirty_two);
        let byte4 = self.builder.and(shifted4, mask);
        hash = self.builder.xor(hash, byte4);
        hash = self.builder.mul(hash, prime);

        // Byte 5
        let forty = self.builder.const_int(40);
        let shifted5 = self.builder.lshr(key, forty);
        let byte5 = self.builder.and(shifted5, mask);
        hash = self.builder.xor(hash, byte5);
        hash = self.builder.mul(hash, prime);

        // Byte 6
        let forty_eight = self.builder.const_int(48);
        let shifted6 = self.builder.lshr(key, forty_eight);
        let byte6 = self.builder.and(shifted6, mask);
        hash = self.builder.xor(hash, byte6);
        hash = self.builder.mul(hash, prime);

        // Byte 7 (most significant)
        let fifty_six = self.builder.const_int(56);
        let shifted7 = self.builder.lshr(key, fifty_six);
        let byte7 = self.builder.and(shifted7, mask);
        hash = self.builder.xor(hash, byte7);
        hash = self.builder.mul(hash, prime);

        // Convert to positive and take modulo capacity
        // Use unsigned remainder for better distribution
        let max_positive = self.builder.const_int(0x7FFFFFFFFFFFFFFF_u64 as i64);
        let hash_positive = self.builder.and(hash, max_positive);
        self.builder.urem(hash_positive, capacity)
    }

    /// Check if an expression is a String type
    fn is_string_expr(&self, expr: &Expr) -> bool {
        self.expr_types.get(&expr.span).map_or(false, |ty| {
            if let crate::typeck::TyKind::Named { name, .. } = &ty.kind {
                name == "String"
            } else {
                false
            }
        })
    }

    /// FNV-1a hash function for String keys
    /// Hashes each byte of the string content
    fn lower_hashmap_hash_string(&mut self, str_ptr: VReg, capacity: VReg) -> VReg {
        // String struct: { ptr: *i8, len: i64, cap: i64 }
        // Load ptr and len from string struct
        let data_field = self.builder.get_field_ptr(str_ptr, 0);
        let data_ptr = self.builder.load(data_field); // data pointer
        let data_base = self.builder.ptrtoint(data_ptr, IrType::I64); // convert to i64 for arithmetic

        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);

        // FNV-1a 64-bit constants
        let offset_basis = self.builder.const_int(0xcbf29ce484222325u64 as i64);
        let prime = self.builder.const_int(0x100000001b3u64 as i64);

        // Loop through each byte
        let loop_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Initialize hash and index
        let hash_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(hash_ptr, offset_basis);

        let idx_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(idx_ptr, zero);

        self.builder.br(loop_block);

        // Loop condition
        self.builder.start_block(loop_block);
        let idx = self.builder.load(idx_ptr);
        let done = self.builder.icmp(CmpOp::Sge, idx, len);
        self.builder.cond_br(done, done_block, body_block);

        // Loop body - use pointer arithmetic (data_base + idx) for byte access
        self.builder.start_block(body_block);
        let byte_addr = self.builder.add(data_base, idx); // data_base + idx (byte offset)
        let byte_ptr = self.builder.inttoptr(byte_addr, IrType::Ptr(Box::new(IrType::I8)));
        let byte = self.builder.load_byte(byte_ptr);
        let byte_i64 = self.builder.zext(byte, IrType::I64);

        let hash = self.builder.load(hash_ptr);
        let hash_xor = self.builder.xor(hash, byte_i64);
        let hash_mul = self.builder.mul(hash_xor, prime);
        self.builder.store(hash_ptr, hash_mul);

        let one = self.builder.const_int(1);
        let next_idx = self.builder.add(idx, one);
        self.builder.store(idx_ptr, next_idx);
        self.builder.br(loop_block);

        // Done
        self.builder.start_block(done_block);
        let final_hash = self.builder.load(hash_ptr);

        // Convert to positive and take modulo capacity
        let max_positive = self.builder.const_int(0x7FFFFFFFFFFFFFFF_u64 as i64);
        let hash_positive = self.builder.and(final_hash, max_positive);
        self.builder.urem(hash_positive, capacity)
    }

    /// Compare two strings for equality
    /// Returns 1 if equal, 0 if not
    fn lower_string_equals(&mut self, str1_ptr: VReg, str2_ptr: VReg) -> VReg {
        // Load lengths
        let len1_field = self.builder.get_field_ptr(str1_ptr, 1);
        let len1 = self.builder.load(len1_field);

        let len2_field = self.builder.get_field_ptr(str2_ptr, 1);
        let len2 = self.builder.load(len2_field);

        // First check if lengths are equal
        let len_eq = self.builder.icmp(CmpOp::Eq, len1, len2);

        let compare_block = self.builder.create_block();
        let not_equal_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(len_eq, compare_block, not_equal_block);

        // Compare block: check byte by byte
        self.builder.start_block(compare_block);

        // Load data base addresses and convert to i64 for pointer arithmetic
        let data1_field = self.builder.get_field_ptr(str1_ptr, 0);
        let data1_ptr = self.builder.load(data1_field);
        let data1_base = self.builder.ptrtoint(data1_ptr, IrType::I64);

        let data2_field = self.builder.get_field_ptr(str2_ptr, 0);
        let data2_ptr = self.builder.load(data2_field);
        let data2_base = self.builder.ptrtoint(data2_ptr, IrType::I64);

        // Use memcmp-like loop
        let loop_block = self.builder.create_block();
        let check_block = self.builder.create_block();
        let equal_block = self.builder.create_block();

        let idx_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(idx_ptr, zero);

        self.builder.br(loop_block);

        // Loop condition
        self.builder.start_block(loop_block);
        let idx = self.builder.load(idx_ptr);
        let loop_done = self.builder.icmp(CmpOp::Sge, idx, len1);
        self.builder.cond_br(loop_done, equal_block, check_block);

        // Check byte - use pointer arithmetic for byte access
        self.builder.start_block(check_block);
        let byte1_addr = self.builder.add(data1_base, idx);
        let byte2_addr = self.builder.add(data2_base, idx);
        let byte1_ptr = self.builder.inttoptr(byte1_addr, IrType::Ptr(Box::new(IrType::I8)));
        let byte2_ptr = self.builder.inttoptr(byte2_addr, IrType::Ptr(Box::new(IrType::I8)));
        let byte1 = self.builder.load_byte(byte1_ptr);
        let byte2 = self.builder.load_byte(byte2_ptr);
        let bytes_eq = self.builder.icmp(CmpOp::Eq, byte1, byte2);

        let next_block = self.builder.create_block();
        self.builder.cond_br(bytes_eq, next_block, not_equal_block);

        // Next iteration
        self.builder.start_block(next_block);
        let one = self.builder.const_int(1);
        let next_idx = self.builder.add(idx, one);
        self.builder.store(idx_ptr, next_idx);
        self.builder.br(loop_block);

        // Equal block
        self.builder.start_block(equal_block);
        let true_val = self.builder.const_int(1);
        let equal_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Not equal block
        self.builder.start_block(not_equal_block);
        let false_val = self.builder.const_int(0);
        let not_equal_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Done block
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (true_val, equal_exit),
            (false_val, not_equal_exit),
        ])
    }

    /// Lower HashMap::new() - creates empty hashmap with default capacity
    fn lower_hashmap_new(&mut self) -> VReg {
        // Create with initial capacity of 16
        let sixteen = self.builder.const_int(16);
        self.lower_hashmap_with_capacity_value(sixteen)
    }

    /// Lower HashMap::with_capacity(cap) - creates hashmap with initial capacity
    fn lower_hashmap_with_capacity(&mut self, args: &[Expr]) -> VReg {
        let cap = if let Some(arg) = args.first() {
            self.lower_expr(arg)
        } else {
            self.builder.const_int(16)
        };
        self.lower_hashmap_with_capacity_value(cap)
    }

    /// Helper: creates hashmap with given capacity value
    fn lower_hashmap_with_capacity_value(&mut self, cap: VReg) -> VReg {
        // Allocate HashMap struct on stack
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.alloca(map_ty.clone());

        // Allocate entries array on heap
        // Each entry is 3 i64s = 24 bytes
        let entry_size = self.builder.const_int(24); // 3 * 8 bytes
        let size = self.builder.mul(cap, entry_size);

        // Allocate with calloc to zero-initialize (all entries start as empty)
        let entries_ptr = self.builder.calloc(size);

        // Store entries pointer (field 0)
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        self.builder.store(entries_field, entries_ptr);

        // Store count = 0 (field 1)
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        let zero = self.builder.const_int(0);
        self.builder.store(count_field, zero);

        // Store capacity (field 2)
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        self.builder.store(cap_field, cap);

        // Track type
        self.vreg_types.insert(map_ptr, map_ty);

        map_ptr
    }

    /// Helper: Resize HashMap when load factor exceeds threshold
    /// Doubles capacity and rehashes all entries
    fn lower_hashmap_resize(&mut self, map_ptr: VReg) {
        // Load old capacity and entries
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        let old_capacity = self.builder.load(cap_field);

        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let old_entries_raw = self.builder.load(entries_field);
        let old_entries = self.builder.inttoptr(old_entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Calculate new capacity = old_capacity * 2
        let two = self.builder.const_int(2);
        let new_capacity = self.builder.mul(old_capacity, two);

        // Allocate new entries array with calloc (zero-initialized)
        let entry_size = self.builder.const_int(24); // 3 * 8 bytes
        let new_size = self.builder.mul(new_capacity, entry_size);
        let new_entries_raw = self.builder.calloc(new_size);
        let new_entries = self.builder.inttoptr(new_entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Loop through old entries and rehash occupied ones
        let loop_block = self.builder.create_block();
        let check_block = self.builder.create_block();
        let rehash_block = self.builder.create_block();
        let next_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Loop counter (index into old entries)
        let idx_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(idx_ptr, zero);

        self.builder.br(loop_block);

        // Loop: check if we've processed all old entries
        self.builder.start_block(loop_block);
        let idx = self.builder.load(idx_ptr);
        let done = self.builder.icmp(CmpOp::Sge, idx, old_capacity);
        self.builder.cond_br(done, done_block, check_block);

        // Check block: see if this entry is occupied
        self.builder.start_block(check_block);
        let three = self.builder.const_int(3);
        let offset = self.builder.mul(idx, three);
        let occ_offset = self.builder.add(offset, two);
        let occupied_ptr = self.builder.get_element_ptr(old_entries, occ_offset);
        let occupied = self.builder.load(occupied_ptr);
        let one = self.builder.const_int(1);
        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);
        self.builder.cond_br(is_occupied, rehash_block, next_block);

        // Rehash block: copy entry to new location
        self.builder.start_block(rehash_block);

        // Load key and value from old entry
        let old_key_ptr = self.builder.get_element_ptr(old_entries, offset);
        let old_key = self.builder.load(old_key_ptr);
        let val_offset = self.builder.add(offset, one);
        let old_val_ptr = self.builder.get_element_ptr(old_entries, val_offset);
        let old_val = self.builder.load(old_val_ptr);

        // Calculate new slot using new capacity
        let new_slot = self.lower_hashmap_hash(old_key, new_capacity);

        // Linear probe to find empty slot in new array
        let probe_loop = self.builder.create_block();
        let probe_next = self.builder.create_block();
        let probe_done = self.builder.create_block();

        let probe_slot_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(probe_slot_ptr, new_slot);

        self.builder.br(probe_loop);

        // Probe loop
        self.builder.start_block(probe_loop);
        let probe_slot = self.builder.load(probe_slot_ptr);
        let probe_offset = self.builder.mul(probe_slot, three);
        let probe_occ_offset = self.builder.add(probe_offset, two);
        let probe_occ_ptr = self.builder.get_element_ptr(new_entries, probe_occ_offset);
        let probe_occ = self.builder.load(probe_occ_ptr);
        let is_empty = self.builder.icmp(CmpOp::Eq, probe_occ, zero);
        self.builder.cond_br(is_empty, probe_done, probe_next);

        // Probe next: continue probing
        self.builder.start_block(probe_next);
        let next_probe_slot = self.builder.add(probe_slot, one);
        let wrapped_probe = self.builder.srem(next_probe_slot, new_capacity);
        self.builder.store(probe_slot_ptr, wrapped_probe);
        self.builder.br(probe_loop);

        // Probe done: insert into new array
        self.builder.start_block(probe_done);
        let final_slot = self.builder.load(probe_slot_ptr);
        let final_offset = self.builder.mul(final_slot, three);
        let new_key_ptr = self.builder.get_element_ptr(new_entries, final_offset);
        self.builder.store(new_key_ptr, old_key);
        let new_val_offset = self.builder.add(final_offset, one);
        let new_val_ptr = self.builder.get_element_ptr(new_entries, new_val_offset);
        self.builder.store(new_val_ptr, old_val);
        let new_occ_offset = self.builder.add(final_offset, two);
        let new_occ_ptr = self.builder.get_element_ptr(new_entries, new_occ_offset);
        self.builder.store(new_occ_ptr, one);
        self.builder.br(next_block);

        // Next block: increment index and continue
        self.builder.start_block(next_block);
        let idx_current = self.builder.load(idx_ptr);
        let idx_next = self.builder.add(idx_current, one);
        self.builder.store(idx_ptr, idx_next);
        self.builder.br(loop_block);

        // Done block: update map with new entries and capacity
        self.builder.start_block(done_block);

        // Free old entries
        self.builder.free(old_entries);

        // Store new entries pointer
        self.builder.store(entries_field, new_entries_raw);

        // Store new capacity
        self.builder.store(cap_field, new_capacity);
    }

    /// Lower HashMap::insert(map, key, value) - inserts key-value pair
    /// Returns Option<V> - Some(old_value) if key existed, None otherwise
    /// Supports both i64 and String keys
    fn lower_hashmap_insert(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 3 {
            return self.create_option_none();
        }

        // Detect if key is a String
        let is_string_key = self.is_string_expr(&args[1]);

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let key = self.lower_expr(&args[1]);
        let value = self.lower_expr(&args[2]);

        // For String keys, key is already an i64 (pointer stored as int)
        // We store it directly. For hash/comparison, we convert to pointer.
        let key_storage = key; // Both i64 keys and String pointers-as-i64

        // === Check if resize is needed (load factor > 75%) ===
        // Resize if: count * 4 >= capacity * 3 (equivalent to count/capacity >= 0.75)
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        let count = self.builder.load(count_field);
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        let capacity = self.builder.load(cap_field);

        let four = self.builder.const_int(4);
        let three = self.builder.const_int(3);
        let count_x4 = self.builder.mul(count, four);
        let cap_x3 = self.builder.mul(capacity, three);
        let needs_resize = self.builder.icmp(CmpOp::Sge, count_x4, cap_x3);

        let resize_block = self.builder.create_block();
        let insert_block = self.builder.create_block();

        self.builder.cond_br(needs_resize, resize_block, insert_block);

        // Resize block
        self.builder.start_block(resize_block);
        self.lower_hashmap_resize(map_ptr);
        self.builder.br(insert_block);

        // Insert block: proceed with insertion
        self.builder.start_block(insert_block);

        // Reload capacity and entries (may have changed after resize)
        let cap_field2 = self.builder.get_field_ptr(map_ptr, 2);
        let capacity = self.builder.load(cap_field2);

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Calculate initial slot: hash(key) % capacity
        let slot = if is_string_key {
            // key is i64 representing pointer, convert to real pointer
            let string_ty = self.string_struct_type();
            let key_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(string_ty)));
            self.lower_hashmap_hash_string(key_ptr, capacity)
        } else {
            self.lower_hashmap_hash(key, capacity)
        };

        // Linear probing loop to find slot
        // We'll probe up to 'capacity' times
        let loop_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let empty_block = self.builder.create_block();
        let update_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Initialize loop counter
        let counter_ptr = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(counter_ptr, zero);

        // Store current slot
        let slot_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(slot_ptr, slot);

        // Result storage (Option)
        let result_ptr = self.builder.alloca(IrType::I64);
        let none_val = self.create_option_none();
        self.builder.store(result_ptr, none_val);

        self.builder.br(loop_block);

        // Loop: check current slot
        self.builder.start_block(loop_block);
        let current_slot = self.builder.load(slot_ptr);

        // Get entry pointer: entries + slot * 3 (each entry is 3 i64s)
        let three = self.builder.const_int(3);
        let offset = self.builder.mul(current_slot, three);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, offset);

        // occupied is at offset + 2
        let two = self.builder.const_int(2);
        let occ_offset = self.builder.add(offset, two);
        let occupied_ptr = self.builder.get_element_ptr(entries_ptr, occ_offset);
        let occupied = self.builder.load(occupied_ptr);

        // Check if empty (occupied == 0) or deleted (occupied == 2)
        let is_empty = self.builder.icmp(CmpOp::Eq, occupied, zero);
        let two_const = self.builder.const_int(2);
        let is_deleted = self.builder.icmp(CmpOp::Eq, occupied, two_const);
        let is_available = self.builder.or(is_empty, is_deleted);

        // Check if slot is occupied
        let one = self.builder.const_int(1);
        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);

        // Branch based on occupied status - IMPORTANT: don't compare keys until we know slot is occupied!
        // This prevents dereferencing null pointers for String keys in empty slots
        let check_key_block = self.builder.create_block();
        let not_occupied_block = self.builder.create_block();

        self.builder.cond_br(is_occupied, check_key_block, not_occupied_block);

        // Check key block: slot is occupied, safe to compare keys
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);

        // Key comparison: use string_equals for String keys, icmp for i64
        let key_matches = if is_string_key {
            let str_ty = self.string_struct_type();
            // Both key and entry_key are i64 (pointers stored as int), convert to real pointers
            let key_as_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(str_ty.clone())));
            let entry_key_ptr_typed = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(str_ty)));
            self.lower_string_equals(key_as_ptr, entry_key_ptr_typed)
        } else {
            self.builder.icmp(CmpOp::Eq, entry_key, key)
        };

        // Convert key_matches to bool if needed (string_equals returns i64 1/0)
        let key_matches_bool = if is_string_key {
            let one_cmp = self.builder.const_int(1);
            self.builder.icmp(CmpOp::Eq, key_matches, one_cmp)
        } else {
            key_matches
        };

        // If key matches, go to found; otherwise continue probing
        self.builder.cond_br(key_matches_bool, found_block, update_block);

        // Not occupied block: check if available (empty or deleted)
        self.builder.start_block(not_occupied_block);
        self.builder.cond_br(is_available, empty_block, update_block);

        // Found block: key exists, update value and return old value
        self.builder.start_block(found_block);
        // Get old value (at offset + 1)
        let val_offset = self.builder.add(offset, one);
        let entry_val_ptr = self.builder.get_element_ptr(entries_ptr, val_offset);
        let old_value = self.builder.load(entry_val_ptr);
        // Store new value
        self.builder.store(entry_val_ptr, value);
        // Store Some(old_value) as result
        let some_old = self.create_option_some(old_value);
        self.builder.store(result_ptr, some_old);
        self.builder.br(done_block);

        // Empty block: insert new entry
        self.builder.start_block(empty_block);
        // Store key at entry_key_ptr (use key_storage for String keys - stored as i64 pointer)
        self.builder.store(entry_key_ptr, key_storage);
        // Store value at offset + 1
        let val_offset2 = self.builder.add(offset, one);
        let entry_val_ptr2 = self.builder.get_element_ptr(entries_ptr, val_offset2);
        self.builder.store(entry_val_ptr2, value);
        // Set occupied to 1
        self.builder.store(occupied_ptr, one);
        // Increment count
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        let count = self.builder.load(count_field);
        let new_count = self.builder.add(count, one);
        self.builder.store(count_field, new_count);
        // Result is None (no old value)
        self.builder.br(done_block);

        // Update block: move to next slot (linear probing)
        self.builder.start_block(update_block);
        let next_slot = self.builder.add(current_slot, one);
        let wrapped_slot = self.builder.srem(next_slot, capacity);
        self.builder.store(slot_ptr, wrapped_slot);
        // Increment counter
        let counter = self.builder.load(counter_ptr);
        let new_counter = self.builder.add(counter, one);
        self.builder.store(counter_ptr, new_counter);
        // Check if we've probed all slots
        let done_probing = self.builder.icmp(CmpOp::Sge, new_counter, capacity);
        self.builder.cond_br(done_probing, done_block, loop_block);

        // Done block: return result
        self.builder.start_block(done_block);
        self.builder.load(result_ptr)
    }

    /// Lower HashMap::get(map, key) - gets value for key as Option<V>
    /// Supports both i64 and String keys
    fn lower_hashmap_get(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.create_option_none();
        }

        // Detect if key is a String
        let is_string_key = self.is_string_expr(&args[1]);

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let key = self.lower_expr(&args[1]);

        // Load capacity
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Check for zero capacity
        let zero = self.builder.const_int(0);
        let is_zero_cap = self.builder.icmp(CmpOp::Eq, capacity, zero);

        let zero_cap_block = self.builder.create_block();
        let search_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_zero_cap, zero_cap_block, search_block);

        // Zero capacity: return None
        self.builder.start_block(zero_cap_block);
        let none1 = self.create_option_none();
        let zero_cap_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Search block
        self.builder.start_block(search_block);

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Calculate initial slot
        let slot = if is_string_key {
            // key is i64 representing pointer, convert to real pointer
            let string_ty = self.string_struct_type();
            let key_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(string_ty)));
            self.lower_hashmap_hash_string(key_ptr, capacity)
        } else {
            self.lower_hashmap_hash(key, capacity)
        };

        // Linear probing loop
        let loop_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();

        let counter_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(counter_ptr, zero);

        let slot_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(slot_ptr, slot);

        self.builder.br(loop_block);

        // Loop block
        self.builder.start_block(loop_block);
        let current_slot = self.builder.load(slot_ptr);

        // Get entry pointer
        let three = self.builder.const_int(3);
        let offset = self.builder.mul(current_slot, three);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, offset);

        // Check occupied
        let two = self.builder.const_int(2);
        let occ_offset = self.builder.add(offset, two);
        let occupied_ptr = self.builder.get_element_ptr(entries_ptr, occ_offset);
        let occupied = self.builder.load(occupied_ptr);

        // If empty (occupied == 0), key not found
        let is_empty = self.builder.icmp(CmpOp::Eq, occupied, zero);

        let check_occupied_block = self.builder.create_block();
        self.builder.cond_br(is_empty, not_found_block, check_occupied_block);

        // Check if slot is occupied (vs deleted/tombstone)
        self.builder.start_block(check_occupied_block);
        let one = self.builder.const_int(1);
        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);

        // Branch based on occupied status - IMPORTANT: don't compare keys until we know slot is occupied!
        // This prevents dereferencing null pointers for String keys in deleted slots
        let check_key_block = self.builder.create_block();
        let next_slot_block = self.builder.create_block();

        self.builder.cond_br(is_occupied, check_key_block, next_slot_block);

        // Check key block: slot is occupied, safe to compare keys
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);

        // Key comparison: use string_equals for String keys, icmp for i64
        let key_matches = if is_string_key {
            let str_ty = self.string_struct_type();
            // Both key and entry_key are i64 (pointers stored as int), convert to real pointers
            let key_as_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(str_ty.clone())));
            let entry_key_ptr_typed = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(str_ty)));
            self.lower_string_equals(key_as_ptr, entry_key_ptr_typed)
        } else {
            self.builder.icmp(CmpOp::Eq, entry_key, key)
        };

        let key_matches_bool = if is_string_key {
            let one_cmp = self.builder.const_int(1);
            self.builder.icmp(CmpOp::Eq, key_matches, one_cmp)
        } else {
            key_matches
        };

        self.builder.cond_br(key_matches_bool, found_block, next_slot_block);

        // Next slot block (continue probing)
        self.builder.start_block(next_slot_block);
        let next_slot = self.builder.add(current_slot, one);
        let wrapped_slot = self.builder.srem(next_slot, capacity);
        self.builder.store(slot_ptr, wrapped_slot);
        let counter = self.builder.load(counter_ptr);
        let new_counter = self.builder.add(counter, one);
        self.builder.store(counter_ptr, new_counter);
        let done_probing = self.builder.icmp(CmpOp::Sge, new_counter, capacity);
        self.builder.cond_br(done_probing, not_found_block, loop_block);

        // Found block: return Some(value)
        self.builder.start_block(found_block);
        let val_offset = self.builder.add(offset, one);
        let entry_val_ptr = self.builder.get_element_ptr(entries_ptr, val_offset);
        let value = self.builder.load(entry_val_ptr);
        let some_value = self.create_option_some(value);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Not found block: return None
        self.builder.start_block(not_found_block);
        let none2 = self.create_option_none();
        let not_found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Done block
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (none1, zero_cap_exit),
            (some_value, found_exit),
            (none2, not_found_exit),
        ])
    }

    /// Lower HashMap::contains_key(map, key) - checks if key exists
    /// Supports both i64 and String keys
    fn lower_hashmap_contains_key(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        // Detect if key is a String
        let is_string_key = self.is_string_expr(&args[1]);

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let key = self.lower_expr(&args[1]);

        // Load capacity
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Check for zero capacity
        let zero = self.builder.const_int(0);
        let is_zero_cap = self.builder.icmp(CmpOp::Eq, capacity, zero);

        let zero_cap_block = self.builder.create_block();
        let search_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_zero_cap, zero_cap_block, search_block);

        // Zero capacity: return false
        self.builder.start_block(zero_cap_block);
        let false_val = self.builder.const_bool(false);
        let zero_cap_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Search block
        self.builder.start_block(search_block);

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Calculate initial slot (hash the key)
        let slot = if is_string_key {
            // key is i64 representing pointer, convert to real pointer
            let string_ty = self.string_struct_type();
            let key_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(string_ty)));
            self.lower_hashmap_hash_string(key_ptr, capacity)
        } else {
            self.lower_hashmap_hash(key, capacity)
        };

        let loop_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();

        let counter_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(counter_ptr, zero);

        let slot_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(slot_ptr, slot);

        self.builder.br(loop_block);

        // Loop block
        self.builder.start_block(loop_block);
        let current_slot = self.builder.load(slot_ptr);

        let three = self.builder.const_int(3);
        let offset = self.builder.mul(current_slot, three);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, offset);

        let two = self.builder.const_int(2);
        let occ_offset = self.builder.add(offset, two);
        let occupied_ptr = self.builder.get_element_ptr(entries_ptr, occ_offset);
        let occupied = self.builder.load(occupied_ptr);

        let is_empty = self.builder.icmp(CmpOp::Eq, occupied, zero);

        let check_occupied_block = self.builder.create_block();
        self.builder.cond_br(is_empty, not_found_block, check_occupied_block);

        // Check if slot is occupied (vs deleted/tombstone)
        self.builder.start_block(check_occupied_block);
        let one = self.builder.const_int(1);
        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);

        // Branch based on occupied status - IMPORTANT: don't compare keys until we know slot is occupied!
        let check_key_block = self.builder.create_block();
        let next_slot_block = self.builder.create_block();

        self.builder.cond_br(is_occupied, check_key_block, next_slot_block);

        // Check key block: slot is occupied, safe to compare keys
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);

        // Key comparison: use string_equals for String keys, icmp for i64
        let key_matches = if is_string_key {
            let str_ty = self.string_struct_type();
            // Both key and entry_key are i64 (pointers stored as int), convert to real pointers
            let key_as_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(str_ty.clone())));
            let entry_key_ptr_typed = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(str_ty)));
            self.lower_string_equals(key_as_ptr, entry_key_ptr_typed)
        } else {
            self.builder.icmp(CmpOp::Eq, entry_key, key)
        };

        let key_matches_bool = if is_string_key {
            let one_cmp = self.builder.const_int(1);
            self.builder.icmp(CmpOp::Eq, key_matches, one_cmp)
        } else {
            key_matches
        };

        self.builder.cond_br(key_matches_bool, found_block, next_slot_block);

        // Next slot block (continue probing)
        self.builder.start_block(next_slot_block);
        let next_slot = self.builder.add(current_slot, one);
        let wrapped_slot = self.builder.srem(next_slot, capacity);
        self.builder.store(slot_ptr, wrapped_slot);
        let counter = self.builder.load(counter_ptr);
        let new_counter = self.builder.add(counter, one);
        self.builder.store(counter_ptr, new_counter);
        let done_probing = self.builder.icmp(CmpOp::Sge, new_counter, capacity);
        self.builder.cond_br(done_probing, not_found_block, loop_block);

        // Found block
        self.builder.start_block(found_block);
        let true_val = self.builder.const_bool(true);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Not found block
        self.builder.start_block(not_found_block);
        let false_val2 = self.builder.const_bool(false);
        let not_found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Done block
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (false_val, zero_cap_exit),
            (true_val, found_exit),
            (false_val2, not_found_exit),
        ])
    }

    /// Lower HashMap::remove(map, key) - removes key and returns Option<V>
    /// Supports both i64 and String keys
    fn lower_hashmap_remove(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.create_option_none();
        }

        // Detect if key is a String
        let is_string_key = self.is_string_expr(&args[1]);

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let key = self.lower_expr(&args[1]);

        // Load capacity
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Check for zero capacity
        let zero = self.builder.const_int(0);
        let is_zero_cap = self.builder.icmp(CmpOp::Eq, capacity, zero);

        let zero_cap_block = self.builder.create_block();
        let search_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_zero_cap, zero_cap_block, search_block);

        // Zero capacity: return None
        self.builder.start_block(zero_cap_block);
        let none1 = self.create_option_none();
        let zero_cap_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Search block
        self.builder.start_block(search_block);

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Calculate initial slot (hash the key)
        let slot = if is_string_key {
            // key is i64 representing pointer, convert to real pointer
            let string_ty = self.string_struct_type();
            let key_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(string_ty)));
            self.lower_hashmap_hash_string(key_ptr, capacity)
        } else {
            self.lower_hashmap_hash(key, capacity)
        };

        let loop_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();

        let counter_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(counter_ptr, zero);

        let slot_ptr = self.builder.alloca(IrType::I64);
        self.builder.store(slot_ptr, slot);

        self.builder.br(loop_block);

        // Loop block
        self.builder.start_block(loop_block);
        let current_slot = self.builder.load(slot_ptr);

        let three = self.builder.const_int(3);
        let offset = self.builder.mul(current_slot, three);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, offset);

        let two = self.builder.const_int(2);
        let occ_offset = self.builder.add(offset, two);
        let occupied_ptr = self.builder.get_element_ptr(entries_ptr, occ_offset);
        let occupied = self.builder.load(occupied_ptr);

        // If empty (occupied == 0), key not found
        let is_empty = self.builder.icmp(CmpOp::Eq, occupied, zero);

        let check_occupied_block = self.builder.create_block();
        self.builder.cond_br(is_empty, not_found_block, check_occupied_block);

        // Check if slot is occupied (vs deleted/tombstone)
        self.builder.start_block(check_occupied_block);
        let one = self.builder.const_int(1);
        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);

        // Branch based on occupied status - IMPORTANT: don't compare keys until we know slot is occupied!
        let check_key_block = self.builder.create_block();
        let next_slot_block = self.builder.create_block();

        self.builder.cond_br(is_occupied, check_key_block, next_slot_block);

        // Check key block: slot is occupied, safe to compare keys
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);

        // Key comparison: use string_equals for String keys, icmp for i64
        let key_matches = if is_string_key {
            let str_ty = self.string_struct_type();
            // Both key and entry_key are i64 (pointers stored as int), convert to real pointers
            let key_as_ptr = self.builder.inttoptr(key, IrType::Ptr(Box::new(str_ty.clone())));
            let entry_key_ptr_typed = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(str_ty)));
            self.lower_string_equals(key_as_ptr, entry_key_ptr_typed)
        } else {
            self.builder.icmp(CmpOp::Eq, entry_key, key)
        };

        let key_matches_bool = if is_string_key {
            let one_cmp = self.builder.const_int(1);
            self.builder.icmp(CmpOp::Eq, key_matches, one_cmp)
        } else {
            key_matches
        };

        // Branch based on key match - we already know slot is occupied
        self.builder.cond_br(key_matches_bool, found_block, next_slot_block);

        self.builder.start_block(next_slot_block);
        let next_slot = self.builder.add(current_slot, one);
        let wrapped_slot = self.builder.srem(next_slot, capacity);
        self.builder.store(slot_ptr, wrapped_slot);
        let counter = self.builder.load(counter_ptr);
        let new_counter = self.builder.add(counter, one);
        self.builder.store(counter_ptr, new_counter);
        let done_probing = self.builder.icmp(CmpOp::Sge, new_counter, capacity);
        self.builder.cond_br(done_probing, not_found_block, loop_block);

        // Found block: mark as deleted (tombstone) and return value
        self.builder.start_block(found_block);
        // Get value first
        let val_offset = self.builder.add(offset, one);
        let entry_val_ptr = self.builder.get_element_ptr(entries_ptr, val_offset);
        let value = self.builder.load(entry_val_ptr);
        // Mark as deleted (occupied = 2)
        self.builder.store(occupied_ptr, two);
        // Decrement count
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        let count = self.builder.load(count_field);
        let new_count = self.builder.sub(count, one);
        self.builder.store(count_field, new_count);
        // Return Some(value)
        let some_value = self.create_option_some(value);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Not found block
        self.builder.start_block(not_found_block);
        let none2 = self.create_option_none();
        let not_found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Done block
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (none1, zero_cap_exit),
            (some_value, found_exit),
            (none2, not_found_exit),
        ])
    }

    /// Lower HashMap::len(map) - returns count of entries
    fn lower_hashmap_len(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        self.builder.load(count_field)
    }

    /// Lower HashMap::is_empty(map) - checks if map is empty
    fn lower_hashmap_is_empty(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(true);
        }

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        let count = self.builder.load(count_field);
        let zero = self.builder.const_int(0);
        self.builder.icmp(CmpOp::Eq, count, zero)
    }

    /// Lower HashMap::clear(map) - removes all entries
    fn lower_hashmap_clear(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));

        // Load capacity and entries
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Zero out all entries using memset
        // Size = capacity * 24 bytes (3 i64s per entry)
        let entry_size = self.builder.const_int(24);
        let size = self.builder.mul(capacity, entry_size);
        let zero = self.builder.const_int(0);
        self.builder.memset(entries_ptr, zero, size);

        // Set count to 0
        let count_field = self.builder.get_field_ptr(map_ptr, 1);
        self.builder.store(count_field, zero);

        self.builder.const_int(0) // Return unit
    }

    /// Lower HashMap::capacity(map) - returns capacity
    fn lower_hashmap_capacity(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let map_raw = self.lower_expr(&args[0]);
        // Convert loaded i64 to pointer (pointers stored as i64)
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_raw, IrType::Ptr(Box::new(map_ty)));
        let cap_field = self.builder.get_field_ptr(map_ptr, 2);
        self.builder.load(cap_field)
    }

    // ============ HashSet Implementation ============
    // HashSet is implemented similarly to HashMap but without values
    // Structure: { entries: *i64, len: i64, cap: i64 }
    // Entry: { occupied: i64, key: i64 } (2 i64s per entry)

    /// Create the HashSet struct type (same as HashMap)
    fn hashset_struct_type(&self) -> IrType {
        IrType::Struct(vec![
            IrType::Ptr(Box::new(IrType::I64)), // entries pointer (each entry is 2 i64s)
            IrType::I64,                         // count - number of occupied entries
            IrType::I64,                         // capacity - total slots
        ])
    }

    /// Entry size in i64 units (occupied + key = 2)
    fn hashset_entry_size(&self) -> i64 {
        2
    }

    /// Lower HashSet::new() - creates empty hashset with capacity 16
    fn lower_hashset_new(&mut self) -> VReg {
        let sixteen = self.builder.const_int(16);
        self.lower_hashset_with_capacity_value(sixteen)
    }

    /// Lower HashSet::with_capacity(cap) - creates hashset with initial capacity
    fn lower_hashset_with_capacity(&mut self, args: &[Expr]) -> VReg {
        let cap = if let Some(arg) = args.first() {
            self.lower_expr(arg)
        } else {
            self.builder.const_int(16)
        };
        self.lower_hashset_with_capacity_value(cap)
    }

    /// Helper: creates hashset with given capacity value
    fn lower_hashset_with_capacity_value(&mut self, cap: VReg) -> VReg {
        // Allocate HashSet struct on stack (like HashMap)
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.alloca(set_ty.clone());

        // Allocate entries array on heap: capacity * 2 i64s (occupied + key per entry)
        // Each entry is 2 * 8 = 16 bytes
        let entry_size = self.builder.const_int(16); // 2 * 8 bytes per entry
        let bytes_needed = self.builder.mul(cap, entry_size);

        // Allocate with calloc to zero-initialize (all entries start as empty)
        let entries_ptr = self.builder.calloc(bytes_needed);

        // Store entries pointer (field 0)
        let entries_field = self.builder.get_field_ptr(set_ptr, 0);
        self.builder.store(entries_field, entries_ptr);

        // Store len = 0 (field 1)
        let len_field = self.builder.get_field_ptr(set_ptr, 1);
        let zero = self.builder.const_int(0);
        self.builder.store(len_field, zero);

        // Store capacity (field 2)
        let cap_field = self.builder.get_field_ptr(set_ptr, 2);
        self.builder.store(cap_field, cap);

        // Track type
        self.vreg_types.insert(set_ptr, set_ty);

        set_ptr
    }

    /// Lower HashSet::insert(set, value) - insert value, returns true if new
    fn lower_hashset_insert(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        let set_raw = self.lower_expr(&args[0]);
        let value = self.lower_expr(&args[1]);

        // Check if key is String type
        let is_string_key = self.expr_types.get(&args[1].span).map_or(false, |ty| {
            use crate::typeck::TyKind;
            matches!(&ty.kind, TyKind::Named { name, .. } if name == "String")
        });

        // Convert loaded i64 to pointer
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));

        // Load capacity
        let cap_field = self.builder.get_field_ptr(set_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Compute hash
        let hash = if is_string_key {
            self.lower_hashmap_hash_string(value, capacity)
        } else {
            self.lower_hashmap_hash(value, capacity)
        };

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(set_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        // Entry size is 2 (occupied, key)
        let entry_size = self.builder.const_int(self.hashset_entry_size());

        // Create blocks for probing loop
        let probe_block = self.builder.create_block();
        let check_key_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let insert_block = self.builder.create_block();
        let next_slot_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Initialize probe index to hash value
        let probe_idx = self.builder.alloca(IrType::I64);
        self.builder.store(probe_idx, hash);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        self.builder.br(probe_block);

        // Probe loop
        self.builder.start_block(probe_block);
        let current_idx = self.builder.load(probe_idx);

        // Calculate entry offset: entries_ptr + (current_idx * entry_size)
        let entry_offset = self.builder.mul(current_idx, entry_size);
        let entry_ptr_calc = self.builder.get_element_ptr(entries_ptr, entry_offset);

        // Load occupied flag (entry[0])
        let occupied = self.builder.load(entry_ptr_calc);

        // Load key (entry[1]) - only if occupied
        let key_offset = self.builder.add(entry_offset, one);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, key_offset);

        // Check if slot is occupied
        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);
        self.builder.cond_br(is_occupied, check_key_block, insert_block);

        // Check key block - compare keys if occupied
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);
        let keys_match = if is_string_key {
            // String comparison
            let str_ptr1 = self.builder.inttoptr(value, IrType::Ptr(Box::new(self.string_struct_type())));
            let str_ptr2 = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(self.string_struct_type())));
            let str_eq = self.lower_string_equals(str_ptr1, str_ptr2);
            self.builder.icmp(CmpOp::Eq, str_eq, one)
        } else {
            self.builder.icmp(CmpOp::Eq, value, entry_key)
        };
        self.builder.cond_br(keys_match, found_block, next_slot_block);

        // Found existing - return false (not inserted)
        self.builder.start_block(found_block);
        let false_val = self.builder.const_bool(false);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Insert into empty slot
        self.builder.start_block(insert_block);
        // Set occupied = 1
        self.builder.store(entry_ptr_calc, one);
        // Set key
        self.builder.store(entry_key_ptr, value);
        // Increment len
        let len_field = self.builder.get_field_ptr(set_ptr, 1);
        let old_len = self.builder.load(len_field);
        let new_len = self.builder.add(old_len, one);
        self.builder.store(len_field, new_len);
        // Return true (was inserted)
        let true_val = self.builder.const_bool(true);
        let insert_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Next slot - linear probing
        self.builder.start_block(next_slot_block);
        let next_idx = self.builder.add(current_idx, one);
        let wrapped_idx = self.builder.srem(next_idx, capacity);
        self.builder.store(probe_idx, wrapped_idx);
        self.builder.br(probe_block);

        // Done - use phi to select result
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (false_val, found_exit),
            (true_val, insert_exit),
        ])
    }

    /// Lower HashSet::contains(set, value) - check if value exists
    fn lower_hashset_contains(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        let set_raw = self.lower_expr(&args[0]);
        let value = self.lower_expr(&args[1]);

        // Check if key is String type
        let is_string_key = self.expr_types.get(&args[1].span).map_or(false, |ty| {
            use crate::typeck::TyKind;
            matches!(&ty.kind, TyKind::Named { name, .. } if name == "String")
        });

        // Convert loaded i64 to pointer
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));

        // Load capacity
        let cap_field = self.builder.get_field_ptr(set_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Compute hash
        let hash = if is_string_key {
            self.lower_hashmap_hash_string(value, capacity)
        } else {
            self.lower_hashmap_hash(value, capacity)
        };

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(set_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        let entry_size = self.builder.const_int(self.hashset_entry_size());

        // Create blocks
        let probe_block = self.builder.create_block();
        let check_key_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();
        let next_slot_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Initialize probe index
        let probe_idx = self.builder.alloca(IrType::I64);
        self.builder.store(probe_idx, hash);

        let zero = self.builder.const_int(0);

        self.builder.br(probe_block);

        // Probe loop
        self.builder.start_block(probe_block);
        let current_idx = self.builder.load(probe_idx);

        let entry_offset = self.builder.mul(current_idx, entry_size);
        let entry_ptr = self.builder.get_element_ptr(entries_ptr, entry_offset);

        let occupied = self.builder.load(entry_ptr);
        let one = self.builder.const_int(1);
        let key_offset = self.builder.add(entry_offset, one);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, key_offset);

        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);
        self.builder.cond_br(is_occupied, check_key_block, not_found_block);

        // Check key
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);
        let keys_match = if is_string_key {
            let str_ptr1 = self.builder.inttoptr(value, IrType::Ptr(Box::new(self.string_struct_type())));
            let str_ptr2 = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(self.string_struct_type())));
            let str_eq = self.lower_string_equals(str_ptr1, str_ptr2);
            // lower_string_equals returns i64 (1 or 0), convert to i1
            self.builder.icmp(CmpOp::Eq, str_eq, one)
        } else {
            self.builder.icmp(CmpOp::Eq, value, entry_key)
        };
        self.builder.cond_br(keys_match, found_block, next_slot_block);

        // Found - return true
        self.builder.start_block(found_block);
        let true_val = self.builder.const_bool(true);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Not found (empty slot) - return false
        self.builder.start_block(not_found_block);
        let false_val = self.builder.const_bool(false);
        let not_found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Next slot
        self.builder.start_block(next_slot_block);
        let next_idx = self.builder.add(current_idx, one);
        let wrapped_idx = self.builder.srem(next_idx, capacity);
        self.builder.store(probe_idx, wrapped_idx);
        self.builder.br(probe_block);

        // Done - use phi to select result
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (true_val, found_exit),
            (false_val, not_found_exit),
        ])
    }

    /// Lower HashSet::remove(set, value) - remove value, returns true if existed
    fn lower_hashset_remove(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_bool(false);
        }

        let set_raw = self.lower_expr(&args[0]);
        let value = self.lower_expr(&args[1]);

        // Check if key is String type
        let is_string_key = self.expr_types.get(&args[1].span).map_or(false, |ty| {
            use crate::typeck::TyKind;
            matches!(&ty.kind, TyKind::Named { name, .. } if name == "String")
        });

        // Convert loaded i64 to pointer
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));

        // Load capacity
        let cap_field = self.builder.get_field_ptr(set_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Compute hash
        let hash = if is_string_key {
            self.lower_hashmap_hash_string(value, capacity)
        } else {
            self.lower_hashmap_hash(value, capacity)
        };

        // Load entries pointer (stored as i64, need inttoptr)
        let entries_field = self.builder.get_field_ptr(set_ptr, 0);
        let entries_raw = self.builder.load(entries_field);
        let entries_ptr = self.builder.inttoptr(entries_raw, IrType::Ptr(Box::new(IrType::I64)));

        let entry_size = self.builder.const_int(self.hashset_entry_size());

        // Create blocks
        let probe_block = self.builder.create_block();
        let check_key_block = self.builder.create_block();
        let found_block = self.builder.create_block();
        let not_found_block = self.builder.create_block();
        let next_slot_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Initialize probe index
        let probe_idx = self.builder.alloca(IrType::I64);
        self.builder.store(probe_idx, hash);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        self.builder.br(probe_block);

        // Probe loop
        self.builder.start_block(probe_block);
        let current_idx = self.builder.load(probe_idx);

        let entry_offset = self.builder.mul(current_idx, entry_size);
        let entry_ptr = self.builder.get_element_ptr(entries_ptr, entry_offset);

        let occupied = self.builder.load(entry_ptr);
        let key_offset = self.builder.add(entry_offset, one);
        let entry_key_ptr = self.builder.get_element_ptr(entries_ptr, key_offset);

        let is_occupied = self.builder.icmp(CmpOp::Eq, occupied, one);
        self.builder.cond_br(is_occupied, check_key_block, not_found_block);

        // Check key
        self.builder.start_block(check_key_block);
        let entry_key = self.builder.load(entry_key_ptr);
        let keys_match = if is_string_key {
            let str_ptr1 = self.builder.inttoptr(value, IrType::Ptr(Box::new(self.string_struct_type())));
            let str_ptr2 = self.builder.inttoptr(entry_key, IrType::Ptr(Box::new(self.string_struct_type())));
            let str_eq = self.lower_string_equals(str_ptr1, str_ptr2);
            self.builder.icmp(CmpOp::Eq, str_eq, one)
        } else {
            self.builder.icmp(CmpOp::Eq, value, entry_key)
        };
        self.builder.cond_br(keys_match, found_block, next_slot_block);

        // Found - remove entry
        self.builder.start_block(found_block);
        // Set occupied = 0
        self.builder.store(entry_ptr, zero);
        // Decrement len
        let len_field = self.builder.get_field_ptr(set_ptr, 1);
        let old_len = self.builder.load(len_field);
        let new_len = self.builder.sub(old_len, one);
        self.builder.store(len_field, new_len);
        // Return true (was removed)
        let true_val = self.builder.const_bool(true);
        let found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Not found - return false
        self.builder.start_block(not_found_block);
        let false_val = self.builder.const_bool(false);
        let not_found_exit = self.builder.current_block_id().unwrap();
        self.builder.br(done_block);

        // Next slot
        self.builder.start_block(next_slot_block);
        let next_idx = self.builder.add(current_idx, one);
        let wrapped_idx = self.builder.srem(next_idx, capacity);
        self.builder.store(probe_idx, wrapped_idx);
        self.builder.br(probe_block);

        // Done - use phi to select result
        self.builder.start_block(done_block);
        self.builder.phi(vec![
            (true_val, found_exit),
            (false_val, not_found_exit),
        ])
    }

    /// Lower HashSet::len(set) - get number of elements
    fn lower_hashset_len(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let set_raw = self.lower_expr(&args[0]);
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));
        let len_field = self.builder.get_field_ptr(set_ptr, 1);
        self.builder.load(len_field)
    }

    /// Lower HashSet::is_empty(set) - check if set is empty
    fn lower_hashset_is_empty(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(true);
        }

        let set_raw = self.lower_expr(&args[0]);
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));
        let len_field = self.builder.get_field_ptr(set_ptr, 1);
        let len = self.builder.load(len_field);
        let zero = self.builder.const_int(0);
        self.builder.icmp(CmpOp::Eq, len, zero)
    }

    /// Lower HashSet::clear(set) - remove all elements
    fn lower_hashset_clear(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let set_raw = self.lower_expr(&args[0]);
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));

        // Load entries and capacity
        let entries_field = self.builder.get_field_ptr(set_ptr, 0);
        let entries_base = self.builder.load(entries_field);
        let cap_field = self.builder.get_field_ptr(set_ptr, 2);
        let capacity = self.builder.load(cap_field);

        // Zero out all entries
        let entry_size = self.builder.const_int(self.hashset_entry_size());
        let total_i64s = self.builder.mul(capacity, entry_size);
        let eight = self.builder.const_int(8);
        let total_bytes = self.builder.mul(total_i64s, eight);
        let zero_i8 = self.builder.const_int(0);
        self.builder.call("memset".to_string(), vec![entries_base, zero_i8, total_bytes]);

        // Set len = 0
        let len_field = self.builder.get_field_ptr(set_ptr, 1);
        let zero = self.builder.const_int(0);
        self.builder.store(len_field, zero);

        zero
    }

    /// Lower HashSet::capacity(set) - get current capacity
    fn lower_hashset_capacity(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let set_raw = self.lower_expr(&args[0]);
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_raw, IrType::Ptr(Box::new(set_ty)));
        let cap_field = self.builder.get_field_ptr(set_ptr, 2);
        self.builder.load(cap_field)
    }

    /// Lower a binary operation
    fn lower_binary_op(&mut self, op: BinaryOp, left: VReg, right: VReg) -> VReg {
        // Check if either operand is a float
        let left_is_float = self.vreg_types.get(&left).map_or(false, |ty| {
            matches!(ty, IrType::F32 | IrType::F64)
        });
        let right_is_float = self.vreg_types.get(&right).map_or(false, |ty| {
            matches!(ty, IrType::F32 | IrType::F64)
        });
        let is_float = left_is_float || right_is_float;

        match op {
            BinaryOp::Add => {
                if is_float {
                    self.builder.fadd(left, right)
                } else {
                    self.builder.add(left, right)
                }
            }
            BinaryOp::Sub => {
                if is_float {
                    self.builder.fsub(left, right)
                } else {
                    self.builder.sub(left, right)
                }
            }
            BinaryOp::Mul => {
                if is_float {
                    self.builder.fmul(left, right)
                } else {
                    self.builder.mul(left, right)
                }
            }
            BinaryOp::Div => {
                if is_float {
                    self.builder.fdiv(left, right)
                } else {
                    self.builder.sdiv(left, right)
                }
            }
            BinaryOp::Rem => {
                // Float modulo would require fmod call, use integer rem for now
                self.builder.srem(left, right)
            }
            BinaryOp::Eq => {
                if is_float {
                    self.builder.fcmp(CmpOp::Eq, left, right)
                } else {
                    self.builder.icmp(CmpOp::Eq, left, right)
                }
            }
            BinaryOp::Ne => {
                if is_float {
                    self.builder.fcmp(CmpOp::Ne, left, right)
                } else {
                    self.builder.icmp(CmpOp::Ne, left, right)
                }
            }
            BinaryOp::Lt => {
                if is_float {
                    self.builder.fcmp(CmpOp::Slt, left, right)
                } else {
                    self.builder.icmp(CmpOp::Slt, left, right)
                }
            }
            BinaryOp::Le => {
                if is_float {
                    self.builder.fcmp(CmpOp::Sle, left, right)
                } else {
                    self.builder.icmp(CmpOp::Sle, left, right)
                }
            }
            BinaryOp::Gt => {
                if is_float {
                    self.builder.fcmp(CmpOp::Sgt, left, right)
                } else {
                    self.builder.icmp(CmpOp::Sgt, left, right)
                }
            }
            BinaryOp::Ge => {
                if is_float {
                    self.builder.fcmp(CmpOp::Sge, left, right)
                } else {
                    self.builder.icmp(CmpOp::Sge, left, right)
                }
            }
            BinaryOp::And => self.builder.and(left, right),
            BinaryOp::Or => self.builder.or(left, right),
            BinaryOp::BitAnd => self.builder.and(left, right),
            BinaryOp::BitOr => self.builder.or(left, right),
            BinaryOp::BitXor => self.builder.xor(left, right),
            BinaryOp::Shl => self.builder.shl(left, right),
            BinaryOp::Shr => self.builder.ashr(left, right),
        }
    }

    /// Lower a unary operation
    fn lower_unary_op(&mut self, op: UnaryOp, operand: VReg) -> VReg {
        match op {
            UnaryOp::Neg => {
                // Check if operand is float
                let is_float = self.vreg_types.get(&operand).map_or(false, |ty| {
                    matches!(ty, IrType::F32 | IrType::F64)
                });
                if is_float {
                    self.builder.fneg(operand)
                } else {
                    self.builder.neg(operand)
                }
            }
            UnaryOp::Not => self.builder.not(operand),
            UnaryOp::Deref => self.builder.load(operand),
            UnaryOp::Ref | UnaryOp::RefMut => operand, // Reference is just the address
        }
    }

    /// Lower an if expression
    fn lower_if(&mut self, condition: &Expr, then_branch: &Block, else_branch: Option<&Expr>) -> VReg {
        let cond = self.lower_expr(condition);

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(cond, then_block, else_block);

        // Then branch
        self.builder.start_block(then_block);
        let then_val = self.lower_block(then_branch).unwrap_or_else(|| self.builder.const_int(0));
        let then_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Else branch
        self.builder.start_block(else_block);
        let else_val = if let Some(else_expr) = else_branch {
            self.lower_expr(else_expr)
        } else {
            self.builder.const_int(0)
        };
        let else_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block with phi
        self.builder.start_block(merge_block);
        self.builder.phi(vec![(then_val, then_exit), (else_val, else_exit)])
    }

    /// Lower a while loop
    fn lower_while(&mut self, condition: &Expr, body: &Block, label: Option<String>) -> VReg {
        let cond_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            exit_block,
            continue_block: cond_block,
            label,
        });

        self.builder.br(cond_block);

        // Condition
        self.builder.start_block(cond_block);
        let cond = self.lower_expr(condition);
        self.builder.cond_br(cond, body_block, exit_block);

        // Body
        self.builder.start_block(body_block);
        self.lower_block(body);
        self.builder.br(cond_block);

        // Pop loop context
        self.loop_stack.pop();

        // Exit
        self.builder.start_block(exit_block);
        self.builder.const_int(0) // Unit
    }

    /// Lower an infinite loop
    fn lower_loop(&mut self, body: &Block, label: Option<String>) -> VReg {
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            exit_block,
            continue_block: body_block,
            label,
        });

        self.builder.br(body_block);

        // Body
        self.builder.start_block(body_block);
        self.lower_block(body);
        self.builder.br(body_block);

        // Pop loop context
        self.loop_stack.pop();

        // Exit (unreachable unless break)
        self.builder.start_block(exit_block);
        self.builder.const_int(0)
    }

    /// Bind a pattern to a value
    fn bind_pattern(&mut self, pattern: &Pattern, slot: VReg) {
        match &pattern.kind {
            PatternKind::Ident { name, .. } => {
                self.locals.insert(name.name.clone(), slot);
            }
            PatternKind::Wildcard => {
                // Ignore
            }
            PatternKind::Tuple(patterns) => {
                // TODO: Destructure tuple
                for (i, p) in patterns.iter().enumerate() {
                    let field_ptr = self.builder.get_field_ptr(slot, i as u32);
                    self.bind_pattern(p, field_ptr);
                }
            }
            _ => {
                // Other patterns not yet implemented
            }
        }
    }

    /// Convert AST type to Ty
    fn ast_type_to_ty(&self, ty: &ast::Type) -> Ty {
        use crate::ast::TypeKind;
        match &ty.kind {
            TypeKind::Path(path) => {
                // Get the type name from path
                let name = &path.segments[0].ident.name;
                match name.as_str() {
                    "i8" => Ty::i8(),
                    "i16" => Ty::i16(),
                    "i32" => Ty::i32(),
                    "i64" => Ty::i64(),
                    "isize" => Ty::isize(),
                    "u8" => Ty::u8(),
                    "u16" => Ty::u16(),
                    "u32" => Ty::u32(),
                    "u64" => Ty::u64(),
                    "usize" => Ty::usize(),
                    "f32" => Ty::f32(),
                    "f64" => Ty::f64(),
                    "bool" => Ty::bool(),
                    "()" | "unit" => Ty::unit(),
                    "Option" | "Result" | "Vec" | "Box" | "String" => {
                        // Handle named types with generics
                        let generics = path.segments[0].generics.as_ref()
                            .map(|gs| gs.iter().map(|g| self.ast_type_to_ty(g)).collect())
                            .unwrap_or_default();
                        Ty::named(name.clone(), generics)
                    }
                    _ => {
                        // User-defined types (structs, enums) - treat as named type
                        let generics = path.segments[0].generics.as_ref()
                            .map(|gs| gs.iter().map(|g| self.ast_type_to_ty(g)).collect())
                            .unwrap_or_default();
                        Ty::named(name.clone(), generics)
                    }
                }
            }
            TypeKind::Reference { mutable, inner } => {
                Ty::reference(self.ast_type_to_ty(inner), *mutable)
            }
            TypeKind::Array { element, size: _ } => {
                // TODO: evaluate size expression
                Ty::array(self.ast_type_to_ty(element), 0)
            }
            TypeKind::Tuple(elems) => {
                Ty::tuple(elems.iter().map(|e| self.ast_type_to_ty(e)).collect())
            }
            TypeKind::FnPtr { params, return_type } => {
                let param_tys: Vec<Ty> = params.iter().map(|p| self.ast_type_to_ty(p)).collect();
                let ret_ty = return_type
                    .as_ref()
                    .map(|t| self.ast_type_to_ty(t))
                    .unwrap_or_else(Ty::unit);
                Ty::function(param_tys, ret_ty)
            }
            TypeKind::Never => Ty::never(),
            TypeKind::Infer => Ty::i64(), // Default for inferred
            _ => Ty::i64(), // Default
        }
    }

    /// Get field index for a struct field access
    /// For now, use a simple heuristic based on field name
    fn get_field_index(&self, object: &Expr, field_name: &str) -> u32 {
        // First, try to look up from specialized struct definitions
        if let Some(mangled_name) = self.get_struct_mangled_name(object) {
            if let Some(idx) = self.get_specialized_field_index(&mangled_name, field_name) {
                return idx;
            }
        }

        // Also check for non-generic structs in specialized_structs
        if let Some(ty) = self.expr_types.get(&object.span) {
            if let TyKind::Named { name, generics } = &ty.kind {
                if generics.is_empty() {
                    // Resolve "Self" to the current impl type
                    let resolved_name = if name == "Self" {
                        self.current_impl_type.as_ref().unwrap_or(name)
                    } else {
                        name
                    };
                    // Non-generic struct - check if we have it registered
                    if let Some(idx) = self.get_specialized_field_index(resolved_name, field_name) {
                        return idx;
                    }
                }
            }
        }

        // Fallback: use common field name patterns
        match field_name {
            "x" | "r" | "red" | "width" | "left" | "first" => 0,
            "y" | "g" | "green" | "height" | "top" | "second" => 1,
            "z" | "blue" | "depth" | "third" => 2,
            "w" | "alpha" | "fourth" => 3,
            "a" => 0,
            "b" => 1,
            "c" => 2,
            "d" => 3,
            "value" => 0,  // Common for wrapper types
            "key" => 0,
            "val" => 1,
            _ => 0,
        }
    }

    /// Lower a for loop: for i in iterable { body }
    /// Supports: Range (start..end), Vec, VecIter
    fn lower_for_range(&mut self, pattern: &Pattern, iterable: &Expr, body: &Block, label: Option<String>) -> VReg {
        // Check if iterable is a Range expression
        if let ExprKind::Range { start: s, end: e, .. } = &iterable.kind {
            return self.lower_for_range_expr(pattern, s, e, body, label);
        }

        // Check if iterable is a Vec or VecIter (from Vec::iter)
        let is_vec_iter = self.expr_types.get(&iterable.span).map_or(false, |ty| {
            use crate::typeck::TyKind;
            matches!(&ty.kind, TyKind::Named { name, .. } if name == "Vec" || name == "VecIter")
        });

        if is_vec_iter {
            return self.lower_for_vec(pattern, iterable, body, label);
        }

        // Fallback: treat as empty loop
        self.builder.const_int(0)
    }

    /// Lower a for-range loop: for i in start..end { body }
    fn lower_for_range_expr(&mut self, pattern: &Pattern, start: &Option<Box<Expr>>, end: &Option<Box<Expr>>, body: &Block, label: Option<String>) -> VReg {
        let start_val = start.as_ref()
            .map(|expr| self.lower_expr(expr))
            .unwrap_or_else(|| self.builder.const_int(0));
        let end_val = end.as_ref()
            .map(|expr| self.lower_expr(expr))
            .unwrap_or_else(|| self.builder.const_int(0));

        // Allocate loop variable
        let loop_var = self.builder.alloca(IrType::I64);
        self.builder.store(loop_var, start_val);

        // Bind pattern to loop variable
        if let PatternKind::Ident { name, .. } = &pattern.kind {
            self.locals.insert(name.name.clone(), loop_var);
        }

        // Create blocks
        let cond_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();
        let incr_block = self.builder.create_block();

        // Push loop context for break/continue
        // continue jumps to increment block (not condition) to properly advance the counter
        self.loop_stack.push(LoopContext {
            exit_block,
            continue_block: incr_block,
            label,
        });

        self.builder.br(cond_block);

        // Condition: i < end
        self.builder.start_block(cond_block);
        let current = self.builder.load(loop_var);
        let cond = self.builder.icmp(CmpOp::Slt, current, end_val);
        self.builder.cond_br(cond, body_block, exit_block);

        // Body
        self.builder.start_block(body_block);
        self.lower_block(body);
        self.builder.br(incr_block);

        // Increment block: i = i + 1
        self.builder.start_block(incr_block);
        let current = self.builder.load(loop_var);
        let one = self.builder.const_int(1);
        let next = self.builder.add(current, one);
        self.builder.store(loop_var, next);
        self.builder.br(cond_block);

        // Pop loop context
        self.loop_stack.pop();

        // Exit
        self.builder.start_block(exit_block);
        self.builder.const_int(0) // Unit
    }

    /// Lower a for loop over Vec: for elem in vec { body }
    fn lower_for_vec(&mut self, pattern: &Pattern, iterable: &Expr, body: &Block, label: Option<String>) -> VReg {
        // Check if it's a VecIter (from Vec::iter()) or direct Vec
        let is_veciter = self.expr_types.get(&iterable.span).map_or(false, |ty| {
            use crate::typeck::TyKind;
            matches!(&ty.kind, TyKind::Named { name, .. } if name == "VecIter")
        });

        // Evaluate the Vec/VecIter
        let iter_or_vec = self.lower_expr(iterable);

        let (data_ptr, len, start_idx) = if is_veciter {
            // VecIter struct: { vec_ptr: i64, index: i64 }
            // Get the underlying Vec pointer
            let vec_ptr_field = self.builder.get_field_ptr(iter_or_vec, 0);
            let vec_ptr = self.builder.load(vec_ptr_field);
            let vec_ptr_typed = self.builder.inttoptr(vec_ptr, IrType::Ptr(Box::new(self.vec_struct_type())));

            // Get starting index from VecIter
            let idx_field = self.builder.get_field_ptr(iter_or_vec, 1);
            let start_idx = self.builder.load(idx_field);

            // Get length from Vec
            let len_field = self.builder.get_field_ptr(vec_ptr_typed, 1);
            let len = self.builder.load(len_field);

            // Get data pointer from Vec
            let ptr_field = self.builder.get_field_ptr(vec_ptr_typed, 0);
            let data_ptr = self.builder.load(ptr_field);
            let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

            (data_ptr, len, start_idx)
        } else {
            // Direct Vec: { ptr: *T, len: i64, cap: i64 }
            let len_field = self.builder.get_field_ptr(iter_or_vec, 1);
            let len = self.builder.load(len_field);

            let ptr_field = self.builder.get_field_ptr(iter_or_vec, 0);
            let data_ptr = self.builder.load(ptr_field);
            let data_ptr = self.builder.inttoptr(data_ptr, IrType::Ptr(Box::new(IrType::I64)));

            let zero = self.builder.const_int(0);
            (data_ptr, len, zero)
        };

        // Allocate index variable (starts at start_idx)
        let index_var = self.builder.alloca(IrType::I64);
        self.builder.store(index_var, start_idx);

        // Allocate element variable for the pattern
        let elem_var = self.builder.alloca(IrType::I64);
        if let PatternKind::Ident { name, .. } = &pattern.kind {
            self.locals.insert(name.name.clone(), elem_var);
        }

        // Create blocks
        let cond_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();
        let incr_block = self.builder.create_block();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            exit_block,
            continue_block: incr_block,
            label,
        });

        self.builder.br(cond_block);

        // Condition: index < len
        self.builder.start_block(cond_block);
        let current_idx = self.builder.load(index_var);
        let cond = self.builder.icmp(CmpOp::Slt, current_idx, len);
        self.builder.cond_br(cond, body_block, exit_block);

        // Body: load element, run body
        self.builder.start_block(body_block);

        // Load current element: data_ptr[index]
        let current_idx = self.builder.load(index_var);
        let elem_ptr = self.builder.get_element_ptr(data_ptr, current_idx);
        let elem = self.builder.load(elem_ptr);
        self.builder.store(elem_var, elem);

        // Execute body
        self.lower_block(body);
        self.builder.br(incr_block);

        // Increment block: index = index + 1
        self.builder.start_block(incr_block);
        let current_idx = self.builder.load(index_var);
        let one = self.builder.const_int(1);
        let next_idx = self.builder.add(current_idx, one);
        self.builder.store(index_var, next_idx);
        self.builder.br(cond_block);

        // Pop loop context
        self.loop_stack.pop();

        // Exit
        self.builder.start_block(exit_block);
        self.builder.const_int(0) // Unit
    }

    /// Convert internal Ty to IR type
    fn ty_to_ir_type(&self, ty: &Ty) -> IrType {
        use crate::typeck::TyKind;
        match &ty.kind {
            TyKind::Unit => IrType::Void,
            TyKind::Bool => IrType::Bool,
            TyKind::Int(int_ty) => {
                use crate::typeck::IntTy;
                match int_ty {
                    IntTy::I8 => IrType::I8,
                    IntTy::I16 => IrType::I16,
                    IntTy::I32 => IrType::I32,
                    IntTy::I64 | IntTy::Isize => IrType::I64,
                    IntTy::I128 => IrType::I64, // TODO: i128 support
                }
            }
            TyKind::Uint(uint_ty) => {
                use crate::typeck::UintTy;
                match uint_ty {
                    UintTy::U8 => IrType::I8, // Use signed types for now
                    UintTy::U16 => IrType::I16,
                    UintTy::U32 => IrType::I32,
                    UintTy::U64 | UintTy::Usize => IrType::I64,
                    UintTy::U128 => IrType::I64,
                }
            }
            TyKind::Float(float_ty) => {
                use crate::typeck::FloatTy;
                match float_ty {
                    FloatTy::F32 => IrType::F32,
                    FloatTy::F64 => IrType::F64,
                }
            }
            TyKind::Ref { inner, .. } => IrType::ptr(self.ty_to_ir_type(inner)),
            TyKind::Array { element, size } => {
                IrType::array(self.ty_to_ir_type(element), *size)
            }
            TyKind::Tuple(elems) => {
                IrType::Struct(elems.iter().map(|e| self.ty_to_ir_type(e)).collect())
            }
            TyKind::Fn { params, ret } => IrType::Fn {
                params: params.iter().map(|p| self.ty_to_ir_type(p)).collect(),
                ret: Box::new(self.ty_to_ir_type(ret)),
            },
            TyKind::Named { name, .. } if name == "Option" || name == "Result" => {
                // Option and Result are represented as pointers to { i32 discriminant, i64 payload }
                let struct_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
                IrType::Ptr(Box::new(struct_ty))
            },
            TyKind::Named { name, .. } if name == "String" => {
                // String = { *i8 data, i64 len, i64 cap }
                let struct_ty = IrType::Struct(vec![
                    IrType::Ptr(Box::new(IrType::I8)),
                    IrType::I64,
                    IrType::I64,
                ]);
                IrType::Ptr(Box::new(struct_ty))
            },
            TyKind::Named { name, .. } if name == "File" => {
                // File = { *i8 handle } - wrapper around FILE*
                let struct_ty = IrType::Struct(vec![
                    IrType::Ptr(Box::new(IrType::I8)),  // FILE* (opaque pointer)
                ]);
                IrType::Ptr(Box::new(struct_ty))
            },
            TyKind::Named { name, .. } if name == "HashMap" => {
                // HashMap = { *Entry entries, i64 count, i64 capacity }
                // Entry = { i64 key, i64 value, i64 occupied } - each entry is 3 i64s
                // For simplicity, entries is just a pointer to contiguous array of Entry structs
                let struct_ty = IrType::Struct(vec![
                    IrType::Ptr(Box::new(IrType::I64)),  // entries pointer (to array of Entry)
                    IrType::I64,                         // count - number of occupied entries
                    IrType::I64,                         // capacity - total slots
                ]);
                IrType::Ptr(Box::new(struct_ty))
            },
            // Channel types (Phase 6) - all represented as i64 (pointer to channel struct)
            TyKind::Named { name, .. } if name == "Channel" || name == "Sender" || name == "Receiver" => {
                IrType::I64  // Pointer to channel structure stored as i64
            },
            // User-defined struct types
            TyKind::Named { name, .. } => {
                // Resolve "Self" to the current impl type
                let resolved_name = if name == "Self" {
                    self.current_impl_type.as_ref().unwrap_or(name)
                } else {
                    name
                };

                // Check if we have a registered struct type
                if let Some(ir_type) = self.struct_types.get(resolved_name) {
                    // User structs are heap-allocated, so variables hold pointers to them
                    IrType::Ptr(Box::new(ir_type.clone()))
                } else {
                    // Fallback to i64 for unknown types
                    IrType::I64
                }
            },
            // Type inference variables - default to concrete types
            TyKind::IntVar => IrType::I64,   // Integer inference variable -> i64
            TyKind::FloatVar => IrType::F64, // Float inference variable -> f64
            _ => IrType::I64, // Default to i64 for unknown types
        }
    }

    /// Lower a match expression
    fn lower_match(&mut self, scrutinee: &Expr, arms: &[ast::MatchArm]) -> VReg {
        // Lower the scrutinee - get the enum pointer
        // For heap-allocated enums (Option, Result, etc.), the variable's slot
        // contains a pointer to the enum struct. We need to load that pointer first.
        let scrutinee_slot = self.lower_expr_place(scrutinee);
        let scrutinee_ptr = self.builder.load(scrutinee_slot);

        // Load the discriminant (first field of the enum struct)
        // The discriminant is stored as i32, but load may return i64
        // Truncate to i32 for correct comparison
        let discrim_ptr = self.builder.get_field_ptr(scrutinee_ptr, 0);
        let discrim_raw = self.builder.load(discrim_ptr);
        let discrim = self.builder.trunc(discrim_raw, IrType::I32);

        // Create blocks for each arm's condition check, body, and the merge block
        let check_blocks: Vec<_> = arms.iter().map(|_| self.builder.create_block()).collect();
        let body_blocks: Vec<_> = arms.iter().map(|_| self.builder.create_block()).collect();
        let merge_block = self.builder.create_block();

        // Create alloca for the result
        let result_slot = self.builder.alloca(IrType::I64);

        // Jump to first check block
        self.builder.br(check_blocks[0]);

        // Generate code for each arm
        for (i, arm) in arms.iter().enumerate() {
            let check_block = check_blocks[i];
            let body_block = body_blocks[i];
            let next_check = if i + 1 < arms.len() {
                check_blocks[i + 1]
            } else {
                merge_block // If no match, fall through to merge (shouldn't happen with exhaustive match)
            };

            // Start the check block
            self.builder.start_block(check_block);

            // Check if pattern matches
            match &arm.pattern.kind {
                PatternKind::Enum { path, fields } => {
                    let variant_name = path.segments.iter()
                        .map(|s| s.ident.name.clone())
                        .collect::<Vec<_>>()
                        .join("::");

                    // Try to find the variant, also checking qualified names for built-in enums
                    let variant_info = self.enum_variants.get(&variant_name).cloned()
                        .or_else(|| {
                            // If not found and it's a short name, try Option:: and Result:: prefixes
                            if path.segments.len() == 1 {
                                let short_name = &path.segments[0].ident.name;
                                self.enum_variants.get(&format!("Option::{}", short_name)).cloned()
                                    .or_else(|| self.enum_variants.get(&format!("Result::{}", short_name)).cloned())
                            } else {
                                None
                            }
                        });

                    if let Some(variant_info) = variant_info {
                        // Discriminant is stored as i32, so truncate the expected value
                        let expected_discrim = self.builder.const_int(variant_info.discriminant as i64);
                        let expected_discrim_i32 = self.builder.trunc(expected_discrim, IrType::I32);
                        let cond = self.builder.icmp(CmpOp::Eq, discrim, expected_discrim_i32);
                        self.builder.cond_br(cond, body_block, next_check);

                        // Body block
                        self.builder.start_block(body_block);

                        // Bind pattern variables (extract payload)
                        if !fields.is_empty() && variant_info.payload_type.is_some() {
                            let payload_ptr = self.builder.get_field_ptr(scrutinee_ptr, 1);
                            if let PatternKind::Ident { name, .. } = &fields[0].kind {
                                self.locals.insert(name.name.clone(), payload_ptr);
                                self.local_types.insert(name.name.clone(), IrType::I64);
                            }
                        }

                        let arm_result = self.lower_expr(&arm.body);
                        self.builder.store(result_slot, arm_result);
                        self.builder.br(merge_block);
                    } else {
                        // Unknown variant, skip to next check
                        self.builder.br(next_check);
                    }
                }
                PatternKind::Wildcard | PatternKind::Ident { .. } => {
                    // Wildcard or binding pattern - always matches, go directly to body
                    self.builder.br(body_block);
                    self.builder.start_block(body_block);

                    // If it's a binding, bind the scrutinee value
                    if let PatternKind::Ident { name, .. } = &arm.pattern.kind {
                        self.locals.insert(name.name.clone(), scrutinee_ptr);
                    }

                    let arm_result = self.lower_expr(&arm.body);
                    self.builder.store(result_slot, arm_result);
                    self.builder.br(merge_block);
                }
                PatternKind::Literal(lit) => {
                    // Literal pattern - compare values
                    let expected = self.lower_literal(lit);
                    let scrutinee_val = self.builder.load(scrutinee_ptr);
                    let cond = self.builder.icmp(CmpOp::Eq, scrutinee_val, expected);
                    self.builder.cond_br(cond, body_block, next_check);

                    self.builder.start_block(body_block);
                    let arm_result = self.lower_expr(&arm.body);
                    self.builder.store(result_slot, arm_result);
                    self.builder.br(merge_block);
                }
                _ => {
                    // Unsupported pattern - skip to next check
                    self.builder.br(next_check);
                }
            }
        }

        // Merge block - load and return result
        self.builder.start_block(merge_block);
        self.builder.load(result_slot)
    }

    /// Lower an enum variant constructor (like Option::Some(42))
    fn lower_enum_variant_constructor(&mut self, _variant_name: &str, variant_info: &EnumVariantInfo, args: &[Expr]) -> VReg {
        // Allocate space for the enum on the HEAP (not stack!)
        // This is critical because Option/Result values may be returned from functions.
        // If we used stack allocation (alloca), returning the pointer would be UB
        // as the stack frame is destroyed on return.
        let enum_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let enum_ptr = self.builder.malloc(enum_ty);

        // Store discriminant
        let discrim_ptr = self.builder.get_field_ptr(enum_ptr, 0);
        let discrim_val = self.builder.const_int(variant_info.discriminant as i64);
        self.vreg_types.insert(discrim_val, IrType::I32);
        // Truncate to i32
        let discrim_i32 = self.builder.trunc(discrim_val, IrType::I32);
        self.builder.store(discrim_ptr, discrim_i32);

        // Store payload if present
        if variant_info.payload_type.is_some() && !args.is_empty() {
            let payload_ptr = self.builder.get_field_ptr(enum_ptr, 1);
            let payload_val = self.lower_expr(&args[0]);
            self.builder.store(payload_ptr, payload_val);
        }

        enum_ptr
    }

    /// Lower a closure expression
    /// Closures are represented as a struct { fn_ptr, env_ptr }
    /// where env is a struct containing captured variables
    fn lower_closure(&mut self, params: &[ast::Param], body: &Expr) -> VReg {
        // Generate unique name for this closure
        let closure_name = format!("__closure_{}", self.closure_counter);
        self.closure_counter += 1;

        // Find free variables (variables used in body but not declared there)
        let param_names: std::collections::HashSet<String> = params.iter()
            .map(|p| p.name.name.clone())
            .collect();
        let free_vars = self.find_free_variables(body, &param_names);

        // Create environment struct type
        // For captured closures, store as i64 (pointer to closure struct)
        let env_field_types: Vec<IrType> = free_vars.iter()
            .map(|(name, _, ty)| {
                if self.closure_locals.contains(name) {
                    // Closures are stored as pointer (i64)
                    IrType::I64
                } else {
                    ty.clone()
                }
            })
            .collect();
        let env_ty = if env_field_types.is_empty() {
            IrType::I64 // Dummy type for empty environment
        } else {
            IrType::Struct(env_field_types)
        };

        // Allocate and populate environment
        let env_ptr = self.builder.alloca(env_ty.clone());
        for (i, (name, slot, _ty)) in free_vars.iter().enumerate() {
            if !free_vars.is_empty() {
                let field_ptr = self.builder.get_field_ptr(env_ptr, i as u32);
                if self.closure_locals.contains(name) {
                    // For captured closures, store the pointer as i64
                    let ptr_as_i64 = self.builder.bitcast(*slot, IrType::I64);
                    self.builder.store(field_ptr, ptr_as_i64);
                } else {
                    // Regular variable - load value and store
                    let val = self.builder.load(*slot);
                    self.builder.store(field_ptr, val);
                }
            }
        }

        // Create the closure struct: { fn_ptr (as i64), env_ptr }
        let closure_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let closure_ptr = self.builder.alloca(closure_ty);

        // Store function pointer
        let fn_ptr_slot = self.builder.get_field_ptr(closure_ptr, 0);
        let fn_ptr = self.builder.func_ref(&closure_name);
        // Bitcast function pointer to i64 for storage
        let fn_as_i64 = self.builder.bitcast(fn_ptr, IrType::I64);
        self.builder.store(fn_ptr_slot, fn_as_i64);

        // Store environment pointer
        let env_ptr_slot = self.builder.get_field_ptr(closure_ptr, 1);
        // Bitcast env_ptr to i64 for storage
        let env_as_i64 = self.builder.bitcast(env_ptr, IrType::I64);
        self.builder.store(env_ptr_slot, env_as_i64);

        // Convert params to IR types
        let ir_params: Vec<(String, IrType)> = params.iter()
            .map(|p| (p.name.name.clone(), self.ty_to_ir_type(&self.ast_type_to_ty(&p.ty))))
            .collect();

        // Identify which captured variables are themselves closures
        let captured_closures: HashSet<String> = free_vars.iter()
            .filter(|(name, _, _)| self.closure_locals.contains(name))
            .map(|(name, _, _)| name.clone())
            .collect();

        // Store pending closure for later generation
        self.pending_closures.push(PendingClosure {
            name: closure_name,
            params: ir_params,
            captures: free_vars,
            captured_closures,
            body: body.clone(),
        });

        closure_ptr
    }

    /// Find free variables in an expression (variables that are used but not defined locally)
    fn find_free_variables(&self, expr: &Expr, bound: &std::collections::HashSet<String>) -> Vec<(String, VReg, IrType)> {
        let mut free = Vec::new();
        self.collect_free_vars(expr, bound, &mut free);
        free
    }

    /// Find free variables in a block (variables that are used but not defined locally)
    fn find_free_variables_in_block(&self, block: &Block, bound: &std::collections::HashSet<String>) -> Vec<(String, VReg, IrType)> {
        let mut free = Vec::new();
        let mut local_bound = bound.clone();

        // Collect free vars from statements, tracking new bindings
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Expr(e) => {
                    self.collect_free_vars(e, &local_bound, &mut free);
                }
                StmtKind::Let { pattern, value, .. } => {
                    // Check the initializer expression first
                    if let Some(val) = value {
                        self.collect_free_vars(val, &local_bound, &mut free);
                    }
                    // Then add the bound variable
                    if let PatternKind::Ident { name, .. } = &pattern.kind {
                        local_bound.insert(name.name.clone());
                    }
                }
                _ => {}
            }
        }

        // Check the block's final expression
        if let Some(expr) = &block.expr {
            self.collect_free_vars(expr, &local_bound, &mut free);
        }

        free
    }

    /// Recursively collect free variables from an expression
    fn collect_free_vars(&self, expr: &Expr, bound: &std::collections::HashSet<String>, free: &mut Vec<(String, VReg, IrType)>) {
        match &expr.kind {
            ExprKind::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident.name;
                    // Check if this variable is not bound and is in our locals
                    if !bound.contains(name) {
                        if let Some(&slot) = self.locals.get(name) {
                            let ty = self.local_types.get(name).cloned().unwrap_or(IrType::I64);
                            // Avoid duplicates
                            if !free.iter().any(|(n, _, _)| n == name) {
                                free.push((name.clone(), slot, ty));
                            }
                        }
                    }
                }
            }
            ExprKind::Binary { left, right, .. } => {
                self.collect_free_vars(left, bound, free);
                self.collect_free_vars(right, bound, free);
            }
            ExprKind::Unary { operand, .. } => {
                self.collect_free_vars(operand, bound, free);
            }
            ExprKind::Call { func, args } => {
                self.collect_free_vars(func, bound, free);
                for arg in args {
                    self.collect_free_vars(arg, bound, free);
                }
            }
            ExprKind::If { condition, then_branch, else_branch } => {
                self.collect_free_vars(condition, bound, free);
                for stmt in &then_branch.stmts {
                    match &stmt.kind {
                        StmtKind::Expr(e) => {
                            self.collect_free_vars(e, bound, free);
                        }
                        StmtKind::Let { value, .. } => {
                            if let Some(val) = value {
                                self.collect_free_vars(val, bound, free);
                            }
                        }
                        _ => {}
                    }
                }
                // Also check the block's final expression
                if let Some(expr) = &then_branch.expr {
                    self.collect_free_vars(expr, bound, free);
                }
                if let Some(else_expr) = else_branch {
                    self.collect_free_vars(else_expr, bound, free);
                }
            }
            ExprKind::Block(block) => {
                for stmt in &block.stmts {
                    match &stmt.kind {
                        StmtKind::Expr(e) => {
                            self.collect_free_vars(e, bound, free);
                        }
                        StmtKind::Let { value, .. } => {
                            // Check the initializer expression for free variables
                            if let Some(val) = value {
                                self.collect_free_vars(val, bound, free);
                            }
                        }
                        _ => {}
                    }
                }
                // Also check the block's final expression
                if let Some(expr) = &block.expr {
                    self.collect_free_vars(expr, bound, free);
                }
            }
            _ => {
                // Other expression types - could add more cases as needed
            }
        }
    }

    // ============ Future<T> Methods ============

    /// Lower Future::poll(future) -> bool
    /// Checks if future is ready (state == 1)
    fn lower_future_poll(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(false);
        }
        let future_ptr = self.lower_expr(&args[0]);

        // Future struct: { state: i32, value: T }
        // state: 0 = Pending, 1 = Ready
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        let state = self.builder.load(state_ptr);

        // Return state == 1
        let one = self.builder.const_int(1);
        self.builder.icmp(CmpOp::Eq, state, one)
    }

    /// Lower Future::is_ready(future) -> bool
    /// Same as poll for our simple implementation
    fn lower_future_is_ready(&mut self, args: &[Expr]) -> VReg {
        self.lower_future_poll(args)
    }

    /// Lower Future::get(future) -> T
    /// Extracts the value from a ready future
    fn lower_future_get(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }
        let future_ptr = self.lower_expr(&args[0]);

        // Future struct: { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
        // Get value field (field 3)
        let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.load(value_ptr)
    }

    // ============ Async Runtime Functions ============

    /// Lower block_on(future) -> T
    /// Runs a future to completion with full executor support:
    /// 1. Initialize the runtime task queue
    /// 2. Run the main future
    /// 3. Run all spawned tasks
    /// 4. Return the result from the main future
    fn lower_block_on(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        // Initialize the async runtime (task queue)
        self.builder.call("__rt_init", vec![]);

        let future_ptr = self.lower_expr(&args[0]);

        // Future struct: { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
        // state: 0 = NotStarted, 1 = Ready, 2 = Pending

        // Create blocks for control flow
        let check_state_block = self.builder.create_block();
        let not_started_block = self.builder.create_block();
        let pending_block = self.builder.create_block();
        let run_tasks_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Result slot
        let result_slot = self.builder.alloca(IrType::I64);

        // Jump to state check
        self.builder.br(check_state_block);

        // Check state
        self.builder.start_block(check_state_block);
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        let state = self.builder.load(state_ptr);

        // Check if Ready (1)
        let one = self.builder.const_int(1);
        let is_ready = self.builder.icmp(CmpOp::Eq, state, one);
        let check_pending_block = self.builder.create_block();
        self.builder.cond_br(is_ready, done_block, check_pending_block);

        // Check if Pending (2)
        self.builder.start_block(check_pending_block);
        let two = self.builder.const_int(2);
        let is_pending = self.builder.icmp(CmpOp::Eq, state, two);
        self.builder.cond_br(is_pending, pending_block, not_started_block);

        // Pending block: process timers and run tasks, then check again
        self.builder.start_block(pending_block);
        // Process expired timers (Phase 5) - this is critical for sleep futures
        self.builder.call("__rt_timer_process", vec![]);
        // Also run spawned tasks if any
        self.builder.call("__rt_run_tasks", vec![]);
        self.builder.br(check_state_block);

        // Not started block: execute the closure
        self.builder.start_block(not_started_block);

        // Load fn_ptr from Future
        let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
        let fn_ptr_i64 = self.builder.load(fn_ptr_field);

        // Check if fn_ptr == -1 (sleep marker - no closure to call)
        let neg_one = self.builder.const_int(-1);
        let is_sleep = self.builder.icmp(CmpOp::Eq, fn_ptr_i64, neg_one);

        let call_closure_block = self.builder.create_block();
        let skip_closure_block = self.builder.create_block();
        self.builder.cond_br(is_sleep, skip_closure_block, call_closure_block);

        // Sleep future in NotStarted state - this shouldn't happen normally
        // but process timers and run tasks and check again
        self.builder.start_block(skip_closure_block);
        self.builder.call("__rt_timer_process", vec![]);
        self.builder.call("__rt_run_tasks", vec![]);
        self.builder.br(check_state_block);

        // Call the closure
        self.builder.start_block(call_closure_block);
        let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
        let env_ptr_i64 = self.builder.load(env_ptr_field);

        // Call the closure: fn_ptr(env_ptr) -> result
        let call_result = self.builder.call_ptr(fn_ptr_i64, vec![env_ptr_i64]);

        // Check if result is pending marker (i64::MIN)
        let pending_marker = self.builder.const_int(i64::MIN);
        let result_is_pending = self.builder.icmp(CmpOp::Eq, call_result, pending_marker);

        let store_result_block = self.builder.create_block();
        let set_pending_block = self.builder.create_block();
        self.builder.cond_br(result_is_pending, set_pending_block, store_result_block);

        // Result is pending - set state to Pending and loop
        self.builder.start_block(set_pending_block);
        self.builder.store(state_ptr, two);
        self.builder.call("__rt_run_tasks", vec![]);
        self.builder.br(check_state_block);

        // Store result in Future's value field and set state to Ready
        self.builder.start_block(store_result_block);
        let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_ptr, call_result);
        self.builder.store(state_ptr, one);
        self.builder.store(result_slot, call_result);
        self.builder.br(done_block);

        // Done block - return the result
        self.builder.start_block(done_block);

        // Load result from the main future's value field
        let value_ptr2 = self.builder.get_field_ptr(future_ptr, 3);
        let existing_value = self.builder.load(value_ptr2);
        existing_value
    }

    /// Lower spawn(future) -> Future<T>
    /// Adds the future to the task queue and returns a handle
    fn lower_spawn(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            // Return empty ready future with 4 fields
            let future_ty = IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]);
            let future_ptr = self.builder.malloc(future_ty.clone());
            let zero = self.builder.const_int(0);
            let one = self.builder.const_int(1);
            // state = 1 (Ready)
            let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
            self.builder.store(state_ptr, one);
            // fn_ptr = 0
            let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
            self.builder.store(fn_ptr_field, zero);
            // env_ptr = 0
            let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
            self.builder.store(env_ptr_field, zero);
            // value = 0
            let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
            self.builder.store(value_ptr, zero);
            return future_ptr;
        }

        // Get the future pointer
        let future_ptr = self.lower_expr(&args[0]);

        // Convert pointer to i64 for storage
        let future_ptr_i64 = self.builder.ptrtoint(future_ptr, IrType::I64);

        // Add to task queue via __rt_spawn
        let _task_id = self.builder.call("__rt_spawn", vec![future_ptr_i64]);

        // Return the original future (caller can await it later)
        future_ptr
    }

    /// Lower select! { a = f1 => body1, b = f2 => body2, ... }
    /// Poll all futures, execute body of first to complete, return result
    fn lower_select(&mut self, arms: &[crate::ast::SelectArm]) -> VReg {
        if arms.is_empty() {
            return self.builder.const_int(0);
        }

        let num_arms = arms.len();

        // Allocate result slot
        let result_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(result_slot, zero);

        // Create future pointers for each arm
        let mut future_ptrs = Vec::new();
        for arm in arms {
            let future_ptr = self.lower_expr(&arm.future);
            future_ptrs.push(future_ptr);
        }

        // Create blocks for control flow
        let poll_loop_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Create body blocks for each arm
        let body_blocks: Vec<_> = (0..num_arms)
            .map(|_| self.builder.create_block())
            .collect();

        // Jump to poll loop
        self.builder.br(poll_loop_block);

        // Poll loop: check each future in order
        self.builder.start_block(poll_loop_block);

        let one = self.builder.const_int(1);

        // For each arm, check if its future is ready
        for (i, future_ptr) in future_ptrs.iter().enumerate() {
            // Get state field (field 0)
            let state_ptr = self.builder.get_field_ptr(*future_ptr, 0);
            let state = self.builder.load(state_ptr);

            // Check if state == Ready (1)
            let is_ready = self.builder.icmp(CmpOp::Eq, state, one);

            // If ready, go to body block
            let next_check_block = if i < num_arms - 1 {
                self.builder.create_block()
            } else {
                // Last arm: if not ready, try to poll and loop back
                self.builder.create_block()
            };

            self.builder.cond_br(is_ready, body_blocks[i], next_check_block);
            self.builder.start_block(next_check_block);
        }

        // None ready yet - try polling each future that's in NotStarted state
        for future_ptr in &future_ptrs {
            let state_ptr = self.builder.get_field_ptr(*future_ptr, 0);
            let state = self.builder.load(state_ptr);

            // If state == 0 (NotStarted), try to execute it
            let is_not_started = self.builder.icmp(CmpOp::Eq, state, zero);

            let try_poll_block = self.builder.create_block();
            let skip_block = self.builder.create_block();

            self.builder.cond_br(is_not_started, try_poll_block, skip_block);

            self.builder.start_block(try_poll_block);

            // Load fn_ptr and env_ptr
            let fn_ptr_field = self.builder.get_field_ptr(*future_ptr, 1);
            let fn_ptr_i64 = self.builder.load(fn_ptr_field);

            // Check if fn_ptr == -1 (sleep marker)
            let neg_one = self.builder.const_int(-1);
            let is_sleep = self.builder.icmp(CmpOp::Eq, fn_ptr_i64, neg_one);

            let call_closure_block = self.builder.create_block();
            let handle_sleep_block = self.builder.create_block();

            self.builder.cond_br(is_sleep, handle_sleep_block, call_closure_block);

            // Handle sleep - just set to pending
            self.builder.start_block(handle_sleep_block);
            let two = self.builder.const_int(2);
            self.builder.store(state_ptr, two);
            self.builder.br(skip_block);

            // Call the closure
            self.builder.start_block(call_closure_block);
            let env_ptr_field = self.builder.get_field_ptr(*future_ptr, 2);
            let env_ptr_i64 = self.builder.load(env_ptr_field);

            let call_result = self.builder.call_ptr(fn_ptr_i64, vec![env_ptr_i64]);

            // Check if result is pending marker
            let pending_marker = self.builder.const_int(i64::MIN);
            let is_pending = self.builder.icmp(CmpOp::Eq, call_result, pending_marker);

            let set_ready_block = self.builder.create_block();
            let set_pending_block = self.builder.create_block();

            self.builder.cond_br(is_pending, set_pending_block, set_ready_block);

            // Set to pending
            self.builder.start_block(set_pending_block);
            let two = self.builder.const_int(2);
            self.builder.store(state_ptr, two);
            self.builder.br(skip_block);

            // Set to ready with value
            self.builder.start_block(set_ready_block);
            self.builder.store(state_ptr, one);
            let value_ptr = self.builder.get_field_ptr(*future_ptr, 3);
            self.builder.store(value_ptr, call_result);
            self.builder.br(skip_block);

            self.builder.start_block(skip_block);
        }

        // Process timers and reactor, then loop back
        self.builder.call("__rt_timer_process", vec![]);
        self.builder.call("__rt_run_tasks", vec![]);
        self.builder.br(poll_loop_block);

        // Generate body blocks for each arm
        for (i, arm) in arms.iter().enumerate() {
            self.builder.start_block(body_blocks[i]);

            // Get the value from the future
            let value_ptr = self.builder.get_field_ptr(future_ptrs[i], 3);
            let value = self.builder.load(value_ptr);

            // Bind the result to the variable
            let slot = self.builder.alloca(IrType::I64);
            self.builder.store(slot, value);
            self.locals.insert(arm.binding.name.clone(), slot);
            self.local_types.insert(arm.binding.name.clone(), IrType::I64);

            // Execute the body
            let body_result = self.lower_expr(&arm.body);
            self.builder.store(result_slot, body_result);
            self.builder.br(done_block);
        }

        // Done block
        self.builder.start_block(done_block);
        self.builder.load(result_slot)
    }

    /// Lower join!(f1, f2, f3, ...)
    /// Poll all futures, wait for all to complete, return tuple of results
    fn lower_join(&mut self, futures: &[Expr]) -> VReg {
        if futures.is_empty() {
            // Return empty tuple (unit)
            return self.builder.const_int(0);
        }

        let num_futures = futures.len();

        // Create future pointers
        let mut future_ptrs = Vec::new();
        for future in futures {
            let future_ptr = self.lower_expr(future);
            future_ptrs.push(future_ptr);
        }

        // Allocate slots to track completion status (0 = not done, 1 = done)
        let mut done_slots = Vec::new();
        for _ in 0..num_futures {
            let slot = self.builder.alloca(IrType::I64);
            let zero = self.builder.const_int(0);
            self.builder.store(slot, zero);
            done_slots.push(slot);
        }

        // Create result tuple on heap
        let tuple_ty = IrType::Struct(vec![IrType::I64; num_futures]);
        let tuple_ptr = self.builder.malloc(tuple_ty);

        // Create blocks
        let poll_loop_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Jump to poll loop
        self.builder.br(poll_loop_block);

        // Poll loop
        self.builder.start_block(poll_loop_block);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        // Check and poll each future
        for (i, future_ptr) in future_ptrs.iter().enumerate() {
            // Check if already done
            let done_val = self.builder.load(done_slots[i]);
            let already_done = self.builder.icmp(CmpOp::Ne, done_val, zero);

            let check_future_block = self.builder.create_block();
            let next_future_block = self.builder.create_block();

            self.builder.cond_br(already_done, next_future_block, check_future_block);

            self.builder.start_block(check_future_block);

            // Get state
            let state_ptr = self.builder.get_field_ptr(*future_ptr, 0);
            let state = self.builder.load(state_ptr);

            // If state == Ready (1), mark done and store value
            let is_ready = self.builder.icmp(CmpOp::Eq, state, one);

            let store_result_block = self.builder.create_block();
            let try_poll_block = self.builder.create_block();

            self.builder.cond_br(is_ready, store_result_block, try_poll_block);

            // Store result
            self.builder.start_block(store_result_block);
            let value_ptr = self.builder.get_field_ptr(*future_ptr, 3);
            let value = self.builder.load(value_ptr);
            let tuple_field_ptr = self.builder.get_field_ptr(tuple_ptr, i as u32);
            self.builder.store(tuple_field_ptr, value);
            self.builder.store(done_slots[i], one);
            self.builder.br(next_future_block);

            // Try to poll if NotStarted
            self.builder.start_block(try_poll_block);
            let is_not_started = self.builder.icmp(CmpOp::Eq, state, zero);

            let exec_block = self.builder.create_block();
            self.builder.cond_br(is_not_started, exec_block, next_future_block);

            self.builder.start_block(exec_block);

            // Load fn_ptr and check for sleep marker
            let fn_ptr_field = self.builder.get_field_ptr(*future_ptr, 1);
            let fn_ptr_i64 = self.builder.load(fn_ptr_field);

            let neg_one = self.builder.const_int(-1);
            let is_sleep = self.builder.icmp(CmpOp::Eq, fn_ptr_i64, neg_one);

            let call_block = self.builder.create_block();
            let sleep_block = self.builder.create_block();

            self.builder.cond_br(is_sleep, sleep_block, call_block);

            // Sleep - set to pending
            self.builder.start_block(sleep_block);
            let two = self.builder.const_int(2);
            self.builder.store(state_ptr, two);
            self.builder.br(next_future_block);

            // Call closure
            self.builder.start_block(call_block);
            let env_ptr_field = self.builder.get_field_ptr(*future_ptr, 2);
            let env_ptr_i64 = self.builder.load(env_ptr_field);
            let call_result = self.builder.call_ptr(fn_ptr_i64, vec![env_ptr_i64]);

            let pending_marker = self.builder.const_int(i64::MIN);
            let is_pending = self.builder.icmp(CmpOp::Eq, call_result, pending_marker);

            let set_ready_block = self.builder.create_block();
            let set_pending_block = self.builder.create_block();

            self.builder.cond_br(is_pending, set_pending_block, set_ready_block);

            // Set pending
            self.builder.start_block(set_pending_block);
            let two = self.builder.const_int(2);
            self.builder.store(state_ptr, two);
            self.builder.br(next_future_block);

            // Set ready
            self.builder.start_block(set_ready_block);
            self.builder.store(state_ptr, one);
            let value_ptr = self.builder.get_field_ptr(*future_ptr, 3);
            self.builder.store(value_ptr, call_result);
            // Also store in tuple and mark done
            let tuple_field_ptr = self.builder.get_field_ptr(tuple_ptr, i as u32);
            self.builder.store(tuple_field_ptr, call_result);
            self.builder.store(done_slots[i], one);
            self.builder.br(next_future_block);

            self.builder.start_block(next_future_block);
        }

        // Check if all are done
        let mut all_done = one;
        for done_slot in &done_slots {
            let done_val = self.builder.load(*done_slot);
            all_done = self.builder.and(all_done, done_val);
        }

        let all_complete = self.builder.icmp(CmpOp::Ne, all_done, zero);

        let continue_block = self.builder.create_block();
        self.builder.cond_br(all_complete, done_block, continue_block);

        // Continue - process timers/tasks and loop
        self.builder.start_block(continue_block);
        self.builder.call("__rt_timer_process", vec![]);
        self.builder.call("__rt_run_tasks", vec![]);
        self.builder.br(poll_loop_block);

        // Done block - return tuple pointer
        self.builder.start_block(done_block);
        tuple_ptr
    }

    /// Lower yield_now() -> Future<()>
    /// Runs all pending tasks and returns a ready future
    /// This allows other spawned tasks to execute before continuing
    fn lower_yield_now(&mut self) -> VReg {
        // Run all pending tasks first (this is the "yield" behavior)
        self.builder.call("__rt_run_tasks", vec![]);

        // Create a ready future with 4 fields
        let future_ty = IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]);
        let future_ptr = self.builder.malloc(future_ty.clone());

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        // Set state to Ready (1)
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        self.builder.store(state_ptr, one);

        // fn_ptr = 0 (not used for ready futures)
        let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
        self.builder.store(fn_ptr_field, zero);

        // env_ptr = 0
        let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
        self.builder.store(env_ptr_field, zero);

        // Set value to 0 (unit)
        let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_ptr, zero);

        future_ptr
    }

    /// Lower sleep_ms(ms) -> Future<()>
    /// Creates an async future that sleeps for given milliseconds using the timer system.
    /// The future is created in Pending state with fn_ptr = -1 (sleep marker).
    /// When the timer expires, __rt_wake will set state to Ready directly.
    fn lower_sleep_ms(&mut self, args: &[Expr]) -> VReg {
        // Create future struct with 4 fields:
        // - state: 0=NotStarted, 1=Ready, 2=Pending
        // - fn_ptr: -1 = sleep marker (no closure to call)
        // - env_ptr: stores the sleep duration in ms
        // - value: result (unit for sleep)
        let future_ty = IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]);
        let future_ptr = self.builder.malloc(future_ty.clone());

        let zero = self.builder.const_int(0);
        let two = self.builder.const_int(2); // Pending state
        let neg_one = self.builder.const_int(-1); // Sleep marker

        // Get milliseconds argument (default to 0 if not provided)
        let ms = if !args.is_empty() {
            self.lower_expr(&args[0])
        } else {
            self.builder.const_int(0)
        };

        // Calculate deadline: now + ms
        let now = self.builder.call("__rt_time_now", vec![]);
        let deadline = self.builder.add(now, ms);

        // Set state to Pending (2) - waiting for timer
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        self.builder.store(state_ptr, two);

        // Store -1 in fn_ptr as sleep marker (indicates no closure to call)
        let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
        self.builder.store(fn_ptr_field, neg_one);

        // Store original ms in env_ptr for debugging/reference
        let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
        self.builder.store(env_ptr_field, ms);

        // Set value to 0 (unit)
        let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_ptr, zero);

        // Register timer with the deadline
        // The future pointer is used as the waker - when timer expires,
        // __rt_timer_process will call __rt_wake(future_ptr) to wake this future
        let future_ptr_i64 = self.builder.ptrtoint(future_ptr, IrType::I64);
        self.builder.call("__rt_timer_register", vec![deadline, future_ptr_i64]);

        future_ptr
    }

    /// Lower pending() -> Future<()>
    /// Creates a future that is in Pending state (state=2)
    /// Used for testing suspension and wake functionality
    /// Note: Does NOT increment __rt_pending_count since this future
    /// is not in the task queue. The count is managed by poll_future/wake.
    fn lower_pending(&mut self) -> VReg {
        // Create future struct with 4 fields
        let future_ty = IrType::Struct(vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64]);
        let future_ptr = self.builder.malloc(future_ty.clone());

        let zero = self.builder.const_int(0);
        let two = self.builder.const_int(2); // Pending state

        // Set state to Pending (2)
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        self.builder.store(state_ptr, two);

        // fn_ptr = 0 (no continuation - just a marker future)
        let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
        self.builder.store(fn_ptr_field, zero);

        // env_ptr = 0
        let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
        self.builder.store(env_ptr_field, zero);

        // value = 0
        let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_ptr, zero);

        future_ptr
    }

    /// Lower wake(future_ptr) -> ()
    /// Wakes a pending future by calling __rt_wake
    fn lower_wake(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        // Get the future pointer (passed as i64)
        let future_ptr = self.lower_expr(&args[0]);

        // Call __rt_wake to wake the future
        self.builder.call("__rt_wake", vec![future_ptr]);

        // Return unit (0)
        self.builder.const_int(0)
    }

    // ============ Time Module Lowering ============

    /// Lower time::now_ms() -> i64
    /// Returns current time in milliseconds since epoch
    fn lower_time_now_ms(&mut self) -> VReg {
        self.builder.call("__rt_time_now", vec![])
    }

    /// Lower time::now_us() -> i64
    /// Returns current time in microseconds since epoch
    fn lower_time_now_us(&mut self) -> VReg {
        self.builder.call("__rt_time_now_us", vec![])
    }

    /// Lower time::now_ns() -> i64
    /// Returns current time in nanoseconds since epoch
    fn lower_time_now_ns(&mut self) -> VReg {
        self.builder.call("__rt_time_now_ns", vec![])
    }

    /// Lower time::elapsed_ms(start: i64) -> i64
    /// Returns milliseconds elapsed since start
    fn lower_time_elapsed_ms(&mut self, args: &[Expr]) -> VReg {
        let start = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };
        let now = self.builder.call("__rt_time_now", vec![]);
        self.builder.sub(now, start)
    }

    /// Lower time::elapsed_us(start: i64) -> i64
    /// Returns microseconds elapsed since start
    fn lower_time_elapsed_us(&mut self, args: &[Expr]) -> VReg {
        let start = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };
        let now = self.builder.call("__rt_time_now_us", vec![]);
        self.builder.sub(now, start)
    }

    /// Lower time::elapsed_ns(start: i64) -> i64
    /// Returns nanoseconds elapsed since start
    fn lower_time_elapsed_ns(&mut self, args: &[Expr]) -> VReg {
        let start = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };
        let now = self.builder.call("__rt_time_now_ns", vec![]);
        self.builder.sub(now, start)
    }

    // ============ Duration Type Lowering ============

    /// Lower Duration::from_secs(secs: i64) -> Duration
    /// Creates a Duration from seconds
    fn lower_duration_from_secs(&mut self, args: &[Expr]) -> VReg {
        let secs = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };
        let zero = self.builder.const_int(0);

        // Allocate Duration struct: { secs: i64, nanos: i64 }
        let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let duration_ptr = self.builder.malloc(duration_ty);

        // Store secs field
        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        self.builder.store(secs_ptr, secs);

        // Store nanos field (0)
        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        self.builder.store(nanos_ptr, zero);

        duration_ptr
    }

    /// Lower Duration::from_millis(millis: i64) -> Duration
    /// Creates a Duration from milliseconds
    fn lower_duration_from_millis(&mut self, args: &[Expr]) -> VReg {
        let millis = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };

        // secs = millis / 1000
        let thousand = self.builder.const_int(1000);
        let secs = self.builder.sdiv(millis, thousand);

        // nanos = (millis % 1000) * 1_000_000
        let rem = self.builder.srem(millis, thousand);
        let million = self.builder.const_int(1_000_000);
        let nanos = self.builder.mul(rem, million);

        // Allocate Duration struct
        let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let duration_ptr = self.builder.malloc(duration_ty);

        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        self.builder.store(secs_ptr, secs);

        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        self.builder.store(nanos_ptr, nanos);

        duration_ptr
    }

    /// Lower Duration::from_micros(micros: i64) -> Duration
    /// Creates a Duration from microseconds
    fn lower_duration_from_micros(&mut self, args: &[Expr]) -> VReg {
        let micros = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };

        // secs = micros / 1_000_000
        let million = self.builder.const_int(1_000_000);
        let secs = self.builder.sdiv(micros, million);

        // nanos = (micros % 1_000_000) * 1000
        let rem = self.builder.srem(micros, million);
        let thousand = self.builder.const_int(1000);
        let nanos = self.builder.mul(rem, thousand);

        // Allocate Duration struct
        let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let duration_ptr = self.builder.malloc(duration_ty);

        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        self.builder.store(secs_ptr, secs);

        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        self.builder.store(nanos_ptr, nanos);

        duration_ptr
    }

    /// Lower Duration::from_nanos(nanos: i64) -> Duration
    /// Creates a Duration from nanoseconds
    fn lower_duration_from_nanos(&mut self, args: &[Expr]) -> VReg {
        let total_nanos = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };

        // secs = nanos / 1_000_000_000
        let billion = self.builder.const_int(1_000_000_000);
        let secs = self.builder.sdiv(total_nanos, billion);

        // nanos = nanos % 1_000_000_000
        let nanos = self.builder.srem(total_nanos, billion);

        // Allocate Duration struct
        let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let duration_ptr = self.builder.malloc(duration_ty);

        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        self.builder.store(secs_ptr, secs);

        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        self.builder.store(nanos_ptr, nanos);

        duration_ptr
    }

    /// Lower Duration::as_secs(d: Duration) -> i64
    /// Returns the total seconds in the duration
    fn lower_duration_as_secs(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let duration_ptr = self.lower_expr(&args[0]);
        let duration_ptr = self.builder.inttoptr(duration_ptr, IrType::ptr(IrType::Struct(vec![IrType::I64, IrType::I64])));
        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        self.builder.load(secs_ptr)
    }

    /// Lower Duration::as_millis(d: Duration) -> i64
    /// Returns the total milliseconds in the duration
    fn lower_duration_as_millis(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let duration_ptr = self.lower_expr(&args[0]);
        let duration_ptr = self.builder.inttoptr(duration_ptr, IrType::ptr(IrType::Struct(vec![IrType::I64, IrType::I64])));

        // Load secs and nanos
        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        let secs = self.builder.load(secs_ptr);
        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        let nanos = self.builder.load(nanos_ptr);

        // millis = secs * 1000 + nanos / 1_000_000
        let thousand = self.builder.const_int(1000);
        let million = self.builder.const_int(1_000_000);
        let secs_ms = self.builder.mul(secs, thousand);
        let nanos_ms = self.builder.sdiv(nanos, million);
        self.builder.add(secs_ms, nanos_ms)
    }

    /// Lower Duration::as_micros(d: Duration) -> i64
    /// Returns the total microseconds in the duration
    fn lower_duration_as_micros(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let duration_ptr = self.lower_expr(&args[0]);
        let duration_ptr = self.builder.inttoptr(duration_ptr, IrType::ptr(IrType::Struct(vec![IrType::I64, IrType::I64])));

        // Load secs and nanos
        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        let secs = self.builder.load(secs_ptr);
        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        let nanos = self.builder.load(nanos_ptr);

        // micros = secs * 1_000_000 + nanos / 1000
        let million = self.builder.const_int(1_000_000);
        let thousand = self.builder.const_int(1000);
        let secs_us = self.builder.mul(secs, million);
        let nanos_us = self.builder.sdiv(nanos, thousand);
        self.builder.add(secs_us, nanos_us)
    }

    /// Lower Duration::as_nanos(d: Duration) -> i64
    /// Returns the total nanoseconds in the duration
    fn lower_duration_as_nanos(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let duration_ptr = self.lower_expr(&args[0]);
        let duration_ptr = self.builder.inttoptr(duration_ptr, IrType::ptr(IrType::Struct(vec![IrType::I64, IrType::I64])));

        // Load secs and nanos
        let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
        let secs = self.builder.load(secs_ptr);
        let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
        let nanos = self.builder.load(nanos_ptr);

        // total_nanos = secs * 1_000_000_000 + nanos
        let billion = self.builder.const_int(1_000_000_000);
        let secs_ns = self.builder.mul(secs, billion);
        self.builder.add(secs_ns, nanos)
    }

    /// Lower Duration::add(a: Duration, b: Duration) -> Duration
    /// Adds two durations together
    fn lower_duration_add(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            // Return zero duration
            let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
            let duration_ptr = self.builder.malloc(duration_ty);
            let zero = self.builder.const_int(0);
            let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
            self.builder.store(secs_ptr, zero);
            let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
            self.builder.store(nanos_ptr, zero);
            return duration_ptr;
        }

        let a_ptr = self.lower_expr(&args[0]);
        let b_ptr = self.lower_expr(&args[1]);

        let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let a_ptr = self.builder.inttoptr(a_ptr, IrType::ptr(duration_ty.clone()));
        let b_ptr = self.builder.inttoptr(b_ptr, IrType::ptr(duration_ty.clone()));

        // Load a.secs, a.nanos
        let a_secs_ptr = self.builder.get_field_ptr(a_ptr, 0);
        let a_secs = self.builder.load(a_secs_ptr);
        let a_nanos_ptr = self.builder.get_field_ptr(a_ptr, 1);
        let a_nanos = self.builder.load(a_nanos_ptr);

        // Load b.secs, b.nanos
        let b_secs_ptr = self.builder.get_field_ptr(b_ptr, 0);
        let b_secs = self.builder.load(b_secs_ptr);
        let b_nanos_ptr = self.builder.get_field_ptr(b_ptr, 1);
        let b_nanos = self.builder.load(b_nanos_ptr);

        // total_nanos = a.nanos + b.nanos
        let total_nanos = self.builder.add(a_nanos, b_nanos);

        // extra_secs = total_nanos / 1_000_000_000
        // result_nanos = total_nanos % 1_000_000_000
        let billion = self.builder.const_int(1_000_000_000);
        let extra_secs = self.builder.sdiv(total_nanos, billion);
        let result_nanos = self.builder.srem(total_nanos, billion);

        // result_secs = a.secs + b.secs + extra_secs
        let secs_sum = self.builder.add(a_secs, b_secs);
        let result_secs = self.builder.add(secs_sum, extra_secs);

        // Allocate result
        let result_ptr = self.builder.malloc(duration_ty);
        let result_secs_ptr = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(result_secs_ptr, result_secs);
        let result_nanos_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(result_nanos_ptr, result_nanos);

        result_ptr
    }

    /// Lower Duration::sub(a: Duration, b: Duration) -> Duration
    /// Subtracts b from a (a - b)
    fn lower_duration_sub(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            // Return zero duration
            let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
            let duration_ptr = self.builder.malloc(duration_ty);
            let zero = self.builder.const_int(0);
            let secs_ptr = self.builder.get_field_ptr(duration_ptr, 0);
            self.builder.store(secs_ptr, zero);
            let nanos_ptr = self.builder.get_field_ptr(duration_ptr, 1);
            self.builder.store(nanos_ptr, zero);
            return duration_ptr;
        }

        let a_ptr = self.lower_expr(&args[0]);
        let b_ptr = self.lower_expr(&args[1]);

        let duration_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let a_ptr = self.builder.inttoptr(a_ptr, IrType::ptr(duration_ty.clone()));
        let b_ptr = self.builder.inttoptr(b_ptr, IrType::ptr(duration_ty.clone()));

        // Load a.secs, a.nanos
        let a_secs_ptr = self.builder.get_field_ptr(a_ptr, 0);
        let a_secs = self.builder.load(a_secs_ptr);
        let a_nanos_ptr = self.builder.get_field_ptr(a_ptr, 1);
        let a_nanos = self.builder.load(a_nanos_ptr);

        // Load b.secs, b.nanos
        let b_secs_ptr = self.builder.get_field_ptr(b_ptr, 0);
        let b_secs = self.builder.load(b_secs_ptr);
        let b_nanos_ptr = self.builder.get_field_ptr(b_ptr, 1);
        let b_nanos = self.builder.load(b_nanos_ptr);

        // nanos_diff = a.nanos - b.nanos
        let nanos_diff = self.builder.sub(a_nanos, b_nanos);

        // secs_diff = a.secs - b.secs
        let secs_diff = self.builder.sub(a_secs, b_secs);

        // If nanos_diff < 0, borrow from seconds
        let zero = self.builder.const_int(0);
        let billion = self.builder.const_int(1_000_000_000);
        let one = self.builder.const_int(1);

        let need_borrow = self.builder.icmp(CmpOp::Slt, nanos_diff, zero);

        // If need_borrow: result_nanos = nanos_diff + 1_000_000_000, result_secs = secs_diff - 1
        // Else: result_nanos = nanos_diff, result_secs = secs_diff
        let borrowed_nanos = self.builder.add(nanos_diff, billion);
        let borrowed_secs = self.builder.sub(secs_diff, one);

        let result_nanos = self.builder.select(need_borrow, borrowed_nanos, nanos_diff);
        let result_secs = self.builder.select(need_borrow, borrowed_secs, secs_diff);

        // Allocate result
        let result_ptr = self.builder.malloc(duration_ty);
        let result_secs_ptr = self.builder.get_field_ptr(result_ptr, 0);
        self.builder.store(result_secs_ptr, result_secs);
        let result_nanos_ptr = self.builder.get_field_ptr(result_ptr, 1);
        self.builder.store(result_nanos_ptr, result_nanos);

        result_ptr
    }

    // ============ Random Module Lowering ============

    /// Lower random::seed(seed: i64)
    /// Sets the seed for the random number generator
    fn lower_random_seed(&mut self, args: &[Expr]) -> VReg {
        let seed = if args.is_empty() {
            self.builder.const_int(12345) // Default seed
        } else {
            self.lower_expr(&args[0])
        };

        // Store seed in global state
        self.builder.call("__rt_random_seed", vec![seed]);
        self.builder.const_int(0) // Return unit
    }

    /// Lower random::next_i64() -> i64
    /// Returns the next random i64
    fn lower_random_next_i64(&mut self) -> VReg {
        self.builder.call("__rt_random_next", vec![])
    }

    /// Lower random::next_f64() -> f64
    /// Returns a random f64 in [0.0, 1.0)
    fn lower_random_next_f64(&mut self) -> VReg {
        self.builder.call("__rt_random_next_f64", vec![])
    }

    /// Lower random::range(min: i64, max: i64) -> i64
    /// Returns a random i64 in [min, max)
    fn lower_random_range(&mut self, args: &[Expr]) -> VReg {
        let min = if args.is_empty() {
            self.builder.const_int(0)
        } else {
            self.lower_expr(&args[0])
        };

        let max = if args.len() < 2 {
            self.builder.const_int(100)
        } else {
            self.lower_expr(&args[1])
        };

        // Get random number
        let rand = self.builder.call("__rt_random_next", vec![]);

        // Make it positive
        let mask = self.builder.const_int(0x7FFFFFFFFFFFFFFF_u64 as i64);
        let positive = self.builder.and(rand, mask);

        // Calculate range and result: min + (positive % (max - min))
        let range = self.builder.sub(max, min);
        let modulo = self.builder.srem(positive, range);
        self.builder.add(min, modulo)
    }

    /// Lower random::coin() -> bool
    /// Returns a random boolean (coin flip)
    fn lower_random_coin(&mut self) -> VReg {
        let rand = self.builder.call("__rt_random_next", vec![]);
        let one = self.builder.const_int(1);
        let bit = self.builder.and(rand, one);
        // Compare bit == 1 to get a proper bool
        self.builder.icmp(CmpOp::Eq, bit, one)
    }

    // ============ Channel Lowering (Phase 6) ============

    /// Lower channel(capacity) -> Channel<T>
    /// Creates a new channel and returns the channel pointer
    fn lower_channel_create(&mut self, args: &[Expr]) -> VReg {
        // Get capacity argument (default to 16 if not provided)
        let capacity = if args.is_empty() {
            self.builder.const_int(16)
        } else {
            self.lower_expr(&args[0])
        };

        // Create and return the channel pointer
        self.builder.call("__rt_channel_create", vec![capacity])
    }

    /// Lower Channel::sender(channel) -> Sender<T>
    /// Get the sender handle from a channel
    fn lower_channel_get_sender(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }
        // Sender is just the channel pointer
        self.lower_expr(&args[0])
    }

    /// Lower Channel::receiver(channel) -> Receiver<T>
    /// Get the receiver handle from a channel
    fn lower_channel_get_receiver(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }
        // Receiver is just the channel pointer
        self.lower_expr(&args[0])
    }

    /// Lower Sender::send(sender, value) -> Future<()>
    /// Returns a Future that completes when the send succeeds
    fn lower_channel_send(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        // Get sender (which is just the channel pointer)
        let sender_ptr = self.lower_expr(&args[0]);
        // Get value to send
        let value = self.lower_expr(&args[1]);

        // Allocate a Future struct to track this send operation
        // Future: {state: i64, fn_ptr: i64, env_ptr: i64, value: i64}
        let future_size = self.builder.const_int(32);
        let future_ptr_raw = self.builder.call("malloc", vec![future_size]);
        let future_ptr_i64 = self.builder.ptrtoint(future_ptr_raw, IrType::I64);
        let future_ptr = self.builder.inttoptr(future_ptr_i64, IrType::ptr(IrType::I64));

        // Initialize future state to NotStarted (0)
        let zero = self.builder.const_int(0);
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        self.builder.store(state_ptr, zero);

        // Store channel_ptr in env_ptr field (we'll use this on retry)
        let env_field = self.builder.get_field_ptr(future_ptr, 2);
        self.builder.store(env_field, sender_ptr);

        // Store value in value field (we'll use this on retry)
        let value_field = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_field, value);

        // Try to send now
        let send_result = self.builder.call("__rt_channel_send", vec![sender_ptr, value, future_ptr_i64]);

        // If send succeeded immediately (result == 1), set state to Ready
        let one = self.builder.const_int(1);
        let sent_immediately = self.builder.icmp(CmpOp::Eq, send_result, one);

        let ready_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(sent_immediately, ready_block, done_block);

        // Set state to Ready (1)
        self.builder.start_block(ready_block);
        self.builder.store(state_ptr, one);
        self.builder.br(done_block);

        self.builder.start_block(done_block);

        // Return the future pointer
        future_ptr_i64
    }

    /// Lower Receiver::recv(receiver) -> Future<Option<T>>
    /// Returns a Future that completes when a value is available
    fn lower_channel_recv(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        // Get receiver (which is just the channel pointer)
        let receiver_ptr = self.lower_expr(&args[0]);

        // Allocate a Future struct to track this recv operation
        let future_size = self.builder.const_int(32);
        let future_ptr_raw = self.builder.call("malloc", vec![future_size]);
        let future_ptr_i64 = self.builder.ptrtoint(future_ptr_raw, IrType::I64);
        let future_ptr = self.builder.inttoptr(future_ptr_i64, IrType::ptr(IrType::I64));

        // Initialize future state to NotStarted (0)
        let zero = self.builder.const_int(0);
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        self.builder.store(state_ptr, zero);

        // Store channel_ptr in env_ptr field
        let env_field = self.builder.get_field_ptr(future_ptr, 2);
        self.builder.store(env_field, receiver_ptr);

        // Try to receive now
        let recv_result = self.builder.call("__rt_channel_recv", vec![receiver_ptr, future_ptr_i64]);

        // Check if we got a value (not i64::MIN)
        let none_marker = self.builder.const_int(i64::MIN);
        let got_value = self.builder.icmp(CmpOp::Ne, recv_result, none_marker);

        let ready_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(got_value, ready_block, done_block);

        // Got value - set state to Ready and store value
        self.builder.start_block(ready_block);
        let one = self.builder.const_int(1);
        self.builder.store(state_ptr, one);

        // Store the received value
        let value_field = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_field, recv_result);

        self.builder.br(done_block);

        self.builder.start_block(done_block);

        // Return the future pointer
        future_ptr_i64
    }

    /// Lower Sender::try_send(sender, value) -> bool
    /// Non-blocking send, returns false if buffer is full
    fn lower_channel_try_send(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let sender_ptr = self.lower_expr(&args[0]);
        let value = self.lower_expr(&args[1]);

        // Call __rt_channel_try_send
        // Returns: 1 = success, 0 = buffer full, -1 = closed
        self.builder.call("__rt_channel_try_send", vec![sender_ptr, value])
    }

    /// Lower Receiver::try_recv(receiver) -> Option<T>
    /// Non-blocking receive, returns None if buffer is empty
    fn lower_channel_try_recv(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let receiver_ptr = self.lower_expr(&args[0]);

        // Call __rt_channel_try_recv
        // Returns value if Some, i64::MIN if None
        let result = self.builder.call("__rt_channel_try_recv", vec![receiver_ptr]);

        // Allocate Option struct using proper typed malloc (like create_option_some)
        let option_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        let opt_ptr = self.builder.malloc(option_ty);

        let none_marker = self.builder.const_int(i64::MIN);
        let is_none = self.builder.icmp(CmpOp::Eq, result, none_marker);

        let none_block = self.builder.create_block();
        let some_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_none, none_block, some_block);

        // None case: tag = 0
        self.builder.start_block(none_block);
        let discrim_ptr_none = self.builder.get_field_ptr(opt_ptr, 0);
        let zero_tag = self.builder.const_int(0);
        let zero_tag_i32 = self.builder.trunc(zero_tag, IrType::I32);
        self.builder.store(discrim_ptr_none, zero_tag_i32);
        self.builder.br(done_block);

        // Some case: tag = 1, value = result
        self.builder.start_block(some_block);
        let discrim_ptr_some = self.builder.get_field_ptr(opt_ptr, 0);
        let one_tag = self.builder.const_int(1);
        let one_tag_i32 = self.builder.trunc(one_tag, IrType::I32);
        self.builder.store(discrim_ptr_some, one_tag_i32);

        // Store payload value
        let payload_ptr = self.builder.get_field_ptr(opt_ptr, 1);
        self.builder.store(payload_ptr, result);

        self.builder.br(done_block);

        self.builder.start_block(done_block);

        opt_ptr
    }

    /// Lower Sender::is_closed(sender) -> bool
    fn lower_sender_is_closed(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let sender_ptr = self.lower_expr(&args[0]);
        let one = self.builder.const_int(1); // is_sender = true

        // Call __rt_channel_is_closed
        self.builder.call("__rt_channel_is_closed", vec![sender_ptr, one])
    }

    /// Lower Receiver::is_closed(receiver) -> bool
    fn lower_receiver_is_closed(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let receiver_ptr = self.lower_expr(&args[0]);
        let zero = self.builder.const_int(0); // is_sender = false

        // Call __rt_channel_is_closed
        self.builder.call("__rt_channel_is_closed", vec![receiver_ptr, zero])
    }

    // ============ TCP I/O Functions (Phase 4.1) ============

    /// Lower TcpListener::bind(addr: String, port: i64) -> Result<TcpListener, i64>
    /// Creates a TCP listening socket bound to addr:port
    fn lower_tcp_listener_bind(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let _addr = self.lower_expr(&args[0]); // String addr (ignored for now, binds to 0.0.0.0)
        let port = self.lower_expr(&args[1]);

        // socket(AF_INET=2, SOCK_STREAM=1, 0)
        let af_inet = self.builder.const_i32(2);
        let sock_stream = self.builder.const_i32(1);
        let zero_i32 = self.builder.const_i32(0);
        let sock_fd = self.builder.call("socket", vec![af_inet, sock_stream, zero_i32]);

        // Check if socket failed (< 0)
        let sock_failed = self.builder.icmp(CmpOp::Slt, sock_fd, zero_i32);

        let socket_error_block = self.builder.create_block();
        let socket_ok_block = self.builder.create_block();
        let bind_error_block = self.builder.create_block();
        let listen_error_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        self.builder.cond_br(sock_failed, socket_error_block, socket_ok_block);

        // Socket error: return Err(1)
        self.builder.start_block(socket_error_block);
        let err1 = self.builder.const_int(1);
        let result_err1 = self.create_result_err(err1);
        let socket_error_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Socket OK: set SO_REUSEADDR and bind
        self.builder.start_block(socket_ok_block);

        // setsockopt(sock_fd, SOL_SOCKET=1, SO_REUSEADDR=2, &one, sizeof(int))
        let sol_socket = self.builder.const_i32(1);
        let so_reuseaddr = self.builder.const_i32(2);
        let one_slot = self.builder.alloca(IrType::I32);
        let one_i32 = self.builder.const_i32(1);
        self.builder.store(one_slot, one_i32);
        let sizeof_int = self.builder.const_i32(4);
        self.builder.call("setsockopt", vec![sock_fd, sol_socket, so_reuseaddr, one_slot, sizeof_int]);

        // Build sockaddr_in structure (16 bytes)
        // struct sockaddr_in { sin_family (2), sin_port (2), sin_addr (4), sin_zero (8) }
        let sockaddr_size = self.builder.const_int(16);
        let sockaddr_raw = self.builder.call("malloc", vec![sockaddr_size]);
        let sockaddr = self.builder.ptrtoint(sockaddr_raw, IrType::I64);

        // sin_family = AF_INET (2) at offset 0
        let sin_family_ptr = self.builder.inttoptr(sockaddr, IrType::Ptr(Box::new(IrType::I16)));
        let af_inet_i16 = self.builder.const_i32(2); // Will be truncated by store
        self.builder.store(sin_family_ptr, af_inet_i16);

        // sin_port = htons(port) at offset 2
        let two = self.builder.const_int(2);
        let sin_port_addr = self.builder.add(sockaddr, two);
        let sin_port_ptr = self.builder.inttoptr(sin_port_addr, IrType::Ptr(Box::new(IrType::I16)));
        let port_i32 = self.builder.trunc(port, IrType::I32);
        let port_htons = self.builder.call("htons", vec![port_i32]);
        self.builder.store(sin_port_ptr, port_htons);

        // sin_addr = INADDR_ANY (0) at offset 4
        let four = self.builder.const_int(4);
        let sin_addr_addr = self.builder.add(sockaddr, four);
        let sin_addr_ptr = self.builder.inttoptr(sin_addr_addr, IrType::Ptr(Box::new(IrType::I32)));
        let inaddr_any = self.builder.const_i32(0);
        self.builder.store(sin_addr_ptr, inaddr_any);

        // Zero out sin_zero (8 bytes at offset 8) - just leave it, bind doesn't care

        // bind(sock_fd, sockaddr, 16)
        let sixteen_i32 = self.builder.const_i32(16);
        let bind_result = self.builder.call("bind", vec![sock_fd, sockaddr_raw, sixteen_i32]);
        let bind_failed = self.builder.icmp(CmpOp::Slt, bind_result, zero_i32);
        self.builder.cond_br(bind_failed, bind_error_block, success_block);

        // Bind error: close socket and return Err(2)
        self.builder.start_block(bind_error_block);
        self.builder.call("close", vec![sock_fd]);
        self.builder.call("free", vec![sockaddr_raw]);
        let err2 = self.builder.const_int(2);
        let result_err2 = self.create_result_err(err2);
        let bind_error_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Success: listen and create TcpListener
        self.builder.start_block(success_block);
        self.builder.call("free", vec![sockaddr_raw]);

        // listen(sock_fd, 128)
        let backlog = self.builder.const_i32(128);
        let listen_result = self.builder.call("listen", vec![sock_fd, backlog]);
        let listen_failed = self.builder.icmp(CmpOp::Slt, listen_result, zero_i32);

        let listen_ok_block = self.builder.create_block();
        self.builder.cond_br(listen_failed, listen_error_block, listen_ok_block);

        // Listen error: close socket and return Err(3)
        self.builder.start_block(listen_error_block);
        self.builder.call("close", vec![sock_fd]);
        let err3 = self.builder.const_int(3);
        let result_err3 = self.create_result_err(err3);
        let listen_error_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Listen OK: Create TcpListener struct { fd: i64 }
        self.builder.start_block(listen_ok_block);
        let listener_size = self.builder.const_int(8);
        let listener_ptr_raw = self.builder.call("malloc", vec![listener_size]);
        let listener_ptr = self.builder.ptrtoint(listener_ptr_raw, IrType::I64);
        let listener_ptr_typed = self.builder.inttoptr(listener_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let sock_fd_i64 = self.builder.sext(sock_fd, IrType::I64);
        self.builder.store(listener_ptr_typed, sock_fd_i64);
        let result_ok = self.create_result_ok(listener_ptr);
        let success_exit = self.builder.current_block_id().unwrap();
        self.builder.br(merge_block);

        // Merge block: PHI at top to select result
        self.builder.start_block(merge_block);
        self.builder.phi(vec![
            (result_err1, socket_error_exit),
            (result_err2, bind_error_exit),
            (result_err3, listen_error_exit),
            (result_ok, success_exit),
        ])
    }

    /// Lower TcpListener::accept(listener) -> Future<Result<TcpStream, i64>>
    /// Async accept - creates a Future that waits for incoming connection
    fn lower_tcp_listener_accept(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let listener = self.lower_expr(&args[0]);

        // Get the socket fd from the TcpListener struct
        let listener_ptr = self.builder.inttoptr(listener, IrType::Ptr(Box::new(IrType::I64)));
        let sock_fd_i64 = self.builder.load(listener_ptr);
        let sock_fd = self.builder.trunc(sock_fd_i64, IrType::I32);

        // Create a Future struct for async accept
        // Future layout: { state: i64, fn_ptr: i64, env_ptr: i64, value: i64 }
        let future_size = self.builder.const_int(32);
        let future_ptr_raw = self.builder.call("malloc", vec![future_size]);
        let future_ptr = self.builder.ptrtoint(future_ptr_raw, IrType::I64);

        // state = 0 (NotStarted), we'll use state to track accept progress
        let state_ptr = self.builder.inttoptr(future_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let zero = self.builder.const_int(0);
        self.builder.store(state_ptr, zero);

        // fn_ptr = special marker for accept (we'll use sock_fd)
        let eight = self.builder.const_int(8);
        let fn_ptr_addr = self.builder.add(future_ptr, eight);
        let fn_ptr_ptr = self.builder.inttoptr(fn_ptr_addr, IrType::Ptr(Box::new(IrType::I64)));
        self.builder.store(fn_ptr_ptr, sock_fd_i64);

        // env_ptr = 0 (unused)
        let sixteen = self.builder.const_int(16);
        let env_ptr_addr = self.builder.add(future_ptr, sixteen);
        let env_ptr_ptr = self.builder.inttoptr(env_ptr_addr, IrType::Ptr(Box::new(IrType::I64)));
        self.builder.store(env_ptr_ptr, zero);

        // value = 0 (will hold Result when ready)
        let twenty_four = self.builder.const_int(24);
        let value_addr = self.builder.add(future_ptr, twenty_four);
        let value_ptr = self.builder.inttoptr(value_addr, IrType::Ptr(Box::new(IrType::I64)));
        self.builder.store(value_ptr, zero);

        // Set socket to non-blocking mode using fcntl
        // fcntl(fd, F_SETFL, O_NONBLOCK)
        let f_getfl = self.builder.const_i32(3); // F_GETFL
        let flags = self.builder.call("fcntl", vec![sock_fd, f_getfl, zero]);
        let o_nonblock = self.builder.const_i32(2048); // O_NONBLOCK
        let new_flags = self.builder.or(flags, o_nonblock);
        let f_setfl = self.builder.const_i32(4); // F_SETFL
        self.builder.call("fcntl", vec![sock_fd, f_setfl, new_flags]);

        // Try accept immediately (non-blocking)
        let null_ptr = self.builder.const_null();
        let client_fd = self.builder.call("accept", vec![sock_fd, null_ptr, null_ptr]);

        // Check if accept succeeded or would block
        let zero_i32 = self.builder.const_i32(0);
        let accept_success = self.builder.icmp(CmpOp::Sge, client_fd, zero_i32);

        let ready_block = self.builder.create_block();
        let pending_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(accept_success, ready_block, pending_block);

        // Ready: accept succeeded immediately
        self.builder.start_block(ready_block);
        // Create TcpStream struct
        let stream_size = self.builder.const_int(8);
        let stream_ptr_raw = self.builder.call("malloc", vec![stream_size]);
        let stream_ptr_int = self.builder.ptrtoint(stream_ptr_raw, IrType::I64);
        let stream_ptr_typed = self.builder.inttoptr(stream_ptr_int, IrType::Ptr(Box::new(IrType::I64)));
        let client_fd_i64 = self.builder.sext(client_fd, IrType::I64);
        self.builder.store(stream_ptr_typed, client_fd_i64);

        // Create Result::Ok(stream_ptr)
        let result_ok = self.create_result_ok(stream_ptr_int);

        // Store in future value and set state to Ready(1)
        let value_ptr_typed = self.builder.inttoptr(value_addr, IrType::Ptr(Box::new(IrType::I64)));
        self.builder.store(value_ptr_typed, result_ok);
        let ready_state = self.builder.const_int(1);
        self.builder.store(state_ptr, ready_state);
        self.builder.br(done_block);

        // Pending: register with reactor and return pending future
        self.builder.start_block(pending_block);
        // Register fd for EPOLLIN (readable) events
        let epollin = self.builder.const_i32(1);
        let epollin_i64 = self.builder.sext(epollin, IrType::I64);
        self.builder.call("__rt_reactor_register", vec![sock_fd_i64, epollin_i64, future_ptr]);

        // Set state to Pending(2)
        let pending_state = self.builder.const_int(2);
        self.builder.store(state_ptr, pending_state);
        self.builder.br(done_block);

        // Done: return future pointer as i64
        self.builder.start_block(done_block);
        future_ptr  // This is already i64 from ptrtoint
    }

    /// Lower TcpListener::close(listener) -> ()
    fn lower_tcp_listener_close(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let listener = self.lower_expr(&args[0]);

        // Get fd from TcpListener struct
        let listener_ptr = self.builder.inttoptr(listener, IrType::Ptr(Box::new(IrType::I64)));
        let sock_fd_i64 = self.builder.load(listener_ptr);
        let sock_fd = self.builder.trunc(sock_fd_i64, IrType::I32);

        // close(fd)
        self.builder.call("close", vec![sock_fd]);

        // free the listener struct (convert i64 to ptr for free)
        self.builder.call("free", vec![listener_ptr]);

        self.builder.const_int(0)
    }

    /// Lower TcpStream::read_string(stream, max_len) -> Future<Result<String, i64>>
    /// Async read - creates a Future that waits for data
    fn lower_tcp_stream_read_string(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let stream = self.lower_expr(&args[0]);
        let max_len = self.lower_expr(&args[1]);

        // Get fd from TcpStream struct
        let stream_ptr = self.builder.inttoptr(stream, IrType::Ptr(Box::new(IrType::I64)));
        let sock_fd_i64 = self.builder.load(stream_ptr);
        let sock_fd = self.builder.trunc(sock_fd_i64, IrType::I32);

        // Set socket to non-blocking
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.const_i32(0);
        let f_getfl = self.builder.const_i32(3);
        let flags = self.builder.call("fcntl", vec![sock_fd, f_getfl, zero_i32]);
        let o_nonblock = self.builder.const_i32(2048);
        let new_flags = self.builder.or(flags, o_nonblock);
        let f_setfl = self.builder.const_i32(4);
        self.builder.call("fcntl", vec![sock_fd, f_setfl, new_flags]);

        // Allocate buffer for read
        let one = self.builder.const_int(1);
        let buf_size = self.builder.add(max_len, one); // +1 for null terminator
        let buffer_raw = self.builder.call("malloc", vec![buf_size]);
        let buffer = self.builder.ptrtoint(buffer_raw, IrType::I64);
        let buffer_ptr = self.builder.inttoptr(buffer, IrType::Ptr(Box::new(IrType::I8)));

        // Try non-blocking read
        let bytes_read = self.builder.call("read", vec![sock_fd, buffer_ptr, max_len]);

        // Check result
        let bytes_read_i64 = self.builder.sext(bytes_read, IrType::I64);
        let read_success = self.builder.icmp(CmpOp::Sgt, bytes_read, zero_i32);

        // Create Future struct
        let future_size = self.builder.const_int(32);
        let future_ptr_raw = self.builder.call("malloc", vec![future_size]);
        let future_ptr = self.builder.ptrtoint(future_ptr_raw, IrType::I64);

        let state_ptr = self.builder.inttoptr(future_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let eight = self.builder.const_int(8);
        let fn_ptr_addr = self.builder.add(future_ptr, eight);
        let fn_ptr_ptr = self.builder.inttoptr(fn_ptr_addr, IrType::Ptr(Box::new(IrType::I64)));
        let sixteen = self.builder.const_int(16);
        let env_ptr_addr = self.builder.add(future_ptr, sixteen);
        let env_ptr_ptr = self.builder.inttoptr(env_ptr_addr, IrType::Ptr(Box::new(IrType::I64)));
        let twenty_four = self.builder.const_int(24);
        let value_addr = self.builder.add(future_ptr, twenty_four);
        let value_ptr = self.builder.inttoptr(value_addr, IrType::Ptr(Box::new(IrType::I64)));

        // Store buffer pointer in fn_ptr for later use
        self.builder.store(fn_ptr_ptr, buffer);
        // Store max_len in env_ptr
        self.builder.store(env_ptr_ptr, max_len);

        let ready_block = self.builder.create_block();
        let pending_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(read_success, ready_block, pending_block);

        // Ready: data available
        self.builder.start_block(ready_block);
        // Null-terminate the string
        let term_pos_int = self.builder.add(buffer, bytes_read_i64);
        let term_pos = self.builder.inttoptr(term_pos_int, IrType::Ptr(Box::new(IrType::I8)));
        let null_byte = self.builder.const_i8(0);
        self.builder.store(term_pos, null_byte);

        // Create String from buffer (buffer is already the string data as i64)
        let result_ok = self.create_result_ok(buffer);
        self.builder.store(value_ptr, result_ok);
        let ready_state = self.builder.const_int(1);
        self.builder.store(state_ptr, ready_state);
        self.builder.br(done_block);

        // Pending: register for read events
        self.builder.start_block(pending_block);
        let epollin = self.builder.const_int(1);
        self.builder.call("__rt_reactor_register", vec![sock_fd_i64, epollin, future_ptr]);
        let pending_state = self.builder.const_int(2);
        self.builder.store(state_ptr, pending_state);
        self.builder.store(value_ptr, zero);
        self.builder.br(done_block);

        // Done
        self.builder.start_block(done_block);
        future_ptr  // Already i64 from ptrtoint
    }

    /// Lower TcpStream::write_string(stream, data) -> Future<Result<i64, i64>>
    /// Async write - creates a Future that waits until write is possible
    fn lower_tcp_stream_write_string(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let stream = self.lower_expr(&args[0]);
        let data = self.lower_expr(&args[1]);

        // Get fd from TcpStream struct
        let stream_ptr = self.builder.inttoptr(stream, IrType::Ptr(Box::new(IrType::I64)));
        let sock_fd_i64 = self.builder.load(stream_ptr);
        let sock_fd = self.builder.trunc(sock_fd_i64, IrType::I32);

        // Set socket to non-blocking
        let zero = self.builder.const_int(0);
        let zero_i32 = self.builder.const_i32(0);
        let f_getfl = self.builder.const_i32(3);
        let flags = self.builder.call("fcntl", vec![sock_fd, f_getfl, zero_i32]);
        let o_nonblock = self.builder.const_i32(2048);
        let new_flags = self.builder.or(flags, o_nonblock);
        let f_setfl = self.builder.const_i32(4);
        self.builder.call("fcntl", vec![sock_fd, f_setfl, new_flags]);

        // Get string length using strlen
        let str_len = self.builder.call("strlen", vec![data]);
        let str_len_i64 = self.builder.sext(str_len, IrType::I64);

        // Try non-blocking write
        let bytes_written = self.builder.call("write", vec![sock_fd, data, str_len]);

        // Check result
        let write_success = self.builder.icmp(CmpOp::Sgt, bytes_written, zero_i32);

        // Create Future struct
        let future_size = self.builder.const_int(32);
        let future_ptr_raw = self.builder.call("malloc", vec![future_size]);
        let future_ptr = self.builder.ptrtoint(future_ptr_raw, IrType::I64);

        let state_ptr = self.builder.inttoptr(future_ptr, IrType::Ptr(Box::new(IrType::I64)));
        let eight = self.builder.const_int(8);
        let fn_ptr_addr = self.builder.add(future_ptr, eight);
        let fn_ptr_ptr = self.builder.inttoptr(fn_ptr_addr, IrType::Ptr(Box::new(IrType::I64)));
        let sixteen = self.builder.const_int(16);
        let env_ptr_addr = self.builder.add(future_ptr, sixteen);
        let env_ptr_ptr = self.builder.inttoptr(env_ptr_addr, IrType::Ptr(Box::new(IrType::I64)));
        let twenty_four = self.builder.const_int(24);
        let value_addr = self.builder.add(future_ptr, twenty_four);
        let value_ptr = self.builder.inttoptr(value_addr, IrType::Ptr(Box::new(IrType::I64)));

        // Store data pointer and length for retry
        self.builder.store(fn_ptr_ptr, data);
        self.builder.store(env_ptr_ptr, str_len_i64);

        let ready_block = self.builder.create_block();
        let pending_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(write_success, ready_block, pending_block);

        // Ready: write succeeded
        self.builder.start_block(ready_block);
        let bytes_written_i64 = self.builder.sext(bytes_written, IrType::I64);
        let result_ok = self.create_result_ok(bytes_written_i64);
        self.builder.store(value_ptr, result_ok);
        let ready_state = self.builder.const_int(1);
        self.builder.store(state_ptr, ready_state);
        self.builder.br(done_block);

        // Pending: register for write events
        self.builder.start_block(pending_block);
        let epollout = self.builder.const_int(4); // EPOLLOUT
        self.builder.call("__rt_reactor_register", vec![sock_fd_i64, epollout, future_ptr]);
        let pending_state = self.builder.const_int(2);
        self.builder.store(state_ptr, pending_state);
        self.builder.store(value_ptr, zero);
        self.builder.br(done_block);

        // Done
        self.builder.start_block(done_block);
        future_ptr  // Already i64 from ptrtoint
    }

    /// Lower TcpStream::close(stream) -> ()
    fn lower_tcp_stream_close(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let stream = self.lower_expr(&args[0]);

        // Get fd from TcpStream struct
        let stream_ptr = self.builder.inttoptr(stream, IrType::Ptr(Box::new(IrType::I64)));
        let sock_fd_i64 = self.builder.load(stream_ptr);
        let sock_fd = self.builder.trunc(sock_fd_i64, IrType::I32);

        // Unregister from reactor
        self.builder.call("__rt_reactor_unregister", vec![sock_fd_i64]);

        // close(fd)
        self.builder.call("close", vec![sock_fd]);

        // free the stream struct (use ptr for free)
        self.builder.call("free", vec![stream_ptr]);

        self.builder.const_int(0)
    }

    /// Register built-in enum types (Option, Result) that are not in the AST
    fn register_builtin_enums(&mut self) {
        // Option<T> has two variants: None (discriminant 0) and Some(T) (discriminant 1)
        self.enum_variants.insert("Option::None".to_string(), EnumVariantInfo {
            discriminant: 0,
            payload_type: None,
        });
        self.enum_variants.insert("Option::Some".to_string(), EnumVariantInfo {
            discriminant: 1,
            payload_type: Some(IrType::I64), // Generic payload represented as i64
        });
        self.enum_types.insert("Option".to_string(), IrType::Struct(vec![IrType::I32, IrType::I64]));

        // Result<T, E> has two variants: Ok(T) (discriminant 0) and Err(E) (discriminant 1)
        self.enum_variants.insert("Result::Ok".to_string(), EnumVariantInfo {
            discriminant: 0,
            payload_type: Some(IrType::I64),
        });
        self.enum_variants.insert("Result::Err".to_string(), EnumVariantInfo {
            discriminant: 1,
            payload_type: Some(IrType::I64),
        });
        self.enum_types.insert("Result".to_string(), IrType::Struct(vec![IrType::I32, IrType::I64]));
    }

    /// Collect enum variant information for lowering
    fn collect_enum_info(&mut self, e: &ast::EnumDef) {
        use crate::ast::VariantKind;
        let enum_name = &e.name.name;

        // Calculate max payload size for the tagged union
        let mut max_payload_size = 0;
        let mut variant_types: Vec<IrType> = Vec::new();

        for (i, variant) in e.variants.iter().enumerate() {
            let variant_name = format!("{}::{}", enum_name, variant.name.name);
            let payload_type = match &variant.kind {
                VariantKind::Unit => None,
                VariantKind::Tuple(types) => {
                    if types.len() == 1 {
                        let ty = self.ast_type_to_ty(&types[0]);
                        let ir_ty = self.ty_to_ir_type(&ty);
                        max_payload_size = max_payload_size.max(ir_ty.size());
                        Some(ir_ty)
                    } else {
                        // Multi-field tuple: struct of types
                        let field_types: Vec<IrType> = types.iter()
                            .map(|t| self.ty_to_ir_type(&self.ast_type_to_ty(t)))
                            .collect();
                        let struct_ty = IrType::Struct(field_types);
                        max_payload_size = max_payload_size.max(struct_ty.size());
                        Some(struct_ty)
                    }
                }
                VariantKind::Struct(fields) => {
                    let field_types: Vec<IrType> = fields.iter()
                        .map(|f| self.ty_to_ir_type(&self.ast_type_to_ty(&f.ty)))
                        .collect();
                    let struct_ty = IrType::Struct(field_types);
                    max_payload_size = max_payload_size.max(struct_ty.size());
                    Some(struct_ty)
                }
            };

            self.enum_variants.insert(variant_name, EnumVariantInfo {
                discriminant: i as u32,
                payload_type: payload_type.clone(),
            });
            variant_types.push(payload_type.unwrap_or(IrType::Void));
        }

        // Create the enum type as a struct: { discriminant: i32, payload: [max_size x i8] }
        // For simplicity, use i64 for payload (can hold most simple types)
        let enum_ty = IrType::Struct(vec![IrType::I32, IrType::I64]);
        self.enum_types.insert(enum_name.clone(), enum_ty);
    }

    // ============ Async Runtime Support ============

    /// Generate async runtime support functions and globals
    /// This creates the infrastructure for spawn() and block_on() to work with multiple tasks
    ///
    /// Future states:
    ///   0 = NotStarted (lazy, not yet polled)
    ///   1 = Ready (completed with value)
    ///   2 = Pending (suspended, waiting to be woken)
    fn generate_async_runtime(&mut self) {
        use super::types::Constant;

        // Add global variables for the task queue
        // __rt_tasks: pointer to array of Future pointers
        self.builder.add_global(
            "__rt_tasks",
            IrType::I64, // Pointer stored as i64
            Some(Constant::Int(0)),
            false,
        );

        // __rt_task_count: number of tasks in queue
        self.builder.add_global(
            "__rt_task_count",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // __rt_task_cap: capacity of task array
        self.builder.add_global(
            "__rt_task_cap",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // __rt_pending_count: number of pending tasks (for tracking progress)
        self.builder.add_global(
            "__rt_pending_count",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // === Reactor globals (Phase 4: Async I/O) ===

        // __rt_epoll_fd: epoll file descriptor (-1 = not initialized)
        // Stored as I64 because the IR load defaults to I64
        self.builder.add_global(
            "__rt_epoll_fd",
            IrType::I64,
            Some(Constant::Int(-1)),
            false,
        );

        // __rt_reactor_fds: array of registered fd entries
        // Each entry: { fd: i32, events: i32, waker_ptr: i64 } = 16 bytes
        self.builder.add_global(
            "__rt_reactor_fds",
            IrType::I64, // Pointer stored as i64
            Some(Constant::Int(0)),
            false,
        );

        // __rt_reactor_count: number of registered fds
        self.builder.add_global(
            "__rt_reactor_count",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // __rt_reactor_cap: capacity of reactor_fds array
        self.builder.add_global(
            "__rt_reactor_cap",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // === Timer globals (Phase 5: Async Timers) ===

        // __rt_timers: pointer to array of timer entries
        // Each entry: { deadline_ms: i64, waker_ptr: i64, active: i64 } = 24 bytes
        self.builder.add_global(
            "__rt_timers",
            IrType::I64, // Pointer stored as i64
            Some(Constant::Int(0)),
            false,
        );

        // __rt_timer_count: number of active timers
        self.builder.add_global(
            "__rt_timer_count",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // __rt_timer_cap: capacity of timer array
        self.builder.add_global(
            "__rt_timer_cap",
            IrType::I64,
            Some(Constant::Int(0)),
            false,
        );

        // Generate __rt_init function
        self.generate_rt_init();

        // Generate __rt_spawn function
        self.generate_rt_spawn();

        // Generate __rt_run_tasks function
        self.generate_rt_run_tasks();

        // Generate __rt_poll_future function
        self.generate_rt_poll_future();

        // Generate __rt_wake function (for waking pending tasks)
        self.generate_rt_wake();

        // Generate reactor runtime functions (Phase 4: Async I/O)
        self.generate_reactor_runtime();
    }

    /// Generate reactor runtime support functions (Phase 4)
    fn generate_reactor_runtime(&mut self) {
        // Generate __rt_reactor_init
        self.generate_rt_reactor_init();

        // Generate __rt_reactor_register
        self.generate_rt_reactor_register();

        // Generate __rt_reactor_poll
        self.generate_rt_reactor_poll();

        // Generate __rt_reactor_unregister
        self.generate_rt_reactor_unregister();

        // Generate timer runtime functions (Phase 5: Async Timers)
        self.generate_timer_runtime();

        // Generate random number generator runtime
        self.generate_random_runtime();

        // Generate format! runtime functions
        self.generate_format_runtime();
    }

    /// Generate timer runtime support functions (Phase 5)
    fn generate_timer_runtime(&mut self) {
        // Generate __rt_time_now (milliseconds)
        self.generate_rt_time_now();

        // Generate __rt_time_now_us (microseconds)
        self.generate_rt_time_now_us();

        // Generate __rt_time_now_ns (nanoseconds)
        self.generate_rt_time_now_ns();

        // Generate __rt_timer_init
        self.generate_rt_timer_init();

        // Generate __rt_timer_register
        self.generate_rt_timer_register();

        // Generate __rt_timer_cancel
        self.generate_rt_timer_cancel();

        // Generate __rt_timer_next_deadline
        self.generate_rt_timer_next_deadline();

        // Generate __rt_timer_process
        self.generate_rt_timer_process();
    }

    /// Generate __rt_reactor_init() - Initialize the epoll reactor
    fn generate_rt_reactor_init(&mut self) {
        let _params = self.builder.start_function("__rt_reactor_init", vec![], IrType::Void);

        // Call epoll_create1(0) to create epoll instance
        let zero_i32 = self.builder.const_i32(0);
        let epoll_fd_i32 = self.builder.call("epoll_create1", vec![zero_i32]);
        // Sign extend to i64 for storage
        let epoll_fd = self.builder.sext(epoll_fd_i32, IrType::I64);

        // Store epoll fd in global (as i64)
        let epoll_fd_global = self.builder.global_ref("__rt_epoll_fd");
        self.builder.store(epoll_fd_global, epoll_fd);

        // Allocate initial reactor_fds array (64 entries * 16 bytes each = 1024 bytes)
        // Each entry: { fd: i32 (4), events: i32 (4), waker_ptr: i64 (8) } = 16 bytes
        let initial_cap = self.builder.const_int(64);
        let entry_size = self.builder.const_int(16);
        let alloc_size = self.builder.mul(initial_cap, entry_size);
        let fds_ptr_raw = self.builder.call("malloc", vec![alloc_size]);
        let fds_ptr = self.builder.ptrtoint(fds_ptr_raw, IrType::I64);

        // Store in globals
        let fds_global = self.builder.global_ref("__rt_reactor_fds");
        self.builder.store(fds_global, fds_ptr);

        let zero = self.builder.const_int(0);
        let count_global = self.builder.global_ref("__rt_reactor_count");
        self.builder.store(count_global, zero);

        let cap_global = self.builder.global_ref("__rt_reactor_cap");
        self.builder.store(cap_global, initial_cap);

        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_reactor_register(fd, events, waker_ptr) -> i64
    /// Registers an fd for events. Returns index in reactor array, or -1 on error.
    /// events: EPOLLIN=1, EPOLLOUT=4
    fn generate_rt_reactor_register(&mut self) {
        let params = self.builder.start_function(
            "__rt_reactor_register",
            vec![IrType::I32, IrType::I32, IrType::I64],
            IrType::I64,
        );
        let fd = params[0];
        let events = params[1];
        let waker_ptr = params[2];

        // Load current count and capacity
        let count_global = self.builder.global_ref("__rt_reactor_count");
        let count = self.builder.load(count_global);

        let cap_global = self.builder.global_ref("__rt_reactor_cap");
        let cap = self.builder.load(cap_global);

        // Check if we need to grow (count >= cap)
        let need_grow_block = self.builder.create_block();
        let store_block = self.builder.create_block();

        let need_grow = self.builder.icmp(CmpOp::Sge, count, cap);
        self.builder.cond_br(need_grow, need_grow_block, store_block);

        // Grow the array
        self.builder.start_block(need_grow_block);
        let two = self.builder.const_int(2);
        let new_cap = self.builder.mul(cap, two);
        let entry_size = self.builder.const_int(16);
        let new_size = self.builder.mul(new_cap, entry_size);

        let fds_global = self.builder.global_ref("__rt_reactor_fds");
        let old_fds = self.builder.load(fds_global);
        let old_fds_ptr = self.builder.inttoptr(old_fds, IrType::Ptr(Box::new(IrType::I8)));
        let new_fds = self.builder.call("realloc", vec![old_fds_ptr, new_size]);
        let new_fds_i64 = self.builder.ptrtoint(new_fds, IrType::I64);

        let fds_global2 = self.builder.global_ref("__rt_reactor_fds");
        self.builder.store(fds_global2, new_fds_i64);

        let cap_global2 = self.builder.global_ref("__rt_reactor_cap");
        self.builder.store(cap_global2, new_cap);

        self.builder.br(store_block);

        // Store the new entry
        self.builder.start_block(store_block);

        // Calculate entry address: reactor_fds + (count * 16)
        let fds_global3 = self.builder.global_ref("__rt_reactor_fds");
        let fds_ptr = self.builder.load(fds_global3);
        let entry_size2 = self.builder.const_int(16);
        let offset = self.builder.mul(count, entry_size2);
        let entry_addr = self.builder.add(fds_ptr, offset);

        // Store fd at entry[0] (offset 0)
        let fd_ptr = self.builder.inttoptr(entry_addr, IrType::ptr(IrType::I32));
        self.builder.store(fd_ptr, fd);

        // Store events at entry[4] (offset 4)
        let four = self.builder.const_int(4);
        let events_addr = self.builder.add(entry_addr, four);
        let events_ptr = self.builder.inttoptr(events_addr, IrType::ptr(IrType::I32));
        self.builder.store(events_ptr, events);

        // Store waker_ptr at entry[8] (offset 8)
        let eight = self.builder.const_int(8);
        let waker_addr = self.builder.add(entry_addr, eight);
        let waker_ptr_ptr = self.builder.inttoptr(waker_addr, IrType::ptr(IrType::I64));
        self.builder.store(waker_ptr_ptr, waker_ptr);

        // Call epoll_ctl to register the fd
        // epoll_ctl(epoll_fd, EPOLL_CTL_ADD=1, fd, &epoll_event)
        // epoll_event layout: { u32 events, u64 data } = 12 bytes (but aligned to 16 on 64-bit)
        // We allocate 16 bytes to hold the struct properly
        let event_alloc = self.builder.alloca(IrType::I64); // First 8 bytes
        let _event_alloc2 = self.builder.alloca(IrType::I64); // Second 8 bytes (for alignment)

        // Store events in epoll_event.events (first 4 bytes)
        let event_ptr = self.builder.inttoptr(event_alloc, IrType::ptr(IrType::I32));
        self.builder.store(event_ptr, events);

        // Store waker_ptr in epoll_event.data (next 8 bytes, at offset 4)
        // Actually epoll_event is: struct { uint32_t events; epoll_data_t data; }
        // epoll_data_t is union { void *ptr; int fd; uint32_t u32; uint64_t u64; }
        // So data starts at offset 4 in the struct
        let event_base = self.builder.ptrtoint(event_alloc, IrType::I64);
        let data_offset = self.builder.const_int(4);
        let data_addr = self.builder.add(event_base, data_offset);
        let data_ptr = self.builder.inttoptr(data_addr, IrType::ptr(IrType::I64));
        self.builder.store(data_ptr, waker_ptr);

        let epoll_fd_global = self.builder.global_ref("__rt_epoll_fd");
        let epoll_fd_i64 = self.builder.load(epoll_fd_global);
        let epoll_fd = self.builder.trunc(epoll_fd_i64, IrType::I32); // Truncate to i32 for epoll_ctl
        let epoll_ctl_add = self.builder.const_i32(1); // EPOLL_CTL_ADD
        let event_ptr_i8 = self.builder.bitcast(event_alloc, IrType::ptr(IrType::I8));
        let _result = self.builder.call("epoll_ctl", vec![epoll_fd, epoll_ctl_add, fd, event_ptr_i8]);

        // Increment count
        let one = self.builder.const_int(1);
        let new_count = self.builder.add(count, one);
        let count_global2 = self.builder.global_ref("__rt_reactor_count");
        self.builder.store(count_global2, new_count);

        // Return the index
        self.builder.ret(Some(count));
        self.builder.finish_function();
    }

    /// Generate __rt_reactor_poll(timeout_ms) -> i64
    /// Polls for events and wakes up waiting tasks.
    /// timeout_ms: -1 = block forever, 0 = non-blocking, >0 = timeout in ms
    /// Returns number of events processed.
    fn generate_rt_reactor_poll(&mut self) {
        let params = self.builder.start_function(
            "__rt_reactor_poll",
            vec![IrType::I32],
            IrType::I64,
        );
        let timeout = params[0];

        // Allocate events array on stack (max 64 events * 12 bytes each)
        // We need to use heap for variable-sized array in practice
        let max_events = self.builder.const_int(64);
        let event_size = self.builder.const_int(12); // sizeof(epoll_event) = 12
        let events_size = self.builder.mul(max_events, event_size);
        let events_ptr_raw = self.builder.call("malloc", vec![events_size]);
        let events_ptr = self.builder.ptrtoint(events_ptr_raw, IrType::I64);

        // Call epoll_wait
        let epoll_fd_global = self.builder.global_ref("__rt_epoll_fd");
        let epoll_fd_i64 = self.builder.load(epoll_fd_global);
        let epoll_fd = self.builder.trunc(epoll_fd_i64, IrType::I32); // Truncate to i32 for epoll_wait
        let events_ptr_i8 = self.builder.inttoptr(events_ptr, IrType::ptr(IrType::I8));
        let max_events_i32 = self.builder.const_i32(64);
        let num_events = self.builder.call("epoll_wait", vec![epoll_fd, events_ptr_i8, max_events_i32, timeout]);

        // Check if num_events <= 0 (no events or error)
        let zero_i32 = self.builder.const_i32(0);
        let no_events = self.builder.icmp(CmpOp::Sle, num_events, zero_i32);

        let early_exit_block = self.builder.create_block();
        let process_events_block = self.builder.create_block();

        self.builder.cond_br(no_events, early_exit_block, process_events_block);

        // Early exit: free events and return 0
        self.builder.start_block(early_exit_block);
        self.builder.free(events_ptr_raw);
        let zero = self.builder.const_int(0);
        self.builder.ret(Some(zero));

        // Process events
        self.builder.start_block(process_events_block);

        // Loop through events and wake up tasks
        // for i in 0..num_events:
        //   waker_ptr = events[i].data
        //   __rt_wake(waker_ptr)
        let loop_init_block = self.builder.create_block();
        let loop_cond_block = self.builder.create_block();
        let loop_body_block = self.builder.create_block();
        let loop_inc_block = self.builder.create_block();
        let loop_exit_block = self.builder.create_block();

        self.builder.br(loop_init_block);

        // Initialize loop counter
        self.builder.start_block(loop_init_block);
        let i_alloc = self.builder.alloca(IrType::I32);
        let zero_i32_init = self.builder.const_i32(0);
        self.builder.store(i_alloc, zero_i32_init);
        self.builder.br(loop_cond_block);

        // Check loop condition
        self.builder.start_block(loop_cond_block);
        let i_val = self.builder.load(i_alloc);
        let continue_loop = self.builder.icmp(CmpOp::Slt, i_val, num_events);
        self.builder.cond_br(continue_loop, loop_body_block, loop_exit_block);

        // Loop body: wake the task
        self.builder.start_block(loop_body_block);
        // Calculate event address: events_ptr + (i * 12)
        let i_i64 = self.builder.sext(i_val, IrType::I64);
        let event_size_body = self.builder.const_int(12);
        let event_offset = self.builder.mul(i_i64, event_size_body);
        let event_addr = self.builder.add(events_ptr, event_offset);
        // Get waker_ptr from event.data (at offset 4)
        let data_offset_body = self.builder.const_int(4);
        let data_addr = self.builder.add(event_addr, data_offset_body);
        let data_ptr = self.builder.inttoptr(data_addr, IrType::ptr(IrType::I64));
        let waker_ptr = self.builder.load(data_ptr);
        // Wake the task
        self.builder.call("__rt_wake", vec![waker_ptr]);
        self.builder.br(loop_inc_block);

        // Increment loop counter
        self.builder.start_block(loop_inc_block);
        let i_val2 = self.builder.load(i_alloc);
        let one_i32 = self.builder.const_i32(1);
        let i_plus_1 = self.builder.add(i_val2, one_i32);
        self.builder.store(i_alloc, i_plus_1);
        self.builder.br(loop_cond_block);

        // Exit loop: free events and return count
        self.builder.start_block(loop_exit_block);
        self.builder.free(events_ptr_raw);
        let result = self.builder.sext(num_events, IrType::I64);
        self.builder.ret(Some(result));

        self.builder.finish_function();
    }

    /// Generate __rt_reactor_unregister(fd) -> i64
    /// Removes an fd from the reactor. Returns 0 on success, -1 on error.
    fn generate_rt_reactor_unregister(&mut self) {
        let params = self.builder.start_function(
            "__rt_reactor_unregister",
            vec![IrType::I32],
            IrType::I64,
        );
        let fd = params[0];

        // Call epoll_ctl with EPOLL_CTL_DEL
        let epoll_fd_global = self.builder.global_ref("__rt_epoll_fd");
        let epoll_fd_i64 = self.builder.load(epoll_fd_global);
        let epoll_fd = self.builder.trunc(epoll_fd_i64, IrType::I32); // Truncate to i32 for epoll_ctl
        let epoll_ctl_del = self.builder.const_i32(2); // EPOLL_CTL_DEL
        let null_ptr = self.builder.const_null();
        let result = self.builder.call("epoll_ctl", vec![epoll_fd, epoll_ctl_del, fd, null_ptr]);

        // Return 0 on success (result == 0), -1 on error
        let zero_i32 = self.builder.const_i32(0);
        let success = self.builder.icmp(CmpOp::Eq, result, zero_i32);

        let success_block = self.builder.create_block();
        let error_block = self.builder.create_block();

        self.builder.cond_br(success, success_block, error_block);

        self.builder.start_block(success_block);
        let zero = self.builder.const_int(0);
        self.builder.ret(Some(zero));

        self.builder.start_block(error_block);
        let neg_one = self.builder.const_int(-1);
        self.builder.ret(Some(neg_one));

        self.builder.finish_function();
    }

    // ============ Timer Runtime Functions (Phase 5: Async Timers) ============

    /// Generate __rt_time_now() -> i64
    /// Returns the current time in milliseconds since epoch using clock_gettime(CLOCK_MONOTONIC)
    fn generate_rt_time_now(&mut self) {
        let _params = self.builder.start_function("__rt_time_now", vec![], IrType::I64);

        // Allocate struct timespec on stack: { tv_sec: i64, tv_nsec: i64 } = 16 bytes
        // Use a struct type to ensure contiguous allocation
        let timespec_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let timespec_ptr = self.builder.alloca(timespec_ty);

        // Call clock_gettime(CLOCK_MONOTONIC=1, &timespec)
        let clock_monotonic = self.builder.const_i32(1);
        let timespec_ptr_i8 = self.builder.bitcast(timespec_ptr, IrType::ptr(IrType::I8));
        self.builder.call("clock_gettime", vec![clock_monotonic, timespec_ptr_i8]);

        // Load tv_sec (field 0)
        let tv_sec_ptr = self.builder.get_field_ptr(timespec_ptr, 0);
        let tv_sec = self.builder.load(tv_sec_ptr);

        // Load tv_nsec (field 1)
        let tv_nsec_ptr = self.builder.get_field_ptr(timespec_ptr, 1);
        let tv_nsec = self.builder.load(tv_nsec_ptr);

        // Convert to milliseconds: (tv_sec * 1000) + (tv_nsec / 1_000_000)
        let thousand = self.builder.const_int(1000);
        let sec_ms = self.builder.mul(tv_sec, thousand);

        let million = self.builder.const_int(1_000_000);
        let nsec_ms = self.builder.sdiv(tv_nsec, million);

        let total_ms = self.builder.add(sec_ms, nsec_ms);

        self.builder.ret(Some(total_ms));
        self.builder.finish_function();
    }

    /// Generate __rt_time_now_us() -> i64
    /// Returns the current time in microseconds since epoch using clock_gettime(CLOCK_MONOTONIC)
    fn generate_rt_time_now_us(&mut self) {
        let _params = self.builder.start_function("__rt_time_now_us", vec![], IrType::I64);

        // Allocate struct timespec on stack: { tv_sec: i64, tv_nsec: i64 } = 16 bytes
        let timespec_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let timespec_ptr = self.builder.alloca(timespec_ty);

        // Call clock_gettime(CLOCK_MONOTONIC=1, &timespec)
        let clock_monotonic = self.builder.const_i32(1);
        let timespec_ptr_i8 = self.builder.bitcast(timespec_ptr, IrType::ptr(IrType::I8));
        self.builder.call("clock_gettime", vec![clock_monotonic, timespec_ptr_i8]);

        // Load tv_sec (field 0)
        let tv_sec_ptr = self.builder.get_field_ptr(timespec_ptr, 0);
        let tv_sec = self.builder.load(tv_sec_ptr);

        // Load tv_nsec (field 1)
        let tv_nsec_ptr = self.builder.get_field_ptr(timespec_ptr, 1);
        let tv_nsec = self.builder.load(tv_nsec_ptr);

        // Convert to microseconds: (tv_sec * 1_000_000) + (tv_nsec / 1_000)
        let million = self.builder.const_int(1_000_000);
        let sec_us = self.builder.mul(tv_sec, million);

        let thousand = self.builder.const_int(1_000);
        let nsec_us = self.builder.sdiv(tv_nsec, thousand);

        let total_us = self.builder.add(sec_us, nsec_us);

        self.builder.ret(Some(total_us));
        self.builder.finish_function();
    }

    /// Generate __rt_time_now_ns() -> i64
    /// Returns the current time in nanoseconds since epoch using clock_gettime(CLOCK_MONOTONIC)
    fn generate_rt_time_now_ns(&mut self) {
        let _params = self.builder.start_function("__rt_time_now_ns", vec![], IrType::I64);

        // Allocate struct timespec on stack: { tv_sec: i64, tv_nsec: i64 } = 16 bytes
        let timespec_ty = IrType::Struct(vec![IrType::I64, IrType::I64]);
        let timespec_ptr = self.builder.alloca(timespec_ty);

        // Call clock_gettime(CLOCK_MONOTONIC=1, &timespec)
        let clock_monotonic = self.builder.const_i32(1);
        let timespec_ptr_i8 = self.builder.bitcast(timespec_ptr, IrType::ptr(IrType::I8));
        self.builder.call("clock_gettime", vec![clock_monotonic, timespec_ptr_i8]);

        // Load tv_sec (field 0)
        let tv_sec_ptr = self.builder.get_field_ptr(timespec_ptr, 0);
        let tv_sec = self.builder.load(tv_sec_ptr);

        // Load tv_nsec (field 1)
        let tv_nsec_ptr = self.builder.get_field_ptr(timespec_ptr, 1);
        let tv_nsec = self.builder.load(tv_nsec_ptr);

        // Convert to nanoseconds: (tv_sec * 1_000_000_000) + tv_nsec
        let billion = self.builder.const_int(1_000_000_000);
        let sec_ns = self.builder.mul(tv_sec, billion);

        let total_ns = self.builder.add(sec_ns, tv_nsec);

        self.builder.ret(Some(total_ns));
        self.builder.finish_function();
    }

    /// Generate __rt_timer_init() - Initialize the timer array
    fn generate_rt_timer_init(&mut self) {
        let _params = self.builder.start_function("__rt_timer_init", vec![], IrType::Void);

        // Allocate initial timer array (32 timers * 24 bytes each = 768 bytes)
        // Each entry: { deadline_ms: i64, waker_ptr: i64, active: i64 }
        let initial_cap = self.builder.const_int(32);
        let entry_size = self.builder.const_int(24);
        let alloc_size = self.builder.mul(initial_cap, entry_size);
        let timers_ptr_raw = self.builder.call("malloc", vec![alloc_size]);
        let timers_ptr = self.builder.ptrtoint(timers_ptr_raw, IrType::I64);

        // Store in globals
        let timers_global = self.builder.global_ref("__rt_timers");
        self.builder.store(timers_global, timers_ptr);

        let zero = self.builder.const_int(0);
        let count_global = self.builder.global_ref("__rt_timer_count");
        self.builder.store(count_global, zero);

        let cap_global = self.builder.global_ref("__rt_timer_cap");
        self.builder.store(cap_global, initial_cap);

        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_timer_register(deadline_ms, waker_ptr) -> i64
    /// Registers a timer with the given deadline. Returns timer ID (index), or -1 on error.
    fn generate_rt_timer_register(&mut self) {
        let params = self.builder.start_function(
            "__rt_timer_register",
            vec![IrType::I64, IrType::I64],
            IrType::I64,
        );
        let deadline_ms = params[0];
        let waker_ptr = params[1];

        // Load current count and capacity
        let count_global = self.builder.global_ref("__rt_timer_count");
        let count = self.builder.load(count_global);

        let cap_global = self.builder.global_ref("__rt_timer_cap");
        let cap = self.builder.load(cap_global);

        // Check if we need to grow (count >= cap)
        let needs_grow = self.builder.icmp(CmpOp::Sge, count, cap);
        let grow_block = self.builder.create_block();
        let store_block = self.builder.create_block();

        self.builder.cond_br(needs_grow, grow_block, store_block);

        // Grow the array (double capacity)
        self.builder.start_block(grow_block);
        let two = self.builder.const_int(2);
        let new_cap = self.builder.mul(cap, two);
        let entry_size = self.builder.const_int(24);
        let new_size = self.builder.mul(new_cap, entry_size);

        let timers_global = self.builder.global_ref("__rt_timers");
        let old_timers = self.builder.load(timers_global);
        let old_timers_ptr = self.builder.inttoptr(old_timers, IrType::Ptr(Box::new(IrType::I8)));
        let new_timers = self.builder.call("realloc", vec![old_timers_ptr, new_size]);
        let new_timers_i64 = self.builder.ptrtoint(new_timers, IrType::I64);

        let timers_global2 = self.builder.global_ref("__rt_timers");
        self.builder.store(timers_global2, new_timers_i64);

        let cap_global2 = self.builder.global_ref("__rt_timer_cap");
        self.builder.store(cap_global2, new_cap);

        self.builder.br(store_block);

        // Store the new timer entry
        self.builder.start_block(store_block);

        // Calculate entry address: timers + (count * 24)
        let timers_global3 = self.builder.global_ref("__rt_timers");
        let timers_ptr = self.builder.load(timers_global3);
        let entry_size2 = self.builder.const_int(24);
        let offset = self.builder.mul(count, entry_size2);
        let entry_addr = self.builder.add(timers_ptr, offset);

        // Store deadline_ms at offset 0
        let deadline_ptr = self.builder.inttoptr(entry_addr, IrType::ptr(IrType::I64));
        self.builder.store(deadline_ptr, deadline_ms);

        // Store waker_ptr at offset 8
        let eight = self.builder.const_int(8);
        let waker_addr = self.builder.add(entry_addr, eight);
        let waker_ptr_ptr = self.builder.inttoptr(waker_addr, IrType::ptr(IrType::I64));
        self.builder.store(waker_ptr_ptr, waker_ptr);

        // Store active=1 at offset 16
        let sixteen = self.builder.const_int(16);
        let active_addr = self.builder.add(entry_addr, sixteen);
        let active_ptr = self.builder.inttoptr(active_addr, IrType::ptr(IrType::I64));
        let one = self.builder.const_int(1);
        self.builder.store(active_ptr, one);

        // Increment count
        let new_count = self.builder.add(count, one);
        let count_global2 = self.builder.global_ref("__rt_timer_count");
        self.builder.store(count_global2, new_count);

        // Return the timer ID (which is the old count, i.e., the index)
        self.builder.ret(Some(count));
        self.builder.finish_function();
    }

    /// Generate __rt_timer_cancel(timer_id) -> i64
    /// Cancels a timer. Returns 0 on success, -1 if timer not found/already cancelled.
    fn generate_rt_timer_cancel(&mut self) {
        let params = self.builder.start_function(
            "__rt_timer_cancel",
            vec![IrType::I64],
            IrType::I64,
        );
        let timer_id = params[0];

        // Load timer count
        let count_global = self.builder.global_ref("__rt_timer_count");
        let count = self.builder.load(count_global);

        // Check if timer_id is valid (0 <= timer_id < count)
        let zero = self.builder.const_int(0);
        let valid_lower = self.builder.icmp(CmpOp::Sge, timer_id, zero);
        let valid_upper = self.builder.icmp(CmpOp::Slt, timer_id, count);
        let valid = self.builder.and(valid_lower, valid_upper);

        let valid_block = self.builder.create_block();
        let invalid_block = self.builder.create_block();

        self.builder.cond_br(valid, valid_block, invalid_block);

        // Valid timer ID - set active to 0
        self.builder.start_block(valid_block);

        let timers_global = self.builder.global_ref("__rt_timers");
        let timers_ptr = self.builder.load(timers_global);
        let entry_size = self.builder.const_int(24);
        let offset = self.builder.mul(timer_id, entry_size);
        let entry_addr = self.builder.add(timers_ptr, offset);

        // Set active=0 at offset 16
        let sixteen = self.builder.const_int(16);
        let active_addr = self.builder.add(entry_addr, sixteen);
        let active_ptr = self.builder.inttoptr(active_addr, IrType::ptr(IrType::I64));
        self.builder.store(active_ptr, zero);

        self.builder.ret(Some(zero));

        // Invalid timer ID
        self.builder.start_block(invalid_block);
        let neg_one = self.builder.const_int(-1);
        self.builder.ret(Some(neg_one));

        self.builder.finish_function();
    }

    /// Generate __rt_timer_next_deadline() -> i64
    /// Returns the next deadline in milliseconds, or -1 if no active timers.
    fn generate_rt_timer_next_deadline(&mut self) {
        let _params = self.builder.start_function("__rt_timer_next_deadline", vec![], IrType::I64);

        // Load timer count
        let count_global = self.builder.global_ref("__rt_timer_count");
        let count = self.builder.load(count_global);

        // Check if count == 0
        let zero = self.builder.const_int(0);
        let has_timers = self.builder.icmp(CmpOp::Sgt, count, zero);

        let loop_block = self.builder.create_block();
        let no_timers_block = self.builder.create_block();

        self.builder.cond_br(has_timers, loop_block, no_timers_block);

        // No timers - return -1
        self.builder.start_block(no_timers_block);
        let neg_one = self.builder.const_int(-1);
        self.builder.ret(Some(neg_one));

        // Loop through timers to find minimum deadline
        self.builder.start_block(loop_block);

        // Initialize min_deadline to i64::MAX
        let min_deadline_slot = self.builder.alloca(IrType::I64);
        let i64_max = self.builder.const_int(i64::MAX);
        self.builder.store(min_deadline_slot, i64_max);

        // Loop counter
        let i_slot = self.builder.alloca(IrType::I64);
        self.builder.store(i_slot, zero);

        let loop_cond_block = self.builder.create_block();
        let loop_body_block = self.builder.create_block();
        let loop_end_block = self.builder.create_block();

        self.builder.br(loop_cond_block);

        // Loop condition: i < count
        self.builder.start_block(loop_cond_block);
        let i_val = self.builder.load(i_slot);
        let cond = self.builder.icmp(CmpOp::Slt, i_val, count);
        self.builder.cond_br(cond, loop_body_block, loop_end_block);

        // Loop body
        self.builder.start_block(loop_body_block);
        let i_current = self.builder.load(i_slot);

        // Calculate entry address
        let timers_global = self.builder.global_ref("__rt_timers");
        let timers_ptr = self.builder.load(timers_global);
        let entry_size = self.builder.const_int(24);
        let offset = self.builder.mul(i_current, entry_size);
        let entry_addr = self.builder.add(timers_ptr, offset);

        // Load active flag at offset 16
        let sixteen = self.builder.const_int(16);
        let active_addr = self.builder.add(entry_addr, sixteen);
        let active_ptr = self.builder.inttoptr(active_addr, IrType::ptr(IrType::I64));
        let active = self.builder.load(active_ptr);

        // Check if active
        let one = self.builder.const_int(1);
        let is_active = self.builder.icmp(CmpOp::Eq, active, one);

        let check_deadline_block = self.builder.create_block();
        let next_iter_block = self.builder.create_block();

        self.builder.cond_br(is_active, check_deadline_block, next_iter_block);

        // Active timer - check if deadline < min_deadline
        self.builder.start_block(check_deadline_block);
        let deadline_ptr = self.builder.inttoptr(entry_addr, IrType::ptr(IrType::I64));
        let deadline = self.builder.load(deadline_ptr);
        let min_deadline = self.builder.load(min_deadline_slot);
        let is_smaller = self.builder.icmp(CmpOp::Slt, deadline, min_deadline);

        let update_min_block = self.builder.create_block();
        self.builder.cond_br(is_smaller, update_min_block, next_iter_block);

        // Update min_deadline
        self.builder.start_block(update_min_block);
        self.builder.store(min_deadline_slot, deadline);
        self.builder.br(next_iter_block);

        // Increment i and continue
        self.builder.start_block(next_iter_block);
        let i_inc = self.builder.load(i_slot);
        let i_next = self.builder.add(i_inc, one);
        self.builder.store(i_slot, i_next);
        self.builder.br(loop_cond_block);

        // Loop end - return min_deadline (or -1 if still i64::MAX)
        self.builder.start_block(loop_end_block);
        let final_min = self.builder.load(min_deadline_slot);
        let found_any = self.builder.icmp(CmpOp::Ne, final_min, i64_max);

        let return_min_block = self.builder.create_block();
        let return_neg_one_block = self.builder.create_block();

        self.builder.cond_br(found_any, return_min_block, return_neg_one_block);

        self.builder.start_block(return_min_block);
        self.builder.ret(Some(final_min));

        self.builder.start_block(return_neg_one_block);
        let neg_one2 = self.builder.const_int(-1);
        self.builder.ret(Some(neg_one2));

        self.builder.finish_function();
    }

    /// Generate __rt_timer_process() -> i64
    /// Processes all expired timers, waking their associated tasks.
    /// Returns the number of timers that were processed.
    fn generate_rt_timer_process(&mut self) {
        let _params = self.builder.start_function("__rt_timer_process", vec![], IrType::I64);

        // Get current time
        let now = self.builder.call("__rt_time_now", vec![]);

        // Load timer count
        let count_global = self.builder.global_ref("__rt_timer_count");
        let count = self.builder.load(count_global);

        // Counter for processed timers
        let processed_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        self.builder.store(processed_slot, zero);

        // Loop counter
        let i_slot = self.builder.alloca(IrType::I64);
        self.builder.store(i_slot, zero);

        let loop_cond_block = self.builder.create_block();
        let loop_body_block = self.builder.create_block();
        let loop_end_block = self.builder.create_block();

        self.builder.br(loop_cond_block);

        // Loop condition: i < count
        self.builder.start_block(loop_cond_block);
        let i_val = self.builder.load(i_slot);
        let cond = self.builder.icmp(CmpOp::Slt, i_val, count);
        self.builder.cond_br(cond, loop_body_block, loop_end_block);

        // Loop body
        self.builder.start_block(loop_body_block);
        let i_current = self.builder.load(i_slot);

        // Calculate entry address
        let timers_global = self.builder.global_ref("__rt_timers");
        let timers_ptr = self.builder.load(timers_global);
        let entry_size = self.builder.const_int(24);
        let offset = self.builder.mul(i_current, entry_size);
        let entry_addr = self.builder.add(timers_ptr, offset);

        // Load active flag at offset 16
        let sixteen = self.builder.const_int(16);
        let active_addr = self.builder.add(entry_addr, sixteen);
        let active_ptr = self.builder.inttoptr(active_addr, IrType::ptr(IrType::I64));
        let active = self.builder.load(active_ptr);

        // Check if active
        let one = self.builder.const_int(1);
        let is_active = self.builder.icmp(CmpOp::Eq, active, one);

        let check_expired_block = self.builder.create_block();
        let next_iter_block = self.builder.create_block();

        self.builder.cond_br(is_active, check_expired_block, next_iter_block);

        // Active timer - check if expired (deadline <= now)
        self.builder.start_block(check_expired_block);
        let deadline_ptr = self.builder.inttoptr(entry_addr, IrType::ptr(IrType::I64));
        let deadline = self.builder.load(deadline_ptr);
        let is_expired = self.builder.icmp(CmpOp::Sle, deadline, now);

        let wake_block = self.builder.create_block();
        self.builder.cond_br(is_expired, wake_block, next_iter_block);

        // Timer expired - wake the task and deactivate timer
        self.builder.start_block(wake_block);

        // Load waker_ptr at offset 8
        let eight = self.builder.const_int(8);
        let waker_addr = self.builder.add(entry_addr, eight);
        let waker_ptr_ptr = self.builder.inttoptr(waker_addr, IrType::ptr(IrType::I64));
        let waker_ptr = self.builder.load(waker_ptr_ptr);

        // Call __rt_wake to wake the future
        self.builder.call("__rt_wake", vec![waker_ptr]);

        // Deactivate timer (set active=0)
        self.builder.store(active_ptr, zero);

        // Increment processed count
        let processed = self.builder.load(processed_slot);
        let processed_inc = self.builder.add(processed, one);
        self.builder.store(processed_slot, processed_inc);

        self.builder.br(next_iter_block);

        // Increment i and continue
        self.builder.start_block(next_iter_block);
        let i_inc = self.builder.load(i_slot);
        let i_next = self.builder.add(i_inc, one);
        self.builder.store(i_slot, i_next);
        self.builder.br(loop_cond_block);

        // Loop end - return processed count
        self.builder.start_block(loop_end_block);
        let final_processed = self.builder.load(processed_slot);
        self.builder.ret(Some(final_processed));

        self.builder.finish_function();
    }

    // ============ Random Number Generator Runtime ============

    /// Generate random number generator runtime functions
    fn generate_random_runtime(&mut self) {
        use super::types::Constant;

        // Create global state for the RNG (xorshift64 state)
        self.builder.add_global(
            "__rt_random_state",
            IrType::I64,
            Some(Constant::Int(0x123456789ABCDEF0_u64 as i64)),
            false,
        );

        // Generate __rt_random_seed
        self.generate_rt_random_seed();

        // Generate __rt_random_next
        self.generate_rt_random_next();

        // Generate __rt_random_next_f64
        self.generate_rt_random_next_f64();
    }

    /// Generate __rt_random_seed(seed: i64)
    /// Sets the seed for the random number generator
    fn generate_rt_random_seed(&mut self) {
        let params = self.builder.start_function(
            "__rt_random_seed",
            vec![IrType::I64],
            IrType::Void,
        );

        let seed = params[0];

        // Store to global state (if seed is 0, xorshift will use initial global value)
        // For simplicity, just store the seed directly - user should provide non-zero seed
        let state_global = self.builder.global_ref("__rt_random_state");
        self.builder.store(state_global, seed);

        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_random_next() -> i64
    /// Xorshift64 algorithm - fast and good quality for non-cryptographic use
    fn generate_rt_random_next(&mut self) {
        let _params = self.builder.start_function("__rt_random_next", vec![], IrType::I64);

        // Load current state
        let state_global = self.builder.global_ref("__rt_random_state");
        let state = self.builder.load(state_global);

        // Xorshift64 algorithm:
        // x ^= x << 13
        // x ^= x >> 7
        // x ^= x << 17

        let thirteen = self.builder.const_int(13);
        let seven = self.builder.const_int(7);
        let seventeen = self.builder.const_int(17);

        // x ^= x << 13
        let shifted1 = self.builder.shl(state, thirteen);
        let state1 = self.builder.xor(state, shifted1);

        // x ^= x >> 7
        let shifted2 = self.builder.ashr(state1, seven);
        let state2 = self.builder.xor(state1, shifted2);

        // x ^= x << 17
        let shifted3 = self.builder.shl(state2, seventeen);
        let state3 = self.builder.xor(state2, shifted3);

        // Store new state
        self.builder.store(state_global, state3);

        // Return the new state as the random number
        self.builder.ret(Some(state3));
        self.builder.finish_function();
    }

    /// Generate __rt_random_next_f64() -> f64
    /// Returns a random f64 in [0.0, 1.0)
    fn generate_rt_random_next_f64(&mut self) {
        let _params = self.builder.start_function("__rt_random_next_f64", vec![], IrType::F64);

        // Get random i64
        let rand = self.builder.call("__rt_random_next", vec![]);

        // Make it positive by masking off sign bit
        let mask = self.builder.const_int(0x7FFFFFFFFFFFFFFF_u64 as i64);
        let positive = self.builder.and(rand, mask);

        // Convert to f64
        let as_f64 = self.builder.sitofp(positive, IrType::F64);

        // Divide by max positive i64 to get [0.0, 1.0)
        let max_val = self.builder.const_float(9223372036854775807.0); // 2^63 - 1
        let result = self.builder.fdiv(as_f64, max_val);

        self.builder.ret(Some(result));
        self.builder.finish_function();
    }

    // ============ Format Runtime Functions ============

    /// Generate format! runtime functions
    fn generate_format_runtime(&mut self) {
        // Generate __format_int_to_string
        self.generate_format_int_to_string();

        // Generate __format_float_to_string
        self.generate_format_float_to_string();
    }

    /// Generate __format_int_to_string(str_ptr, value, width, zero_pad, fmt_type)
    /// Appends formatted integer to the String
    /// fmt_type: 0=decimal, 1=hex_lower, 2=hex_upper, 3=binary, 4=octal
    fn generate_format_int_to_string(&mut self) {
        let params = self.builder.start_function(
            "__format_int_to_string",
            vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64, IrType::I64],
            IrType::Void,
        );

        let str_ptr = params[0];
        let value = params[1];
        let width = params[2];
        let zero_pad = params[3];
        let fmt_type = params[4];

        // Allocate buffer on stack (64 bytes is enough for any formatted integer)
        let buffer_size = self.builder.const_int(64);
        let buffer = self.builder.malloc_array(IrType::I8, buffer_size);

        // Use sprintf to format the integer
        // Select format string based on fmt_type
        let fmt_decimal = self.builder.add_string_constant("%lld");
        let fmt_decimal_ptr = self.builder.global_string_ptr(&fmt_decimal);
        let fmt_hex_lower = self.builder.add_string_constant("%llx");
        let fmt_hex_lower_ptr = self.builder.global_string_ptr(&fmt_hex_lower);
        let fmt_hex_upper = self.builder.add_string_constant("%llX");
        let fmt_hex_upper_ptr = self.builder.global_string_ptr(&fmt_hex_upper);
        let fmt_octal = self.builder.add_string_constant("%llo");
        let fmt_octal_ptr = self.builder.global_string_ptr(&fmt_octal);

        // Select format based on fmt_type
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        let two = self.builder.const_int(2);
        let four = self.builder.const_int(4);

        let is_hex_lower = self.builder.icmp(CmpOp::Eq, fmt_type, one);
        let is_hex_upper = self.builder.icmp(CmpOp::Eq, fmt_type, two);
        let is_octal = self.builder.icmp(CmpOp::Eq, fmt_type, four);

        // Choose format string (nested selects)
        let fmt_ptr1 = self.builder.select(is_octal, fmt_octal_ptr, fmt_decimal_ptr);
        let fmt_ptr2 = self.builder.select(is_hex_upper, fmt_hex_upper_ptr, fmt_ptr1);
        let fmt_ptr = self.builder.select(is_hex_lower, fmt_hex_lower_ptr, fmt_ptr2);

        // Call sprintf(buffer, format, value)
        let _result = self.builder.call("sprintf", vec![buffer, fmt_ptr, value]);

        // Calculate actual length using strlen
        let len = self.builder.call("strlen", vec![buffer]);

        // Handle width padding if needed
        let need_padding = self.builder.icmp(CmpOp::Sgt, width, len);

        let pad_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(need_padding, pad_block, done_block);

        // Pad block: add padding characters using memset
        self.builder.start_block(pad_block);
        let pad_count = self.builder.sub(width, len);

        // Determine padding char: '0' if zero_pad, ' ' otherwise
        let zero_char = self.builder.const_int(48); // '0'
        let space_char = self.builder.const_int(32); // ' '
        let is_zero_pad = self.builder.icmp(CmpOp::Ne, zero_pad, zero);
        let pad_char = self.builder.select(is_zero_pad, zero_char, space_char);
        let pad_char_i32 = self.builder.trunc(pad_char, IrType::I32);

        // Add padding using memset
        let pad_buffer = self.builder.malloc_array(IrType::I8, pad_count);
        self.builder.call("memset", vec![pad_buffer, pad_char_i32, pad_count]);

        // Push padding first
        self.builder.call("__string_push_bytes", vec![str_ptr, pad_buffer, pad_count]);
        self.builder.call("free", vec![pad_buffer]);
        self.builder.br(done_block);

        // Done block: push the formatted value
        self.builder.start_block(done_block);
        self.builder.call("__string_push_bytes", vec![str_ptr, buffer, len]);
        self.builder.call("free", vec![buffer]);

        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __format_float_to_string(str_ptr, value, precision, width)
    /// Appends formatted float to the String
    fn generate_format_float_to_string(&mut self) {
        let params = self.builder.start_function(
            "__format_float_to_string",
            vec![IrType::I64, IrType::F64, IrType::I64, IrType::I64],
            IrType::Void,
        );

        let str_ptr = params[0];
        let value = params[1];
        let _precision = params[2];
        let _width = params[3];

        // Allocate buffer (64 bytes is enough)
        let buffer_size = self.builder.const_int(64);
        let buffer = self.builder.malloc_array(IrType::I8, buffer_size);

        // Use %g format for general float formatting
        let fmt_str = self.builder.add_string_constant("%g");
        let fmt_ptr = self.builder.global_string_ptr(&fmt_str);

        // Call sprintf(buffer, format, value)
        let _result = self.builder.call("sprintf", vec![buffer, fmt_ptr, value]);

        // Calculate length
        let len = self.builder.call("strlen", vec![buffer]);

        // Push to string
        self.builder.call("__string_push_bytes", vec![str_ptr, buffer, len]);
        self.builder.call("free", vec![buffer]);

        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_init() - Initialize the task queue
    fn generate_rt_init(&mut self) {
        let params = self.builder.start_function("__rt_init", vec![], IrType::Void);
        let _ = params;

        // Check if already initialized (task array pointer is non-zero)
        let tasks_global = self.builder.global_ref("__rt_tasks");
        let current_tasks = self.builder.load(tasks_global);
        let zero = self.builder.const_int(0);
        let is_initialized = self.builder.icmp(CmpOp::Ne, current_tasks, zero);

        let init_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // Skip initialization if already done
        self.builder.cond_br(is_initialized, done_block, init_block);

        // Perform initialization
        self.builder.start_block(init_block);

        // Allocate initial task array (64 tasks)
        let initial_cap = self.builder.const_int(64);
        let ptr_size = self.builder.const_int(8); // sizeof(pointer) = 8
        let alloc_size = self.builder.mul(initial_cap, ptr_size);
        let tasks_ptr_raw = self.builder.call("malloc", vec![alloc_size]);
        let tasks_ptr = self.builder.ptrtoint(tasks_ptr_raw, IrType::I64);

        // Store in globals
        self.builder.store(tasks_global, tasks_ptr);

        let count_global = self.builder.global_ref("__rt_task_count");
        self.builder.store(count_global, zero);

        let cap_global = self.builder.global_ref("__rt_task_cap");
        self.builder.store(cap_global, initial_cap);

        // Initialize the reactor (Phase 4: Async I/O)
        self.builder.call("__rt_reactor_init", vec![]);

        // Initialize the timer system (Phase 5: Async Timers)
        self.builder.call("__rt_timer_init", vec![]);

        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_spawn(future_ptr) -> task_id
    fn generate_rt_spawn(&mut self) {
        let params = self.builder.start_function("__rt_spawn", vec![IrType::I64], IrType::I64);
        let future_ptr = params[0];

        // Load current count and capacity
        let count_global = self.builder.global_ref("__rt_task_count");
        let count = self.builder.load(count_global);

        let cap_global = self.builder.global_ref("__rt_task_cap");
        let cap = self.builder.load(cap_global);

        // Check if we need to grow (count >= cap)
        let need_grow_block = self.builder.create_block();
        let store_block = self.builder.create_block();

        let need_grow = self.builder.icmp(CmpOp::Sge, count, cap);
        self.builder.cond_br(need_grow, need_grow_block, store_block);

        // Grow the array (handles uninitialized case where cap=0)
        self.builder.start_block(need_grow_block);

        // new_cap = max(cap * 2, 8) to handle cap=0 case
        let two = self.builder.const_int(2);
        let doubled_cap = self.builder.mul(cap, two);
        let min_cap = self.builder.const_int(8);

        // If doubled_cap < 8, use 8; otherwise use doubled_cap
        let use_min_block = self.builder.create_block();
        let use_doubled_block = self.builder.create_block();
        let continue_grow_block = self.builder.create_block();

        let is_less = self.builder.icmp(CmpOp::Slt, doubled_cap, min_cap);
        self.builder.cond_br(is_less, use_min_block, use_doubled_block);

        // Use minimum capacity (8)
        self.builder.start_block(use_min_block);
        self.builder.br(continue_grow_block);

        // Use doubled capacity
        self.builder.start_block(use_doubled_block);
        self.builder.br(continue_grow_block);

        // Continue with the chosen capacity
        self.builder.start_block(continue_grow_block);
        // PHI node: new_cap = min_cap if is_less else doubled_cap
        let new_cap = self.builder.phi(vec![(min_cap, use_min_block), (doubled_cap, use_doubled_block)]);

        let ptr_size = self.builder.const_int(8);
        let new_size = self.builder.mul(new_cap, ptr_size);

        let tasks_global = self.builder.global_ref("__rt_tasks");
        let old_tasks = self.builder.load(tasks_global);
        let old_tasks_ptr = self.builder.inttoptr(old_tasks, IrType::Ptr(Box::new(IrType::I8)));
        let new_tasks = self.builder.call("realloc", vec![old_tasks_ptr, new_size]);
        let new_tasks_i64 = self.builder.ptrtoint(new_tasks, IrType::I64);

        // Store new pointer and capacity
        let tasks_global2 = self.builder.global_ref("__rt_tasks");
        self.builder.store(tasks_global2, new_tasks_i64);

        let cap_global2 = self.builder.global_ref("__rt_task_cap");
        self.builder.store(cap_global2, new_cap);

        self.builder.br(store_block);

        // Store the future pointer
        self.builder.start_block(store_block);
        let tasks_global3 = self.builder.global_ref("__rt_tasks");
        let tasks_ptr = self.builder.load(tasks_global3);

        let count_global2 = self.builder.global_ref("__rt_task_count");
        let current_count = self.builder.load(count_global2);

        // Calculate offset: tasks_ptr + count * 8
        let ptr_size2 = self.builder.const_int(8);
        let offset = self.builder.mul(current_count, ptr_size2);
        let slot_ptr = self.builder.add(tasks_ptr, offset);
        let slot_ptr_typed = self.builder.inttoptr(slot_ptr, IrType::ptr(IrType::I64));
        self.builder.store(slot_ptr_typed, future_ptr);

        // Increment count
        let one = self.builder.const_int(1);
        let new_count = self.builder.add(current_count, one);
        self.builder.store(count_global2, new_count);

        // Return task_id (which is the old count)
        self.builder.ret(Some(current_count));
        self.builder.finish_function();
    }

    /// Generate __rt_run_tasks() - Run all tasks until all are Ready
    /// This implements a simple event loop that keeps polling tasks
    /// until there are no more Pending tasks
    fn generate_rt_run_tasks(&mut self) {
        let params = self.builder.start_function("__rt_run_tasks", vec![], IrType::Void);
        let _ = params;

        // Outer loop: keep running while there are pending tasks
        let outer_loop_start = self.builder.create_block();
        let inner_loop_start = self.builder.create_block();
        let inner_loop_body = self.builder.create_block();
        let inner_loop_end = self.builder.create_block();
        let outer_loop_end = self.builder.create_block();

        // progress = false (track if any task made progress this iteration)
        let progress_slot = self.builder.alloca(IrType::I64);
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        self.builder.br(outer_loop_start);

        // Outer loop: reset progress flag and run inner loop
        self.builder.start_block(outer_loop_start);
        self.builder.store(progress_slot, zero);

        // i = 0
        let i_slot = self.builder.alloca(IrType::I64);
        self.builder.store(i_slot, zero);
        self.builder.br(inner_loop_start);

        // Inner loop condition: i < count
        self.builder.start_block(inner_loop_start);
        let i = self.builder.load(i_slot);

        let count_global = self.builder.global_ref("__rt_task_count");
        let count = self.builder.load(count_global);

        let cond = self.builder.icmp(CmpOp::Slt, i, count);
        self.builder.cond_br(cond, inner_loop_body, inner_loop_end);

        // Inner loop body: poll task[i]
        self.builder.start_block(inner_loop_body);

        let tasks_global = self.builder.global_ref("__rt_tasks");
        let tasks_ptr = self.builder.load(tasks_global);

        let i_val = self.builder.load(i_slot);
        let ptr_size = self.builder.const_int(8);
        let offset = self.builder.mul(i_val, ptr_size);
        let slot_addr = self.builder.add(tasks_ptr, offset);
        let slot_ptr = self.builder.inttoptr(slot_addr, IrType::ptr(IrType::I64));
        let future_ptr = self.builder.load(slot_ptr);

        // Call __rt_poll_future(future_ptr) - returns 0 if Ready, 1 if Pending
        let poll_result = self.builder.call("__rt_poll_future", vec![future_ptr]);

        // If poll returned Ready (0), we made progress
        let is_ready = self.builder.icmp(CmpOp::Eq, poll_result, zero);
        let made_progress_block = self.builder.create_block();
        let continue_block = self.builder.create_block();

        self.builder.cond_br(is_ready, made_progress_block, continue_block);

        // Made progress - set flag
        self.builder.start_block(made_progress_block);
        self.builder.store(progress_slot, one);
        self.builder.br(continue_block);

        // Continue: i++
        self.builder.start_block(continue_block);
        let i_plus_one = self.builder.add(i_val, one);
        self.builder.store(i_slot, i_plus_one);
        self.builder.br(inner_loop_start);

        // Inner loop end - check if we should continue outer loop
        self.builder.start_block(inner_loop_end);

        // Check if there are still pending tasks
        let pending_global = self.builder.global_ref("__rt_pending_count");
        let pending = self.builder.load(pending_global);
        let has_pending = self.builder.icmp(CmpOp::Sgt, pending, zero);

        let progress = self.builder.load(progress_slot);
        let made_progress = self.builder.icmp(CmpOp::Ne, progress, zero);

        // If we made progress, continue immediately
        let check_reactor_block = self.builder.create_block();
        let after_reactor_block = self.builder.create_block();

        self.builder.cond_br(made_progress, outer_loop_start, check_reactor_block);

        // No direct progress - poll the reactor for I/O events (Phase 4)
        self.builder.start_block(check_reactor_block);

        // Only poll if there are pending tasks
        let skip_poll_block = self.builder.create_block();
        self.builder.cond_br(has_pending, after_reactor_block, skip_poll_block);

        self.builder.start_block(after_reactor_block);

        // Phase 5: Process expired timers first
        let timer_wakeups = self.builder.call("__rt_timer_process", vec![]);

        // Check if timers made progress
        let zero_timers = self.builder.const_int(0);
        let timers_made_progress = self.builder.icmp(CmpOp::Sgt, timer_wakeups, zero_timers);

        let poll_reactor_block = self.builder.create_block();
        let check_combined_progress = self.builder.create_block();

        // If timers woke tasks, continue loop immediately
        self.builder.cond_br(timers_made_progress, outer_loop_start, poll_reactor_block);

        // Poll reactor with timeout based on next timer deadline
        self.builder.start_block(poll_reactor_block);

        // Get next timer deadline
        let next_deadline = self.builder.call("__rt_timer_next_deadline", vec![]);

        // Calculate timeout: if next_deadline == -1, use 0 (non-blocking)
        // else: timeout = max(0, min(next_deadline - now, 10000))
        let neg_one = self.builder.const_int(-1);
        let has_timer = self.builder.icmp(CmpOp::Ne, next_deadline, neg_one);

        // Get current time
        let now = self.builder.call("__rt_time_now", vec![]);

        // Calculate timeout in ms: next_deadline - now
        let timeout_ms = self.builder.sub(next_deadline, now);

        // Clamp to 0 if negative (deadline already passed)
        let zero_i64 = self.builder.const_int(0);
        let timeout_is_neg = self.builder.icmp(CmpOp::Slt, timeout_ms, zero_i64);
        let timeout_clamped = self.builder.select(timeout_is_neg, zero_i64, timeout_ms);

        // Cap at 10 seconds max
        let max_timeout = self.builder.const_int(10000);
        let timeout_is_big = self.builder.icmp(CmpOp::Sgt, timeout_clamped, max_timeout);
        let timeout_capped = self.builder.select(timeout_is_big, max_timeout, timeout_clamped);

        // If no timer, use 0 timeout (non-blocking)
        let zero_timeout = self.builder.const_int(0);
        let final_timeout = self.builder.select(has_timer, timeout_capped, zero_timeout);

        // Truncate to i32 for epoll_wait
        let timeout_i32 = self.builder.trunc(final_timeout, IrType::I32);

        let reactor_events = self.builder.call("__rt_reactor_poll", vec![timeout_i32]);

        // Process timers again after epoll_wait returns (some may have expired during wait)
        let timer_wakeups2 = self.builder.call("__rt_timer_process", vec![]);

        // Check combined progress
        self.builder.br(check_combined_progress);
        self.builder.start_block(check_combined_progress);

        // If reactor or timers woke up any tasks, continue the loop
        let zero_events = self.builder.const_int(0);
        let reactor_made_progress = self.builder.icmp(CmpOp::Sgt, reactor_events, zero_events);
        let timers_made_progress2 = self.builder.icmp(CmpOp::Sgt, timer_wakeups2, zero_events);
        let any_progress = self.builder.or(reactor_made_progress, timers_made_progress2);
        self.builder.cond_br(any_progress, outer_loop_start, outer_loop_end);

        // Skip poll - no pending tasks
        self.builder.start_block(skip_poll_block);
        self.builder.br(outer_loop_end);

        // Outer loop end
        self.builder.start_block(outer_loop_end);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_poll_future(future_ptr) -> i64
    /// Poll a single future. Returns:
    ///   0 = Ready (completed)
    ///   1 = Pending (needs to be polled again later)
    fn generate_rt_poll_future(&mut self) {
        let params = self.builder.start_function("__rt_poll_future", vec![IrType::I64], IrType::I64);
        let future_ptr_i64 = params[0];

        // Convert to pointer
        let future_ptr = self.builder.inttoptr(future_ptr_i64, IrType::ptr(IrType::I64));

        // Check state
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        let state = self.builder.load(state_ptr);

        let not_started_block = self.builder.create_block();
        let pending_block = self.builder.create_block();
        let ready_block = self.builder.create_block();

        // state == 0 -> not started, execute
        // state == 1 -> already ready
        // state == 2 -> pending
        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);
        let two = self.builder.const_int(2);

        let is_not_started = self.builder.icmp(CmpOp::Eq, state, zero);
        self.builder.cond_br(is_not_started, not_started_block, pending_block);

        // Check if pending or ready
        self.builder.start_block(pending_block);
        let is_pending = self.builder.icmp(CmpOp::Eq, state, two);

        let return_pending_block = self.builder.create_block();
        self.builder.cond_br(is_pending, return_pending_block, ready_block);

        // Pending: return 1
        self.builder.start_block(return_pending_block);
        self.builder.ret(Some(one));

        // Not started: execute the closure
        self.builder.start_block(not_started_block);

        // Load fn_ptr and env_ptr
        let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
        let fn_ptr_i64 = self.builder.load(fn_ptr_field);

        let env_ptr_field = self.builder.get_field_ptr(future_ptr, 2);
        let env_ptr_i64 = self.builder.load(env_ptr_field);

        // Call the closure - it returns a packed value:
        // High bit (bit 63) = 1 if Pending, 0 if Ready
        // Lower 63 bits = value (if Ready)
        let call_result = self.builder.call_ptr(fn_ptr_i64, vec![env_ptr_i64]);

        // Check if result indicates Pending (high bit set)
        // For simplicity in Phase 3, we use a convention:
        // If the async body returns i64::MIN (0x8000_0000_0000_0000), it means Pending
        // Otherwise, the value is stored and state is set to Ready
        let pending_marker = self.builder.const_int(i64::MIN);
        let is_result_pending = self.builder.icmp(CmpOp::Eq, call_result, pending_marker);

        let store_ready_block = self.builder.create_block();
        let set_pending_block = self.builder.create_block();

        self.builder.cond_br(is_result_pending, set_pending_block, store_ready_block);

        // Set to pending state
        self.builder.start_block(set_pending_block);
        self.builder.store(state_ptr, two);
        // Increment pending count
        let pending_count_global = self.builder.global_ref("__rt_pending_count");
        let pending_count = self.builder.load(pending_count_global);
        let new_pending_count = self.builder.add(pending_count, one);
        self.builder.store(pending_count_global, new_pending_count);
        // Return 1 (Pending)
        self.builder.ret(Some(one));

        // Store result and set state to Ready
        self.builder.start_block(store_ready_block);
        let value_ptr = self.builder.get_field_ptr(future_ptr, 3);
        self.builder.store(value_ptr, call_result);
        self.builder.store(state_ptr, one);
        self.builder.br(ready_block);

        // Ready block - return 0
        self.builder.start_block(ready_block);
        self.builder.ret(Some(zero));
        self.builder.finish_function();
    }

    /// Generate __rt_wake(future_ptr) - Wake a pending future
    /// For regular futures: sets state back to NotStarted so it will be re-polled
    /// For sleep futures (fn_ptr == -1): sets state directly to Ready
    fn generate_rt_wake(&mut self) {
        let params = self.builder.start_function("__rt_wake", vec![IrType::I64], IrType::Void);
        let future_ptr_i64 = params[0];

        // Convert to pointer
        let future_ptr = self.builder.inttoptr(future_ptr_i64, IrType::ptr(IrType::I64));

        // Get state pointer
        let state_ptr = self.builder.get_field_ptr(future_ptr, 0);
        let state = self.builder.load(state_ptr);

        // Only wake if currently Pending (state == 2)
        let two = self.builder.const_int(2);
        let is_pending = self.builder.icmp(CmpOp::Eq, state, two);

        let wake_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_pending, wake_block, done_block);

        // Wake: check if this is a sleep future (fn_ptr == -1)
        self.builder.start_block(wake_block);

        let fn_ptr_field = self.builder.get_field_ptr(future_ptr, 1);
        let fn_ptr = self.builder.load(fn_ptr_field);
        let neg_one = self.builder.const_int(-1);
        let is_sleep = self.builder.icmp(CmpOp::Eq, fn_ptr, neg_one);

        let sleep_wake_block = self.builder.create_block();
        let regular_wake_block = self.builder.create_block();

        self.builder.cond_br(is_sleep, sleep_wake_block, regular_wake_block);

        // Sleep future: set state directly to Ready (1)
        self.builder.start_block(sleep_wake_block);
        let one = self.builder.const_int(1);
        self.builder.store(state_ptr, one);
        self.builder.br(done_block);

        // Regular future: set state back to NotStarted (0) so it will be re-polled
        self.builder.start_block(regular_wake_block);
        let zero = self.builder.const_int(0);
        self.builder.store(state_ptr, zero);

        // Decrement pending count
        let pending_count_global = self.builder.global_ref("__rt_pending_count");
        let pending_count = self.builder.load(pending_count_global);
        let one2 = self.builder.const_int(1);
        let new_pending_count = self.builder.sub(pending_count, one2);
        self.builder.store(pending_count_global, new_pending_count);

        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    // ============ Channel Runtime (Phase 6) ============
    //
    // Channel structure layout (all fields are i64):
    //   [0] buffer: pointer to circular buffer of values
    //   [1] capacity: maximum number of elements
    //   [2] count: current number of elements in buffer
    //   [3] head: read index (where to recv from)
    //   [4] tail: write index (where to send to)
    //   [5] senders_waiting: pointer to array of future pointers waiting to send
    //   [6] senders_count: number of senders waiting
    //   [7] receivers_waiting: pointer to array of future pointers waiting to recv
    //   [8] receivers_count: number of receivers waiting
    //   [9] sender_ref_count: number of Sender handles
    //   [10] receiver_ref_count: number of Receiver handles
    //
    // Sender structure: just wraps the channel pointer
    //   [0] channel_ptr: pointer to channel
    //
    // Receiver structure: just wraps the channel pointer
    //   [0] channel_ptr: pointer to channel

    /// Generate channel runtime support functions
    fn generate_channel_runtime(&mut self) {
        // Generate __rt_channel_create
        self.generate_rt_channel_create();

        // Generate __rt_channel_try_send
        self.generate_rt_channel_try_send();

        // Generate __rt_channel_try_recv
        self.generate_rt_channel_try_recv();

        // Generate __rt_channel_send (async version)
        self.generate_rt_channel_send();

        // Generate __rt_channel_recv (async version)
        self.generate_rt_channel_recv();

        // Generate __rt_channel_wake_sender
        self.generate_rt_channel_wake_sender();

        // Generate __rt_channel_wake_receiver
        self.generate_rt_channel_wake_receiver();

        // Generate __rt_channel_is_closed
        self.generate_rt_channel_is_closed();
    }

    /// Generate __rt_channel_create(capacity) -> channel_ptr
    /// Creates a new channel with the given capacity and returns pointer to it (as i64)
    fn generate_rt_channel_create(&mut self) {
        let params = self.builder.start_function("__rt_channel_create", vec![IrType::I64], IrType::I64);
        let capacity = params[0];

        // Channel layout offsets (in bytes):
        // [0] buffer, [8] capacity, [16] count, [24] head, [32] tail,
        // [40] senders_waiting, [48] senders_count, [56] receivers_waiting,
        // [64] receivers_count, [72] sender_ref_count, [80] receiver_ref_count

        // Allocate channel structure (11 fields * 8 bytes = 88 bytes)
        let channel_size = self.builder.const_int(88);
        let channel_ptr_raw = self.builder.call("malloc", vec![channel_size]);
        let channel_ptr = self.builder.ptrtoint(channel_ptr_raw, IrType::I64);

        // Allocate buffer: capacity * 8 bytes
        let elem_size = self.builder.const_int(8);
        let buffer_size = self.builder.mul(capacity, elem_size);
        let buffer_ptr_raw = self.builder.call("malloc", vec![buffer_size]);
        let buffer_ptr = self.builder.ptrtoint(buffer_ptr_raw, IrType::I64);

        // Allocate waiter arrays (initial capacity = 8 pointers each)
        let waiter_cap = self.builder.const_int(64);
        let senders_waiting_raw = self.builder.call("malloc", vec![waiter_cap]);
        let senders_waiting = self.builder.ptrtoint(senders_waiting_raw, IrType::I64);
        let receivers_waiting_raw = self.builder.call("malloc", vec![waiter_cap]);
        let receivers_waiting = self.builder.ptrtoint(receivers_waiting_raw, IrType::I64);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        // Helper to store i64 at channel_ptr + offset
        // Store buffer pointer [offset 0]
        let ptr0 = self.builder.inttoptr(channel_ptr, IrType::ptr(IrType::I64));
        self.builder.store(ptr0, buffer_ptr);

        // Store capacity [offset 8]
        let off8 = self.builder.const_int(8);
        let addr8 = self.builder.add(channel_ptr, off8);
        let ptr8 = self.builder.inttoptr(addr8, IrType::ptr(IrType::I64));
        self.builder.store(ptr8, capacity);

        // Store count = 0 [offset 16]
        let off16 = self.builder.const_int(16);
        let addr16 = self.builder.add(channel_ptr, off16);
        let ptr16 = self.builder.inttoptr(addr16, IrType::ptr(IrType::I64));
        self.builder.store(ptr16, zero);

        // Store head = 0 [offset 24]
        let off24 = self.builder.const_int(24);
        let addr24 = self.builder.add(channel_ptr, off24);
        let ptr24 = self.builder.inttoptr(addr24, IrType::ptr(IrType::I64));
        self.builder.store(ptr24, zero);

        // Store tail = 0 [offset 32]
        let off32 = self.builder.const_int(32);
        let addr32 = self.builder.add(channel_ptr, off32);
        let ptr32 = self.builder.inttoptr(addr32, IrType::ptr(IrType::I64));
        self.builder.store(ptr32, zero);

        // Store senders_waiting pointer [offset 40]
        let off40 = self.builder.const_int(40);
        let addr40 = self.builder.add(channel_ptr, off40);
        let ptr40 = self.builder.inttoptr(addr40, IrType::ptr(IrType::I64));
        self.builder.store(ptr40, senders_waiting);

        // Store senders_count = 0 [offset 48]
        let off48 = self.builder.const_int(48);
        let addr48 = self.builder.add(channel_ptr, off48);
        let ptr48 = self.builder.inttoptr(addr48, IrType::ptr(IrType::I64));
        self.builder.store(ptr48, zero);

        // Store receivers_waiting pointer [offset 56]
        let off56 = self.builder.const_int(56);
        let addr56 = self.builder.add(channel_ptr, off56);
        let ptr56 = self.builder.inttoptr(addr56, IrType::ptr(IrType::I64));
        self.builder.store(ptr56, receivers_waiting);

        // Store receivers_count = 0 [offset 64]
        let off64 = self.builder.const_int(64);
        let addr64 = self.builder.add(channel_ptr, off64);
        let ptr64 = self.builder.inttoptr(addr64, IrType::ptr(IrType::I64));
        self.builder.store(ptr64, zero);

        // Store sender_ref_count = 1 [offset 72]
        let off72 = self.builder.const_int(72);
        let addr72 = self.builder.add(channel_ptr, off72);
        let ptr72 = self.builder.inttoptr(addr72, IrType::ptr(IrType::I64));
        self.builder.store(ptr72, one);

        // Store receiver_ref_count = 1 [offset 80]
        let off80 = self.builder.const_int(80);
        let addr80 = self.builder.add(channel_ptr, off80);
        let ptr80 = self.builder.inttoptr(addr80, IrType::ptr(IrType::I64));
        self.builder.store(ptr80, one);

        self.builder.ret(Some(channel_ptr));
        self.builder.finish_function();
    }

    /// Generate __rt_channel_try_send(channel_ptr, value) -> i64
    /// Attempts to send without blocking.
    /// Returns: 1 = success, 0 = buffer full, -1 = closed
    fn generate_rt_channel_try_send(&mut self) {
        let params = self.builder.start_function("__rt_channel_try_send", vec![IrType::I64, IrType::I64], IrType::I64);
        let channel_ptr_i64 = params[0];
        let value = params[1];

        let eight = self.builder.const_int(8);

        // Channel layout offsets (in bytes):
        // [0] buffer, [8] capacity, [16] count, [24] head, [32] tail,
        // [40] senders_waiting, [48] senders_count, [56] receivers_waiting,
        // [64] receivers_count, [72] sender_ref_count, [80] receiver_ref_count

        // Check if receiver is closed (receiver_ref_count == 0) at offset 80
        let rrc_offset = self.builder.const_int(80);
        let rrc_addr = self.builder.add(channel_ptr_i64, rrc_offset);
        let rrc_ptr = self.builder.inttoptr(rrc_addr, IrType::ptr(IrType::I64));
        let rrc = self.builder.load(rrc_ptr);
        let zero = self.builder.const_int(0);
        let is_closed = self.builder.icmp(CmpOp::Eq, rrc, zero);

        let closed_block = self.builder.create_block();
        let check_full_block = self.builder.create_block();

        self.builder.cond_br(is_closed, closed_block, check_full_block);

        // Return -1 (closed)
        self.builder.start_block(closed_block);
        let minus_one = self.builder.const_int(-1);
        self.builder.ret(Some(minus_one));

        // Check if buffer is full
        self.builder.start_block(check_full_block);

        // Load count (offset 16)
        let count_offset = self.builder.const_int(16);
        let count_addr = self.builder.add(channel_ptr_i64, count_offset);
        let count_ptr = self.builder.inttoptr(count_addr, IrType::ptr(IrType::I64));
        let count = self.builder.load(count_ptr);

        // Load capacity (offset 8)
        let cap_offset = self.builder.const_int(8);
        let cap_addr = self.builder.add(channel_ptr_i64, cap_offset);
        let cap_ptr = self.builder.inttoptr(cap_addr, IrType::ptr(IrType::I64));
        let cap = self.builder.load(cap_ptr);

        let is_full = self.builder.icmp(CmpOp::Sge, count, cap);

        let full_block = self.builder.create_block();
        let send_block = self.builder.create_block();

        self.builder.cond_br(is_full, full_block, send_block);

        // Return 0 (buffer full)
        self.builder.start_block(full_block);
        self.builder.ret(Some(zero));

        // Perform the send
        self.builder.start_block(send_block);

        // Load tail index (offset 32)
        let tail_offset = self.builder.const_int(32);
        let tail_addr = self.builder.add(channel_ptr_i64, tail_offset);
        let tail_ptr = self.builder.inttoptr(tail_addr, IrType::ptr(IrType::I64));
        let tail = self.builder.load(tail_ptr);

        // Load buffer pointer (offset 0)
        let buffer_ptr = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));
        let buffer = self.builder.load(buffer_ptr);

        // buffer[tail] = value
        let elem_offset = self.builder.mul(tail, eight);
        let slot_addr = self.builder.add(buffer, elem_offset);
        let slot_ptr = self.builder.inttoptr(slot_addr, IrType::ptr(IrType::I64));
        self.builder.store(slot_ptr, value);

        // tail = (tail + 1) % capacity
        let one = self.builder.const_int(1);
        let new_tail = self.builder.add(tail, one);
        let wrapped_tail = self.builder.srem(new_tail, cap);
        self.builder.store(tail_ptr, wrapped_tail);

        // count++
        let new_count = self.builder.add(count, one);
        self.builder.store(count_ptr, new_count);

        // Wake a waiting receiver if any
        self.builder.call("__rt_channel_wake_receiver", vec![channel_ptr_i64]);

        // Return 1 (success)
        self.builder.ret(Some(one));
        self.builder.finish_function();
    }

    /// Generate __rt_channel_try_recv(channel_ptr) -> i64
    /// Attempts to receive without blocking.
    /// Returns: value if Some, i64::MIN if None (empty)
    fn generate_rt_channel_try_recv(&mut self) {
        let params = self.builder.start_function("__rt_channel_try_recv", vec![IrType::I64], IrType::I64);
        let channel_ptr_i64 = params[0];

        let eight = self.builder.const_int(8);

        // Channel layout offsets (in bytes, each field is i64 = 8 bytes):
        // [0*8] buffer, [1*8] capacity, [2*8] count, [3*8] head, [4*8] tail, ...

        // Load count (field 2, offset 16)
        let count_offset = self.builder.const_int(16);
        let count_addr = self.builder.add(channel_ptr_i64, count_offset);
        let count_ptr = self.builder.inttoptr(count_addr, IrType::ptr(IrType::I64));
        let count = self.builder.load(count_ptr);

        // Check if buffer is empty
        let zero = self.builder.const_int(0);
        let is_empty = self.builder.icmp(CmpOp::Eq, count, zero);

        let empty_block = self.builder.create_block();
        let recv_block = self.builder.create_block();

        self.builder.cond_br(is_empty, empty_block, recv_block);

        // Buffer is empty - return None marker
        self.builder.start_block(empty_block);
        let none_marker = self.builder.const_int(i64::MIN);
        self.builder.ret(Some(none_marker));

        // Perform the receive
        self.builder.start_block(recv_block);

        // Load buffer pointer (field 0, offset 0)
        let buffer_ptr = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));
        let buffer = self.builder.load(buffer_ptr);

        // Load head index (field 3, offset 24)
        let head_offset = self.builder.const_int(24);
        let head_addr = self.builder.add(channel_ptr_i64, head_offset);
        let head_ptr = self.builder.inttoptr(head_addr, IrType::ptr(IrType::I64));
        let head = self.builder.load(head_ptr);

        // value = buffer[head]
        let elem_offset = self.builder.mul(head, eight);
        let slot_addr = self.builder.add(buffer, elem_offset);
        let slot_ptr = self.builder.inttoptr(slot_addr, IrType::ptr(IrType::I64));
        let value = self.builder.load(slot_ptr);

        // Load capacity (field 1, offset 8)
        let cap_offset = self.builder.const_int(8);
        let cap_addr = self.builder.add(channel_ptr_i64, cap_offset);
        let cap_ptr = self.builder.inttoptr(cap_addr, IrType::ptr(IrType::I64));
        let cap = self.builder.load(cap_ptr);

        // head = (head + 1) % capacity
        let one = self.builder.const_int(1);
        let new_head = self.builder.add(head, one);
        let wrapped_head = self.builder.srem(new_head, cap);
        self.builder.store(head_ptr, wrapped_head);

        // count--
        let new_count = self.builder.sub(count, one);
        self.builder.store(count_ptr, new_count);

        // Wake a waiting sender if any
        self.builder.call("__rt_channel_wake_sender", vec![channel_ptr_i64]);

        // Return the value
        self.builder.ret(Some(value));
        self.builder.finish_function();
    }

    /// Generate __rt_channel_send(channel_ptr, value, future_ptr) -> i64
    /// Async send - registers waiter if buffer is full.
    /// Returns: 1 = sent immediately, 0 = pending (added to waiters), -1 = closed
    fn generate_rt_channel_send(&mut self) {
        let params = self.builder.start_function("__rt_channel_send", vec![IrType::I64, IrType::I64, IrType::I64], IrType::I64);
        let channel_ptr_i64 = params[0];
        let value = params[1];
        let future_ptr = params[2];

        // Try non-blocking send first
        let try_result = self.builder.call("__rt_channel_try_send", vec![channel_ptr_i64, value]);

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        // Check if it was full (result == 0)
        let is_full = self.builder.icmp(CmpOp::Eq, try_result, zero);

        let add_waiter_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_full, add_waiter_block, done_block);

        // Add this future to senders_waiting list
        self.builder.start_block(add_waiter_block);

        let channel = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));

        // Get senders_waiting array and count
        let sw_field = self.builder.get_field_ptr(channel, 5);
        let sw_ptr = self.builder.load(sw_field);
        let sc_field = self.builder.get_field_ptr(channel, 6);
        let sc = self.builder.load(sc_field);

        // Store future_ptr at senders_waiting[senders_count]
        let elem_size = self.builder.const_int(8);
        let offset = self.builder.mul(sc, elem_size);
        let slot_addr = self.builder.add(sw_ptr, offset);
        let slot_ptr = self.builder.inttoptr(slot_addr, IrType::ptr(IrType::I64));
        self.builder.store(slot_ptr, future_ptr);

        // senders_count++
        let new_sc = self.builder.add(sc, one);
        self.builder.store(sc_field, new_sc);

        // Set future state to Pending (2)
        let future = self.builder.inttoptr(future_ptr, IrType::ptr(IrType::I64));
        let state_ptr = self.builder.get_field_ptr(future, 0);
        let two = self.builder.const_int(2);
        self.builder.store(state_ptr, two);

        // Increment pending count
        let pending_global = self.builder.global_ref("__rt_pending_count");
        let pending = self.builder.load(pending_global);
        let new_pending = self.builder.add(pending, one);
        self.builder.store(pending_global, new_pending);

        // Return 0 (pending)
        self.builder.ret(Some(zero));

        // Done - return the try_send result
        self.builder.start_block(done_block);
        self.builder.ret(Some(try_result));

        self.builder.finish_function();
    }

    /// Generate __rt_channel_recv(channel_ptr, future_ptr) -> i64
    /// Async receive - registers waiter if buffer is empty.
    /// Returns: value if available, i64::MIN if pending or closed
    fn generate_rt_channel_recv(&mut self) {
        let params = self.builder.start_function("__rt_channel_recv", vec![IrType::I64, IrType::I64], IrType::I64);
        let channel_ptr_i64 = params[0];
        let future_ptr = params[1];

        // Try non-blocking receive first
        let try_result = self.builder.call("__rt_channel_try_recv", vec![channel_ptr_i64]);

        // Check if it was empty (result == i64::MIN)
        let none_marker = self.builder.const_int(i64::MIN);
        let is_empty = self.builder.icmp(CmpOp::Eq, try_result, none_marker);

        let check_closed_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_empty, check_closed_block, done_block);

        // Check if channel is closed (all senders dropped)
        self.builder.start_block(check_closed_block);

        let channel = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));
        let src_field = self.builder.get_field_ptr(channel, 9);
        let src = self.builder.load(src_field);
        let zero = self.builder.const_int(0);
        let is_closed = self.builder.icmp(CmpOp::Eq, src, zero);

        let closed_block = self.builder.create_block();
        let add_waiter_block = self.builder.create_block();

        self.builder.cond_br(is_closed, closed_block, add_waiter_block);

        // Closed and empty - return None marker
        self.builder.start_block(closed_block);
        self.builder.ret(Some(none_marker));

        // Add this future to receivers_waiting list
        self.builder.start_block(add_waiter_block);

        // Get receivers_waiting array and count
        let rw_field = self.builder.get_field_ptr(channel, 7);
        let rw_ptr = self.builder.load(rw_field);
        let rc_field = self.builder.get_field_ptr(channel, 8);
        let rc = self.builder.load(rc_field);

        // Store future_ptr at receivers_waiting[receivers_count]
        let one = self.builder.const_int(1);
        let elem_size = self.builder.const_int(8);
        let offset = self.builder.mul(rc, elem_size);
        let slot_addr = self.builder.add(rw_ptr, offset);
        let slot_ptr = self.builder.inttoptr(slot_addr, IrType::ptr(IrType::I64));
        self.builder.store(slot_ptr, future_ptr);

        // receivers_count++
        let new_rc = self.builder.add(rc, one);
        self.builder.store(rc_field, new_rc);

        // Set future state to Pending (2)
        let future = self.builder.inttoptr(future_ptr, IrType::ptr(IrType::I64));
        let state_ptr = self.builder.get_field_ptr(future, 0);
        let two = self.builder.const_int(2);
        self.builder.store(state_ptr, two);

        // Increment pending count
        let pending_global = self.builder.global_ref("__rt_pending_count");
        let pending = self.builder.load(pending_global);
        let new_pending = self.builder.add(pending, one);
        self.builder.store(pending_global, new_pending);

        // Return pending marker
        self.builder.ret(Some(none_marker));

        // Done - return the try_recv result
        self.builder.start_block(done_block);
        self.builder.ret(Some(try_result));

        self.builder.finish_function();
    }

    /// Generate __rt_channel_wake_sender(channel_ptr)
    /// Wakes one sender waiting to send (called when space becomes available)
    fn generate_rt_channel_wake_sender(&mut self) {
        let params = self.builder.start_function("__rt_channel_wake_sender", vec![IrType::I64], IrType::Void);
        let channel_ptr_i64 = params[0];

        let channel = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));

        // Check if there are senders waiting
        let sc_field = self.builder.get_field_ptr(channel, 6);
        let sc = self.builder.load(sc_field);
        let zero = self.builder.const_int(0);
        let has_waiters = self.builder.icmp(CmpOp::Sgt, sc, zero);

        let wake_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(has_waiters, wake_block, done_block);

        // Wake the first sender in queue
        self.builder.start_block(wake_block);

        // Get senders_waiting[0]
        let sw_field = self.builder.get_field_ptr(channel, 5);
        let sw_ptr = self.builder.load(sw_field);
        let sw_typed = self.builder.inttoptr(sw_ptr, IrType::ptr(IrType::I64));
        let future_ptr = self.builder.load(sw_typed);

        // Shift remaining waiters left (simple approach: move all elements)
        // For simplicity, just decrement count and shift array
        let one = self.builder.const_int(1);
        let new_sc = self.builder.sub(sc, one);
        self.builder.store(sc_field, new_sc);

        // TODO: Actually shift the array (for now we just wake first and leave gaps)
        // This is simplified - production code would use proper queue

        // Wake the future using __rt_wake
        self.builder.call("__rt_wake", vec![future_ptr]);

        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_channel_wake_receiver(channel_ptr)
    /// Wakes one receiver waiting to receive (called when data becomes available)
    fn generate_rt_channel_wake_receiver(&mut self) {
        let params = self.builder.start_function("__rt_channel_wake_receiver", vec![IrType::I64], IrType::Void);
        let channel_ptr_i64 = params[0];

        let channel = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));

        // Check if there are receivers waiting
        let rc_field = self.builder.get_field_ptr(channel, 8);
        let rc = self.builder.load(rc_field);
        let zero = self.builder.const_int(0);
        let has_waiters = self.builder.icmp(CmpOp::Sgt, rc, zero);

        let wake_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(has_waiters, wake_block, done_block);

        // Wake the first receiver in queue
        self.builder.start_block(wake_block);

        // Get receivers_waiting[0]
        let rw_field = self.builder.get_field_ptr(channel, 7);
        let rw_ptr = self.builder.load(rw_field);
        let rw_typed = self.builder.inttoptr(rw_ptr, IrType::ptr(IrType::I64));
        let future_ptr = self.builder.load(rw_typed);

        // Decrement receivers_count
        let one = self.builder.const_int(1);
        let new_rc = self.builder.sub(rc, one);
        self.builder.store(rc_field, new_rc);

        // Wake the future using __rt_wake
        self.builder.call("__rt_wake", vec![future_ptr]);

        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __rt_channel_is_closed(channel_ptr, is_sender) -> i64
    /// Returns 1 if channel is closed from perspective of sender/receiver
    fn generate_rt_channel_is_closed(&mut self) {
        let params = self.builder.start_function("__rt_channel_is_closed", vec![IrType::I64, IrType::I64], IrType::I64);
        let channel_ptr_i64 = params[0];
        let is_sender = params[1];

        let channel = self.builder.inttoptr(channel_ptr_i64, IrType::ptr(IrType::I64));

        let zero = self.builder.const_int(0);
        let one = self.builder.const_int(1);

        // If is_sender, check receiver_ref_count; else check sender_ref_count
        let check_sender_block = self.builder.create_block();
        let check_receiver_block = self.builder.create_block();
        let return_closed_block = self.builder.create_block();
        let return_open_block = self.builder.create_block();

        let is_sender_check = self.builder.icmp(CmpOp::Ne, is_sender, zero);
        self.builder.cond_br(is_sender_check, check_sender_block, check_receiver_block);

        // For sender: check if receiver_ref_count == 0
        self.builder.start_block(check_sender_block);
        let rrc_field = self.builder.get_field_ptr(channel, 10);
        let rrc = self.builder.load(rrc_field);
        let is_closed_s = self.builder.icmp(CmpOp::Eq, rrc, zero);
        self.builder.cond_br(is_closed_s, return_closed_block, return_open_block);

        // For receiver: check if sender_ref_count == 0
        self.builder.start_block(check_receiver_block);
        let src_field = self.builder.get_field_ptr(channel, 9);
        let src = self.builder.load(src_field);
        let is_closed_r = self.builder.icmp(CmpOp::Eq, src, zero);
        self.builder.cond_br(is_closed_r, return_closed_block, return_open_block);

        self.builder.start_block(return_closed_block);
        self.builder.ret(Some(one));

        self.builder.start_block(return_open_block);
        self.builder.ret(Some(zero));

        self.builder.finish_function();
    }

    // ============ HARC Drop Functions ============

    /// Generate drop functions for all RC types
    fn generate_drop_functions(&mut self) {
        self.generate_drop_vec();
        self.generate_drop_string();
        self.generate_drop_hashmap();
        self.generate_drop_hashset();
        self.generate_drop_box();
        self.generate_drop_file();
        self.generate_drop_generic();
    }

    /// Generate __drop_vec(ptr) - deallocates Vec data buffer
    fn generate_drop_vec(&mut self) {
        let params = self.builder.start_function("__drop_vec", vec![IrType::I64], IrType::Void);
        let vec_ptr_i64 = params[0];

        // Vec struct: { data_ptr: *T, len: i64, cap: i64 }
        let vec_ty = self.vec_struct_type();
        let vec_ptr = self.builder.inttoptr(vec_ptr_i64, IrType::Ptr(Box::new(vec_ty.clone())));
        self.vreg_types.insert(vec_ptr, vec_ty);

        // Get data_ptr field
        let data_field = self.builder.get_field_ptr(vec_ptr, 0);
        let data_ptr = self.builder.load(data_field);

        // Free the data buffer if not null
        // Convert pointer to i64 for comparison since icmp expects integers
        let data_ptr_i64 = self.builder.ptrtoint(data_ptr, IrType::I64);
        let zero = self.builder.const_int(0);
        let is_null = self.builder.icmp(CmpOp::Eq, data_ptr_i64, zero);

        let free_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_null, done_block, free_block);

        self.builder.start_block(free_block);
        // data_ptr is already a pointer, use it directly
        self.builder.free(data_ptr);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __drop_string(ptr) - deallocates String data buffer
    fn generate_drop_string(&mut self) {
        let params = self.builder.start_function("__drop_string", vec![IrType::I64], IrType::Void);
        let str_ptr_i64 = params[0];

        // String struct: { data_ptr: *u8, len: i64, cap: i64 }
        let str_ty = self.string_struct_type();
        let str_ptr = self.builder.inttoptr(str_ptr_i64, IrType::Ptr(Box::new(str_ty.clone())));
        self.vreg_types.insert(str_ptr, str_ty);

        // Get cap field - only free if cap > 0 (cap=0 means literal/non-allocated)
        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        let cap = self.builder.load(cap_field);

        let zero = self.builder.const_int(0);
        let has_capacity = self.builder.icmp(CmpOp::Sgt, cap, zero);

        let check_ptr_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        // If cap <= 0, skip to done (no allocated memory to free)
        self.builder.cond_br(has_capacity, check_ptr_block, done_block);

        // Check if pointer is not null
        self.builder.start_block(check_ptr_block);
        let data_field = self.builder.get_field_ptr(str_ptr, 0);
        let data_ptr = self.builder.load(data_field);
        // Convert pointer to i64 for comparison since icmp expects integers
        let data_ptr_i64 = self.builder.ptrtoint(data_ptr, IrType::I64);
        let is_null = self.builder.icmp(CmpOp::Eq, data_ptr_i64, zero);

        let free_block = self.builder.create_block();
        self.builder.cond_br(is_null, done_block, free_block);

        self.builder.start_block(free_block);
        // data_ptr is already a pointer, use it directly
        self.builder.free(data_ptr);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __drop_hashmap(ptr) - deallocates HashMap entries array
    fn generate_drop_hashmap(&mut self) {
        let params = self.builder.start_function("__drop_hashmap", vec![IrType::I64], IrType::Void);
        let map_ptr_i64 = params[0];

        // HashMap struct: { entries_ptr: *Entry, count: i64, capacity: i64 }
        let map_ty = self.hashmap_struct_type();
        let map_ptr = self.builder.inttoptr(map_ptr_i64, IrType::Ptr(Box::new(map_ty.clone())));
        self.vreg_types.insert(map_ptr, map_ty);

        // Get entries_ptr field
        let entries_field = self.builder.get_field_ptr(map_ptr, 0);
        let entries_ptr = self.builder.load(entries_field);

        // Free the entries array if not null
        // Convert pointer to i64 for comparison since icmp expects integers
        let entries_ptr_i64 = self.builder.ptrtoint(entries_ptr, IrType::I64);
        let zero = self.builder.const_int(0);
        let is_null = self.builder.icmp(CmpOp::Eq, entries_ptr_i64, zero);

        let free_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_null, done_block, free_block);

        self.builder.start_block(free_block);
        // entries_ptr is already a pointer, use it directly
        self.builder.free(entries_ptr);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __drop_hashset(ptr) - deallocates HashSet entries array
    fn generate_drop_hashset(&mut self) {
        let params = self.builder.start_function("__drop_hashset", vec![IrType::I64], IrType::Void);
        let set_ptr_i64 = params[0];

        // HashSet struct: { entries_ptr: *Entry, count: i64, capacity: i64 }
        let set_ty = self.hashset_struct_type();
        let set_ptr = self.builder.inttoptr(set_ptr_i64, IrType::Ptr(Box::new(set_ty.clone())));
        self.vreg_types.insert(set_ptr, set_ty);

        // Get entries_ptr field
        let entries_field = self.builder.get_field_ptr(set_ptr, 0);
        let entries_ptr = self.builder.load(entries_field);

        // Free the entries array if not null
        // Convert pointer to i64 for comparison since icmp expects integers
        let entries_ptr_i64 = self.builder.ptrtoint(entries_ptr, IrType::I64);
        let zero = self.builder.const_int(0);
        let is_null = self.builder.icmp(CmpOp::Eq, entries_ptr_i64, zero);

        let free_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_null, done_block, free_block);

        self.builder.start_block(free_block);
        // entries_ptr is already a pointer, use it directly
        self.builder.free(entries_ptr);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __drop_box(ptr) - deallocates Box inner value
    fn generate_drop_box(&mut self) {
        let params = self.builder.start_function("__drop_box", vec![IrType::I64], IrType::Void);
        let box_ptr_i64 = params[0];

        // Box is just a pointer to the value - free it directly
        let box_ptr = self.builder.inttoptr(box_ptr_i64, IrType::ptr(IrType::I64));
        self.builder.free(box_ptr);

        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __drop_file(ptr) - closes file handle
    fn generate_drop_file(&mut self) {
        let params = self.builder.start_function("__drop_file", vec![IrType::I64], IrType::Void);
        let file_ptr_i64 = params[0];

        // File struct: { fd: i64, flags: i64 }
        let file_ptr = self.builder.inttoptr(file_ptr_i64, IrType::ptr(IrType::I64));

        // Get fd field
        let fd_field = self.builder.get_field_ptr(file_ptr, 0);
        let fd_i64 = self.builder.load(fd_field);

        // Close the file descriptor if valid (fd >= 0)
        let zero = self.builder.const_int(0);
        let is_valid = self.builder.icmp(CmpOp::Sge, fd_i64, zero);

        let close_block = self.builder.create_block();
        let done_block = self.builder.create_block();

        self.builder.cond_br(is_valid, close_block, done_block);

        self.builder.start_block(close_block);
        // Truncate fd to i32 for close() syscall
        let fd_i32 = self.builder.trunc(fd_i64, IrType::I32);
        self.builder.call("close", vec![fd_i32]);
        self.builder.br(done_block);

        self.builder.start_block(done_block);
        self.builder.ret(None);
        self.builder.finish_function();
    }

    /// Generate __drop_generic(ptr) - fallback for unknown types
    fn generate_drop_generic(&mut self) {
        let _params = self.builder.start_function("__drop_generic", vec![IrType::I64], IrType::Void);
        // Generic drop does nothing - memory will be freed by RC header
        self.builder.ret(None);
        self.builder.finish_function();
    }

    // ============ Built-in Macro Helpers ============

    /// Lower vec![] macro - create an empty or initialized Vec
    fn lower_builtin_vec_macro(&mut self, tokens: &[crate::ast::MacroToken]) -> VReg {
        use crate::ast::MacroToken;
        use crate::token::TokenKind;

        // Call Vec::new() to create the vec
        let vec_ptr = self.builder.call("__vec_new", vec![]);

        // If there are tokens (elements), parse and push them
        if !tokens.is_empty() {
            // Split tokens by comma to get individual elements
            let elements = self.split_macro_tokens_by_comma(tokens);

            for elem_tokens in elements {
                // Try to convert tokens to a value and push it
                if let Some(value) = self.macro_tokens_to_value(&elem_tokens) {
                    // Push the value to the vec
                    self.builder.call("__vec_push_i64", vec![vec_ptr, value]);
                }
            }
        }

        vec_ptr
    }

    /// Split macro tokens by comma separator
    fn split_macro_tokens_by_comma(&self, tokens: &[crate::ast::MacroToken]) -> Vec<Vec<crate::ast::MacroToken>> {
        use crate::ast::MacroToken;
        use crate::token::TokenKind;

        let mut result = vec![];
        let mut current = vec![];
        let mut depth = 0;

        for token in tokens {
            match token {
                MacroToken::Token(TokenKind::Comma, _) if depth == 0 => {
                    if !current.is_empty() {
                        result.push(std::mem::take(&mut current));
                    }
                }
                MacroToken::Token(TokenKind::LParen, _) |
                MacroToken::Token(TokenKind::LBracket, _) |
                MacroToken::Token(TokenKind::LBrace, _) => {
                    depth += 1;
                    current.push(token.clone());
                }
                MacroToken::Token(TokenKind::RParen, _) |
                MacroToken::Token(TokenKind::RBracket, _) |
                MacroToken::Token(TokenKind::RBrace, _) => {
                    depth -= 1;
                    current.push(token.clone());
                }
                MacroToken::Group { .. } => {
                    current.push(token.clone());
                }
                _ => {
                    current.push(token.clone());
                }
            }
        }

        if !current.is_empty() {
            result.push(current);
        }

        result
    }

    /// Convert macro tokens to a VReg value (for simple expressions)
    fn macro_tokens_to_value(&mut self, tokens: &[crate::ast::MacroToken]) -> Option<VReg> {
        use crate::ast::MacroToken;
        use crate::token::TokenKind;

        if tokens.is_empty() {
            return None;
        }

        // Handle single token cases
        if tokens.len() == 1 {
            match &tokens[0] {
                MacroToken::IntLit(n, _) => {
                    return Some(self.builder.const_int(*n));
                }
                MacroToken::FloatLit(f, _) => {
                    return Some(self.builder.const_float(*f));
                }
                MacroToken::Token(TokenKind::True, _) => {
                    return Some(self.builder.const_int(1));
                }
                MacroToken::Token(TokenKind::False, _) => {
                    return Some(self.builder.const_int(0));
                }
                MacroToken::StrLit(s, _) => {
                    // Create a string literal
                    let name = self.builder.add_string_constant(s);
                    let data_ptr = self.builder.global_string_ptr(&name);
                    let str_len = self.builder.const_int(s.len() as i64);
                    let zero = self.builder.const_int(0);

                    // Create String struct on stack
                    let str_ty = self.string_struct_type();
                    let str_ptr = self.builder.alloca(str_ty.clone());
                    self.vreg_types.insert(str_ptr, str_ty);

                    // Store fields: ptr, len, cap=0 (global constant)
                    let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
                    self.builder.store(ptr_field, data_ptr);
                    let len_field = self.builder.get_field_ptr(str_ptr, 1);
                    self.builder.store(len_field, str_len);
                    let cap_field = self.builder.get_field_ptr(str_ptr, 2);
                    self.builder.store(cap_field, zero);

                    return Some(str_ptr);
                }
                MacroToken::CharLit(c, _) => {
                    return Some(self.builder.const_int(*c as i64));
                }
                MacroToken::Ident(name, _) => {
                    // Variable reference - look up in locals
                    if let Some(&slot) = self.locals.get(name) {
                        let value = self.builder.load(slot);
                        return Some(value);
                    }
                }
                _ => {}
            }
        }

        // For more complex expressions, we'd need full expression parsing
        // For now, return None for unsupported cases
        None
    }

    /// Lower println!/print! macro
    fn lower_builtin_print_macro(&mut self, tokens: &[crate::ast::MacroToken], newline: bool) -> VReg {
        use crate::ast::MacroToken;

        if tokens.is_empty() {
            if newline {
                // Print empty line
                let nl_str = self.builder.add_string_constant("\n");
                let nl_ptr = self.builder.global_string_ptr(&nl_str);
                self.builder.call("__print_str", vec![nl_ptr]);
            }
            return self.builder.const_int(0);
        }

        // Get the first token - should be the format string
        if let Some(first) = tokens.first() {
            match first {
                MacroToken::StrLit(s, _) => {
                    // Simple case: println!("hello")
                    let output = if newline {
                        format!("{}\n", s)
                    } else {
                        s.clone()
                    };
                    let str_name = self.builder.add_string_constant(&output);
                    let str_ptr = self.builder.global_string_ptr(&str_name);
                    self.builder.call("__print_str", vec![str_ptr]);
                }
                MacroToken::Ident(name, _) => {
                    // println!(var) - print the variable
                    if let Some(&slot) = self.locals.get(name) {
                        let value = self.builder.load(slot);
                        // For now, print as integer
                        self.builder.call("__print_i64", vec![value]);
                        if newline {
                            let nl = self.builder.add_string_constant("\n");
                            let nl_ptr = self.builder.global_string_ptr(&nl);
                            self.builder.call("__print_str", vec![nl_ptr]);
                        }
                    }
                }
                MacroToken::IntLit(n, _) => {
                    let value = self.builder.const_int(*n);
                    self.builder.call("__print_i64", vec![value]);
                    if newline {
                        let nl = self.builder.add_string_constant("\n");
                        let nl_ptr = self.builder.global_string_ptr(&nl);
                        self.builder.call("__print_str", vec![nl_ptr]);
                    }
                }
                _ => {}
            }
        }

        self.builder.const_int(0)
    }

    /// Lower eprintln!/eprint! macro (to stderr)
    fn lower_builtin_eprint_macro(&mut self, tokens: &[crate::ast::MacroToken], newline: bool) -> VReg {
        use crate::ast::MacroToken;
        use crate::token::TokenKind;

        if tokens.is_empty() {
            if newline {
                let nl_str = self.builder.add_string_constant("\n");
                let nl_ptr = self.builder.global_string_ptr(&nl_str);
                self.builder.call("__eprint_str", vec![nl_ptr]);
            }
            return self.builder.const_int(0);
        }

        if let Some(first) = tokens.first() {
            if let MacroToken::StrLit(s, _) = first {
                let output = if newline {
                    format!("{}\n", s)
                } else {
                    s.clone()
                };
                let str_name = self.builder.add_string_constant(&output);
                let str_ptr = self.builder.global_string_ptr(&str_name);
                self.builder.call("__eprint_str", vec![str_ptr]);
            }
        }

        self.builder.const_int(0)
    }

    /// Lower format! macro - returns String
    /// Simplified single-sprintf approach for stability
    fn lower_builtin_format_macro(&mut self, tokens: &[crate::ast::MacroToken]) -> VReg {
        use crate::ast::MacroToken;
        use crate::token::TokenKind;

        if tokens.is_empty() {
            return self.lower_string_new();
        }

        // Parse tokens: first should be format string, rest are arguments
        let format_str = match tokens.first() {
            Some(MacroToken::StrLit(s, _)) => s.clone(),
            _ => return self.lower_string_new(),
        };

        // Collect arguments from remaining tokens (skip format string and commas)
        let mut args: Vec<VReg> = Vec::new();
        let mut i = 1;

        while i < tokens.len() {
            // Skip comma tokens
            if let MacroToken::Token(TokenKind::Comma, _) = &tokens[i] {
                i += 1;
                continue;
            }

            // Get value for this token
            if let Some(val) = self.lower_macro_token_to_value(&tokens[i]) {
                args.push(val);
            }
            i += 1;
        }

        // If no placeholders, just create String from literal
        if !format_str.contains('{') {
            let name = self.builder.add_string_constant(&format_str);
            let data_ptr = self.builder.global_string_ptr(&name);
            let str_len = self.builder.const_int(format_str.len() as i64);
            let zero = self.builder.const_int(0);

            // Create String struct on stack
            let str_ty = self.string_struct_type();
            let str_ptr = self.builder.alloca(str_ty.clone());
            self.vreg_types.insert(str_ptr, str_ty);

            // Store fields: ptr, len, cap=0 (global constant, don't free)
            let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
            self.builder.store(ptr_field, data_ptr);
            let len_field = self.builder.get_field_ptr(str_ptr, 1);
            self.builder.store(len_field, str_len);
            let cap_field = self.builder.get_field_ptr(str_ptr, 2);
            self.builder.store(cap_field, zero);

            return str_ptr;
        }

        // Convert format string with advanced specifiers:
        // {} -> %lld (integer)
        // {:04} -> %04lld (zero-padded integer)
        // {:.2} -> %.2f (float with precision)
        // {:08.2} -> %08.2f (width and precision)
        // {:x} -> %llx (hex)
        // {:X} -> %llX (uppercase hex)
        // {:o} -> %llo (octal)
        // {:b} -> custom binary (not native printf)
        // {{ -> {, }} -> }, % -> %%
        let mut printf_fmt = String::new();
        let mut placeholder_count = 0;
        let mut format_specs: Vec<FormatSpec> = Vec::new();
        let mut chars = format_str.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                if let Some(&'{') = chars.peek() {
                    chars.next();
                    printf_fmt.push('{');
                    continue;
                }

                // Parse placeholder content until }
                let mut placeholder_content = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2 == '}' {
                        chars.next();
                        break;
                    }
                    placeholder_content.push(chars.next().unwrap());
                }

                // Parse format spec from placeholder content
                let spec = if placeholder_content.is_empty() {
                    FormatSpec::default()
                } else if placeholder_content.starts_with(':') {
                    self.parse_format_spec(&placeholder_content[1..])
                } else if let Some(colon_pos) = placeholder_content.find(':') {
                    self.parse_format_spec(&placeholder_content[colon_pos + 1..])
                } else {
                    FormatSpec::default()
                };

                // Only add format if we have an argument for this placeholder
                if placeholder_count < args.len() {
                    // Generate printf format based on spec
                    let fmt = self.format_spec_to_printf(&spec);
                    printf_fmt.push_str(&fmt);
                }
                format_specs.push(spec);
                placeholder_count += 1;
            } else if c == '}' {
                if let Some(&'}') = chars.peek() {
                    chars.next();
                }
                printf_fmt.push('}');
            } else if c == '%' {
                printf_fmt.push_str("%%");
            } else {
                printf_fmt.push(c);
            }
        }

        // Allocate a fixed-size buffer on the heap (using direct malloc like elsewhere)
        let buffer_size = self.builder.const_int(1024);
        let buffer_raw = self.builder.call("malloc", vec![buffer_size]);
        let buffer_i64 = self.builder.ptrtoint(buffer_raw, IrType::I64);
        let buffer = self.builder.inttoptr(buffer_i64, IrType::Ptr(Box::new(IrType::I8)));

        // Create printf format string constant
        let fmt_name = self.builder.add_string_constant(&printf_fmt);
        let fmt_ptr = self.builder.global_string_ptr(&fmt_name);

        // Call sprintf with all args
        match args.len() {
            0 => {
                self.builder.call("sprintf", vec![buffer, fmt_ptr]);
            }
            1 => {
                self.builder.call("sprintf", vec![buffer, fmt_ptr, args[0]]);
            }
            2 => {
                self.builder.call("sprintf", vec![buffer, fmt_ptr, args[0], args[1]]);
            }
            3 => {
                self.builder.call("sprintf", vec![buffer, fmt_ptr, args[0], args[1], args[2]]);
            }
            4 => {
                self.builder.call("sprintf", vec![buffer, fmt_ptr, args[0], args[1], args[2], args[3]]);
            }
            5 => {
                self.builder.call("sprintf", vec![buffer, fmt_ptr, args[0], args[1], args[2], args[3], args[4]]);
            }
            _ => {
                // Fallback: just use the format string as-is
                self.builder.call("sprintf", vec![buffer, fmt_ptr]);
            }
        }

        // Get length with strlen
        let len = self.builder.call("strlen", vec![buffer]);

        // Create String from buffer (buffer becomes String's data, no copy needed)
        // String struct: {ptr, len, cap}
        let str_ty = self.string_struct_type();
        let str_ptr = self.builder.alloca(str_ty.clone());

        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        self.builder.store(ptr_field, buffer);

        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        self.builder.store(len_field, len);

        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        self.builder.store(cap_field, buffer_size);

        self.vreg_types.insert(str_ptr, str_ty);

        str_ptr
    }

    /// Parse format placeholder content: {arg:spec} or {} or {0} or {name}
    /// Returns (optional_arg_specifier, format_spec)
    fn parse_format_placeholder(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> (Option<String>, FormatSpec) {
        let mut arg_part = String::new();
        let mut spec = FormatSpec::default();

        // Read until : or }
        while let Some(&c) = chars.peek() {
            if c == ':' || c == '}' {
                break;
            }
            arg_part.push(chars.next().unwrap());
        }

        // Parse format specifier after :
        if chars.peek() == Some(&':') {
            chars.next(); // consume ':'

            // Parse format specifier
            let mut spec_str = String::new();
            while let Some(&c) = chars.peek() {
                if c == '}' {
                    break;
                }
                spec_str.push(chars.next().unwrap());
            }

            spec = self.parse_format_spec(&spec_str);
        }

        // Consume closing }
        if chars.peek() == Some(&'}') {
            chars.next();
        }

        let arg = if arg_part.is_empty() { None } else { Some(arg_part) };
        (arg, spec)
    }

    /// Parse format specifier: "04", ".2", "x", "08x", etc.
    fn parse_format_spec(&self, spec: &str) -> FormatSpec {
        let mut result = FormatSpec::default();

        let mut chars = spec.chars().peekable();

        // Check for fill/alignment
        if let Some(&c) = chars.peek() {
            if c == '0' {
                result.zero_pad = true;
                chars.next();
            } else if c == '<' || c == '>' || c == '^' {
                result.align = Some(c);
                chars.next();
            }
        }

        // Parse width
        let mut width_str = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                width_str.push(chars.next().unwrap());
            } else {
                break;
            }
        }
        if !width_str.is_empty() {
            result.width = width_str.parse().ok();
        }

        // Check for precision
        if chars.peek() == Some(&'.') {
            chars.next();
            let mut prec_str = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    prec_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            if !prec_str.is_empty() {
                result.precision = prec_str.parse().ok();
            }
        }

        // Check for type specifier
        if let Some(c) = chars.next() {
            result.type_spec = Some(c);
        }

        result
    }

    /// Convert FormatSpec to printf format string
    /// Examples:
    ///   {} -> %lld
    ///   {:04} -> %04lld
    ///   {:.2} -> %.2f
    ///   {:08.2} -> %08.2f
    ///   {:x} -> %llx
    ///   {:X} -> %llX
    ///   {:o} -> %llo
    fn format_spec_to_printf(&self, spec: &FormatSpec) -> String {
        let mut fmt = String::from("%");

        // Add zero padding flag
        if spec.zero_pad {
            fmt.push('0');
        }

        // Add alignment (printf uses - for left align)
        if let Some(align) = spec.align {
            if align == '<' {
                fmt.push('-');
            }
            // '>' is default (right align), '^' (center) not supported in printf
        }

        // Add width
        if let Some(width) = spec.width {
            fmt.push_str(&width.to_string());
        }

        // Add precision (for floats)
        if let Some(precision) = spec.precision {
            fmt.push('.');
            fmt.push_str(&precision.to_string());
        }

        // Determine type specifier
        match spec.type_spec {
            Some('x') => fmt.push_str("llx"),      // lowercase hex
            Some('X') => fmt.push_str("llX"),      // uppercase hex
            Some('o') => fmt.push_str("llo"),      // octal
            Some('e') => fmt.push('e'),            // scientific notation
            Some('E') => fmt.push('E'),            // scientific notation uppercase
            Some('f') => fmt.push('f'),            // float
            Some('s') => fmt.push('s'),            // string
            _ => {
                // Default: if precision is set, assume float, else integer
                if spec.precision.is_some() {
                    fmt.push('f');
                } else {
                    fmt.push_str("lld");
                }
            }
        }

        fmt
    }

    /// Convert a macro token to a VReg value
    fn lower_macro_token_to_value(&mut self, token: &crate::ast::MacroToken) -> Option<VReg> {
        use crate::ast::MacroToken;

        match token {
            MacroToken::IntLit(n, _) => Some(self.builder.const_int(*n)),
            MacroToken::FloatLit(f, _) => Some(self.builder.const_float(*f)),
            MacroToken::StrLit(s, _) => {
                // Create a String from the literal
                let name = self.builder.add_string_constant(s);
                let data_ptr = self.builder.global_string_ptr(&name);
                let str_len = self.builder.const_int(s.len() as i64);
                let zero = self.builder.const_int(0);

                // Create String struct on stack
                let str_ty = self.string_struct_type();
                let str_ptr = self.builder.alloca(str_ty.clone());
                self.vreg_types.insert(str_ptr, str_ty);

                // Store fields: ptr, len, cap=0 (global constant)
                let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
                self.builder.store(ptr_field, data_ptr);
                let len_field = self.builder.get_field_ptr(str_ptr, 1);
                self.builder.store(len_field, str_len);
                let cap_field = self.builder.get_field_ptr(str_ptr, 2);
                self.builder.store(cap_field, zero);

                Some(str_ptr)
            }
            MacroToken::Ident(name, _) => {
                if let Some(&slot) = self.locals.get(name) {
                    Some(self.builder.load(slot))
                } else {
                    None
                }
            }
            MacroToken::Group { tokens, .. } => {
                // Try to parse as expression
                if let Some(expr) = crate::macro_expand::tokens_to_expr(tokens, crate::span::Span::new(0, 0)) {
                    Some(self.lower_expr(&expr))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Append a literal string to the format buffer
    fn format_append_literal(&mut self, str_ptr: VReg, literal: &str) {
        let const_str = self.builder.add_string_constant(literal);
        let const_ptr = self.builder.global_string_ptr(&const_str);
        let byte_len = self.builder.const_int(literal.len() as i64);
        self.format_push_bytes(str_ptr, const_ptr, byte_len);
    }

    /// Inline implementation of pushing bytes to a String (replaces __string_push_bytes)
    /// Simplified: always allocates new buffer to avoid complex control flow
    fn format_push_bytes(&mut self, str_ptr: VReg, src_ptr: VReg, src_len: VReg) {
        // Load current len
        let len_field = self.builder.get_field_ptr(str_ptr, 1);
        let len = self.builder.load(len_field);

        // Load cap
        let cap_field = self.builder.get_field_ptr(str_ptr, 2);
        let cap = self.builder.load(cap_field);

        // Load data pointer
        let ptr_field = self.builder.get_field_ptr(str_ptr, 0);
        let data_ptr = self.builder.load(ptr_field);

        // Calculate new length
        let new_len = self.builder.add(len, src_len);

        // new_cap = max(new_len + 16, cap * 2, 32)
        let sixteen = self.builder.const_int(16);
        let padded = self.builder.add(new_len, sixteen);
        let two = self.builder.const_int(2);
        let doubled = self.builder.mul(cap, two);

        use super::instr::CmpOp;
        let use_doubled = self.builder.icmp(CmpOp::Sgt, doubled, padded);
        let cap1 = self.builder.select(use_doubled, doubled, padded);
        let thirty_two = self.builder.const_int(32);
        let use_min = self.builder.icmp(CmpOp::Sgt, thirty_two, cap1);
        let new_cap = self.builder.select(use_min, thirty_two, cap1);

        // Use realloc (handles NULL, might reuse same buffer)
        let new_data = self.builder.realloc(data_ptr, new_cap);

        // Store new pointer and cap
        self.builder.store(ptr_field, new_data);
        self.builder.store(cap_field, new_cap);

        // Calculate destination: new_data + len
        let dest = self.builder.get_byte_ptr(new_data, len);

        // Copy source bytes
        self.builder.memcpy(dest, src_ptr, src_len);

        // Update len
        self.builder.store(len_field, new_len);
    }

    /// Append a formatted value to the format buffer
    fn format_append_value(&mut self, str_ptr: VReg, value: VReg, spec: &FormatSpec) {
        // Determine value type from vreg_types
        let is_float = self.vreg_types.get(&value).map_or(false, |ty| {
            matches!(ty, IrType::F32 | IrType::F64)
        });
        let is_string = self.vreg_types.get(&value).map_or(false, |ty| {
            if let IrType::Struct(fields) = ty {
                // String is {ptr, len, cap}
                fields.len() == 3
            } else {
                false
            }
        });

        if is_string {
            // Append string directly
            // Get the data pointer from String struct
            let data_ptr_ptr = self.builder.get_field_ptr(value, 0);
            let data_ptr = self.builder.load(data_ptr_ptr);
            let len_ptr = self.builder.get_field_ptr(value, 1);
            let len = self.builder.load(len_ptr);
            self.format_push_bytes(str_ptr, data_ptr, len);
        } else if is_float {
            // Format float
            self.format_append_float(str_ptr, value, spec);
        } else {
            // Format integer (default)
            self.format_append_int(str_ptr, value, spec);
        }
    }

    /// Append formatted integer to string (inline implementation)
    fn format_append_int(&mut self, str_ptr: VReg, value: VReg, spec: &FormatSpec) {
        // Use sprintf directly with appropriate format string
        let buffer_size = self.builder.const_int(64);
        let buffer = self.builder.malloc_array(IrType::I8, buffer_size);

        // Select format string based on type specifier and width
        let fmt_str = match (spec.type_spec, spec.width, spec.zero_pad) {
            (Some('x'), Some(w), true) => format!("%0{}llx", w),
            (Some('x'), Some(w), false) => format!("%{}llx", w),
            (Some('x'), None, _) => "%llx".to_string(),
            (Some('X'), Some(w), true) => format!("%0{}llX", w),
            (Some('X'), Some(w), false) => format!("%{}llX", w),
            (Some('X'), None, _) => "%llX".to_string(),
            (Some('o'), Some(w), true) => format!("%0{}llo", w),
            (Some('o'), Some(w), false) => format!("%{}llo", w),
            (Some('o'), None, _) => "%llo".to_string(),
            (_, Some(w), true) => format!("%0{}lld", w),
            (_, Some(w), false) => format!("%{}lld", w),
            _ => "%lld".to_string(),
        };

        let fmt_name = self.builder.add_string_constant(&fmt_str);
        let fmt_ptr = self.builder.global_string_ptr(&fmt_name);

        // Call sprintf
        self.builder.call("sprintf", vec![buffer, fmt_ptr, value]);

        // Get length and push to string
        let len = self.builder.call("strlen", vec![buffer]);
        self.format_push_bytes(str_ptr, buffer, len);
        // Note: not freeing buffer to avoid heap corruption with multiple reallocs
        // Minor memory leak but stable
    }

    /// Append formatted float to string (inline implementation)
    fn format_append_float(&mut self, str_ptr: VReg, value: VReg, spec: &FormatSpec) {
        let buffer_size = self.builder.const_int(64);
        let buffer = self.builder.malloc_array(IrType::I8, buffer_size);

        // Select format string based on precision
        let fmt_str = match spec.precision {
            Some(p) => format!("%.{}g", p),
            None => "%g".to_string(),
        };

        let fmt_name = self.builder.add_string_constant(&fmt_str);
        let fmt_ptr = self.builder.global_string_ptr(&fmt_name);

        // Call sprintf
        self.builder.call("sprintf", vec![buffer, fmt_ptr, value]);

        // Get length and push to string
        let len = self.builder.call("strlen", vec![buffer]);
        self.format_push_bytes(str_ptr, buffer, len);
        self.builder.call("free", vec![buffer]);
    }

    /// Lower panic! macro - aborts the program
    fn lower_builtin_panic_macro(&mut self, tokens: &[crate::ast::MacroToken]) -> VReg {
        use crate::ast::MacroToken;

        // Print panic message if provided
        if let Some(first) = tokens.first() {
            if let MacroToken::StrLit(s, _) = first {
                let msg = format!("panic: {}\n", s);
                let str_name = self.builder.add_string_constant(&msg);
                let str_ptr = self.builder.global_string_ptr(&str_name);
                self.builder.call("__eprint_str", vec![str_ptr]);
            }
        } else {
            let msg = self.builder.add_string_constant("panic: explicit panic\n");
            let msg_ptr = self.builder.global_string_ptr(&msg);
            self.builder.call("__eprint_str", vec![msg_ptr]);
        }

        // Exit with code 1
        let exit_code = self.builder.const_int(1);
        self.builder.call("exit", vec![exit_code]);
        self.builder.const_int(0)
    }

    /// Lower assert! macro - checks condition, panics if false
    fn lower_builtin_assert_macro(&mut self, tokens: &[crate::ast::MacroToken], _debug_only: bool) -> VReg {
        // Try to evaluate the condition
        if let Some(cond_value) = self.macro_tokens_to_value(tokens) {
            // Create assertion check
            let zero = self.builder.const_int(0);
            let is_false = self.builder.icmp(crate::ir::CmpOp::Eq, cond_value, zero);

            // Branch: if false, panic
            let panic_block = self.builder.create_block();
            let continue_block = self.builder.create_block();
            self.builder.cond_br(is_false, panic_block, continue_block);

            // Panic block
            self.builder.start_block(panic_block);
            let msg = self.builder.add_string_constant("assertion failed\n");
            let msg_ptr = self.builder.global_string_ptr(&msg);
            self.builder.call("__eprint_str", vec![msg_ptr]);
            let exit_code = self.builder.const_int(1);
            self.builder.call("exit", vec![exit_code]);
            self.builder.br(continue_block);

            // Continue block
            self.builder.start_block(continue_block);
        }

        self.builder.const_int(0)
    }

    /// Lower assert_eq!/assert_ne! macro
    fn lower_builtin_assert_eq_macro(&mut self, tokens: &[crate::ast::MacroToken], is_eq: bool) -> VReg {
        // Split by comma to get left and right expressions
        let parts = self.split_macro_tokens_by_comma(tokens);

        if parts.len() >= 2 {
            if let (Some(left), Some(right)) = (
                self.macro_tokens_to_value(&parts[0]),
                self.macro_tokens_to_value(&parts[1]),
            ) {
                // Compare values
                let cmp_op = if is_eq {
                    crate::ir::CmpOp::Ne // assert_eq fails if NOT equal
                } else {
                    crate::ir::CmpOp::Eq // assert_ne fails if equal
                };
                let should_panic = self.builder.icmp(cmp_op, left, right);

                let panic_block = self.builder.create_block();
                let continue_block = self.builder.create_block();
                self.builder.cond_br(should_panic, panic_block, continue_block);

                // Panic block
                self.builder.start_block(panic_block);
                let msg = if is_eq {
                    self.builder.add_string_constant("assertion failed: values not equal\n")
                } else {
                    self.builder.add_string_constant("assertion failed: values are equal\n")
                };
                let msg_ptr = self.builder.global_string_ptr(&msg);
                self.builder.call("__eprint_str", vec![msg_ptr]);
                let exit_code = self.builder.const_int(1);
                self.builder.call("exit", vec![exit_code]);
                self.builder.br(continue_block);

                // Continue block
                self.builder.start_block(continue_block);
            }
        }

        self.builder.const_int(0)
    }

    // ============ Math Functions Implementation ============

    /// Lower f64 unary math function (sin, cos, sqrt, etc.)
    fn lower_math_f64_unary(&mut self, args: &[Expr], func_name: &str) -> VReg {
        self.builder.declare_math();

        if args.is_empty() {
            return self.builder.const_float(0.0);
        }

        let x = self.lower_expr(&args[0]);
        self.builder.call_f64(func_name, vec![x])
    }

    /// Lower f64 binary math function (pow, atan2, etc.)
    fn lower_math_f64_binary(&mut self, args: &[Expr], func_name: &str) -> VReg {
        self.builder.declare_math();

        if args.len() < 2 {
            return self.builder.const_float(0.0);
        }

        let x = self.lower_expr(&args[0]);
        let y = self.lower_expr(&args[1]);
        self.builder.call_f64(func_name, vec![x, y])
    }

    /// Lower f32 unary math function
    fn lower_math_f32_unary(&mut self, args: &[Expr], func_name: &str) -> VReg {
        self.builder.declare_math();

        if args.is_empty() {
            return self.builder.const_float32(0.0);
        }

        let x = self.lower_expr(&args[0]);
        self.builder.call_f32(func_name, vec![x])
    }

    /// Lower f32 binary math function
    fn lower_math_f32_binary(&mut self, args: &[Expr], func_name: &str) -> VReg {
        self.builder.declare_math();

        if args.len() < 2 {
            return self.builder.const_float32(0.0);
        }

        let x = self.lower_expr(&args[0]);
        let y = self.lower_expr(&args[1]);
        self.builder.call_f32(func_name, vec![x, y])
    }

    /// Lower f64::log(x, base) - logarithm with arbitrary base
    /// Implemented as ln(x) / ln(base)
    fn lower_math_log_base(&mut self, args: &[Expr]) -> VReg {
        self.builder.declare_math();

        if args.len() < 2 {
            return self.builder.const_float(0.0);
        }

        let x = self.lower_expr(&args[0]);
        let base = self.lower_expr(&args[1]);

        let ln_x = self.builder.call_f64("log", vec![x]);
        let ln_base = self.builder.call_f64("log", vec![base]);
        self.builder.fdiv(ln_x, ln_base)
    }

    /// Lower f64::is_nan(x) - check if NaN
    /// Use unordered comparison (UNO) which returns true if either operand is NaN
    fn lower_math_is_nan(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_bool(false);
        }

        let x = self.lower_expr(&args[0]);
        // UNO comparison returns true if either operand is NaN
        // x uno x is true iff x is NaN
        self.builder.fcmp(CmpOp::Uno, x, x)
    }

    /// Lower f64::is_infinite(x) - check if positive or negative infinity
    fn lower_math_is_infinite(&mut self, args: &[Expr]) -> VReg {
        self.builder.declare_math();

        if args.is_empty() {
            return self.builder.const_bool(false);
        }

        let x = self.lower_expr(&args[0]);
        let abs_x = self.builder.call_f64("fabs", vec![x]);

        // Check if |x| == infinity
        // Use a very large number as infinity representation
        let inf = self.builder.const_float(f64::INFINITY);
        self.builder.fcmp(CmpOp::Eq, abs_x, inf)
    }

    /// Lower f64::is_finite(x) - check if not NaN and not infinite
    fn lower_math_is_finite(&mut self, args: &[Expr]) -> VReg {
        self.builder.declare_math();

        if args.is_empty() {
            return self.builder.const_bool(true);
        }

        let x = self.lower_expr(&args[0]);

        // is_finite = !is_nan && !is_infinite
        // x is finite if x == x (not NaN) and |x| != infinity

        // Check not NaN: x == x
        let not_nan = self.builder.fcmp(CmpOp::Eq, x, x);

        // Check not infinite: |x| != infinity
        let abs_x = self.builder.call_f64("fabs", vec![x]);
        let inf = self.builder.const_float(f64::INFINITY);
        let not_inf = self.builder.fcmp(CmpOp::Ne, abs_x, inf);

        // Return not_nan && not_inf
        self.builder.and(not_nan, not_inf)
    }

    /// Lower f64::PI constant
    fn lower_math_const_pi(&mut self) -> VReg {
        self.builder.const_float(std::f64::consts::PI)
    }

    /// Lower f64::E constant
    fn lower_math_const_e(&mut self) -> VReg {
        self.builder.const_float(std::f64::consts::E)
    }

    /// Lower f64::INFINITY constant
    fn lower_math_const_infinity(&mut self) -> VReg {
        self.builder.const_float(f64::INFINITY)
    }

    /// Lower f64::NEG_INFINITY constant
    fn lower_math_const_neg_infinity(&mut self) -> VReg {
        self.builder.const_float(f64::NEG_INFINITY)
    }

    /// Lower f64::NAN constant
    fn lower_math_const_nan(&mut self) -> VReg {
        self.builder.const_float(f64::NAN)
    }

    /// Lower i64::abs(x) - absolute value for i64
    fn lower_math_i64_abs(&mut self, args: &[Expr]) -> VReg {
        if args.is_empty() {
            return self.builder.const_int(0);
        }

        let x = self.lower_expr(&args[0]);
        let zero = self.builder.const_int(0);

        // if x < 0 then -x else x
        let is_neg = self.builder.icmp(CmpOp::Slt, x, zero);
        let neg_x = self.builder.neg(x);
        self.builder.select(is_neg, neg_x, x)
    }

    /// Lower i64::min(x, y)
    fn lower_math_i64_min(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let x = self.lower_expr(&args[0]);
        let y = self.lower_expr(&args[1]);

        let cmp = self.builder.icmp(CmpOp::Slt, x, y);
        self.builder.select(cmp, x, y)
    }

    /// Lower i64::max(x, y)
    fn lower_math_i64_max(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let x = self.lower_expr(&args[0]);
        let y = self.lower_expr(&args[1]);

        let cmp = self.builder.icmp(CmpOp::Sgt, x, y);
        self.builder.select(cmp, x, y)
    }

    /// Lower i64::pow(base, exp) - integer power
    /// Uses binary exponentiation for efficiency
    fn lower_math_i64_pow(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_int(0);
        }

        let base = self.lower_expr(&args[0]);
        let exp = self.lower_expr(&args[1]);

        // For simplicity, convert to f64, use pow, convert back
        // This works for reasonable values
        self.builder.declare_math();

        let base_f64 = self.builder.sitofp(base, IrType::F64);
        let exp_f64 = self.builder.sitofp(exp, IrType::F64);
        let result_f64 = self.builder.call_f64("pow", vec![base_f64, exp_f64]);
        self.builder.fptosi(result_f64, IrType::I64)
    }

    /// Lower i32::abs(x)
    fn lower_math_i32_abs(&mut self, args: &[Expr]) -> VReg {
        self.builder.declare_math();

        if args.is_empty() {
            return self.builder.const_i32(0);
        }

        let x = self.lower_expr(&args[0]);
        // Call libc abs
        self.builder.call("abs", vec![x])
    }

    /// Lower i32::min(x, y)
    fn lower_math_i32_min(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_i32(0);
        }

        let x = self.lower_expr(&args[0]);
        let y = self.lower_expr(&args[1]);

        let cmp = self.builder.icmp(CmpOp::Slt, x, y);
        self.builder.select(cmp, x, y)
    }

    /// Lower i32::max(x, y)
    fn lower_math_i32_max(&mut self, args: &[Expr]) -> VReg {
        if args.len() < 2 {
            return self.builder.const_i32(0);
        }

        let x = self.lower_expr(&args[0]);
        let y = self.lower_expr(&args[1]);

        let cmp = self.builder.icmp(CmpOp::Sgt, x, y);
        self.builder.select(cmp, x, y)
    }
}

/// Print module in a readable format
pub fn print_module(module: &Module) -> String {
    let mut output = String::new();
    output.push_str(&format!("module {}\n\n", module.name));

    for func in &module.functions {
        if func.is_external {
            output.push_str(&format!("declare {} ", func.name));
        } else {
            output.push_str(&format!("define {} ", func.name));
        }

        output.push_str("(");
        for (i, (vreg, ty)) in func.params.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("{} {}", ty, vreg));
        }
        output.push_str(&format!(") -> {} ", func.ret_type));

        if func.is_external {
            output.push_str("\n");
            continue;
        }

        output.push_str("{\n");
        for block in &func.blocks {
            output.push_str(&format!("  {}:\n", block.id));
            for instr in &block.instructions {
                output.push_str(&format!("    {}\n", instr));
            }
            if let Some(ref term) = block.terminator {
                output.push_str(&format!("    {}\n", term));
            }
        }
        output.push_str("}\n\n");
    }

    output
}
