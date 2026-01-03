//! Monomorphization Collector for Genesis Lang
//!
//! This module collects all instantiations of generic types and functions
//! to enable monomorphization during code generation.
//!
//! # How it works
//!
//! 1. Collect all generic type/function definitions from the AST
//! 2. Scan resolved expression types for Named types with generics
//! 3. For each unique instantiation, record the concrete type arguments
//! 4. Provide mangled names for specialized versions
//!
//! # Example
//!
//! ```text
//! struct Wrapper<T> { value: T }
//! fn main() {
//!     let w1 = Wrapper { value: 42 }      // Wrapper<i64>
//!     let w2 = Wrapper { value: "hi" }    // Wrapper<String>
//! }
//! ```
//!
//! This produces:
//! - struct_instances: { "Wrapper" -> [[i64], [String]] }
//! - Mangled names: "Wrapper$i64", "Wrapper$String"

use std::collections::{HashMap, HashSet};

use crate::ast::{Program, Item, FnDef, StructDef, EnumDef};
use crate::span::Span;
use super::ty::{Ty, TyKind};
use super::context::TypeContext;

/// Collects monomorphization instances for generic types and functions
#[derive(Debug, Clone)]
pub struct MonomorphCollector {
    /// Struct instantiations: struct_name -> set of type argument lists
    /// e.g., "Wrapper" -> { [Ty::i64], [Ty::string] }
    pub struct_instances: HashMap<String, HashSet<Vec<Ty>>>,

    /// Function instantiations: fn_name -> set of type argument lists
    /// e.g., "identity" -> { [Ty::i64], [Ty::bool] }
    pub fn_instances: HashMap<String, HashSet<Vec<Ty>>>,

    /// Generic struct definitions (structs that have type parameters)
    generic_structs: HashMap<String, GenericDef>,

    /// Generic function definitions
    generic_fns: HashMap<String, GenericDef>,

    /// Generic enum definitions
    generic_enums: HashMap<String, GenericDef>,
}

/// Information about a generic definition
#[derive(Debug, Clone)]
pub struct GenericDef {
    pub name: String,
    pub type_params: Vec<String>,
}

impl MonomorphCollector {
    /// Create a new empty collector
    pub fn new() -> Self {
        Self {
            struct_instances: HashMap::new(),
            fn_instances: HashMap::new(),
            generic_structs: HashMap::new(),
            generic_fns: HashMap::new(),
            generic_enums: HashMap::new(),
        }
    }

    /// Collect monomorphization instances from a program and its resolved types
    ///
    /// # Arguments
    /// * `program` - The parsed AST
    /// * `expr_types` - Map from expression spans to resolved types (from type checking)
    /// * `ctx` - Type context with registered types and functions
    pub fn collect(
        program: &Program,
        expr_types: &HashMap<Span, Ty>,
        ctx: &TypeContext,
    ) -> Self {
        let mut collector = Self::new();

        // Phase 1: Collect generic definitions from AST
        collector.collect_generic_definitions(program);

        // Phase 2: Collect instantiations from resolved expression types
        collector.collect_instantiations(expr_types, ctx);

        collector
    }

    /// Collect generic type and function definitions from the AST
    fn collect_generic_definitions(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    if let Some(ref generics) = s.generics {
                        if !generics.params.is_empty() {
                            let type_params: Vec<String> = generics.params
                                .iter()
                                .map(|p| p.name.name.clone())
                                .collect();
                            self.generic_structs.insert(
                                s.name.name.clone(),
                                GenericDef {
                                    name: s.name.name.clone(),
                                    type_params,
                                },
                            );
                        }
                    }
                }
                Item::Enum(e) => {
                    if let Some(ref generics) = e.generics {
                        if !generics.params.is_empty() {
                            let type_params: Vec<String> = generics.params
                                .iter()
                                .map(|p| p.name.name.clone())
                                .collect();
                            self.generic_enums.insert(
                                e.name.name.clone(),
                                GenericDef {
                                    name: e.name.name.clone(),
                                    type_params,
                                },
                            );
                        }
                    }
                }
                Item::Function(f) => {
                    if let Some(ref generics) = f.generics {
                        if !generics.params.is_empty() {
                            let type_params: Vec<String> = generics.params
                                .iter()
                                .map(|p| p.name.name.clone())
                                .collect();
                            self.generic_fns.insert(
                                f.name.name.clone(),
                                GenericDef {
                                    name: f.name.name.clone(),
                                    type_params,
                                },
                            );
                        }
                    }
                }
                Item::Mod(m) => {
                    // Process items in inline modules
                    if let Some(ref items) = m.items {
                        let sub_program = Program {
                            items: items.clone(),
                            span: m.span,
                        };
                        self.collect_generic_definitions(&sub_program);
                    }
                }
                _ => {}
            }
        }
    }

    /// Collect instantiations from resolved expression types
    fn collect_instantiations(&mut self, expr_types: &HashMap<Span, Ty>, _ctx: &TypeContext) {
        for (_span, ty) in expr_types {
            self.collect_from_type(ty);
        }
    }

    /// Recursively collect instantiations from a type
    fn collect_from_type(&mut self, ty: &Ty) {
        match &ty.kind {
            TyKind::Named { name, generics } if !generics.is_empty() => {
                // Check if this is a user-defined generic struct
                if self.generic_structs.contains_key(name) {
                    // Ensure all type arguments are concrete (no type variables)
                    if generics.iter().all(|g| !g.has_vars() && !self.is_generic_param(g)) {
                        self.struct_instances
                            .entry(name.clone())
                            .or_insert_with(HashSet::new)
                            .insert(generics.clone());
                    }
                }

                // Check if this is a user-defined generic enum
                if self.generic_enums.contains_key(name) {
                    if generics.iter().all(|g| !g.has_vars() && !self.is_generic_param(g)) {
                        self.struct_instances
                            .entry(name.clone())
                            .or_insert_with(HashSet::new)
                            .insert(generics.clone());
                    }
                }

                // Recursively collect from type arguments
                for generic in generics {
                    self.collect_from_type(generic);
                }
            }
            TyKind::Ref { inner, .. } => self.collect_from_type(inner),
            TyKind::Array { element, .. } => self.collect_from_type(element),
            TyKind::Slice { element } => self.collect_from_type(element),
            TyKind::Tuple(elements) => {
                for elem in elements {
                    self.collect_from_type(elem);
                }
            }
            TyKind::Fn { params, ret } => {
                for param in params {
                    self.collect_from_type(param);
                }
                self.collect_from_type(ret);
            }
            _ => {}
        }
    }

    /// Check if a type is a generic parameter (not a concrete type)
    fn is_generic_param(&self, ty: &Ty) -> bool {
        matches!(&ty.kind, TyKind::Generic { .. })
    }

    /// Generate a mangled name for a specialized type
    ///
    /// # Examples
    /// - `mangle("Wrapper", [i64])` -> `"Wrapper$i64"`
    /// - `mangle("Pair", [i64, String])` -> `"Pair$i64$String"`
    /// - `mangle("identity", [bool])` -> `"identity$bool"`
    pub fn mangle(base: &str, type_args: &[Ty]) -> String {
        if type_args.is_empty() {
            base.to_string()
        } else {
            let args = type_args
                .iter()
                .map(|t| Self::ty_to_suffix(t))
                .collect::<Vec<_>>()
                .join("$");
            format!("{}${}", base, args)
        }
    }

    /// Convert a type to a suffix string for mangling
    fn ty_to_suffix(ty: &Ty) -> String {
        match &ty.kind {
            TyKind::Unit => "unit".to_string(),
            TyKind::Bool => "bool".to_string(),
            TyKind::Char => "char".to_string(),
            TyKind::Str => "str".to_string(),
            TyKind::Int(int_ty) => format!("{:?}", int_ty).to_lowercase(),
            TyKind::Uint(uint_ty) => format!("{:?}", uint_ty).to_lowercase(),
            TyKind::Float(float_ty) => format!("{:?}", float_ty).to_lowercase(),
            TyKind::Named { name, generics } => {
                if generics.is_empty() {
                    name.clone()
                } else {
                    let args = generics
                        .iter()
                        .map(|g| Self::ty_to_suffix(g))
                        .collect::<Vec<_>>()
                        .join("_");
                    format!("{}__{}", name, args)
                }
            }
            TyKind::Ref { inner, mutable } => {
                let prefix = if *mutable { "refmut" } else { "ref" };
                format!("{}_{}", prefix, Self::ty_to_suffix(inner))
            }
            TyKind::Array { element, size } => {
                format!("arr{}_{}", size, Self::ty_to_suffix(element))
            }
            TyKind::Slice { element } => {
                format!("slice_{}", Self::ty_to_suffix(element))
            }
            TyKind::Tuple(elements) => {
                let elems = elements
                    .iter()
                    .map(|e| Self::ty_to_suffix(e))
                    .collect::<Vec<_>>()
                    .join("_");
                format!("tup_{}", elems)
            }
            TyKind::Fn { params, ret } => {
                let ps = params
                    .iter()
                    .map(|p| Self::ty_to_suffix(p))
                    .collect::<Vec<_>>()
                    .join("_");
                format!("fn_{}_{}", ps, Self::ty_to_suffix(ret))
            }
            TyKind::Generic { name } => name.clone(),
            TyKind::Var(v) => format!("var{}", v.0),
            TyKind::IntVar => "intvar".to_string(),
            TyKind::FloatVar => "floatvar".to_string(),
            TyKind::Never => "never".to_string(),
            TyKind::Actor { name } => format!("actor_{}", name),
            TyKind::Error => "error".to_string(),
            TyKind::Projection { base_ty, assoc_name, .. } => {
                format!("{}__{}", Self::ty_to_suffix(base_ty), assoc_name)
            }
            TyKind::TraitObject { trait_name } => {
                format!("dyn_{}", trait_name)
            }
        }
    }

    /// Check if a struct name is a user-defined generic struct
    pub fn is_generic_struct(&self, name: &str) -> bool {
        self.generic_structs.contains_key(name)
    }

    /// Check if a function name is a user-defined generic function
    pub fn is_generic_fn(&self, name: &str) -> bool {
        self.generic_fns.contains_key(name)
    }

    /// Get type parameters for a generic struct
    pub fn get_struct_type_params(&self, name: &str) -> Option<&[String]> {
        self.generic_structs.get(name).map(|d| d.type_params.as_slice())
    }

    /// Get type parameters for a generic function
    pub fn get_fn_type_params(&self, name: &str) -> Option<&[String]> {
        self.generic_fns.get(name).map(|d| d.type_params.as_slice())
    }

    /// Get all struct instantiations for a given generic struct
    pub fn get_struct_instantiations(&self, name: &str) -> Option<&HashSet<Vec<Ty>>> {
        self.struct_instances.get(name)
    }

    /// Get all function instantiations for a given generic function
    pub fn get_fn_instantiations(&self, name: &str) -> Option<&HashSet<Vec<Ty>>> {
        self.fn_instances.get(name)
    }

    /// Register a function call with inferred type arguments
    pub fn register_fn_call(&mut self, name: &str, type_args: Vec<Ty>) {
        if self.generic_fns.contains_key(name) && !type_args.is_empty() {
            // Ensure all type arguments are concrete
            if type_args.iter().all(|t| !t.has_vars() && !self.is_generic_param(t)) {
                self.fn_instances
                    .entry(name.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(type_args);
            }
        }
    }

    /// Get summary statistics
    pub fn stats(&self) -> MonomorphStats {
        MonomorphStats {
            generic_structs: self.generic_structs.len(),
            generic_fns: self.generic_fns.len(),
            struct_instantiations: self.struct_instances.values().map(|s| s.len()).sum(),
            fn_instantiations: self.fn_instances.values().map(|s| s.len()).sum(),
        }
    }
}

impl Default for MonomorphCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about monomorphization
#[derive(Debug, Clone)]
pub struct MonomorphStats {
    pub generic_structs: usize,
    pub generic_fns: usize,
    pub struct_instantiations: usize,
    pub fn_instantiations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mangle_simple() {
        let type_args = vec![Ty::i64()];
        assert_eq!(MonomorphCollector::mangle("Wrapper", &type_args), "Wrapper$i64");
    }

    #[test]
    fn test_mangle_multiple() {
        let type_args = vec![Ty::i64(), Ty::bool()];
        assert_eq!(MonomorphCollector::mangle("Pair", &type_args), "Pair$i64$bool");
    }

    #[test]
    fn test_mangle_nested() {
        let inner = Ty::named("Option".to_string(), vec![Ty::i64()]);
        let type_args = vec![inner];
        assert_eq!(MonomorphCollector::mangle("Box", &type_args), "Box$Option__i64");
    }

    #[test]
    fn test_mangle_empty() {
        assert_eq!(MonomorphCollector::mangle("Foo", &[]), "Foo");
    }
}
