//! Type Checker for Genesis Lang
//!
//! This module implements type checking and inference for the Genesis programming language.
//! It uses a Hindley-Milner style type inference algorithm with extensions for:
//! - Structs and enums
//! - Generics with bounds
//! - Ownership and borrowing (basic)
//! - Actor types
//!
//! # Architecture
//!
//! The type checker follows a two-phase approach:
//! 1. **Collection Phase**: Gather type information from all definitions
//! 2. **Checking Phase**: Verify expressions and infer types
//!
//! # References
//! - [Rust Compiler Type Inference](https://rustc-dev-guide.rust-lang.org/type-inference.html)
//! - [Hindley-Milner Type System](https://en.wikipedia.org/wiki/Hindley–Milner_type_system)

mod ty;
mod context;
mod infer;
mod unify;
mod check;
mod error;
mod ownership;
mod exhaustiveness;
mod monomorph;
mod variance;

pub use ty::{Ty, TyKind, TyVar, Substitution, IntTy, UintTy, FloatTy};
pub use context::{TypeContext, Scope, Symbol, SymbolKind, ModuleDef, ImportedSymbol, ImportKind};
pub use infer::TypeInference;
pub use error::{TypeError, TypeResult};
pub use ownership::{OwnershipChecker, OwnershipState, LifetimeId, ActiveBorrow};
pub use monomorph::{MonomorphCollector, MonomorphStats};
pub use variance::{Variance, VarianceAnalyzer, is_subtype, types_compatible};

use crate::ast::Program;
use crate::span::Span;

/// Main entry point for type checking a program
pub fn check_program(program: &Program) -> Result<TypedProgram, Vec<TypeError>> {
    let mut checker = TypeChecker::new();
    checker.check(program)
}

/// A type-checked program with resolved types
#[derive(Debug)]
pub struct TypedProgram {
    /// Map from expression spans to their types
    pub expr_types: std::collections::HashMap<Span, Ty>,
    /// Map from variable names to their types in each scope
    pub symbol_types: std::collections::HashMap<String, Ty>,
    /// Monomorphization information for generic types and functions
    pub monomorph: MonomorphCollector,
    /// Generic function calls: call_span -> (fn_name, type_args)
    pub generic_fn_calls: std::collections::HashMap<Span, (String, Vec<Ty>)>,
    /// Any warnings generated during type checking
    pub warnings: Vec<TypeWarning>,
}

/// Type warning (non-fatal issues)
#[derive(Debug, Clone)]
pub struct TypeWarning {
    pub message: String,
    pub span: Span,
}

/// The main type checker orchestrator
pub struct TypeChecker {
    /// Type inference engine
    inference: TypeInference,
    /// Ownership checker
    ownership: OwnershipChecker,
    /// Collected errors
    errors: Vec<TypeError>,
    /// Collected warnings
    warnings: Vec<TypeWarning>,
}

impl TypeChecker {
    /// Create a new type checker
    pub fn new() -> Self {
        Self {
            inference: TypeInference::new(),
            ownership: OwnershipChecker::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Check a complete program (alias for backwards compatibility)
    pub fn check_program(&mut self, program: &Program) -> Result<TypedProgram, Vec<TypeError>> {
        self.check(program)
    }

    /// Check a complete program
    pub fn check(&mut self, program: &Program) -> Result<TypedProgram, Vec<TypeError>> {
        // Phase 1: Collect all type definitions
        self.collect_definitions(program);

        // Phase 2: Check all items
        for item in &program.items {
            if let Err(e) = self.check_item(item) {
                self.errors.push(e);
            }
        }

        // Phase 3: Verify ownership
        if let Err(e) = self.ownership.verify() {
            self.errors.extend(e);
        }

        if self.errors.is_empty() {
            let expr_types = self.inference.get_expr_types();

            // Phase 4: Collect monomorphization instances
            let mut monomorph = MonomorphCollector::collect(
                program,
                &expr_types,
                &self.inference.ctx,
            );

            // Phase 5: Register generic function calls from type inference
            let generic_fn_calls = self.inference.get_generic_fn_calls();
            for (_span, (fn_name, type_args)) in &generic_fn_calls {
                monomorph.register_fn_call(fn_name, type_args.clone());
            }

            Ok(TypedProgram {
                expr_types,
                symbol_types: self.inference.get_symbol_types(),
                monomorph,
                generic_fn_calls,
                warnings: std::mem::take(&mut self.warnings),
            })
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Collect type definitions from the program
    fn collect_definitions(&mut self, program: &Program) {
        use crate::ast::Item;

        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    self.inference.register_struct(s);
                }
                Item::Enum(e) => {
                    self.inference.register_enum(e);
                }
                Item::Function(f) => {
                    self.inference.register_function(f);
                }
                Item::Trait(t) => {
                    self.inference.register_trait(t);
                }
                Item::TypeAlias(t) => {
                    self.inference.register_type_alias(t);
                }
                Item::Actor(a) => {
                    self.inference.register_actor(a);
                }
                Item::Impl(i) => {
                    self.inference.register_impl(i);
                }
                Item::Const(c) => {
                    self.inference.register_const(c);
                }
                Item::Use(u) => {
                    self.register_use(u);
                }
                Item::Mod(m) => {
                    self.register_mod(m);
                }
                Item::Macro(m) => {
                    self.inference.register_macro(m);
                }
            }
        }
    }

    /// Check a single item
    fn check_item(&mut self, item: &crate::ast::Item) -> TypeResult<()> {
        use crate::ast::Item;

        match item {
            Item::Function(f) => self.check_function(f),
            Item::Struct(s) => self.check_struct(s),
            Item::Enum(e) => self.check_enum(e),
            Item::Impl(i) => self.check_impl(i),
            Item::Trait(t) => self.check_trait(t),
            Item::Const(c) => self.check_const(c),
            Item::Actor(a) => self.check_actor(a),
            Item::TypeAlias(_) | Item::Use(_) | Item::Mod(_) | Item::Macro(_) => Ok(()),
        }
    }

    fn check_function(&mut self, f: &crate::ast::FnDef) -> TypeResult<()> {
        self.inference.check_function(f, &mut self.ownership)
    }

    fn check_struct(&mut self, s: &crate::ast::StructDef) -> TypeResult<()> {
        self.inference.check_struct(s)
    }

    fn check_enum(&mut self, e: &crate::ast::EnumDef) -> TypeResult<()> {
        self.inference.check_enum(e)
    }

    fn check_impl(&mut self, i: &crate::ast::ImplDef) -> TypeResult<()> {
        self.inference.check_impl(i, &mut self.ownership)
    }

    fn check_trait(&mut self, t: &crate::ast::TraitDef) -> TypeResult<()> {
        self.inference.check_trait(t)
    }

    fn check_const(&mut self, c: &crate::ast::ConstDef) -> TypeResult<()> {
        self.inference.check_const(c)
    }

    fn check_actor(&mut self, a: &crate::ast::ActorDef) -> TypeResult<()> {
        self.inference.check_actor(a, &mut self.ownership)
    }

    /// Register a use statement
    fn register_use(&mut self, u: &crate::ast::UseDef) {
        // Extract path segments
        let segments: Vec<String> = u.path.segments.iter()
            .map(|s| s.ident.name.clone())
            .collect();

        // Determine the alias (use provided alias or last segment)
        let alias = u.alias.as_ref()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| segments.last().cloned().unwrap_or_default());

        // Determine the kind based on what we can find
        let kind = if segments.len() > 0 {
            let last = segments.last().unwrap();
            // Check if it's a known type
            if self.inference.ctx.type_exists(last) {
                ImportKind::Type
            } else if self.inference.ctx.lookup_function(last).is_some() {
                ImportKind::Function
            } else if self.inference.ctx.lookup_module(last).is_some() {
                ImportKind::Module
            } else {
                // Assume it's a type for now
                ImportKind::Type
            }
        } else {
            ImportKind::Type
        };

        self.inference.ctx.register_import(&alias, segments, kind);
    }

    /// Register a module definition
    fn register_mod(&mut self, m: &crate::ast::ModDef) {
        let name = m.name.name.clone();

        // If it's an inline module with items, process them
        if let Some(ref items) = m.items {
            let mut types = Vec::new();
            let mut functions = Vec::new();

            // First, register all items in the module
            for item in items {
                match item {
                    crate::ast::Item::Struct(s) => {
                        types.push(s.name.name.clone());
                        self.inference.register_struct_with_prefix(s, &name);
                    }
                    crate::ast::Item::Enum(e) => {
                        types.push(e.name.name.clone());
                        self.inference.register_enum_with_prefix(e, &name);
                    }
                    crate::ast::Item::Function(f) => {
                        functions.push(f.name.name.clone());
                        self.inference.register_function_with_prefix(f, &name);
                    }
                    _ => {
                        // Recursively handle other items
                    }
                }
            }

            // Register the module itself
            self.inference.ctx.register_module(&name, ModuleDef {
                name: name.clone(),
                types,
                functions,
                is_pub: m.is_pub,
            });
        } else {
            // External module (file-based) - not yet supported
            // Just register an empty module placeholder
            self.inference.ctx.register_module(&name, ModuleDef {
                name: name.clone(),
                types: Vec::new(),
                functions: Vec::new(),
                is_pub: m.is_pub,
            });
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn check_ok(source: &str) -> TypedProgram {
        let (program, errors) = parser::parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        check_program(&program).expect("Type check failed")
    }

    fn check_err(source: &str) -> Vec<TypeError> {
        let (program, errors) = parser::parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        check_program(&program).expect_err("Expected type error")
    }

    #[test]
    fn test_simple_function() {
        let _ = check_ok("fn main() { let x: i32 = 5 }");
    }

    #[test]
    fn test_type_mismatch() {
        let errors = check_err("fn main() { let x: i32 = true }");
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_arithmetic() {
        let _ = check_ok("fn add(a: i32, b: i32) -> i32 { a + b }");
    }

    #[test]
    fn test_struct_literal() {
        let _ = check_ok(r#"
            struct Point { x: i32, y: i32 }
            fn main() {
                let p = Point { x: 1, y: 2 }
            }
        "#);
    }

    #[test]
    fn test_module_public_function() {
        let _ = check_ok(r#"
            mod math {
                pub fn add(a: i64, b: i64) -> i64 { a + b }
            }
            fn main() -> i64 {
                math::add(1, 2)
            }
        "#);
    }

    #[test]
    fn test_module_private_function_error() {
        let errors = check_err(r#"
            mod math {
                fn private_add(a: i64, b: i64) -> i64 { a + b }
            }
            fn main() -> i64 {
                math::private_add(1, 2)
            }
        "#);
        assert!(!errors.is_empty());
        let error_msg = format!("{}", errors[0]);
        assert!(error_msg.contains("private"), "Error should mention 'private': {}", error_msg);
    }

    #[test]
    fn test_module_private_internal_call() {
        // Private functions can be called from within the same module
        let _ = check_ok(r#"
            mod math {
                fn private_helper(x: i64) -> i64 { x * 2 }
                pub fn double(x: i64) -> i64 { private_helper(x) }
            }
            fn main() -> i64 {
                math::double(5)
            }
        "#);
    }

    // ============ Exhaustiveness Tests ============

    #[test]
    fn test_match_bool_exhaustive() {
        let _ = check_ok(r#"
            fn test(x: bool) -> i64 {
                match x {
                    true => 1,
                    false => 0,
                }
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_match_bool_with_wildcard() {
        let _ = check_ok(r#"
            fn test(x: bool) -> i64 {
                match x {
                    true => 1,
                    _ => 0,
                }
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_match_bool_missing_case() {
        // Only matching true should fail
        let errors = check_err(r#"
            fn test(x: bool) -> i64 {
                match x {
                    true => 1,
                }
            }
            fn main() {}
        "#);
        assert!(!errors.is_empty());
        let error_msg = format!("{}", errors[0]);
        assert!(error_msg.contains("non-exhaustive") || error_msg.contains("false"),
            "Error should mention non-exhaustive: {}", error_msg);
    }

    #[test]
    fn test_match_option_exhaustive() {
        let _ = check_ok(r#"
            fn test(opt: Option<i64>) -> i64 {
                match opt {
                    Some(v) => v,
                    None => 0,
                }
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_match_option_missing_none() {
        let errors = check_err(r#"
            fn test(opt: Option<i64>) -> i64 {
                match opt {
                    Some(v) => v,
                }
            }
            fn main() {}
        "#);
        assert!(!errors.is_empty());
        let error_msg = format!("{}", errors[0]);
        assert!(error_msg.contains("non-exhaustive") || error_msg.contains("None"),
            "Error should mention non-exhaustive or None: {}", error_msg);
    }

    #[test]
    fn test_match_option_with_wildcard() {
        // Wildcard makes it exhaustive
        let _ = check_ok(r#"
            fn test(opt: Option<i64>) -> i64 {
                match opt {
                    Some(v) => v,
                    _ => 0,
                }
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_match_result_exhaustive() {
        let _ = check_ok(r#"
            fn test(res: Result<i64, i64>) -> i64 {
                match res {
                    Ok(v) => v,
                    Err(e) => e,
                }
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_match_binding_exhaustive() {
        // A simple binding should be exhaustive
        let _ = check_ok(r#"
            fn test(x: i64) -> i64 {
                match x {
                    n => n,
                }
            }
            fn main() {}
        "#);
    }

    // ============ Trait Tests ============

    #[test]
    fn test_trait_definition() {
        // Basic trait definition should work
        let _ = check_ok(r#"
            trait Speak {
                fn speak(&self) -> i64
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_trait_impl_complete() {
        // Complete trait implementation should work
        let _ = check_ok(r#"
            trait Speak {
                fn speak(&self) -> i64
            }

            struct Dog {}

            impl Speak for Dog {
                fn speak(&self) -> i64 {
                    1
                }
            }
            fn main() {}
        "#);
    }

    #[test]
    fn test_trait_impl_missing_method() {
        // Missing trait method should fail
        let errors = check_err(r#"
            trait Speak {
                fn speak(&self) -> i64
                fn name(&self) -> i64
            }

            struct Dog {}

            impl Speak for Dog {
                fn speak(&self) -> i64 {
                    1
                }
            }
            fn main() {}
        "#);
        assert!(!errors.is_empty());
        let error_msg = format!("{}", errors[0]);
        assert!(error_msg.contains("missing") || error_msg.contains("name"),
            "Error should mention missing method: {}", error_msg);
    }

    #[test]
    fn test_trait_not_found() {
        // Implementing non-existent trait should fail
        let errors = check_err(r#"
            struct Dog {}

            impl UnknownTrait for Dog {
                fn speak(&self) -> i64 {
                    1
                }
            }
            fn main() {}
        "#);
        assert!(!errors.is_empty());
        let error_msg = format!("{}", errors[0]);
        assert!(error_msg.contains("not found") || error_msg.contains("UnknownTrait"),
            "Error should mention trait not found: {}", error_msg);
    }

    #[test]
    fn test_trait_method_call() {
        // Calling trait method on implementing type should work
        let _ = check_ok(r#"
            trait Speak {
                fn speak(&self) -> i64
            }

            struct Dog {}

            impl Speak for Dog {
                fn speak(&self) -> i64 {
                    42
                }
            }

            fn main() -> i64 {
                let d = Dog {}
                d.speak()
            }
        "#);
    }

    #[test]
    fn test_multiple_trait_impls() {
        // Multiple traits for same type should work
        let _ = check_ok(r#"
            trait Speak {
                fn speak(&self) -> i64
            }

            trait Walk {
                fn walk(&self) -> i64
            }

            struct Dog {}

            impl Speak for Dog {
                fn speak(&self) -> i64 { 1 }
            }

            impl Walk for Dog {
                fn walk(&self) -> i64 { 2 }
            }

            fn main() -> i64 {
                let d = Dog {}
                d.speak() + d.walk()
            }
        "#);
    }

    #[test]
    fn test_inherent_and_trait_impl() {
        // Both inherent impl and trait impl should work
        let _ = check_ok(r#"
            trait Speak {
                fn speak(&self) -> i64
            }

            struct Dog {
                age: i64,
            }

            impl Dog {
                fn get_age(&self) -> i64 {
                    self.age
                }
            }

            impl Speak for Dog {
                fn speak(&self) -> i64 {
                    42
                }
            }

            fn main() -> i64 {
                let d = Dog { age: 5 }
                d.get_age() + d.speak()
            }
        "#);
    }

    // ============ Generic Struct Tests ============

    #[test]
    fn test_generic_struct_simple() {
        let _ = check_ok(r#"
            struct Wrapper<T> {
                value: T
            }
            fn main() -> i64 {
                let w = Wrapper { value: 42 }
                w.value
            }
        "#);
    }

    #[test]
    fn test_generic_struct_two_params() {
        let _ = check_ok(r#"
            struct Pair<A, B> {
                first: A,
                second: B
            }
            fn main() -> i64 {
                let p = Pair { first: 10, second: 20 }
                p.first + p.second
            }
        "#);
    }
}
