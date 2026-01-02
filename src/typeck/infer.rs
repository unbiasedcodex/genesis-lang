//! Type Inference Engine
//!
//! This module implements type inference using an extended Hindley-Milner algorithm.
//! It handles:
//! - Expression type inference
//! - Pattern matching
//! - Generic instantiation
//! - Method resolution

use std::collections::HashMap;

use crate::ast::{
    self, BinaryOp, Block, ConstDef, EnumDef, Expr, ExprKind, FnDef, Generics, ImplDef, Literal, Pattern,
    PatternKind, Stmt, StmtKind, StructDef, TraitDef, Type as AstType, TypeKind as AstTypeKind,
    UnaryOp, ActorDef, TypeAlias, MacroInvocation,
};
use crate::span::Span;
use crate::typeck::context::{TypeContext, FnSig, ImplDef as CtxImplDef, ActorDef as CtxActorDef, TraitDef as CtxTraitDef, MessageDef};
use crate::typeck::error::{TypeError, TypeResult};
use crate::typeck::exhaustiveness::{ExhaustivenessChecker, format_missing_patterns};
use crate::typeck::ownership::OwnershipChecker;
use crate::typeck::ty::{Ty, TyKind};
use crate::typeck::unify::Unifier;
use crate::macro_expand::MacroExpander;

/// The type inference engine
pub struct TypeInference {
    /// Type context with all definitions
    pub ctx: TypeContext,
    /// The unifier for type unification
    unifier: Unifier,
    /// Map from expression spans to inferred types
    expr_types: HashMap<Span, Ty>,
    /// Current return type (for checking return statements)
    current_return_type: Option<Ty>,
    /// Are we inside a loop?
    in_loop: bool,
    /// Generic function calls: span -> (fn_name, type_args with type vars)
    /// These are resolved to concrete types in get_generic_fn_calls
    generic_fn_calls_raw: HashMap<Span, (String, Vec<Ty>)>,
    /// Macro expander for user-defined macros
    macro_expander: MacroExpander,
    /// Current Self type when inside an impl block (for Self::method() calls)
    current_self_type: Option<Ty>,
}

impl TypeInference {
    /// Create a new type inference engine
    pub fn new() -> Self {
        Self {
            ctx: TypeContext::new(),
            unifier: Unifier::new(),
            expr_types: HashMap::new(),
            current_return_type: None,
            in_loop: false,
            generic_fn_calls_raw: HashMap::new(),
            macro_expander: MacroExpander::new(),
            current_self_type: None,
        }
    }

    /// Unify two types with type alias expansion
    /// This expands type aliases before unification so that
    /// `type Meters = i64; let x: Meters = 10; let y: i64 = x` works correctly.
    fn unify_with_alias_expansion(&mut self, t1: &Ty, t2: &Ty, span: Span) -> TypeResult<Ty> {
        let t1_expanded = self.ctx.expand_type_alias(t1);
        let t2_expanded = self.ctx.expand_type_alias(t2);
        self.unifier.unify(&t1_expanded, &t2_expanded, span)
    }

    /// Get inferred expression types
    pub fn get_expr_types(&self) -> HashMap<Span, Ty> {
        self.expr_types.clone()
    }

    /// Get generic function calls for monomorphization
    /// This resolves type variables to concrete types (defaulting IntVar to i64, FloatVar to f64)
    pub fn get_generic_fn_calls(&self) -> HashMap<Span, (String, Vec<Ty>)> {
        let mut result = HashMap::new();
        for (span, (fn_name, type_args)) in &self.generic_fn_calls_raw {
            // Apply unifier to resolve type variables
            let resolved_args: Vec<Ty> = type_args
                .iter()
                .map(|t| self.unifier.apply_with_defaults(t))
                .collect();

            // Only include if all types are now concrete
            if resolved_args.iter().all(|t| !t.has_vars() && !t.has_generics()) {
                result.insert(*span, (fn_name.clone(), resolved_args));
            }
        }
        result
    }

    /// Get symbol types
    pub fn get_symbol_types(&self) -> HashMap<String, Ty> {
        let types = HashMap::new();
        // This would collect all variable types - simplified for now
        types
    }

    // ============ Registration Phase ============

    pub fn register_struct(&mut self, s: &StructDef) {
        let generics: Vec<String> = s
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Enter generic scope so T is recognized as a generic parameter
        self.ctx.enter_generic_scope(&s.generics);

        let fields: Vec<(String, Ty)> = s
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), self.ast_type_to_ty(&f.ty)))
            .collect();

        // Extract generic defaults
        let generic_defaults = self.extract_generic_defaults(&s.generics);

        // Leave generic scope
        self.ctx.leave_generic_scope();

        let module = self.ctx.current_module().cloned();
        self.ctx.register_struct(&s.name.name, generics, generic_defaults, fields, s.is_pub, module);
    }

    pub fn register_enum(&mut self, e: &EnumDef) {
        let generics: Vec<String> = e
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Extract generic defaults
        let generic_defaults = self.extract_generic_defaults(&e.generics);

        let variants: Vec<(String, Vec<Ty>)> = e
            .variants
            .iter()
            .map(|v| {
                let tys = match &v.kind {
                    ast::VariantKind::Unit => vec![],
                    ast::VariantKind::Tuple(types) => {
                        types.iter().map(|t| self.ast_type_to_ty(t)).collect()
                    }
                    ast::VariantKind::Struct(fields) => {
                        fields.iter().map(|f| self.ast_type_to_ty(&f.ty)).collect()
                    }
                };
                (v.name.name.clone(), tys)
            })
            .collect();

        let module = self.ctx.current_module().cloned();
        self.ctx.register_enum(&e.name.name, generics, generic_defaults, variants, e.is_pub, module);
    }

    pub fn register_function(&mut self, f: &FnDef) {
        let generics: Vec<String> = f
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Enter generic scope so T is recognized as a generic parameter
        self.ctx.enter_generic_scope(&f.generics);

        let params: Vec<Ty> = f.params.iter().map(|p| self.ast_type_to_ty(&p.ty)).collect();
        let base_ret = f
            .return_type
            .as_ref()
            .map(|t| self.ast_type_to_ty(t))
            .unwrap_or_else(Ty::unit);

        // Leave generic scope
        self.ctx.leave_generic_scope();

        // Async functions return Future<T> instead of T
        let ret = if f.is_async {
            Ty::future(base_ret)
        } else {
            base_ret
        };

        let module = self.ctx.current_module().cloned();
        // Extract generic bounds and defaults from AST
        let generic_bounds = self.extract_generic_bounds(&f.generics);
        let generic_defaults = self.extract_generic_defaults(&f.generics);

        self.ctx.register_function(
            &f.name.name,
            FnSig {
                name: f.name.name.clone(),
                generics,
                generic_bounds,
                generic_defaults,
                params,
                ret,
                is_method: false,
                is_async: f.is_async,
                is_pub: f.is_pub,
                module,
            },
        );
    }

    /// Register a struct with a module prefix (for inline modules)
    pub fn register_struct_with_prefix(&mut self, s: &StructDef, prefix: &str) {
        let full_name = format!("{}::{}", prefix, s.name.name);
        let generics: Vec<String> = s
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Enter generic scope so T is recognized as a generic parameter
        self.ctx.enter_generic_scope(&s.generics);

        let fields: Vec<(String, Ty)> = s
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), self.ast_type_to_ty(&f.ty)))
            .collect();

        // Extract generic defaults
        let generic_defaults = self.extract_generic_defaults(&s.generics);

        // Leave generic scope
        self.ctx.leave_generic_scope();

        // Register with full name (Module::Type)
        self.ctx.register_struct(&full_name, generics, generic_defaults, fields, s.is_pub, Some(prefix.to_string()));
    }

    /// Register an enum with a module prefix
    pub fn register_enum_with_prefix(&mut self, e: &EnumDef, prefix: &str) {
        let full_name = format!("{}::{}", prefix, e.name.name);
        let generics: Vec<String> = e
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Extract generic defaults
        let generic_defaults = self.extract_generic_defaults(&e.generics);

        let variants: Vec<(String, Vec<Ty>)> = e
            .variants
            .iter()
            .map(|v| {
                let tys = match &v.kind {
                    ast::VariantKind::Unit => vec![],
                    ast::VariantKind::Tuple(types) => {
                        types.iter().map(|t| self.ast_type_to_ty(t)).collect()
                    }
                    ast::VariantKind::Struct(fields) => {
                        fields.iter().map(|f| self.ast_type_to_ty(&f.ty)).collect()
                    }
                };
                (v.name.name.clone(), tys)
            })
            .collect();

        self.ctx.register_enum(&full_name, generics, generic_defaults, variants, e.is_pub, Some(prefix.to_string()));
    }

    /// Register a function with a module prefix
    pub fn register_function_with_prefix(&mut self, f: &FnDef, prefix: &str) {
        let full_name = format!("{}::{}", prefix, f.name.name);
        let generics: Vec<String> = f
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Enter generic scope so T is recognized as a generic parameter
        self.ctx.enter_generic_scope(&f.generics);

        let params: Vec<Ty> = f.params.iter().map(|p| self.ast_type_to_ty(&p.ty)).collect();
        let base_ret = f
            .return_type
            .as_ref()
            .map(|t| self.ast_type_to_ty(t))
            .unwrap_or_else(Ty::unit);

        // Leave generic scope
        self.ctx.leave_generic_scope();

        let ret = if f.is_async {
            Ty::future(base_ret)
        } else {
            base_ret
        };

        // Extract generic bounds and defaults from AST
        let generic_bounds = self.extract_generic_bounds(&f.generics);
        let generic_defaults = self.extract_generic_defaults(&f.generics);

        self.ctx.register_function(
            &full_name,
            FnSig {
                name: full_name.clone(),
                generics,
                generic_bounds,
                generic_defaults,
                params,
                ret,
                is_method: false,
                is_async: f.is_async,
                is_pub: f.is_pub,
                module: Some(prefix.to_string()),
            },
        );
    }

    pub fn register_trait(&mut self, t: &TraitDef) {
        use crate::typeck::context::AssociatedTypeDef as CtxAssocTypeDef;

        let generics: Vec<String> = t
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        let super_traits: Vec<String> = t
            .super_traits
            .iter()
            .filter_map(|st| {
                if let AstTypeKind::Path(path) = &st.kind {
                    Some(path.segments[0].ident.name.clone())
                } else {
                    None
                }
            })
            .collect();

        let methods: Vec<FnSig> = t
            .items
            .iter()
            .filter_map(|item| {
                if let ast::TraitItem::Function(sig) = item {
                    let params: Vec<Ty> = sig.params.iter().map(|p| self.ast_type_to_ty(&p.ty)).collect();
                    let ret = sig
                        .return_type
                        .as_ref()
                        .map(|rt| self.ast_type_to_ty(rt))
                        .unwrap_or_else(Ty::unit);
                    Some(FnSig {
                        name: sig.name.name.clone(),
                        generics: vec![],
                        generic_bounds: HashMap::new(),
                        generic_defaults: HashMap::new(),
                        params,
                        ret,
                        is_method: true,
                        is_async: false,
                        is_pub: true, // Trait methods are always public
                        module: None,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Extract associated types from trait items
        let associated_types: HashMap<String, CtxAssocTypeDef> = t
            .items
            .iter()
            .filter_map(|item| {
                if let ast::TraitItem::AssociatedType(assoc) = item {
                    let bounds: Vec<String> = assoc.bounds.iter()
                        .filter_map(|b| {
                            if let AstTypeKind::Path(path) = &b.kind {
                                Some(path.segments[0].ident.name.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    let default = assoc.default.as_ref().map(|ty| self.ast_type_to_ty(ty));
                    Some((assoc.name.name.clone(), CtxAssocTypeDef { bounds, default }))
                } else {
                    None
                }
            })
            .collect();

        self.ctx.register_trait(
            &t.name.name,
            CtxTraitDef {
                name: t.name.name.clone(),
                generics,
                super_traits,
                methods,
                associated_types,
            },
        );
    }

    pub fn register_type_alias(&mut self, t: &TypeAlias) {
        let generics: Vec<String> = t
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        // Extract generic defaults
        let generic_defaults = self.extract_generic_defaults(&t.generics);

        let target = self.ast_type_to_ty(&t.ty);
        let module = self.ctx.current_module().cloned();
        self.ctx.register_type_alias(&t.name.name, generics, generic_defaults, target, t.is_pub, module);
    }

    pub fn register_macro(&mut self, m: &crate::ast::MacroDef) {
        self.ctx.register_macro(m);
        // Also register with the macro expander for expansion
        self.macro_expander.register(m.clone());
    }

    pub fn register_actor(&mut self, a: &ActorDef) {
        let generics: Vec<String> = a
            .generics
            .as_ref()
            .map(|g| g.params.iter().map(|p| p.name.name.clone()).collect())
            .unwrap_or_default();

        let state: Vec<(String, Ty)> = a
            .state
            .iter()
            .map(|f| (f.name.name.clone(), self.ast_type_to_ty(&f.ty)))
            .collect();

        // Extract message definitions from message handlers
        let messages: Vec<MessageDef> = a
            .receive
            .iter()
            .filter_map(|h| {
                self.extract_message_def(&h.pattern)
            })
            .collect();

        self.ctx.register_actor(
            &a.name.name,
            CtxActorDef {
                name: a.name.name.clone(),
                generics,
                state,
                messages,
            },
        );
    }

    pub fn register_impl(&mut self, i: &ImplDef) {
        let self_type = if let AstTypeKind::Path(path) = &i.self_type.kind {
            path.segments[0].ident.name.clone()
        } else {
            return;
        };

        let trait_name = i.trait_.as_ref().and_then(|t| {
            if let AstTypeKind::Path(path) = &t.kind {
                Some(path.segments[0].ident.name.clone())
            } else {
                None
            }
        });

        let methods: Vec<FnSig> = i
            .items
            .iter()
            .filter_map(|item| {
                if let ast::ImplItem::Function(f) = item {
                    let params: Vec<Ty> = f.params.iter().map(|p| self.ast_type_to_ty(&p.ty)).collect();
                    let base_ret = f
                        .return_type
                        .as_ref()
                        .map(|rt| self.ast_type_to_ty(rt))
                        .unwrap_or_else(Ty::unit);
                    // Async methods return Future<T>
                    let ret = if f.is_async {
                        Ty::future(base_ret)
                    } else {
                        base_ret
                    };
                    Some(FnSig {
                        name: f.name.name.clone(),
                        generics: vec![],
                        generic_bounds: HashMap::new(),
                        generic_defaults: HashMap::new(),
                        params,
                        ret,
                        is_method: true,
                        is_async: f.is_async,
                        is_pub: f.is_pub,
                        module: None, // Methods inherit visibility from impl block
                    })
                } else {
                    None
                }
            })
            .collect();

        // Extract associated type implementations from impl items
        let associated_types: HashMap<String, Ty> = i
            .items
            .iter()
            .filter_map(|item| {
                if let ast::ImplItem::TypeAlias(ta) = item {
                    let ty = self.ast_type_to_ty(&ta.ty);
                    Some((ta.name.name.clone(), ty))
                } else {
                    None
                }
            })
            .collect();

        self.ctx.register_impl(CtxImplDef {
            self_type,
            trait_name,
            methods,
            associated_types,
        });
    }

    pub fn register_const(&mut self, c: &ConstDef) {
        let ty = c
            .ty
            .as_ref()
            .map(|t| self.ast_type_to_ty(t))
            .unwrap_or_else(Ty::fresh_var);
        self.ctx.define_var(&c.name.name, ty, false);
    }

    // ============ Generic Bounds Utilities ============

    /// Extract generic bounds from AST Generics into a HashMap
    /// This includes both inline bounds (T: Clone) and where clause bounds (where T: Clone)
    fn extract_generic_bounds(&self, generics: &Option<Generics>) -> HashMap<String, Vec<String>> {
        let mut bounds_map: HashMap<String, Vec<String>> = HashMap::new();
        if let Some(g) = generics {
            // 1. Extract inline bounds from generic params: <T: Clone, U: Debug>
            for param in &g.params {
                let param_name = param.name.name.clone();
                let bounds: Vec<String> = param.bounds.iter()
                    .filter_map(|bound| {
                        // Bounds are stored as Type, extract the trait name from Path
                        match &bound.kind {
                            AstTypeKind::Path(path) => {
                                Some(path.segments[0].ident.name.clone())
                            }
                            _ => None,
                        }
                    })
                    .collect();
                if !bounds.is_empty() {
                    bounds_map.entry(param_name).or_default().extend(bounds);
                }
            }

            // 2. Extract where clause bounds: where T: Clone + Debug, U: Display
            if let Some(where_clause) = &g.where_clause {
                for predicate in &where_clause.predicates {
                    // Extract the type name (e.g., "T" from "T: Clone")
                    let ty_name = match &predicate.ty.kind {
                        AstTypeKind::Path(path) => {
                            path.segments[0].ident.name.clone()
                        }
                        _ => continue,
                    };

                    // Extract trait bounds
                    let bounds: Vec<String> = predicate.bounds.iter()
                        .filter_map(|bound| {
                            match &bound.kind {
                                AstTypeKind::Path(path) => {
                                    Some(path.segments[0].ident.name.clone())
                                }
                                _ => None,
                            }
                        })
                        .collect();

                    if !bounds.is_empty() {
                        bounds_map.entry(ty_name).or_default().extend(bounds);
                    }
                }
            }
        }
        bounds_map
    }

    /// Extract default types for generic parameters from AST Generics
    fn extract_generic_defaults(&self, generics: &Option<Generics>) -> HashMap<String, Ty> {
        let mut defaults_map: HashMap<String, Ty> = HashMap::new();
        if let Some(g) = generics {
            for param in &g.params {
                if let Some(default_type) = &param.default {
                    let ty = self.ast_type_to_ty(default_type);
                    defaults_map.insert(param.name.name.clone(), ty);
                }
            }
        }
        defaults_map
    }

    /// Apply default type parameters when fewer type arguments are provided than expected.
    /// Example: If Container<T = i64> is used as Container, this fills in T = i64.
    fn apply_default_type_params(&self, type_name: &str, mut provided: Vec<Ty>) -> Vec<Ty> {
        // Look up the type definition using public accessor
        if let Some(typedef) = self.ctx.lookup_type(type_name) {
            let expected_count = typedef.generics.len();
            let provided_count = provided.len();

            // If fewer type arguments provided than expected, apply defaults
            if provided_count < expected_count {
                // Clone the needed data to avoid borrow issues
                let generics = typedef.generics.clone();
                let defaults = typedef.generic_defaults.clone();

                for i in provided_count..expected_count {
                    let param_name = &generics[i];
                    if let Some(default_ty) = defaults.get(param_name) {
                        provided.push(default_ty.clone());
                    } else {
                        // No default, leave as type variable (will be inferred or cause error)
                        provided.push(Ty::fresh_var());
                    }
                }
            }
        }
        provided
    }

    /// Check that concrete types satisfy trait bounds when calling a generic function
    /// generic_mapping: maps generic param names to concrete types
    /// bounds: maps generic param names to required trait names
    pub fn check_trait_bounds(
        &self,
        generic_mapping: &HashMap<String, Ty>,
        bounds: &HashMap<String, Vec<String>>,
        span: Span,
    ) -> TypeResult<()> {
        use crate::typeck::check::satisfies_bound;

        for (generic_name, required_traits) in bounds {
            if let Some(concrete_ty) = generic_mapping.get(generic_name) {
                // Apply unifier to get the resolved type
                let resolved_ty = self.unifier.apply(concrete_ty);

                for trait_name in required_traits {
                    // First check built-in traits
                    let satisfies_builtin = satisfies_bound(&resolved_ty, trait_name);

                    // Then check user-defined trait implementations
                    let satisfies_user_defined = self.ctx.implements_trait(&resolved_ty, trait_name);

                    if !satisfies_builtin && !satisfies_user_defined {
                        return Err(TypeError::trait_not_implemented(
                            resolved_ty.clone(),
                            trait_name.clone(),
                            span,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    // ============ Checking Phase ============

    pub fn check_function(&mut self, f: &FnDef, ownership: &mut OwnershipChecker) -> TypeResult<()> {
        self.check_function_with_self(f, ownership, None)
    }

    pub fn check_function_with_self(&mut self, f: &FnDef, ownership: &mut OwnershipChecker, self_ty: Option<&Ty>) -> TypeResult<()> {
        self.ctx.enter_scope();
        self.ctx.enter_generic_scope(&f.generics);

        // Add parameters to scope
        for param in &f.params {
            let ty = self.resolve_self_type(&param.ty, self_ty);
            self.ctx.define_var(&param.name.name, ty.clone(), param.is_mut);
            ownership.define(&param.name.name, param.is_mut);
        }

        // Set expected return type
        let expected_ret = f
            .return_type
            .as_ref()
            .map(|t| self.resolve_self_type(t, self_ty))
            .unwrap_or_else(Ty::unit);
        self.current_return_type = Some(expected_ret.clone());

        // Check body
        let body_ty = self.infer_block(&f.body)?;

        // Unify with expected return type
        self.unify_with_alias_expansion(&body_ty, &expected_ret, f.span)?;

        self.current_return_type = None;
        self.ctx.leave_generic_scope();
        self.ctx.leave_scope();
        Ok(())
    }

    /// Resolve Self type references in a type
    fn resolve_self_type(&self, ast_ty: &AstType, self_ty: Option<&Ty>) -> Ty {
        match &ast_ty.kind {
            AstTypeKind::Path(path) => {
                let name = &path.segments[0].ident.name;
                // Check if it's Self and we have a self_ty
                if name == "Self" {
                    if let Some(ty) = self_ty {
                        return ty.clone();
                    }
                }
                // Otherwise use normal conversion
                self.ast_type_to_ty(ast_ty)
            }
            AstTypeKind::Reference { mutable, inner } => {
                let inner_ty = self.resolve_self_type(inner, self_ty);
                Ty::reference(inner_ty, *mutable)
            }
            AstTypeKind::SelfType => {
                if let Some(ty) = self_ty {
                    ty.clone()
                } else {
                    Ty::generic("Self".to_string())
                }
            }
            AstTypeKind::Projection { base, assoc_name } => {
                let base_ty = self.resolve_self_type(base, self_ty);
                // Try to resolve the projection if we have a concrete type
                self.resolve_projection(&base_ty, assoc_name)
            }
            _ => self.ast_type_to_ty(ast_ty),
        }
    }

    /// Resolve a projection type (e.g., Self::Item) to its concrete type
    fn resolve_projection(&self, base_ty: &Ty, assoc_name: &str) -> Ty {
        // Get the type name from base_ty
        let type_name = match &base_ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            TyKind::Generic { name, .. } => {
                // If it's still generic (like Self), we can't resolve yet
                return Ty::projection(base_ty.clone(), assoc_name.to_string());
            }
            _ => return Ty::projection(base_ty.clone(), assoc_name.to_string()),
        };

        // Try to resolve the associated type from impl
        if let Some(resolved) = self.ctx.resolve_associated_type_by_name(&type_name, assoc_name) {
            return resolved;
        }

        // Could not resolve, keep as projection
        Ty::projection(base_ty.clone(), assoc_name.to_string())
    }

    /// Resolve projection types in a Ty, substituting Self with self_ty
    fn resolve_ty_projection(&self, ty: &Ty, self_ty: &Ty) -> Ty {
        match &ty.kind {
            TyKind::Projection { base_ty, assoc_name, .. } => {
                // Resolve the base type first
                let resolved_base = self.resolve_ty_projection(base_ty, self_ty);

                // If base is Self (generic), substitute with self_ty
                let actual_base = match &resolved_base.kind {
                    TyKind::Generic { name } if name == "Self" => self_ty.clone(),
                    _ => resolved_base,
                };

                // Now resolve the projection
                self.resolve_projection(&actual_base, assoc_name)
            }
            TyKind::Generic { name } if name == "Self" => self_ty.clone(),
            TyKind::Ref { inner, mutable } => {
                Ty::reference(self.resolve_ty_projection(inner, self_ty), *mutable)
            }
            TyKind::Array { element, size } => {
                Ty::array(self.resolve_ty_projection(element, self_ty), *size)
            }
            TyKind::Slice { element } => {
                Ty::slice(self.resolve_ty_projection(element, self_ty))
            }
            TyKind::Tuple(elements) => {
                Ty::tuple(elements.iter().map(|e| self.resolve_ty_projection(e, self_ty)).collect())
            }
            TyKind::Fn { params, ret } => {
                Ty::function(
                    params.iter().map(|p| self.resolve_ty_projection(p, self_ty)).collect(),
                    self.resolve_ty_projection(ret, self_ty),
                )
            }
            TyKind::Named { name, generics } => {
                Ty::named(
                    name.clone(),
                    generics.iter().map(|g| self.resolve_ty_projection(g, self_ty)).collect(),
                )
            }
            _ => ty.clone(),
        }
    }

    pub fn check_struct(&mut self, _s: &StructDef) -> TypeResult<()> {
        // Struct types are already registered; just validate field types exist
        Ok(())
    }

    pub fn check_enum(&mut self, _e: &EnumDef) -> TypeResult<()> {
        // Enum types are already registered
        Ok(())
    }

    pub fn check_impl(&mut self, i: &ImplDef, ownership: &mut OwnershipChecker) -> TypeResult<()> {
        self.ctx.enter_generic_scope(&i.generics);

        // Get the Self type for this impl block
        let self_ty = self.ast_type_to_ty(&i.self_type);
        let self_type_name = if let AstTypeKind::Path(path) = &i.self_type.kind {
            path.segments[0].ident.name.clone()
        } else {
            "Unknown".to_string()
        };

        // Set the current Self type for Self::method() resolution
        let old_self_type = self.current_self_type.take();
        self.current_self_type = Some(self_ty.clone());

        // If this is a trait implementation, verify it
        if let Some(trait_type) = &i.trait_ {
            let trait_name = if let AstTypeKind::Path(path) = &trait_type.kind {
                path.segments[0].ident.name.clone()
            } else {
                return Err(TypeError::trait_not_found("Unknown".to_string(), i.span));
            };

            // Look up the trait
            if let Some(trait_def) = self.ctx.lookup_trait(&trait_name).cloned() {
                // Collect implemented method names
                let impl_methods: std::collections::HashSet<String> = i.items.iter()
                    .filter_map(|item| {
                        if let ast::ImplItem::Function(f) = item {
                            Some(f.name.name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                // Check that all required trait methods are implemented
                for trait_method in &trait_def.methods {
                    if !impl_methods.contains(&trait_method.name) {
                        return Err(TypeError::missing_trait_method(
                            trait_name.clone(),
                            trait_method.name.clone(),
                            self_type_name.clone(),
                            i.span,
                        ));
                    }

                    // Find the impl method and check signature compatibility
                    if let Some(ast::ImplItem::Function(impl_fn)) = i.items.iter().find(|item| {
                        if let ast::ImplItem::Function(f) = item {
                            f.name.name == trait_method.name
                        } else {
                            false
                        }
                    }) {
                        // Check parameter count (excluding self)
                        // For impl methods, first param is often self
                        let impl_params: Vec<_> = impl_fn.params.iter()
                            .filter(|p| p.name.name != "self")
                            .collect();

                        // For trait method, params are Vec<Ty>, and is_method indicates if self is implicit
                        let trait_param_count = if trait_method.is_method {
                            trait_method.params.len().saturating_sub(1) // Subtract self
                        } else {
                            trait_method.params.len()
                        };

                        if impl_params.len() != trait_param_count {
                            return Err(TypeError::trait_method_signature_mismatch(
                                trait_name.clone(),
                                trait_method.name.clone(),
                                format!("{} parameter(s)", trait_param_count),
                                format!("{} parameter(s)", impl_params.len()),
                                impl_fn.span,
                            ));
                        }

                        // Check return type compatibility
                        let impl_ret = impl_fn.return_type.as_ref()
                            .map(|rt| self.ast_type_to_ty(rt))
                            .unwrap_or_else(Ty::unit);
                        let trait_ret = trait_method.ret.clone();

                        // Allow some flexibility in return types
                        if !self.types_compatible(&impl_ret, &trait_ret) {
                            return Err(TypeError::trait_method_signature_mismatch(
                                trait_name.clone(),
                                trait_method.name.clone(),
                                format!("-> {}", trait_ret),
                                format!("-> {}", impl_ret),
                                impl_fn.span,
                            ));
                        }
                    }
                }

                // Check that all required associated types are implemented
                let impl_assoc_types: std::collections::HashSet<String> = i.items.iter()
                    .filter_map(|item| {
                        if let ast::ImplItem::TypeAlias(ta) = item {
                            Some(ta.name.name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                for (assoc_name, assoc_def) in &trait_def.associated_types {
                    // Check if the associated type is implemented (or has a default)
                    if !impl_assoc_types.contains(assoc_name) && assoc_def.default.is_none() {
                        return Err(TypeError::missing_associated_type(
                            trait_name.clone(),
                            assoc_name.clone(),
                            self_type_name.clone(),
                            i.span,
                        ));
                    }
                }
            } else {
                return Err(TypeError::trait_not_found(trait_name, i.span));
            }
        }

        // Check all methods in the impl
        for item in &i.items {
            if let ast::ImplItem::Function(f) = item {
                self.check_function_with_self(f, ownership, Some(&self_ty))?;
            }
        }

        // Restore the old Self type
        self.current_self_type = old_self_type;
        self.ctx.leave_generic_scope();
        Ok(())
    }

    /// Check if two types are compatible for trait impl
    fn types_compatible(&self, ty1: &Ty, ty2: &Ty) -> bool {
        // Exact match
        if ty1 == ty2 {
            return true;
        }

        // Generic types are compatible with any concrete type
        if matches!(&ty2.kind, TyKind::Generic { .. }) {
            return true;
        }
        if matches!(&ty1.kind, TyKind::Generic { .. }) {
            return true;
        }

        // Type variables are compatible
        if matches!(&ty1.kind, TyKind::Var(_)) || matches!(&ty2.kind, TyKind::Var(_)) {
            return true;
        }

        // Named types with same name are compatible (generics may differ)
        if let (TyKind::Named { name: n1, .. }, TyKind::Named { name: n2, .. }) = (&ty1.kind, &ty2.kind) {
            return n1 == n2;
        }

        false
    }

    pub fn check_trait(&mut self, _t: &TraitDef) -> TypeResult<()> {
        // Trait is already registered
        Ok(())
    }

    pub fn check_const(&mut self, c: &ConstDef) -> TypeResult<()> {
        let inferred = self.infer_expr(&c.value)?;
        if let Some(declared) = &c.ty {
            let declared_ty = self.ast_type_to_ty(declared);
            self.unify_with_alias_expansion(&inferred, &declared_ty, c.span)?;
        }
        Ok(())
    }

    pub fn check_actor(&mut self, a: &ActorDef, _ownership: &mut OwnershipChecker) -> TypeResult<()> {
        self.ctx.enter_scope();
        self.ctx.enter_generic_scope(&a.generics);

        // Add state fields to scope
        for field in &a.state {
            let ty = self.ast_type_to_ty(&field.ty);
            self.ctx.define_var(&field.name.name, ty, true);
        }

        // Check message handlers
        for handler in &a.receive {
            let _ = self.infer_expr(&handler.body)?;
        }

        self.ctx.leave_generic_scope();
        self.ctx.leave_scope();
        Ok(())
    }

    // ============ Expression Inference ============

    pub fn infer_expr(&mut self, expr: &Expr) -> TypeResult<Ty> {
        let ty = self.infer_expr_kind(&expr.kind, expr.span)?;
        let resolved = self.unifier.apply(&ty);
        self.expr_types.insert(expr.span, resolved.clone());
        Ok(resolved)
    }

    fn infer_expr_kind(&mut self, kind: &ExprKind, span: Span) -> TypeResult<Ty> {
        match kind {
            ExprKind::Literal(lit) => self.infer_literal(lit),

            ExprKind::Path(path) => {
                // For single-segment paths, use simple resolution
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident.name;

                    // Check if it's Self (returns the current Self type in impl blocks)
                    if name == "Self" {
                        if let Some(self_ty) = &self.current_self_type {
                            return Ok(self_ty.clone());
                        }
                        // If no Self type, treat as a generic type parameter
                        return Ok(Ty::generic("Self".to_string()));
                    }

                    // Check if it's a variable
                    if let Some(symbol) = self.ctx.lookup_var(name) {
                        return Ok(symbol.ty.clone());
                    }

                    // Check if it's a function
                    if let Some(sig) = self.ctx.lookup_function(name) {
                        return Ok(Ty::function(sig.params.clone(), sig.ret.clone()));
                    }

                    // Check if it's a type (enum variant)
                    if let Some(_) = self.ctx.lookup_type(name) {
                        return Ok(Ty::named(name.clone(), vec![]));
                    }

                    // Check if it's an imported symbol
                    if let Some(import) = self.ctx.lookup_import(name) {
                        let original_name = import.original_path.last().unwrap();
                        if let Some(sig) = self.ctx.lookup_function(original_name) {
                            return Ok(Ty::function(sig.params.clone(), sig.ret.clone()));
                        }
                        if let Some(_) = self.ctx.lookup_type(original_name) {
                            return Ok(Ty::named(original_name.clone(), vec![]));
                        }
                    }

                    Err(TypeError::undefined_variable(name.clone(), span))
                } else {
                    // Multi-segment path like Module::function or Type::variant
                    let full_path: Vec<String> = path.segments.iter()
                        .map(|s| s.ident.name.clone())
                        .collect();
                    let full_name = full_path.join("::");

                    // Check if first segment is Self (for Self::method() calls)
                    let first = &path.segments[0].ident.name;
                    if first == "Self" {
                        if let Some(self_ty) = &self.current_self_type.clone() {
                            // Get the type name from the current Self type
                            let type_name = match &self_ty.kind {
                                TyKind::Named { name, .. } => name.clone(),
                                _ => "Self".to_string(),
                            };

                            // Replace Self with the actual type name
                            let mut resolved_path = full_path.clone();
                            resolved_path[0] = type_name.clone();
                            let resolved_name = resolved_path.join("::");

                            // Check if it's a function on the Self type (e.g., Self::new -> MyType::new)
                            // First try lookup_function (for built-in types like Option, Result)
                            if let Some(sig) = self.ctx.lookup_function(&resolved_name) {
                                return Ok(Ty::function(sig.params.clone(), sig.ret.clone()));
                            }

                            // Also check impl methods (for user-defined types like Counter, Point)
                            if path.segments.len() == 2 {
                                let method_name = &path.segments[1].ident.name;
                                if let Some(sig) = self.ctx.find_method(&self_ty, method_name) {
                                    // Substitute Self with the concrete type in params and return type
                                    let params: Vec<Ty> = sig.params.iter()
                                        .map(|p| self.substitute_self_type(p, &self_ty))
                                        .collect();
                                    let ret = self.substitute_self_type(&sig.ret, &self_ty);
                                    return Ok(Ty::function(params, ret));
                                }
                            }

                            // Check if it's an enum variant on Self type
                            if let Some(type_def) = self.ctx.lookup_type(&type_name) {
                                if let crate::typeck::context::TypeDefKind::Enum { variants } = &type_def.kind {
                                    if path.segments.len() == 2 {
                                        let variant_name = &path.segments[1].ident.name;
                                        let generics_count = type_def.generics.len();
                                        let variants_cloned = variants.clone();

                                        let generic_vars: Vec<Ty> = (0..generics_count)
                                            .map(|_| Ty::fresh_var())
                                            .collect();
                                        for (name, fields) in &variants_cloned {
                                            if name == variant_name {
                                                if fields.is_empty() {
                                                    return Ok(Ty::named(type_name.clone(), generic_vars));
                                                } else {
                                                    let instantiated_fields: Vec<Ty> = fields.iter()
                                                        .map(|f| self.instantiate_generics(f))
                                                        .collect();
                                                    let ret_ty = Ty::named(type_name.clone(), generic_vars);
                                                    let instantiated_ret = self.instantiate_generics(&ret_ty);
                                                    return Ok(Ty::function(instantiated_fields, instantiated_ret));
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            return Err(TypeError::undefined_function(resolved_name, span));
                        }
                        // No Self type set - error
                        return Err(TypeError::undefined_variable("Self".to_string(), span));
                    }

                    // Check if first segment is a module
                    if self.ctx.lookup_module(first).is_some() {
                        // Look up the function with qualified name
                        if let Some(sig) = self.ctx.lookup_function(&full_name) {
                            // Check visibility
                            if !self.ctx.is_visible(sig.module.as_ref(), sig.is_pub) {
                                return Err(TypeError::private_access(
                                    full_name.clone(),
                                    sig.module.as_ref().unwrap_or(&first.to_string()).clone(),
                                    span,
                                ));
                            }
                            return Ok(Ty::function(sig.params.clone(), sig.ret.clone()));
                        }
                        // Look up the type with qualified name
                        if let Some(type_def) = self.ctx.lookup_type(&full_name) {
                            // Check visibility
                            if !self.ctx.is_visible(type_def.module.as_ref(), type_def.is_pub) {
                                return Err(TypeError::private_access(
                                    full_name.clone(),
                                    type_def.module.as_ref().unwrap_or(&first.to_string()).clone(),
                                    span,
                                ));
                            }
                            return Ok(Ty::named(full_name.clone(), vec![]));
                        }
                    }

                    // Check if it's an enum variant path (e.g., Option::Some)
                    if let Some(type_def) = self.ctx.lookup_type(first) {
                        if let crate::typeck::context::TypeDefKind::Enum { variants } = &type_def.kind {
                            let variant_name = &path.segments[1].ident.name;
                            // Clone data we need before releasing the borrow
                            let generics_count = type_def.generics.len();
                            let variants_cloned = variants.clone();

                            // Create fresh type variables for each generic parameter
                            let generic_vars: Vec<Ty> = (0..generics_count)
                                .map(|_| Ty::fresh_var())
                                .collect();
                            for (name, fields) in &variants_cloned {
                                if name == variant_name {
                                    if fields.is_empty() {
                                        // Unit variant (e.g., None) - return enum type with fresh generics
                                        return Ok(Ty::named(first.clone(), generic_vars));
                                    } else {
                                        // Enum variant constructor - instantiate field types
                                        let instantiated_fields: Vec<Ty> = fields.iter()
                                            .map(|f| self.instantiate_generics(f))
                                            .collect();
                                        let ret_ty = Ty::named(first.clone(), generic_vars);
                                        let instantiated_ret = self.instantiate_generics(&ret_ty);
                                        return Ok(Ty::function(instantiated_fields, instantiated_ret));
                                    }
                                }
                            }
                        }
                    }

                    // Check if it's an associated function (e.g., Box::new, Vec::new, Option::is_some, Result::is_ok)
                    if let Some(sig) = self.ctx.lookup_function(&full_name) {
                        return Ok(Ty::function(sig.params.clone(), sig.ret.clone()));
                    }

                    // Check if it's an impl method on a type (e.g., Counter::new, Point::origin)
                    if path.segments.len() == 2 {
                        let type_name = first;
                        let method_name = &path.segments[1].ident.name;
                        // Create a type for the lookup
                        let ty = Ty::named(type_name.clone(), vec![]);
                        if let Some(sig) = self.ctx.find_method(&ty, method_name) {
                            // Substitute Self with the concrete type in params and return type
                            let params: Vec<Ty> = sig.params.iter()
                                .map(|p| self.substitute_self_type(p, &ty))
                                .collect();
                            let ret = self.substitute_self_type(&sig.ret, &ty);
                            return Ok(Ty::function(params, ret));
                        }
                    }

                    Err(TypeError::undefined_variable(full_name, span))
                }
            }

            ExprKind::Binary { op, left, right } => {
                let left_ty = self.infer_expr(left)?;
                let right_ty = self.infer_expr(right)?;
                self.infer_binary_op(*op, &left_ty, &right_ty, span)
            }

            ExprKind::Unary { op, operand } => {
                let operand_ty = self.infer_expr(operand)?;
                self.infer_unary_op(*op, &operand_ty, span)
            }

            ExprKind::Call { func, args } => {
                let func_ty = self.infer_expr(func)?;
                let arg_tys: Vec<Ty> = args.iter().map(|a| self.infer_expr(a)).collect::<Result<_, _>>()?;

                // Get function name and signature if it's a path (for generic function tracking and bounds checking)
                let (func_name, func_sig) = if let ExprKind::Path(path) = &func.kind {
                    let name = path.segments.iter()
                        .map(|s| s.ident.name.as_str())
                        .collect::<Vec<_>>()
                        .join("::");
                    let sig = self.ctx.lookup_function(&name).cloned();
                    (Some(name), sig)
                } else {
                    (None, None)
                };

                match &func_ty.kind {
                    TyKind::Fn { params, ret } => {
                        if params.len() != arg_tys.len() {
                            return Err(TypeError::wrong_arg_count(params.len(), arg_tys.len(), span));
                        }

                        // Check if this is a generic function
                        let is_generic = params.iter().any(|p| p.has_generics()) ||
                                         ret.has_generics();

                        // Create mapping from generic names to fresh type variables
                        let mut generic_vars: HashMap<String, Ty> = HashMap::new();

                        // Instantiate generic parameters with fresh type variables
                        let instantiated_params: Vec<Ty> = params.iter()
                            .map(|p| self.instantiate_generics_with_tracking(p, &mut generic_vars))
                            .collect();
                        let instantiated_ret = self.instantiate_generics_with_tracking(ret, &mut generic_vars);

                        for (param, arg) in instantiated_params.iter().zip(arg_tys.iter()) {
                            self.unify_with_alias_expansion(param, arg, span)?;
                        }

                        // Check trait bounds after unification
                        if is_generic {
                            if let Some(ref sig) = func_sig {
                                if !sig.generic_bounds.is_empty() {
                                    // Build mapping from generic param names to resolved concrete types
                                    let resolved_mapping: HashMap<String, Ty> = generic_vars.iter()
                                        .map(|(name, ty)| (name.clone(), self.unifier.apply(ty)))
                                        .collect();

                                    self.check_trait_bounds(&resolved_mapping, &sig.generic_bounds, span)?;
                                }
                            }
                        }

                        // If this is a generic function, register the call with type vars for later resolution
                        if is_generic {
                            if let Some(ref fn_name) = func_name {
                                // Apply unifier to get partially resolved types (may still have IntVar/FloatVar)
                                let type_args: Vec<Ty> = generic_vars.values()
                                    .map(|t| self.unifier.apply(t))
                                    .collect();

                                // Register for later resolution when defaults are applied
                                self.generic_fn_calls_raw.insert(span, (fn_name.clone(), type_args));
                            }
                        }

                        Ok(self.unifier.apply(&instantiated_ret))
                    }
                    TyKind::Var(_) => {
                        // Create fresh return type
                        let ret = Ty::fresh_var();
                        let expected = Ty::function(arg_tys, ret.clone());
                        self.unify_with_alias_expansion(&func_ty, &expected, span)?;
                        Ok(ret)
                    }
                    _ => Err(TypeError::not_callable(func_ty, span)),
                }
            }

            ExprKind::MethodCall { receiver, method, args } => {
                let receiver_ty = self.infer_expr(receiver)?;
                let arg_tys: Vec<Ty> = args.iter().map(|a| self.infer_expr(a)).collect::<Result<_, _>>()?;

                // Auto-dereference for method lookup (handles &self, &mut self, etc.)
                let lookup_ty = match &receiver_ty.kind {
                    TyKind::Ref { inner, .. } => (**inner).clone(),
                    _ => receiver_ty.clone(),
                };

                if let Some(method_sig) = self.ctx.find_method(&lookup_ty, &method.name).cloned() {
                    // Skip self parameter
                    let expected_params: Vec<Ty> = if method_sig.is_method && !method_sig.params.is_empty() {
                        method_sig.params[1..].to_vec()
                    } else {
                        method_sig.params.clone()
                    };

                    if expected_params.len() != arg_tys.len() {
                        return Err(TypeError::wrong_arg_count(expected_params.len(), arg_tys.len(), span));
                    }

                    for (param, arg) in expected_params.iter().zip(arg_tys.iter()) {
                        self.unify_with_alias_expansion(param, arg, span)?;
                    }

                    // Resolve projections in the return type (e.g., Self::Item -> concrete type)
                    let ret_ty = self.resolve_ty_projection(&method_sig.ret, &lookup_ty);
                    Ok(ret_ty)
                } else {
                    Err(TypeError::undefined_method(lookup_ty, method.name.clone(), span))
                }
            }

            ExprKind::Field { object, field } => {
                let obj_ty = self.infer_expr(object)?;

                // Auto-dereference for field access (handles &self, &mut self, etc.)
                let actual_ty = match &obj_ty.kind {
                    TyKind::Ref { inner, .. } => (**inner).clone(),
                    _ => obj_ty.clone(),
                };

                if let TyKind::Named { name, generics } = &actual_ty.kind {
                    if let Some(fields) = self.ctx.get_struct_fields(name) {
                        // Get the generic parameter names from the type definition
                        let type_def = self.ctx.lookup_type(name);
                        let generic_names: Vec<String> = type_def
                            .map(|td| td.generics.clone())
                            .unwrap_or_default();

                        // Build substitution map: generic_name -> concrete_type
                        let subst_map: std::collections::HashMap<String, Ty> = generic_names
                            .iter()
                            .zip(generics.iter())
                            .map(|(name, ty)| (name.clone(), ty.clone()))
                            .collect();

                        for (fname, fty) in fields {
                            if fname == &field.name {
                                // Substitute generic parameters with concrete types
                                let resolved_ty = self.substitute_type_params(fty, &subst_map);
                                return Ok(resolved_ty);
                            }
                        }
                    }
                }

                // Handle tuple indexing (x.0, x.1, etc.)
                if let TyKind::Tuple(elems) = &actual_ty.kind {
                    if let Ok(idx) = field.name.parse::<usize>() {
                        if idx < elems.len() {
                            return Ok(elems[idx].clone());
                        }
                    }
                }

                Err(TypeError::undefined_field(obj_ty, field.name.clone(), span))
            }

            ExprKind::Index { object, index } => {
                let obj_ty = self.infer_expr(object)?;
                let idx_ty = self.infer_expr(index)?;

                // Index must be usize
                self.unify_with_alias_expansion(&idx_ty, &Ty::usize(), span)?;

                match &obj_ty.kind {
                    TyKind::Array { element, .. } => Ok((**element).clone()),
                    TyKind::Slice { element } => Ok((**element).clone()),
                    TyKind::Named { name, generics } if name == "Vec" => {
                        if let Some(elem) = generics.first() {
                            Ok(elem.clone())
                        } else {
                            Ok(Ty::fresh_var())
                        }
                    }
                    // String indexing returns u8 (byte)
                    TyKind::Named { name, .. } if name == "String" => Ok(Ty::u8()),
                    _ => Err(TypeError::not_indexable(obj_ty, span)),
                }
            }

            ExprKind::Struct { path, fields } => {
                let name = &path.segments[0].ident.name;

                // Look up type definition to get generic parameters
                let type_def = self.ctx.lookup_type(name).cloned();

                if let Some(struct_fields) = self.ctx.get_struct_fields(name) {
                    // Create fresh type variables for each generic parameter
                    let generic_params = type_def.as_ref()
                        .map(|td| td.generics.clone())
                        .unwrap_or_default();

                    let generic_vars: Vec<Ty> = generic_params
                        .iter()
                        .map(|_| Ty::fresh_var())
                        .collect();

                    // Build substitution map: generic_name -> fresh_var
                    let subst_map: std::collections::HashMap<String, Ty> = generic_params
                        .iter()
                        .zip(generic_vars.iter())
                        .map(|(name, ty)| (name.clone(), ty.clone()))
                        .collect();

                    // Substitute generics in field types
                    let struct_fields_substituted: HashMap<String, Ty> = struct_fields
                        .iter()
                        .map(|(n, t)| {
                            let substituted = self.substitute_type_params(t, &subst_map);
                            (n.clone(), substituted)
                        })
                        .collect();

                    for (field_name, field_expr) in fields {
                        if let Some(expected_ty) = struct_fields_substituted.get(&field_name.name) {
                            let actual_ty = self.infer_expr(field_expr)?;
                            self.unify_with_alias_expansion(expected_ty, &actual_ty, span)?;
                        } else {
                            return Err(TypeError::extra_field(name.clone(), field_name.name.clone(), span));
                        }
                    }

                    // Apply unifier to resolve the generic variables
                    let resolved_generics: Vec<Ty> = generic_vars
                        .iter()
                        .map(|gv| self.unifier.apply(gv))
                        .collect();

                    Ok(Ty::named(name.clone(), resolved_generics))
                } else {
                    Err(TypeError::undefined_type(name.clone(), span))
                }
            }

            ExprKind::Array(elements) => {
                if elements.is_empty() {
                    Ok(Ty::array(Ty::fresh_var(), 0))
                } else {
                    let elem_ty = self.infer_expr(&elements[0])?;
                    for elem in &elements[1..] {
                        let ty = self.infer_expr(elem)?;
                        self.unify_with_alias_expansion(&elem_ty, &ty, span)?;
                    }
                    Ok(Ty::array(elem_ty, elements.len()))
                }
            }

            ExprKind::Tuple(elements) => {
                let tys: Vec<Ty> = elements.iter().map(|e| self.infer_expr(e)).collect::<Result<_, _>>()?;
                Ok(Ty::tuple(tys))
            }

            ExprKind::Block(block) => self.infer_block(block),

            ExprKind::If { condition, then_branch, else_branch } => {
                let cond_ty = self.infer_expr(condition)?;
                self.unify_with_alias_expansion(&cond_ty, &Ty::bool(), condition.span)?;

                let then_ty = self.infer_block(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let else_ty = self.infer_expr(else_expr)?;
                    self.unify_with_alias_expansion(&then_ty, &else_ty, span)
                } else {
                    // No else branch - if must be unit
                    self.unify_with_alias_expansion(&then_ty, &Ty::unit(), span)?;
                    Ok(Ty::unit())
                }
            }

            ExprKind::Match { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expr(scrutinee)?;

                if arms.is_empty() {
                    return Ok(Ty::never());
                }

                let mut result_ty: Option<Ty> = None;
                for arm in arms {
                    self.check_pattern(&arm.pattern, &scrutinee_ty)?;

                    self.ctx.enter_scope();
                    self.bind_pattern(&arm.pattern, &scrutinee_ty)?;
                    let body_ty = self.infer_expr(&arm.body)?;
                    self.ctx.leave_scope();

                    if let Some(ref prev) = result_ty {
                        self.unify_with_alias_expansion(prev, &body_ty, arm.span)?;
                    } else {
                        result_ty = Some(body_ty);
                    }
                }

                // Check exhaustiveness
                let resolved_ty = self.unifier.apply(&scrutinee_ty);
                let patterns: Vec<&Pattern> = arms.iter().map(|a| &a.pattern).collect();
                let checker = ExhaustivenessChecker::new(&self.ctx);
                let exhaustiveness = checker.check(&patterns, &resolved_ty);

                if !exhaustiveness.is_exhaustive {
                    let missing = format_missing_patterns(&exhaustiveness.missing);
                    return Err(TypeError::non_exhaustive_match(
                        resolved_ty.clone(),
                        missing,
                        span,
                    ));
                }

                Ok(result_ty.unwrap_or_else(Ty::never))
            }

            ExprKind::Loop { body, .. } => {
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                let _ = self.infer_block(body)?;
                self.in_loop = prev_in_loop;
                // Loop returns never unless broken
                Ok(Ty::never())
            }

            ExprKind::While { condition, body, .. } => {
                let cond_ty = self.infer_expr(condition)?;
                self.unify_with_alias_expansion(&cond_ty, &Ty::bool(), condition.span)?;

                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                let _ = self.infer_block(body)?;
                self.in_loop = prev_in_loop;

                Ok(Ty::unit())
            }

            ExprKind::For { pattern, iterable, body, .. } => {
                let iter_ty = self.infer_expr(iterable)?;

                // Determine element type based on iterable type
                let elem_ty = match &iter_ty.kind {
                    TyKind::Array { element, .. } => (**element).clone(),
                    TyKind::Slice { element } => (**element).clone(),
                    TyKind::Named { name, generics } if name == "Vec" => {
                        generics.first().cloned().unwrap_or_else(Ty::fresh_var)
                    }
                    TyKind::Named { name, generics } if name == "VecIter" => {
                        // VecIter<T> iterates over T (from Vec::iter)
                        generics.first().cloned().unwrap_or_else(Ty::fresh_var)
                    }
                    TyKind::Named { name, generics } if name == "Range" => {
                        // Range<T> iterates over T
                        generics.first().cloned().unwrap_or_else(|| Ty::i32())
                    }
                    TyKind::Named { name, generics } if name == "Option" => {
                        // Option<T> can be iterated (yields T once or nothing)
                        generics.first().cloned().unwrap_or_else(Ty::fresh_var)
                    }
                    _ => Ty::fresh_var(), // Assume it's some kind of iterator
                };

                self.ctx.enter_scope();
                self.bind_pattern(pattern, &elem_ty)?;

                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                let _ = self.infer_block(body)?;
                self.in_loop = prev_in_loop;

                self.ctx.leave_scope();
                Ok(Ty::unit())
            }

            ExprKind::Break { value, .. } => {
                if !self.in_loop {
                    return Err(TypeError::break_outside_loop(span));
                }
                if let Some(val) = value {
                    let _ = self.infer_expr(val)?;
                }
                Ok(Ty::never())
            }

            ExprKind::Continue { .. } => {
                if !self.in_loop {
                    return Err(TypeError::continue_outside_loop(span));
                }
                Ok(Ty::never())
            }

            ExprKind::Return { value } => {
                let ret_ty = if let Some(val) = value {
                    self.infer_expr(val)?
                } else {
                    Ty::unit()
                };

                if let Some(expected) = self.current_return_type.clone() {
                    self.unify_with_alias_expansion(&ret_ty, &expected, span)?;
                }

                Ok(Ty::never())
            }

            ExprKind::Closure { params, body } => {
                self.ctx.enter_scope();

                let param_tys: Vec<Ty> = params
                    .iter()
                    .map(|p| {
                        let ty = self.ast_type_to_ty(&p.ty);
                        self.ctx.define_var(&p.name.name, ty.clone(), p.is_mut);
                        ty
                    })
                    .collect();

                let body_ty = self.infer_expr(body)?;
                self.ctx.leave_scope();

                Ok(Ty::function(param_tys, body_ty))
            }

            ExprKind::Ref { mutable, operand } => {
                let operand_ty = self.infer_expr(operand)?;
                Ok(Ty::reference(operand_ty, *mutable))
            }

            ExprKind::Deref { operand } => {
                let operand_ty = self.infer_expr(operand)?;
                // Apply unifier to resolve type variables
                let resolved_ty = self.unifier.apply(&operand_ty);
                match &resolved_ty.kind {
                    TyKind::Ref { inner, .. } => Ok((**inner).clone()),
                    // Support dereferencing Box<T> and other smart pointers
                    TyKind::Named { name, generics } if name == "Box" => {
                        if let Some(inner) = generics.first() {
                            Ok(self.unifier.apply(inner))
                        } else {
                            Ok(Ty::fresh_var())
                        }
                    }
                    TyKind::Var(_) => {
                        let inner = Ty::fresh_var();
                        let ref_ty = Ty::reference(inner.clone(), false);
                        self.unify_with_alias_expansion(&operand_ty, &ref_ty, span)?;
                        Ok(inner)
                    }
                    _ => Err(TypeError::cannot_deref(resolved_ty, span)),
                }
            }

            ExprKind::Cast { expr, ty } => {
                let from_ty = self.infer_expr(expr)?;
                let to_ty = self.ast_type_to_ty(ty);

                // Check valid casts
                if self.is_valid_cast(&from_ty, &to_ty) {
                    Ok(to_ty)
                } else {
                    Err(TypeError::invalid_cast(from_ty, to_ty, span))
                }
            }

            ExprKind::Range { start, end, inclusive: _ } => {
                let elem_ty = if let Some(s) = start {
                    self.infer_expr(s)?
                } else if let Some(e) = end {
                    self.infer_expr(e)?
                } else {
                    Ty::new(TyKind::IntVar)
                };

                if let Some(s) = start {
                    let s_ty = self.infer_expr(s)?;
                    self.unify_with_alias_expansion(&elem_ty, &s_ty, span)?;
                }
                if let Some(e) = end {
                    let e_ty = self.infer_expr(e)?;
                    self.unify_with_alias_expansion(&elem_ty, &e_ty, span)?;
                }

                // Return a Range type (for now, just use the element type)
                Ok(Ty::named("Range".to_string(), vec![elem_ty]))
            }

            ExprKind::Await { operand } => {
                let operand_ty = self.infer_expr(operand)?;

                // Apply substitution to get resolved type
                let resolved_ty = self.unifier.apply(&operand_ty);

                // Await unwraps Future<T> to T
                match &resolved_ty.kind {
                    TyKind::Named { name, generics } if name == "Future" => {
                        if !generics.is_empty() {
                            Ok(generics[0].clone())
                        } else {
                            // Future with no inner type - use fresh var
                            Ok(Ty::fresh_var())
                        }
                    }
                    TyKind::Var(_) => {
                        // Type is still unknown - create Future constraint
                        let inner = Ty::fresh_var();
                        let future_ty = Ty::future(inner.clone());
                        self.unify_with_alias_expansion(&operand_ty, &future_ty, span)?;
                        Ok(inner)
                    }
                    _ => Err(TypeError::await_on_non_future(resolved_ty, span)),
                }
            }

            ExprKind::Try { operand } => {
                let operand_ty = self.infer_expr(operand)?;

                // Apply substitution to get resolved type
                let resolved_ty = self.unifier.apply(&operand_ty);

                // Try unwraps Result<T, E> or Option<T>
                match &resolved_ty.kind {
                    TyKind::Named { name, generics } if name == "Result" => {
                        if !generics.is_empty() {
                            Ok(generics[0].clone())
                        } else {
                            Ok(Ty::fresh_var())
                        }
                    }
                    TyKind::Named { name, generics } if name == "Option" => {
                        if !generics.is_empty() {
                            Ok(generics[0].clone())
                        } else {
                            Ok(Ty::fresh_var())
                        }
                    }
                    TyKind::Var(_) => {
                        // Type is still unknown - assume Result and create constraint
                        let ok_ty = Ty::fresh_var();
                        let err_ty = Ty::fresh_var();
                        let result_ty = Ty::result(ok_ty.clone(), err_ty);
                        self.unify_with_alias_expansion(&operand_ty, &result_ty, span)?;
                        Ok(ok_ty)
                    }
                    _ => Err(TypeError::try_on_non_result(resolved_ty, span)),
                }
            }

            ExprKind::Spawn { actor, args } => {
                let name = &actor.segments[0].ident.name;
                if let Some(actor_def) = self.ctx.lookup_actor(name) {
                    // Validate arguments against actor state fields
                    let state_types: Vec<Ty> = actor_def.state.iter()
                        .map(|(_, ty)| ty.clone())
                        .collect();

                    // Check argument count
                    if args.len() != state_types.len() {
                        return Err(TypeError::wrong_arg_count(state_types.len(), args.len(), span));
                    }

                    // Type check each argument against corresponding state field
                    for (arg, expected_ty) in args.iter().zip(state_types.iter()) {
                        let arg_ty = self.infer_expr(arg)?;
                        self.unify_with_alias_expansion(&arg_ty, expected_ty, arg.span)?;
                    }

                    Ok(Ty::actor(name.clone()))
                } else {
                    Err(TypeError::undefined_type(name.clone(), span))
                }
            }

            ExprKind::SpawnTask { future } => {
                // spawn(future_expr) - spawn a task for concurrent execution
                let future_ty = self.infer_expr(future)?;
                let resolved_ty = self.unifier.apply(&future_ty);

                // Verify the operand is a Future<T>
                match &resolved_ty.kind {
                    TyKind::Named { name, generics: _ } if name == "Future" => {
                        // spawn returns the same Future<T> type
                        Ok(resolved_ty)
                    }
                    _ => {
                        // Try to unify with a fresh Future type
                        let inner_ty = Ty::fresh_var();
                        let expected_future = Ty::future(inner_ty);
                        self.unify_with_alias_expansion(&future_ty, &expected_future, span)?;
                        Ok(self.unifier.apply(&expected_future))
                    }
                }
            }

            ExprKind::Send { target, message } => {
                let target_ty = self.infer_expr(target)?;
                let _message_ty = self.infer_expr(message)?;

                if let TyKind::Actor { name: actor_name } = &target_ty.kind {
                    // Extract message name if it's a struct/enum expression
                    let message_name = self.extract_message_name(message);

                    // Validate message exists if we can determine the name
                    if let Some(msg_name) = &message_name {
                        if self.ctx.lookup_actor_message(actor_name, msg_name).is_none() {
                            return Err(TypeError::invalid_message(
                                actor_name.clone(),
                                msg_name.clone(),
                                span,
                            ));
                        }
                    }

                    Ok(Ty::unit())
                } else {
                    Err(TypeError::not_an_actor(target_ty, span))
                }
            }

            ExprKind::SendRecv { target, message } => {
                let target_ty = self.infer_expr(target)?;
                let _message_ty = self.infer_expr(message)?;

                if let TyKind::Actor { name: actor_name } = &target_ty.kind {
                    // Extract message name if it's a struct/enum expression
                    let message_name = self.extract_message_name(message);

                    // Look up message and get response type if available
                    if let Some(msg_name) = &message_name {
                        if let Some(msg_def) = self.ctx.lookup_actor_message(actor_name, msg_name) {
                            // If message has a defined response type, use it
                            if let Some(ref resp_ty) = msg_def.response_ty {
                                return Ok(resp_ty.clone());
                            }
                        } else {
                            return Err(TypeError::invalid_message(
                                actor_name.clone(),
                                msg_name.clone(),
                                span,
                            ));
                        }
                    }

                    // Default: return fresh type variable
                    Ok(Ty::fresh_var())
                } else {
                    Err(TypeError::not_an_actor(target_ty, span))
                }
            }

            ExprKind::Reply { value } => {
                let _ = self.infer_expr(value)?;
                Ok(Ty::never())
            }

            ExprKind::Assign { target, value } => {
                let target_ty = self.infer_expr(target)?;
                let value_ty = self.infer_expr(value)?;
                self.unify_with_alias_expansion(&target_ty, &value_ty, span)?;
                Ok(Ty::unit())
            }

            ExprKind::AssignOp { op: _, target, value } => {
                let target_ty = self.infer_expr(target)?;
                let value_ty = self.infer_expr(value)?;
                self.unify_with_alias_expansion(&target_ty, &value_ty, span)?;
                Ok(Ty::unit())
            }

            ExprKind::Select { arms } => {
                // select! returns the type of the body expressions
                // All bodies should have compatible types
                let mut result_ty = None;
                for arm in arms {
                    // Infer future type (should be Future<T>)
                    let future_ty = self.infer_expr(&arm.future)?;

                    // Enter scope for the binding
                    self.ctx.enter_scope();

                    // Bind the result variable - extract inner type from Future<T> or use as-is
                    let inner_ty = if let TyKind::Named { generics, .. } = &future_ty.kind {
                        generics.first().cloned().unwrap_or(Ty::i64())
                    } else {
                        Ty::i64()
                    };
                    self.ctx.define_var(&arm.binding.name, inner_ty, false);

                    // Infer body type
                    let body_ty = self.infer_expr(&arm.body)?;
                    self.ctx.leave_scope();

                    if let Some(ref prev_ty) = result_ty {
                        self.unify_with_alias_expansion(prev_ty, &body_ty, arm.span)?;
                    } else {
                        result_ty = Some(body_ty);
                    }
                }
                Ok(result_ty.unwrap_or_else(Ty::unit))
            }

            ExprKind::Join { futures } => {
                // join! returns a tuple of the future result types
                let mut tys = Vec::new();
                for future in futures {
                    let future_ty = self.infer_expr(future)?;
                    // Extract T from Future<T>, or use the type as-is
                    tys.push(future_ty);
                }
                if tys.is_empty() {
                    Ok(Ty::unit())
                } else if tys.len() == 1 {
                    Ok(tys.pop().unwrap())
                } else {
                    Ok(Ty::tuple(tys))
                }
            }

            ExprKind::MacroCall(invocation) => {
                // Macro calls should be expanded before type checking
                // For now, we return a type variable that will be unified later
                // Built-in macros like vec![] will be handled specially
                let macro_name = &invocation.name.name;
                match macro_name.as_str() {
                    "vec" => {
                        // vec![] creates a Vec<T> where T is inferred from elements
                        let elem_ty = Ty::fresh_var();
                        Ok(Ty::named("Vec".to_string(), vec![elem_ty]))
                    }
                    "println" | "print" | "eprintln" | "eprint" => {
                        // These macros return unit
                        Ok(Ty::unit())
                    }
                    "format" => {
                        // format! returns String
                        Ok(Ty::named("String".to_string(), vec![]))
                    }
                    "panic" | "todo" | "unreachable" => {
                        // These macros never return (diverge), treat as unit for now
                        Ok(Ty::unit())
                    }
                    "assert" | "assert_eq" | "assert_ne" | "debug_assert" => {
                        // Assertion macros return unit
                        Ok(Ty::unit())
                    }
                    _ => {
                        // User-defined macro - try to expand it (with recursive expansion)
                        if self.macro_expander.has_macro(macro_name) {
                            // Expand the macro recursively
                            match self.macro_expander.expand_recursive(invocation) {
                                Ok(expanded_tokens) => {
                                    // Try to convert tokens directly to an expression
                                    if let Some(expr) = crate::macro_expand::tokens_to_expr(&expanded_tokens, span) {
                                        // Type-check the expanded expression
                                        self.infer_expr(&expr)
                                    } else {
                                        // Tokens couldn't be directly converted - try re-parsing
                                        let source = crate::macro_expand::tokens_to_source(&expanded_tokens);
                                        match self.parse_and_infer_expr(&source, span) {
                                            Ok(ty) => Ok(ty),
                                            Err(_) => {
                                                // Fallback: return fresh type var
                                                Ok(Ty::fresh_var())
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    Err(TypeError::macro_expansion_error(e.message(), span))
                                }
                            }
                        } else if self.ctx.lookup_macro(macro_name).is_some() {
                            // Macro is registered but not in expander (shouldn't happen)
                            Ok(Ty::fresh_var())
                        } else {
                            Err(TypeError::undefined_macro(macro_name.clone(), span))
                        }
                    }
                }
            }
        }
    }

    /// Parse a source string as an expression and infer its type
    /// Used for macro expansion when tokens_to_expr can't handle the complexity
    fn parse_and_infer_expr(&mut self, source: &str, _span: Span) -> TypeResult<Ty> {
        use crate::parser::Parser;

        // Parse the expanded macro output as an expression
        let mut parser = Parser::new(source);

        match parser.parse_expr() {
            Ok(expr) => self.infer_expr(&expr),
            Err(_) => {
                // Parse error - return a fresh type variable as fallback
                Ok(Ty::fresh_var())
            }
        }
    }

    fn infer_literal(&self, lit: &Literal) -> TypeResult<Ty> {
        Ok(match lit {
            Literal::Int(_) => Ty::new(TyKind::IntVar),
            Literal::Float(_) => Ty::new(TyKind::FloatVar),
            Literal::String(_) => Ty::str(),
            Literal::Char(_) => Ty::char(),
            Literal::Bool(_) => Ty::bool(),
        })
    }

    fn infer_binary_op(&mut self, op: BinaryOp, left: &Ty, right: &Ty, span: Span) -> TypeResult<Ty> {
        match op {
            // Arithmetic: both operands must be numeric, result is same type
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                let unified = self.unify_with_alias_expansion(left, right, span)?;
                if !unified.is_numeric() && !matches!(unified.kind, TyKind::Var(_)) {
                    return Err(TypeError::binary_op_mismatch(
                        format!("{:?}", op),
                        left.clone(),
                        right.clone(),
                        span,
                    ));
                }
                Ok(unified)
            }

            // Comparison: both operands must be same type, result is bool
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                self.unify_with_alias_expansion(left, right, span)?;
                Ok(Ty::bool())
            }

            // Logical: both must be bool, result is bool
            BinaryOp::And | BinaryOp::Or => {
                self.unify_with_alias_expansion(left, &Ty::bool(), span)?;
                self.unify_with_alias_expansion(right, &Ty::bool(), span)?;
                Ok(Ty::bool())
            }

            // Bitwise: both must be integers, result is same type
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
                let unified = self.unify_with_alias_expansion(left, right, span)?;
                if !unified.is_integer() && !matches!(unified.kind, TyKind::Var(_) | TyKind::IntVar) {
                    return Err(TypeError::binary_op_mismatch(
                        format!("{:?}", op),
                        left.clone(),
                        right.clone(),
                        span,
                    ));
                }
                Ok(unified)
            }
        }
    }

    fn infer_unary_op(&mut self, op: UnaryOp, operand: &Ty, span: Span) -> TypeResult<Ty> {
        match op {
            UnaryOp::Neg => {
                if operand.is_numeric() || matches!(operand.kind, TyKind::Var(_) | TyKind::IntVar | TyKind::FloatVar) {
                    Ok(operand.clone())
                } else {
                    Err(TypeError::unary_op_mismatch("-".to_string(), operand.clone(), span))
                }
            }
            UnaryOp::Not => {
                if operand.is_bool() || operand.is_integer() || matches!(operand.kind, TyKind::Var(_)) {
                    Ok(operand.clone())
                } else {
                    Err(TypeError::unary_op_mismatch("!".to_string(), operand.clone(), span))
                }
            }
            UnaryOp::Deref => {
                match &operand.kind {
                    TyKind::Ref { inner, .. } => Ok((**inner).clone()),
                    // Support dereferencing Box<T> and other smart pointers
                    TyKind::Named { name, generics } if name == "Box" => {
                        if let Some(inner) = generics.first() {
                            Ok(inner.clone())
                        } else {
                            // Box without generic - return unknown type
                            Ok(Ty::fresh_var())
                        }
                    }
                    _ => Err(TypeError::cannot_deref(operand.clone(), span)),
                }
            }
            UnaryOp::Ref => Ok(Ty::reference(operand.clone(), false)),
            UnaryOp::RefMut => Ok(Ty::reference(operand.clone(), true)),
        }
    }

    fn infer_block(&mut self, block: &Block) -> TypeResult<Ty> {
        self.ctx.enter_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt)?;
        }

        let result = if let Some(expr) = &block.expr {
            self.infer_expr(expr)?
        } else {
            Ty::unit()
        };

        self.ctx.leave_scope();
        Ok(result)
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> TypeResult<()> {
        match &stmt.kind {
            StmtKind::Let { pattern, ty, value } => {
                let declared_ty = ty.as_ref().map(|t| self.ast_type_to_ty(t));
                let inferred_ty = if let Some(val) = value {
                    let val_ty = self.infer_expr(val)?;
                    if let Some(ref decl) = declared_ty {
                        self.unify_with_alias_expansion(decl, &val_ty, stmt.span)?
                    } else {
                        val_ty
                    }
                } else if let Some(decl) = declared_ty {
                    decl
                } else {
                    Ty::fresh_var()
                };

                self.bind_pattern(pattern, &inferred_ty)?;
                Ok(())
            }
            StmtKind::Expr(expr) => {
                let _ = self.infer_expr(expr)?;
                Ok(())
            }
            StmtKind::Item(_) => Ok(()),
        }
    }

    // ============ Pattern Handling ============

    fn check_pattern(&mut self, pattern: &Pattern, expected: &Ty) -> TypeResult<()> {
        match &pattern.kind {
            PatternKind::Wildcard => Ok(()),
            PatternKind::Ident { .. } => Ok(()),
            PatternKind::Literal(lit) => {
                let lit_ty = self.infer_literal(lit)?;
                self.unify_with_alias_expansion(&lit_ty, expected, pattern.span)?;
                Ok(())
            }
            PatternKind::Tuple(patterns) => {
                if let TyKind::Tuple(elem_tys) = &expected.kind {
                    if patterns.len() != elem_tys.len() {
                        return Err(TypeError::pattern_type_mismatch(
                            expected.clone(),
                            Ty::tuple(vec![Ty::fresh_var(); patterns.len()]),
                            pattern.span,
                        ));
                    }
                    for (p, t) in patterns.iter().zip(elem_tys.iter()) {
                        self.check_pattern(p, t)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, ty: &Ty) -> TypeResult<()> {
        match &pattern.kind {
            PatternKind::Wildcard => Ok(()),
            PatternKind::Ident { name, mutable } => {
                self.ctx.define_var(&name.name, ty.clone(), *mutable);
                Ok(())
            }
            PatternKind::Tuple(patterns) => {
                if let TyKind::Tuple(elem_tys) = &ty.kind {
                    for (p, t) in patterns.iter().zip(elem_tys.iter()) {
                        self.bind_pattern(p, t)?;
                    }
                }
                Ok(())
            }
            PatternKind::Enum { path, fields } => {
                // Get the enum type and variant
                let variant_name = &path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();

                // Try to get the variant field types from the matched type
                if let TyKind::Named { name, generics } = &ty.kind {
                    if let Some(variants) = self.ctx.get_enum_variants(name) {
                        for (vname, vtypes) in variants {
                            if vname == variant_name {
                                // Substitute generic parameters
                                let substituted_types: Vec<Ty> = vtypes.iter().map(|vt| {
                                    self.substitute_generics(vt, name, generics)
                                }).collect();

                                // Bind each field pattern to its corresponding type
                                for (p, t) in fields.iter().zip(substituted_types.iter()) {
                                    self.bind_pattern(p, t)?;
                                }
                                return Ok(());
                            }
                        }
                    }
                }
                Ok(())
            }
            PatternKind::Struct { fields, .. } => {
                // Bind struct field patterns
                if let TyKind::Named { name, .. } = &ty.kind {
                    // Clone to avoid borrow conflict
                    let struct_fields: Vec<(String, Ty)> = self.ctx.get_struct_fields(name)
                        .map(|f| f.to_vec())
                        .unwrap_or_default();

                    for (field_name, field_pattern) in fields {
                        for (fname, fty) in &struct_fields {
                            if fname == &field_name.name {
                                self.bind_pattern(field_pattern, fty)?;
                                break;
                            }
                        }
                    }
                }
                Ok(())
            }
            PatternKind::Literal(_) => Ok(()),
            _ => Ok(()),
        }
    }

    /// Substitute generic type parameters in a type
    fn substitute_generics(&self, ty: &Ty, enum_name: &str, generics: &[Ty]) -> Ty {
        match &ty.kind {
            TyKind::Generic { name } => {
                // Map generic name to position and get the substitution
                if let Some(type_def) = self.ctx.lookup_type(enum_name) {
                    for (i, gname) in type_def.generics.iter().enumerate() {
                        if gname == name {
                            if i < generics.len() {
                                return generics[i].clone();
                            }
                        }
                    }
                }
                ty.clone()
            }
            _ => ty.clone(),
        }
    }

    /// Substitute type parameters using a substitution map
    /// Used for generic struct instantiation
    fn substitute_type_params(&self, ty: &Ty, subst_map: &std::collections::HashMap<String, Ty>) -> Ty {
        match &ty.kind {
            TyKind::Generic { name } => {
                // Look up in substitution map
                subst_map.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            TyKind::Named { name, generics } => {
                // Recursively substitute in generic arguments
                let new_generics: Vec<Ty> = generics
                    .iter()
                    .map(|g| self.substitute_type_params(g, subst_map))
                    .collect();
                Ty::named(name.clone(), new_generics)
            }
            TyKind::Ref { inner, mutable } => {
                Ty::reference(self.substitute_type_params(inner, subst_map), *mutable)
            }
            TyKind::Array { element, size } => {
                Ty::array(self.substitute_type_params(element, subst_map), *size)
            }
            TyKind::Slice { element } => {
                Ty::slice(self.substitute_type_params(element, subst_map))
            }
            TyKind::Tuple(elements) => {
                let new_elements: Vec<Ty> = elements
                    .iter()
                    .map(|e| self.substitute_type_params(e, subst_map))
                    .collect();
                Ty::tuple(new_elements)
            }
            TyKind::Fn { params, ret } => {
                let new_params: Vec<Ty> = params
                    .iter()
                    .map(|p| self.substitute_type_params(p, subst_map))
                    .collect();
                let new_ret = self.substitute_type_params(ret, subst_map);
                Ty::function(new_params, new_ret)
            }
            // For all other types, return as-is
            _ => ty.clone(),
        }
    }

    /// Extract message name from a message expression
    fn extract_message_name(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Path(path) => {
                // Simple identifier message
                path.segments.last().map(|s| s.ident.name.clone())
            }
            ExprKind::Struct { path, .. } => {
                // Struct literal message
                path.segments.last().map(|s| s.ident.name.clone())
            }
            ExprKind::Call { func, .. } => {
                // Enum variant message like Foo::Bar(x)
                if let ExprKind::Path(path) = &func.kind {
                    path.segments.last().map(|s| s.ident.name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract message definition from a pattern
    fn extract_message_def(&self, pattern: &Pattern) -> Option<MessageDef> {
        match &pattern.kind {
            PatternKind::Ident { name, .. } => {
                // Simple message name (no fields)
                Some(MessageDef {
                    name: name.name.clone(),
                    fields: vec![],
                    response_ty: None,
                })
            }
            PatternKind::Enum { path, fields } => {
                // Enum-style message with fields
                let name = path.segments.last()
                    .map(|s| s.ident.name.clone())
                    .unwrap_or_default();
                Some(MessageDef {
                    name,
                    fields: vec![Ty::fresh_var(); fields.len()], // Field types need inference
                    response_ty: None,
                })
            }
            PatternKind::Struct { path, fields, .. } => {
                // Struct-style message with named fields
                let name = path.segments.last()
                    .map(|s| s.ident.name.clone())
                    .unwrap_or_default();
                Some(MessageDef {
                    name,
                    fields: vec![Ty::fresh_var(); fields.len()],
                    response_ty: None,
                })
            }
            _ => None,
        }
    }

    /// Instantiate generic type parameters with fresh type variables
    /// This is used when calling generic functions to create fresh instances
    fn instantiate_generics(&mut self, ty: &Ty) -> Ty {
        match &ty.kind {
            TyKind::Generic { .. } => {
                // Replace generic parameter with a fresh type variable
                Ty::fresh_var()
            }
            TyKind::Ref { inner, mutable } => {
                let inner_inst = self.instantiate_generics(inner);
                Ty::reference(inner_inst, *mutable)
            }
            TyKind::Array { element, size } => {
                let elem_inst = self.instantiate_generics(element);
                Ty::array(elem_inst, *size)
            }
            TyKind::Slice { element } => {
                let elem_inst = self.instantiate_generics(element);
                Ty::slice(elem_inst)
            }
            TyKind::Tuple(elems) => {
                let elems_inst: Vec<Ty> = elems.iter()
                    .map(|e| self.instantiate_generics(e))
                    .collect();
                Ty::tuple(elems_inst)
            }
            TyKind::Fn { params, ret } => {
                let params_inst: Vec<Ty> = params.iter()
                    .map(|p| self.instantiate_generics(p))
                    .collect();
                let ret_inst = self.instantiate_generics(ret);
                Ty::function(params_inst, ret_inst)
            }
            TyKind::Named { name, generics } => {
                let generics_inst: Vec<Ty> = generics.iter()
                    .map(|g| self.instantiate_generics(g))
                    .collect();
                Ty::named(name.clone(), generics_inst)
            }
            // For all other types, return as-is
            _ => ty.clone(),
        }
    }

    /// Instantiate generic type parameters with fresh type variables
    /// while tracking the mapping from generic names to type variables.
    /// This is used to collect type arguments for monomorphization.
    fn instantiate_generics_with_tracking(&mut self, ty: &Ty, tracking: &mut HashMap<String, Ty>) -> Ty {
        match &ty.kind {
            TyKind::Generic { name } => {
                // Check if we already have a variable for this generic
                if let Some(var) = tracking.get(name) {
                    var.clone()
                } else {
                    // Create a fresh type variable and track it
                    let fresh = Ty::fresh_var();
                    tracking.insert(name.clone(), fresh.clone());
                    fresh
                }
            }
            TyKind::Ref { inner, mutable } => {
                let inner_inst = self.instantiate_generics_with_tracking(inner, tracking);
                Ty::reference(inner_inst, *mutable)
            }
            TyKind::Array { element, size } => {
                let elem_inst = self.instantiate_generics_with_tracking(element, tracking);
                Ty::array(elem_inst, *size)
            }
            TyKind::Slice { element } => {
                let elem_inst = self.instantiate_generics_with_tracking(element, tracking);
                Ty::slice(elem_inst)
            }
            TyKind::Tuple(elems) => {
                let elems_inst: Vec<Ty> = elems.iter()
                    .map(|e| self.instantiate_generics_with_tracking(e, tracking))
                    .collect();
                Ty::tuple(elems_inst)
            }
            TyKind::Fn { params, ret } => {
                let params_inst: Vec<Ty> = params.iter()
                    .map(|p| self.instantiate_generics_with_tracking(p, tracking))
                    .collect();
                let ret_inst = self.instantiate_generics_with_tracking(ret, tracking);
                Ty::function(params_inst, ret_inst)
            }
            TyKind::Named { name, generics } => {
                let generics_inst: Vec<Ty> = generics.iter()
                    .map(|g| self.instantiate_generics_with_tracking(g, tracking))
                    .collect();
                Ty::named(name.clone(), generics_inst)
            }
            // For all other types, return as-is
            _ => ty.clone(),
        }
    }

    /// Substitute Self type with a concrete type
    /// This is used when resolving Self::method() calls inside impl blocks
    fn substitute_self_type(&self, ty: &Ty, concrete: &Ty) -> Ty {
        match &ty.kind {
            TyKind::Named { name, generics } if name == "Self" => {
                // Replace Self with the concrete type
                concrete.clone()
            }
            TyKind::Generic { name } if name == "Self" => {
                // Also handle generic Self
                concrete.clone()
            }
            TyKind::Ref { inner, mutable } => {
                let inner_subst = self.substitute_self_type(inner, concrete);
                Ty::reference(inner_subst, *mutable)
            }
            TyKind::Array { element, size } => {
                let elem_subst = self.substitute_self_type(element, concrete);
                Ty::array(elem_subst, *size)
            }
            TyKind::Slice { element } => {
                let elem_subst = self.substitute_self_type(element, concrete);
                Ty::slice(elem_subst)
            }
            TyKind::Tuple(elems) => {
                let elems_subst: Vec<Ty> = elems.iter()
                    .map(|e| self.substitute_self_type(e, concrete))
                    .collect();
                Ty::tuple(elems_subst)
            }
            TyKind::Fn { params, ret } => {
                let params_subst: Vec<Ty> = params.iter()
                    .map(|p| self.substitute_self_type(p, concrete))
                    .collect();
                let ret_subst = self.substitute_self_type(ret, concrete);
                Ty::function(params_subst, ret_subst)
            }
            TyKind::Named { name, generics } => {
                let generics_subst: Vec<Ty> = generics.iter()
                    .map(|g| self.substitute_self_type(g, concrete))
                    .collect();
                Ty::named(name.clone(), generics_subst)
            }
            // For all other types, return as-is
            _ => ty.clone(),
        }
    }

    // ============ Constant Evaluation ============

    /// Evaluate a constant expression to a usize value
    /// Returns None if the expression cannot be evaluated at compile time
    fn const_eval_usize(&self, expr: &Expr) -> Option<usize> {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int(n) => {
                    if *n >= 0 && *n <= usize::MAX as i128 {
                        Some(*n as usize)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            ExprKind::Binary { op, left, right } => {
                let l = self.const_eval_usize(left)?;
                let r = self.const_eval_usize(right)?;
                match op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    BinaryOp::Div => {
                        if r == 0 {
                            None
                        } else {
                            l.checked_div(r)
                        }
                    }
                    _ => None,
                }
            }
            ExprKind::Unary { op, operand } => {
                let val = self.const_eval_usize(operand)?;
                match op {
                    UnaryOp::Neg => None, // Can't negate usize
                    _ => Some(val),
                }
            }
            // Parentheses are already handled by the parser
            _ => None,
        }
    }

    // ============ Type Conversion ============

    fn ast_type_to_ty(&self, ast_ty: &AstType) -> Ty {
        match &ast_ty.kind {
            AstTypeKind::Path(path) => {
                let name = &path.segments[0].ident.name;
                let generics: Vec<Ty> = path.segments[0]
                    .generics
                    .as_ref()
                    .map(|gs| gs.iter().map(|g| self.ast_type_to_ty(g)).collect())
                    .unwrap_or_default();

                // Check for primitive types
                match name.as_str() {
                    "i8" => Ty::i8(),
                    "i16" => Ty::i16(),
                    "i32" => Ty::i32(),
                    "i64" => Ty::i64(),
                    "i128" => Ty::i128(),
                    "isize" => Ty::isize(),
                    "u8" => Ty::u8(),
                    "u16" => Ty::u16(),
                    "u32" => Ty::u32(),
                    "u64" => Ty::u64(),
                    "u128" => Ty::u128(),
                    "usize" => Ty::usize(),
                    "f32" => Ty::f32(),
                    "f64" => Ty::f64(),
                    "bool" => Ty::bool(),
                    "char" => Ty::char(),
                    "str" => Ty::str(),
                    _ => {
                        // Check if it's a generic parameter
                        if self.ctx.is_generic(name) {
                            Ty::generic(name.clone())
                        } else {
                            // Apply default type parameters if fewer arguments are provided
                            let final_generics = self.apply_default_type_params(name, generics);
                            Ty::named(name.clone(), final_generics)
                        }
                    }
                }
            }
            AstTypeKind::Reference { mutable, inner } => {
                Ty::reference(self.ast_type_to_ty(inner), *mutable)
            }
            AstTypeKind::Array { element, size } => {
                // Evaluate array size at compile time
                let size_val = self.const_eval_usize(size).unwrap_or(0);
                Ty::array(self.ast_type_to_ty(element), size_val)
            }
            AstTypeKind::Slice { element } => Ty::slice(self.ast_type_to_ty(element)),
            AstTypeKind::Tuple(elems) => {
                Ty::tuple(elems.iter().map(|e| self.ast_type_to_ty(e)).collect())
            }
            AstTypeKind::FnPtr { params, return_type } => {
                let param_tys: Vec<Ty> = params.iter().map(|p| self.ast_type_to_ty(p)).collect();
                let ret = return_type
                    .as_ref()
                    .map(|r| self.ast_type_to_ty(r))
                    .unwrap_or_else(Ty::unit);
                Ty::function(param_tys, ret)
            }
            AstTypeKind::Never => Ty::never(),
            AstTypeKind::Infer => Ty::fresh_var(),
            AstTypeKind::Option(inner) => Ty::option(self.ast_type_to_ty(inner)),
            AstTypeKind::Result { ok, err } => {
                Ty::result(self.ast_type_to_ty(ok), self.ast_type_to_ty(err))
            }
            AstTypeKind::SelfType => {
                // Self type - create a placeholder that will be resolved later
                // Actual resolution happens in resolve_self_type when self_ty is known
                Ty::generic("Self".to_string())
            }
            AstTypeKind::Projection { base, assoc_name } => {
                let base_ty = self.ast_type_to_ty(base);
                Ty::projection(base_ty, assoc_name.clone())
            }
        }
    }

    fn is_valid_cast(&self, from: &Ty, to: &Ty) -> bool {
        // Allow numeric casts
        if from.is_numeric() && to.is_numeric() {
            return true;
        }

        // Allow casting integers to/from bool
        if (from.is_integer() && to.is_bool()) || (from.is_bool() && to.is_integer()) {
            return true;
        }

        // IntVar can cast to any numeric
        if matches!(from.kind, TyKind::IntVar) && to.is_numeric() {
            return true;
        }
        if matches!(to.kind, TyKind::IntVar) && from.is_numeric() {
            return true;
        }

        // Same type is always valid
        from == to
    }
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}
