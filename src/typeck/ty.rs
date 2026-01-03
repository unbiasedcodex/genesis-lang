//! Internal Type Representation for Genesis Lang
//!
//! This module defines the internal representation of types used during type checking.
//! These types are different from the AST types (`ast::Type`) - they are resolved and
//! canonical representations used for type checking and inference.
//!
//! # Type Variables
//!
//! Type variables (`TyVar`) represent unknown types that will be resolved during
//! type inference. They are written as `?T0`, `?T1`, etc. in debug output.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// Counter for generating unique type variable IDs
static NEXT_TY_VAR: AtomicU32 = AtomicU32::new(0);

/// A type variable - represents an unknown type to be inferred
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(pub u32);

impl TyVar {
    /// Create a fresh type variable
    pub fn fresh() -> Self {
        TyVar(NEXT_TY_VAR.fetch_add(1, Ordering::SeqCst))
    }

    /// Reset the counter (for testing)
    #[cfg(test)]
    pub fn reset_counter() {
        NEXT_TY_VAR.store(0, Ordering::SeqCst);
    }
}

impl fmt::Display for TyVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?T{}", self.0)
    }
}

/// The main type representation used during type checking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ty {
    pub kind: TyKind,
}

impl Ty {
    /// Create a new type
    pub fn new(kind: TyKind) -> Self {
        Self { kind }
    }

    // ============ Primitive Type Constructors ============

    pub fn unit() -> Self {
        Self::new(TyKind::Unit)
    }

    pub fn bool() -> Self {
        Self::new(TyKind::Bool)
    }

    pub fn char() -> Self {
        Self::new(TyKind::Char)
    }

    pub fn str() -> Self {
        Self::new(TyKind::Str)
    }

    pub fn i8() -> Self {
        Self::new(TyKind::Int(IntTy::I8))
    }

    pub fn i16() -> Self {
        Self::new(TyKind::Int(IntTy::I16))
    }

    pub fn i32() -> Self {
        Self::new(TyKind::Int(IntTy::I32))
    }

    pub fn i64() -> Self {
        Self::new(TyKind::Int(IntTy::I64))
    }

    pub fn i128() -> Self {
        Self::new(TyKind::Int(IntTy::I128))
    }

    pub fn isize() -> Self {
        Self::new(TyKind::Int(IntTy::Isize))
    }

    pub fn u8() -> Self {
        Self::new(TyKind::Uint(UintTy::U8))
    }

    pub fn u16() -> Self {
        Self::new(TyKind::Uint(UintTy::U16))
    }

    pub fn u32() -> Self {
        Self::new(TyKind::Uint(UintTy::U32))
    }

    pub fn u64() -> Self {
        Self::new(TyKind::Uint(UintTy::U64))
    }

    pub fn u128() -> Self {
        Self::new(TyKind::Uint(UintTy::U128))
    }

    pub fn usize() -> Self {
        Self::new(TyKind::Uint(UintTy::Usize))
    }

    pub fn f32() -> Self {
        Self::new(TyKind::Float(FloatTy::F32))
    }

    pub fn f64() -> Self {
        Self::new(TyKind::Float(FloatTy::F64))
    }

    pub fn never() -> Self {
        Self::new(TyKind::Never)
    }

    // ============ Compound Type Constructors ============

    pub fn var(v: TyVar) -> Self {
        Self::new(TyKind::Var(v))
    }

    pub fn fresh_var() -> Self {
        Self::var(TyVar::fresh())
    }

    pub fn reference(inner: Ty, mutable: bool) -> Self {
        Self::new(TyKind::Ref {
            inner: Box::new(inner),
            mutable,
        })
    }

    pub fn array(element: Ty, size: usize) -> Self {
        Self::new(TyKind::Array {
            element: Box::new(element),
            size,
        })
    }

    pub fn slice(element: Ty) -> Self {
        Self::new(TyKind::Slice {
            element: Box::new(element),
        })
    }

    pub fn tuple(elements: Vec<Ty>) -> Self {
        Self::new(TyKind::Tuple(elements))
    }

    pub fn function(params: Vec<Ty>, ret: Ty) -> Self {
        Self::new(TyKind::Fn {
            params,
            ret: Box::new(ret),
        })
    }

    pub fn named(name: String, generics: Vec<Ty>) -> Self {
        Self::new(TyKind::Named { name, generics })
    }

    pub fn option(inner: Ty) -> Self {
        Self::named("Option".to_string(), vec![inner])
    }

    pub fn result(ok: Ty, err: Ty) -> Self {
        Self::named("Result".to_string(), vec![ok, err])
    }

    pub fn future(inner: Ty) -> Self {
        Self::named("Future".to_string(), vec![inner])
    }

    pub fn actor(name: String) -> Self {
        Self::new(TyKind::Actor { name })
    }

    pub fn generic(name: String) -> Self {
        Self::new(TyKind::Generic { name })
    }

    /// Associated type projection: `Self::Item` or `T::Item`
    pub fn projection(base_ty: Ty, assoc_name: String) -> Self {
        Self::new(TyKind::Projection {
            base_ty: Box::new(base_ty),
            trait_name: None,
            assoc_name,
        })
    }

    /// Associated type projection with explicit trait: `<T as Trait>::Item`
    pub fn projection_with_trait(base_ty: Ty, trait_name: String, assoc_name: String) -> Self {
        Self::new(TyKind::Projection {
            base_ty: Box::new(base_ty),
            trait_name: Some(trait_name),
            assoc_name,
        })
    }

    /// Trait object: `dyn Trait`
    pub fn trait_object(trait_name: String) -> Self {
        Self::new(TyKind::TraitObject { trait_name })
    }

    // ============ Type Predicates ============

    pub fn is_var(&self) -> bool {
        matches!(self.kind, TyKind::Var(_))
    }

    pub fn is_never(&self) -> bool {
        matches!(self.kind, TyKind::Never)
    }

    pub fn is_unit(&self) -> bool {
        matches!(self.kind, TyKind::Unit)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self.kind, TyKind::Bool)
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self.kind,
            TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) | TyKind::IntVar | TyKind::FloatVar
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.kind, TyKind::Int(_) | TyKind::Uint(_) | TyKind::IntVar)
    }

    pub fn is_float(&self) -> bool {
        matches!(self.kind, TyKind::Float(_) | TyKind::FloatVar)
    }

    pub fn is_reference(&self) -> bool {
        matches!(self.kind, TyKind::Ref { .. })
    }

    /// Check if this type contains any type variables
    pub fn has_vars(&self) -> bool {
        match &self.kind {
            TyKind::Var(_) | TyKind::IntVar | TyKind::FloatVar => true,
            TyKind::Ref { inner, .. } => inner.has_vars(),
            TyKind::Array { element, .. } => element.has_vars(),
            TyKind::Slice { element } => element.has_vars(),
            TyKind::Tuple(elements) => elements.iter().any(|t| t.has_vars()),
            TyKind::Fn { params, ret } => {
                params.iter().any(|t| t.has_vars()) || ret.has_vars()
            }
            TyKind::Named { generics, .. } => generics.iter().any(|t| t.has_vars()),
            _ => false,
        }
    }

    /// Check if this type contains any generic type parameters
    pub fn has_generics(&self) -> bool {
        match &self.kind {
            TyKind::Generic { .. } => true,
            TyKind::Ref { inner, .. } => inner.has_generics(),
            TyKind::Array { element, .. } => element.has_generics(),
            TyKind::Slice { element } => element.has_generics(),
            TyKind::Tuple(elements) => elements.iter().any(|t| t.has_generics()),
            TyKind::Fn { params, ret } => {
                params.iter().any(|t| t.has_generics()) || ret.has_generics()
            }
            TyKind::Named { generics, .. } => generics.iter().any(|t| t.has_generics()),
            _ => false,
        }
    }

    /// Apply a substitution to this type
    pub fn apply(&self, subst: &Substitution) -> Ty {
        match &self.kind {
            TyKind::Var(v) => subst.get(*v).cloned().unwrap_or_else(|| self.clone()),
            TyKind::Ref { inner, mutable } => {
                Ty::reference(inner.apply(subst), *mutable)
            }
            TyKind::Array { element, size } => {
                Ty::array(element.apply(subst), *size)
            }
            TyKind::Slice { element } => {
                Ty::slice(element.apply(subst))
            }
            TyKind::Tuple(elements) => {
                Ty::tuple(elements.iter().map(|t| t.apply(subst)).collect())
            }
            TyKind::Fn { params, ret } => {
                Ty::function(
                    params.iter().map(|t| t.apply(subst)).collect(),
                    ret.apply(subst),
                )
            }
            TyKind::Named { name, generics } => {
                Ty::named(
                    name.clone(),
                    generics.iter().map(|t| t.apply(subst)).collect(),
                )
            }
            _ => self.clone(),
        }
    }

    /// Collect all type variables in this type
    pub fn free_vars(&self) -> Vec<TyVar> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut Vec<TyVar>) {
        match &self.kind {
            TyKind::Var(v) => {
                if !vars.contains(v) {
                    vars.push(*v);
                }
            }
            TyKind::Ref { inner, .. } => inner.collect_vars(vars),
            TyKind::Array { element, .. } => element.collect_vars(vars),
            TyKind::Slice { element } => element.collect_vars(vars),
            TyKind::Tuple(elements) => {
                for e in elements {
                    e.collect_vars(vars);
                }
            }
            TyKind::Fn { params, ret } => {
                for p in params {
                    p.collect_vars(vars);
                }
                ret.collect_vars(vars);
            }
            TyKind::Named { generics, .. } => {
                for g in generics {
                    g.collect_vars(vars);
                }
            }
            _ => {}
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            TyKind::Unit => write!(f, "()"),
            TyKind::Bool => write!(f, "bool"),
            TyKind::Char => write!(f, "char"),
            TyKind::Str => write!(f, "str"),
            TyKind::Int(t) => write!(f, "{}", t),
            TyKind::Uint(t) => write!(f, "{}", t),
            TyKind::Float(t) => write!(f, "{}", t),
            TyKind::Never => write!(f, "!"),
            TyKind::Var(v) => write!(f, "{}", v),
            TyKind::IntVar => write!(f, "{{integer}}"),
            TyKind::FloatVar => write!(f, "{{float}}"),
            TyKind::Ref { inner, mutable } => {
                if *mutable {
                    write!(f, "&mut {}", inner)
                } else {
                    write!(f, "&{}", inner)
                }
            }
            TyKind::Array { element, size } => write!(f, "[{}; {}]", element, size),
            TyKind::Slice { element } => write!(f, "[{}]", element),
            TyKind::Tuple(elements) => {
                write!(f, "(")?;
                for (i, e) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            TyKind::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            TyKind::Named { name, generics } => {
                write!(f, "{}", name)?;
                if !generics.is_empty() {
                    write!(f, "<")?;
                    for (i, g) in generics.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", g)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            TyKind::Actor { name } => write!(f, "actor {}", name),
            TyKind::Generic { name } => write!(f, "{}", name),
            TyKind::Projection { base_ty, trait_name, assoc_name } => {
                if let Some(trait_name) = trait_name {
                    write!(f, "<{} as {}>::{}", base_ty, trait_name, assoc_name)
                } else {
                    write!(f, "{}::{}", base_ty, assoc_name)
                }
            }
            TyKind::TraitObject { trait_name } => write!(f, "dyn {}", trait_name),
            TyKind::Error => write!(f, "{{error}}"),
        }
    }
}

/// The kind of a type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TyKind {
    // ============ Primitives ============
    /// Unit type: `()`
    Unit,
    /// Boolean: `bool`
    Bool,
    /// Character: `char`
    Char,
    /// String: `str`
    Str,
    /// Signed integers
    Int(IntTy),
    /// Unsigned integers
    Uint(UintTy),
    /// Floating point
    Float(FloatTy),
    /// Never type: `!`
    Never,

    // ============ Type Variables ============
    /// Unknown type to be inferred
    Var(TyVar),
    /// Integer type variable (can unify with any integer type)
    IntVar,
    /// Float type variable (can unify with any float type)
    FloatVar,

    // ============ Compound Types ============
    /// Reference: `&T` or `&mut T`
    Ref { inner: Box<Ty>, mutable: bool },
    /// Array: `[T; N]`
    Array { element: Box<Ty>, size: usize },
    /// Slice: `[T]`
    Slice { element: Box<Ty> },
    /// Tuple: `(T1, T2, ...)`
    Tuple(Vec<Ty>),
    /// Function type: `fn(T1, T2) -> R`
    Fn { params: Vec<Ty>, ret: Box<Ty> },

    // ============ Named Types ============
    /// Named type: struct, enum, type alias with optional generics
    Named { name: String, generics: Vec<Ty> },
    /// Actor type
    Actor { name: String },
    /// Generic type parameter: `T`
    Generic { name: String },

    // ============ Associated Types ============
    /// Associated type projection: `Self::Item` or `T::Item`
    /// base_ty is the type (Self or T), assoc_name is the associated type name
    Projection {
        base_ty: Box<Ty>,
        trait_name: Option<String>,  // Optional trait for disambiguation
        assoc_name: String,
    },

    // ============ Trait Objects ============
    /// Trait object: `dyn Trait`
    TraitObject { trait_name: String },

    // ============ Error ============
    /// Error type (for error recovery)
    Error,
}

/// Signed integer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

impl fmt::Display for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntTy::I8 => write!(f, "i8"),
            IntTy::I16 => write!(f, "i16"),
            IntTy::I32 => write!(f, "i32"),
            IntTy::I64 => write!(f, "i64"),
            IntTy::I128 => write!(f, "i128"),
            IntTy::Isize => write!(f, "isize"),
        }
    }
}

/// Unsigned integer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UintTy {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}

impl fmt::Display for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UintTy::U8 => write!(f, "u8"),
            UintTy::U16 => write!(f, "u16"),
            UintTy::U32 => write!(f, "u32"),
            UintTy::U64 => write!(f, "u64"),
            UintTy::U128 => write!(f, "u128"),
            UintTy::Usize => write!(f, "usize"),
        }
    }
}

/// Floating point types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatTy {
    F32,
    F64,
}

impl fmt::Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatTy::F32 => write!(f, "f32"),
            FloatTy::F64 => write!(f, "f64"),
        }
    }
}

/// A substitution maps type variables to types
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    mapping: HashMap<TyVar, Ty>,
}

impl Substitution {
    /// Create an empty substitution
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
        }
    }

    /// Add a mapping to the substitution
    pub fn insert(&mut self, var: TyVar, ty: Ty) {
        self.mapping.insert(var, ty);
    }

    /// Get the type for a variable
    pub fn get(&self, var: TyVar) -> Option<&Ty> {
        self.mapping.get(&var)
    }

    /// Check if a variable is in the substitution
    pub fn contains(&self, var: TyVar) -> bool {
        self.mapping.contains_key(&var)
    }

    /// Compose two substitutions: self ∘ other
    /// Applies self to the range of other, then combines
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();

        // Apply self to each type in other
        for (var, ty) in &other.mapping {
            result.insert(*var, ty.apply(self));
        }

        // Add mappings from self that aren't in other
        for (var, ty) in &self.mapping {
            if !result.contains(*var) {
                result.insert(*var, ty.clone());
            }
        }

        result
    }

    /// Get all mappings
    pub fn iter(&self) -> impl Iterator<Item = (&TyVar, &Ty)> {
        self.mapping.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ty_display() {
        assert_eq!(Ty::i32().to_string(), "i32");
        assert_eq!(Ty::bool().to_string(), "bool");
        assert_eq!(Ty::unit().to_string(), "()");
        assert_eq!(Ty::reference(Ty::i32(), false).to_string(), "&i32");
        assert_eq!(Ty::reference(Ty::i32(), true).to_string(), "&mut i32");
        assert_eq!(Ty::array(Ty::u8(), 10).to_string(), "[u8; 10]");
        assert_eq!(Ty::tuple(vec![Ty::i32(), Ty::bool()]).to_string(), "(i32, bool)");
        assert_eq!(
            Ty::function(vec![Ty::i32(), Ty::i32()], Ty::i32()).to_string(),
            "fn(i32, i32) -> i32"
        );
        assert_eq!(
            Ty::named("Vec".to_string(), vec![Ty::i32()]).to_string(),
            "Vec<i32>"
        );
    }

    #[test]
    fn test_substitution() {
        TyVar::reset_counter();
        let v0 = TyVar::fresh();
        let v1 = TyVar::fresh();

        let mut subst = Substitution::new();
        subst.insert(v0, Ty::i32());
        subst.insert(v1, Ty::bool());

        let ty = Ty::tuple(vec![Ty::var(v0), Ty::var(v1)]);
        let applied = ty.apply(&subst);

        assert_eq!(applied.to_string(), "(i32, bool)");
    }

    #[test]
    fn test_free_vars() {
        TyVar::reset_counter();
        let v0 = TyVar::fresh();
        let v1 = TyVar::fresh();

        let ty = Ty::function(vec![Ty::var(v0)], Ty::var(v1));
        let vars = ty.free_vars();

        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&v0));
        assert!(vars.contains(&v1));
    }
}
