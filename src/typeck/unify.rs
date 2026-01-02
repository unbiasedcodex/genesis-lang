//! Type Unification
//!
//! This module implements the unification algorithm for type inference.
//! Unification attempts to make two types equal by finding a substitution
//! that maps type variables to concrete types.
//!
//! # Algorithm
//!
//! The unification algorithm is based on Robinson's unification algorithm,
//! extended to handle:
//! - Primitive types
//! - Compound types (arrays, tuples, functions)
//! - Generic types
//! - Integer and float type variables
//! - Variance checking for references and functions

#![allow(dead_code)]

use crate::span::Span;
use crate::typeck::error::{TypeError, TypeResult};
use crate::typeck::ty::{Substitution, Ty, TyKind, TyVar};
use crate::typeck::variance::is_subtype;

/// The unifier manages type unification
pub struct Unifier {
    /// The current substitution
    subst: Substitution,
}

impl Unifier {
    /// Create a new unifier
    pub fn new() -> Self {
        Self {
            subst: Substitution::new(),
        }
    }

    /// Get the current substitution
    pub fn substitution(&self) -> &Substitution {
        &self.subst
    }

    /// Get the current substitution (consuming)
    pub fn into_substitution(self) -> Substitution {
        self.subst
    }

    /// Unify two types, returning the unified type
    pub fn unify(&mut self, t1: &Ty, t2: &Ty, span: Span) -> TypeResult<Ty> {
        // Apply current substitution first
        let t1 = t1.apply(&self.subst);
        let t2 = t2.apply(&self.subst);

        self.unify_impl(&t1, &t2, span)
    }

    fn unify_impl(&mut self, t1: &Ty, t2: &Ty, span: Span) -> TypeResult<Ty> {
        // If they're already equal, we're done
        if t1 == t2 {
            return Ok(t1.clone());
        }

        match (&t1.kind, &t2.kind) {
            // Type variable on the left
            (TyKind::Var(v), _) => {
                self.unify_var(*v, t2, span)
            }

            // Type variable on the right
            (_, TyKind::Var(v)) => {
                self.unify_var(*v, t1, span)
            }

            // Integer type variable
            (TyKind::IntVar, TyKind::Int(_)) |
            (TyKind::IntVar, TyKind::Uint(_)) => Ok(t2.clone()),
            (TyKind::Int(_), TyKind::IntVar) |
            (TyKind::Uint(_), TyKind::IntVar) => Ok(t1.clone()),
            (TyKind::IntVar, TyKind::IntVar) => Ok(Ty::i32()), // Default to i32

            // Float type variable
            (TyKind::FloatVar, TyKind::Float(_)) => Ok(t2.clone()),
            (TyKind::Float(_), TyKind::FloatVar) => Ok(t1.clone()),
            (TyKind::FloatVar, TyKind::FloatVar) => Ok(Ty::f64()), // Default to f64

            // Integer and float vars together - float wins
            (TyKind::IntVar, TyKind::FloatVar) |
            (TyKind::FloatVar, TyKind::IntVar) => Ok(Ty::f64()),

            // Never type unifies with anything
            (TyKind::Never, _) => Ok(t2.clone()),
            (_, TyKind::Never) => Ok(t1.clone()),

            // Error type unifies with anything (error recovery)
            (TyKind::Error, _) => Ok(t2.clone()),
            (_, TyKind::Error) => Ok(t1.clone()),

            // References - with variance checking
            (TyKind::Ref { inner: i1, mutable: m1 }, TyKind::Ref { inner: i2, mutable: m2 }) => {
                match (m1, m2) {
                    // Both immutable: covariant - allow subtyping
                    (false, false) => {
                        // Try exact unification first
                        if let Ok(inner) = self.unify_impl(i1, i2, span) {
                            Ok(Ty::reference(inner, false))
                        } else if is_subtype(i1, i2) {
                            // Allow if inner is subtype (covariant)
                            Ok(t2.clone())
                        } else {
                            Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span))
                        }
                    }
                    // Both mutable: invariant - must match exactly
                    (true, true) => {
                        let inner = self.unify_impl(i1, i2, span)?;
                        Ok(Ty::reference(inner, true))
                    }
                    // Mutable to immutable: allowed (coercion)
                    (true, false) => {
                        if let Ok(inner) = self.unify_impl(i1, i2, span) {
                            Ok(Ty::reference(inner, false))
                        } else if is_subtype(i1, i2) {
                            Ok(t2.clone())
                        } else {
                            Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span))
                        }
                    }
                    // Immutable to mutable: NOT allowed
                    (false, true) => {
                        Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span))
                    }
                }
            }

            // Arrays
            (TyKind::Array { element: e1, size: s1 }, TyKind::Array { element: e2, size: s2 }) => {
                if s1 != s2 {
                    return Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span));
                }
                let element = self.unify_impl(e1, e2, span)?;
                Ok(Ty::array(element, *s1))
            }

            // Slices
            (TyKind::Slice { element: e1 }, TyKind::Slice { element: e2 }) => {
                let element = self.unify_impl(e1, e2, span)?;
                Ok(Ty::slice(element))
            }

            // Tuples
            (TyKind::Tuple(elems1), TyKind::Tuple(elems2)) => {
                if elems1.len() != elems2.len() {
                    return Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span));
                }
                let mut unified = Vec::with_capacity(elems1.len());
                for (e1, e2) in elems1.iter().zip(elems2.iter()) {
                    unified.push(self.unify_impl(e1, e2, span)?);
                }
                Ok(Ty::tuple(unified))
            }

            // Functions
            (TyKind::Fn { params: p1, ret: r1 }, TyKind::Fn { params: p2, ret: r2 }) => {
                if p1.len() != p2.len() {
                    return Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span));
                }
                let mut params = Vec::with_capacity(p1.len());
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    params.push(self.unify_impl(param1, param2, span)?);
                }
                let ret = self.unify_impl(r1, r2, span)?;
                Ok(Ty::function(params, ret))
            }

            // Named types
            (TyKind::Named { name: n1, generics: g1 }, TyKind::Named { name: n2, generics: g2 }) => {
                if n1 != n2 {
                    return Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span));
                }
                if g1.len() != g2.len() {
                    return Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span));
                }
                let mut generics = Vec::with_capacity(g1.len());
                for (gen1, gen2) in g1.iter().zip(g2.iter()) {
                    generics.push(self.unify_impl(gen1, gen2, span)?);
                }
                Ok(Ty::named(n1.clone(), generics))
            }

            // Generic parameters
            (TyKind::Generic { name: n1 }, TyKind::Generic { name: n2 }) if n1 == n2 => {
                Ok(t1.clone())
            }

            // Actors
            (TyKind::Actor { name: n1 }, TyKind::Actor { name: n2 }) if n1 == n2 => {
                Ok(t1.clone())
            }

            // Primitives must match exactly
            (TyKind::Unit, TyKind::Unit) => Ok(Ty::unit()),
            (TyKind::Bool, TyKind::Bool) => Ok(Ty::bool()),
            (TyKind::Char, TyKind::Char) => Ok(Ty::char()),
            (TyKind::Str, TyKind::Str) => Ok(Ty::str()),
            (TyKind::Int(i1), TyKind::Int(i2)) if i1 == i2 => Ok(t1.clone()),
            (TyKind::Uint(u1), TyKind::Uint(u2)) if u1 == u2 => Ok(t1.clone()),
            (TyKind::Float(f1), TyKind::Float(f2)) if f1 == f2 => Ok(t1.clone()),

            // No match
            _ => Err(TypeError::type_mismatch(t1.clone(), t2.clone(), span)),
        }
    }

    /// Unify a type variable with a type
    fn unify_var(&mut self, var: TyVar, ty: &Ty, span: Span) -> TypeResult<Ty> {
        // Check if var is already bound
        if let Some(bound) = self.subst.get(var).cloned() {
            return self.unify_impl(&bound, ty, span);
        }

        // Occurs check: ensure var doesn't appear in ty
        if self.occurs_in(var, ty) {
            return Err(TypeError::infinite_type(
                format!("?T{}", var.0),
                ty.clone(),
                span,
            ));
        }

        // Bind the variable
        self.subst.insert(var, ty.clone());
        Ok(ty.clone())
    }

    /// Check if a type variable occurs in a type (occurs check)
    fn occurs_in(&self, var: TyVar, ty: &Ty) -> bool {
        match &ty.kind {
            TyKind::Var(v) => {
                if var == *v {
                    return true;
                }
                // Check if v is bound to something containing var
                if let Some(bound) = self.subst.get(*v) {
                    return self.occurs_in(var, bound);
                }
                false
            }
            TyKind::Ref { inner, .. } => self.occurs_in(var, inner),
            TyKind::Array { element, .. } => self.occurs_in(var, element),
            TyKind::Slice { element } => self.occurs_in(var, element),
            TyKind::Tuple(elems) => elems.iter().any(|e| self.occurs_in(var, e)),
            TyKind::Fn { params, ret } => {
                params.iter().any(|p| self.occurs_in(var, p)) || self.occurs_in(var, ret)
            }
            TyKind::Named { generics, .. } => generics.iter().any(|g| self.occurs_in(var, g)),
            _ => false,
        }
    }

    /// Apply the current substitution to a type
    pub fn apply(&self, ty: &Ty) -> Ty {
        ty.apply(&self.subst)
    }

    /// Apply the current substitution to a type, defaulting IntVar to i64 and FloatVar to f64
    pub fn apply_with_defaults(&self, ty: &Ty) -> Ty {
        let applied = ty.apply(&self.subst);
        self.default_type_vars(&applied)
    }

    /// Default type variables: IntVar -> i64, FloatVar -> f64
    fn default_type_vars(&self, ty: &Ty) -> Ty {
        match &ty.kind {
            TyKind::IntVar => Ty::i64(),
            TyKind::FloatVar => Ty::f64(),
            TyKind::Var(_) => ty.clone(), // Cannot default generic vars
            TyKind::Ref { inner, mutable } => Ty {
                kind: TyKind::Ref {
                    inner: Box::new(self.default_type_vars(inner)),
                    mutable: *mutable,
                },
            },
            TyKind::Array { element, size } => Ty {
                kind: TyKind::Array {
                    element: Box::new(self.default_type_vars(element)),
                    size: *size,
                },
            },
            TyKind::Slice { element } => Ty {
                kind: TyKind::Slice {
                    element: Box::new(self.default_type_vars(element)),
                },
            },
            TyKind::Tuple(elements) => Ty {
                kind: TyKind::Tuple(elements.iter().map(|t| self.default_type_vars(t)).collect()),
            },
            TyKind::Fn { params, ret } => Ty {
                kind: TyKind::Fn {
                    params: params.iter().map(|t| self.default_type_vars(t)).collect(),
                    ret: Box::new(self.default_type_vars(ret)),
                },
            },
            TyKind::Named { name, generics } => Ty {
                kind: TyKind::Named {
                    name: name.clone(),
                    generics: generics.iter().map(|t| self.default_type_vars(t)).collect(),
                },
            },
            _ => ty.clone(),
        }
    }

    /// Check if two types can be unified without actually unifying them
    pub fn can_unify(&self, t1: &Ty, t2: &Ty) -> bool {
        let mut temp = Unifier {
            subst: self.subst.clone(),
        };
        temp.unify(t1, t2, Span::new(0, 0)).is_ok()
    }
}

impl Default for Unifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a type is assignable to another (one-way compatibility)
pub fn is_assignable(from: &Ty, to: &Ty) -> bool {
    // Same types are always assignable
    if from == to {
        return true;
    }

    match (&from.kind, &to.kind) {
        // Never is assignable to anything
        (TyKind::Never, _) => true,

        // Error type is assignable to anything (error recovery)
        (TyKind::Error, _) | (_, TyKind::Error) => true,

        // Integer literal can be assigned to any integer type
        (TyKind::IntVar, TyKind::Int(_)) |
        (TyKind::IntVar, TyKind::Uint(_)) => true,

        // Float literal can be assigned to any float type
        (TyKind::FloatVar, TyKind::Float(_)) => true,

        // References: &T is assignable to &T
        (TyKind::Ref { inner: i1, mutable: m1 }, TyKind::Ref { inner: i2, mutable: m2 }) => {
            // Can coerce &mut T to &T
            (*m1 || !*m2) && is_assignable(i1, i2)
        }

        // Arrays must have same size
        (TyKind::Array { element: e1, size: s1 }, TyKind::Array { element: e2, size: s2 }) => {
            s1 == s2 && is_assignable(e1, e2)
        }

        // Slices
        (TyKind::Slice { element: e1 }, TyKind::Slice { element: e2 }) => {
            is_assignable(e1, e2)
        }

        // Tuples
        (TyKind::Tuple(elems1), TyKind::Tuple(elems2)) => {
            elems1.len() == elems2.len()
                && elems1.iter().zip(elems2.iter()).all(|(e1, e2)| is_assignable(e1, e2))
        }

        // Functions (contravariant in params, covariant in return)
        (TyKind::Fn { params: p1, ret: r1 }, TyKind::Fn { params: p2, ret: r2 }) => {
            p1.len() == p2.len()
                && p1.iter().zip(p2.iter()).all(|(a, b)| is_assignable(b, a))
                && is_assignable(r1, r2)
        }

        // Named types must match
        (TyKind::Named { name: n1, generics: g1 }, TyKind::Named { name: n2, generics: g2 }) => {
            n1 == n2
                && g1.len() == g2.len()
                && g1.iter().zip(g2.iter()).all(|(a, b)| a == b)
        }

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_same() {
        let mut unifier = Unifier::new();
        let result = unifier.unify(&Ty::i32(), &Ty::i32(), Span::new(0, 0));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Ty::i32());
    }

    #[test]
    fn test_unify_var() {
        TyVar::reset_counter();
        let mut unifier = Unifier::new();
        let var = Ty::fresh_var();
        let result = unifier.unify(&var, &Ty::i32(), Span::new(0, 0));
        assert!(result.is_ok());
        assert_eq!(unifier.apply(&var), Ty::i32());
    }

    #[test]
    fn test_unify_int_var() {
        let mut unifier = Unifier::new();
        let int_var = Ty::new(TyKind::IntVar);
        let result = unifier.unify(&int_var, &Ty::i64(), Span::new(0, 0));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Ty::i64());
    }

    #[test]
    fn test_unify_mismatch() {
        let mut unifier = Unifier::new();
        let result = unifier.unify(&Ty::i32(), &Ty::bool(), Span::new(0, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_tuple() {
        TyVar::reset_counter();
        let mut unifier = Unifier::new();
        let v = Ty::fresh_var();
        let t1 = Ty::tuple(vec![Ty::i32(), v.clone()]);
        let t2 = Ty::tuple(vec![Ty::i32(), Ty::bool()]);
        let result = unifier.unify(&t1, &t2, Span::new(0, 0));
        assert!(result.is_ok());
        assert_eq!(unifier.apply(&v), Ty::bool());
    }

    #[test]
    fn test_occurs_check() {
        TyVar::reset_counter();
        let mut unifier = Unifier::new();
        let v = Ty::fresh_var();
        let recursive = Ty::tuple(vec![v.clone()]);
        let result = unifier.unify(&v, &recursive, Span::new(0, 0));
        assert!(result.is_err()); // Should fail occurs check
    }
}
