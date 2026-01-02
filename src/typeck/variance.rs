//! Variance Analysis for Genesis Lang
//!
//! This module implements variance checking for generic types.
//! Variance determines how subtyping relationships are preserved
//! when types are used in different positions.
//!
//! # Variance Kinds
//!
//! - **Covariant**: If A <: B, then F<A> <: F<B>
//!   - Examples: `&T`, return types, read-only containers
//!
//! - **Contravariant**: If A <: B, then F<B> <: F<A>
//!   - Examples: function parameter types
//!
//! - **Invariant**: F<A> and F<B> are only related if A == B
//!   - Examples: `&mut T`, mutable containers
//!
//! - **Bivariant**: F<A> and F<B> are always related
//!   - Examples: unused/phantom type parameters
//!
//! # Genesis Lang Specifics
//!
//! Genesis doesn't have class inheritance, so subtyping is limited to:
//! - Never type (`!`) is a subtype of all types
//! - Reference coercions based on mutability
//! - Numeric coercions (handled separately)

use crate::typeck::ty::{Ty, TyKind};
use std::collections::HashMap;

/// Variance of a type parameter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variance {
    /// Covariant: preserves subtyping direction
    /// If A <: B, then F<A> <: F<B>
    Covariant,

    /// Contravariant: reverses subtyping direction
    /// If A <: B, then F<B> <: F<A>
    Contravariant,

    /// Invariant: no subtyping relationship
    /// F<A> and F<B> only related if A == B
    Invariant,

    /// Bivariant: always related (unused type params)
    Bivariant,
}

impl Variance {
    /// Combine two variances (used when composing type constructors)
    ///
    /// | self \\ other | Co  | Contra | Inv | Bi  |
    /// |---------------|-----|--------|-----|-----|
    /// | Covariant     | Co  | Contra | Inv | Bi  |
    /// | Contravariant | Contra | Co  | Inv | Bi  |
    /// | Invariant     | Inv | Inv    | Inv | Inv |
    /// | Bivariant     | Bi  | Bi     | Bi  | Bi  |
    pub fn compose(self, other: Variance) -> Variance {
        match (self, other) {
            // Bivariant composes to bivariant (or itself)
            (Variance::Bivariant, _) | (_, Variance::Bivariant) => Variance::Bivariant,

            // Invariant dominates
            (Variance::Invariant, _) | (_, Variance::Invariant) => Variance::Invariant,

            // Covariant is identity
            (Variance::Covariant, v) | (v, Variance::Covariant) => v,

            // Contravariant flips
            (Variance::Contravariant, Variance::Contravariant) => Variance::Covariant,
        }
    }

    /// Join two variances (used when a param appears multiple times)
    ///
    /// Takes the most restrictive variance.
    pub fn join(self, other: Variance) -> Variance {
        match (self, other) {
            // Same variance stays the same
            (v1, v2) if v1 == v2 => v1,

            // Bivariant joins with anything
            (Variance::Bivariant, v) | (v, Variance::Bivariant) => v,

            // Otherwise invariant (conflicting uses)
            _ => Variance::Invariant,
        }
    }
}

impl std::fmt::Display for Variance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variance::Covariant => write!(f, "covariant"),
            Variance::Contravariant => write!(f, "contravariant"),
            Variance::Invariant => write!(f, "invariant"),
            Variance::Bivariant => write!(f, "bivariant"),
        }
    }
}

/// Analyzer for computing variance of type parameters
pub struct VarianceAnalyzer {
    /// Computed variances for each type parameter
    variances: HashMap<String, Variance>,
}

impl VarianceAnalyzer {
    /// Create a new variance analyzer
    pub fn new() -> Self {
        Self {
            variances: HashMap::new(),
        }
    }

    /// Compute variance of a type parameter within a type
    ///
    /// # Arguments
    /// * `ty` - The type to analyze
    /// * `param` - The type parameter name to compute variance for
    /// * `position` - Current variance position (starts as Covariant)
    ///
    /// # Returns
    /// The variance of `param` within `ty`
    pub fn compute_variance(&mut self, ty: &Ty, param: &str, position: Variance) -> Variance {
        match &ty.kind {
            // If this is the parameter we're looking for
            TyKind::Generic { name } if name == param => position,

            // Other generic parameters don't contribute
            TyKind::Generic { .. } => Variance::Bivariant,

            // Primitives don't contain type parameters
            TyKind::Unit
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Str
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Never
            | TyKind::Error => Variance::Bivariant,

            // Type variables don't affect variance analysis
            TyKind::Var(_) | TyKind::IntVar | TyKind::FloatVar => Variance::Bivariant,

            // Immutable reference: covariant
            // &T preserves subtyping: if A <: B then &A <: &B
            TyKind::Ref { inner, mutable: false } => {
                self.compute_variance(inner, param, position.compose(Variance::Covariant))
            }

            // Mutable reference: invariant
            // &mut T requires exact type match
            TyKind::Ref { inner, mutable: true } => {
                self.compute_variance(inner, param, position.compose(Variance::Invariant))
            }

            // Array: covariant in element type (immutable semantics)
            TyKind::Array { element, .. } => {
                self.compute_variance(element, param, position.compose(Variance::Covariant))
            }

            // Slice: covariant in element type
            TyKind::Slice { element } => {
                self.compute_variance(element, param, position.compose(Variance::Covariant))
            }

            // Tuple: covariant in all element types
            TyKind::Tuple(elems) => {
                let mut result = Variance::Bivariant;
                for elem in elems {
                    let v = self.compute_variance(elem, param, position.compose(Variance::Covariant));
                    result = result.join(v);
                }
                result
            }

            // Function: contravariant in params, covariant in return
            TyKind::Fn { params, ret } => {
                let mut result = Variance::Bivariant;

                // Parameters are contravariant
                for p in params {
                    let v = self.compute_variance(p, param, position.compose(Variance::Contravariant));
                    result = result.join(v);
                }

                // Return type is covariant
                let ret_v = self.compute_variance(ret, param, position.compose(Variance::Covariant));
                result.join(ret_v)
            }

            // Named types: variance depends on the specific type
            TyKind::Named { name, generics } => {
                let mut result = Variance::Bivariant;

                // Special-case known covariant types
                let is_covariant_type = matches!(
                    name.as_str(),
                    "Option" | "Result" | "Vec" | "Box" | "Future"
                );

                for g in generics {
                    let inner_variance = if is_covariant_type {
                        Variance::Covariant
                    } else {
                        // Default to invariant for user-defined types (safest)
                        Variance::Invariant
                    };
                    let v = self.compute_variance(g, param, position.compose(inner_variance));
                    result = result.join(v);
                }
                result
            }

            // Actors: invariant (actors have identity)
            TyKind::Actor { .. } => Variance::Bivariant,

            // Projection: treat as invariant (complex case)
            TyKind::Projection { .. } => Variance::Invariant,
        }
    }

    /// Get the computed variance for a parameter
    pub fn get_variance(&self, param: &str) -> Option<Variance> {
        self.variances.get(param).copied()
    }
}

impl Default for VarianceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a type is a subtype of another, considering variance
///
/// In Genesis, subtyping is limited to:
/// - Never type is subtype of all types
/// - Reference variance rules
/// - Exact matches for most types
pub fn is_subtype(sub: &Ty, sup: &Ty) -> bool {
    // Never is subtype of everything
    if matches!(sub.kind, TyKind::Never) {
        return true;
    }

    // Error type matches anything (for error recovery)
    if matches!(sub.kind, TyKind::Error) || matches!(sup.kind, TyKind::Error) {
        return true;
    }

    match (&sub.kind, &sup.kind) {
        // Exact type match
        _ if sub == sup => true,

        // References: check variance
        (TyKind::Ref { inner: sub_inner, mutable: sub_mut },
         TyKind::Ref { inner: sup_inner, mutable: sup_mut }) => {
            match (sub_mut, sup_mut) {
                // Immutable to immutable: covariant
                (false, false) => is_subtype(sub_inner, sup_inner),

                // Mutable to immutable: allowed (coercion)
                (true, false) => is_subtype(sub_inner, sup_inner),

                // Mutable to mutable: invariant (exact match required)
                (true, true) => sub_inner == sup_inner,

                // Immutable to mutable: not allowed
                (false, true) => false,
            }
        }

        // Function subtyping: contravariant params, covariant return
        (TyKind::Fn { params: sub_params, ret: sub_ret },
         TyKind::Fn { params: sup_params, ret: sup_ret }) => {
            if sub_params.len() != sup_params.len() {
                return false;
            }

            // Contravariant in parameters
            for (sub_p, sup_p) in sub_params.iter().zip(sup_params.iter()) {
                if !is_subtype(sup_p, sub_p) {  // Note: reversed!
                    return false;
                }
            }

            // Covariant in return
            is_subtype(sub_ret, sup_ret)
        }

        // Arrays: covariant if same size
        (TyKind::Array { element: sub_elem, size: sub_size },
         TyKind::Array { element: sup_elem, size: sup_size }) => {
            sub_size == sup_size && is_subtype(sub_elem, sup_elem)
        }

        // Slices: covariant
        (TyKind::Slice { element: sub_elem },
         TyKind::Slice { element: sup_elem }) => {
            is_subtype(sub_elem, sup_elem)
        }

        // Tuples: covariant in all elements
        (TyKind::Tuple(sub_elems), TyKind::Tuple(sup_elems)) => {
            if sub_elems.len() != sup_elems.len() {
                return false;
            }
            sub_elems.iter().zip(sup_elems.iter())
                .all(|(s, t)| is_subtype(s, t))
        }

        // Named types: covariant for known types, invariant for user types
        (TyKind::Named { name: sub_name, generics: sub_gens },
         TyKind::Named { name: sup_name, generics: sup_gens }) => {
            if sub_name != sup_name || sub_gens.len() != sup_gens.len() {
                return false;
            }

            // Known covariant types
            let is_covariant = matches!(
                sub_name.as_str(),
                "Option" | "Result" | "Vec" | "Box" | "Future"
            );

            if is_covariant {
                // Covariant: all generic args must be subtypes
                sub_gens.iter().zip(sup_gens.iter())
                    .all(|(s, t)| is_subtype(s, t))
            } else {
                // Invariant: all generic args must match exactly
                sub_gens.iter().zip(sup_gens.iter())
                    .all(|(s, t)| s == t)
            }
        }

        // Actors must match exactly
        (TyKind::Actor { name: n1 }, TyKind::Actor { name: n2 }) => n1 == n2,

        // No subtyping relationship
        _ => false,
    }
}

/// Check if two types are compatible considering variance
/// This is a bidirectional check used during type inference
pub fn types_compatible(t1: &Ty, t2: &Ty, variance: Variance) -> bool {
    match variance {
        Variance::Covariant => is_subtype(t1, t2),
        Variance::Contravariant => is_subtype(t2, t1),
        Variance::Invariant => t1 == t2,
        Variance::Bivariant => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_compose() {
        // Covariant is identity
        assert_eq!(Variance::Covariant.compose(Variance::Covariant), Variance::Covariant);
        assert_eq!(Variance::Covariant.compose(Variance::Contravariant), Variance::Contravariant);

        // Contravariant flips
        assert_eq!(Variance::Contravariant.compose(Variance::Covariant), Variance::Contravariant);
        assert_eq!(Variance::Contravariant.compose(Variance::Contravariant), Variance::Covariant);

        // Invariant dominates
        assert_eq!(Variance::Invariant.compose(Variance::Covariant), Variance::Invariant);
        assert_eq!(Variance::Covariant.compose(Variance::Invariant), Variance::Invariant);
    }

    #[test]
    fn test_variance_join() {
        // Same stays same
        assert_eq!(Variance::Covariant.join(Variance::Covariant), Variance::Covariant);

        // Different becomes invariant
        assert_eq!(Variance::Covariant.join(Variance::Contravariant), Variance::Invariant);

        // Bivariant joins with anything
        assert_eq!(Variance::Bivariant.join(Variance::Covariant), Variance::Covariant);
    }

    #[test]
    fn test_is_subtype_never() {
        let never = Ty::never();
        let i64 = Ty::i64();
        let str = Ty::str();

        assert!(is_subtype(&never, &i64));
        assert!(is_subtype(&never, &str));
        assert!(is_subtype(&never, &never));
    }

    #[test]
    fn test_is_subtype_refs() {
        let i64 = Ty::i64();
        let ref_i64 = Ty::reference(i64.clone(), false);
        let mut_ref_i64 = Ty::reference(i64.clone(), true);

        // Same types
        assert!(is_subtype(&ref_i64, &ref_i64));
        assert!(is_subtype(&mut_ref_i64, &mut_ref_i64));

        // &mut T -> &T (coercion)
        assert!(is_subtype(&mut_ref_i64, &ref_i64));

        // &T -> &mut T (not allowed)
        assert!(!is_subtype(&ref_i64, &mut_ref_i64));
    }

    #[test]
    fn test_compute_variance_ref() {
        let mut analyzer = VarianceAnalyzer::new();

        // &T is covariant in T
        let ref_t = Ty::reference(Ty::generic("T".to_string()), false);
        let v = analyzer.compute_variance(&ref_t, "T", Variance::Covariant);
        assert_eq!(v, Variance::Covariant);

        // &mut T is invariant in T
        let mut_ref_t = Ty::reference(Ty::generic("T".to_string()), true);
        let v = analyzer.compute_variance(&mut_ref_t, "T", Variance::Covariant);
        assert_eq!(v, Variance::Invariant);
    }

    #[test]
    fn test_compute_variance_fn() {
        let mut analyzer = VarianceAnalyzer::new();

        // fn(T) -> U is contravariant in T, covariant in U
        let fn_ty = Ty::function(
            vec![Ty::generic("T".to_string())],
            Ty::generic("U".to_string()),
        );

        let v_t = analyzer.compute_variance(&fn_ty, "T", Variance::Covariant);
        assert_eq!(v_t, Variance::Contravariant);

        let v_u = analyzer.compute_variance(&fn_ty, "U", Variance::Covariant);
        assert_eq!(v_u, Variance::Covariant);
    }
}
