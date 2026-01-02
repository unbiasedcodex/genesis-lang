//! Type Checking Utilities
//!
//! This module provides additional type checking utilities and helpers.

#![allow(dead_code)]

use crate::typeck::ty::Ty;

/// Check if a type satisfies a trait bound
pub fn satisfies_bound(ty: &Ty, bound: &str) -> bool {
    // For now, implement some basic built-in traits
    match bound {
        "Copy" => is_copy(ty),
        "Clone" => is_clone(ty),
        "Debug" => true, // Assume everything is Debug
        "Display" => is_display(ty),
        "Default" => is_default(ty),
        "Eq" => is_eq(ty),
        "PartialEq" => is_partial_eq(ty),
        "Ord" => is_ord(ty),
        "PartialOrd" => is_partial_ord(ty),
        "Hash" => is_hash(ty),
        "Send" => is_send(ty),
        "Sync" => is_sync(ty),
        _ => false,
    }
}

/// Check if a type is Copy
pub fn is_copy(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        // Primitives are Copy
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::Never => true,

        // References are Copy
        TyKind::Ref { mutable, .. } => !mutable,

        // Arrays of Copy types are Copy
        TyKind::Array { element, .. } => is_copy(element),

        // Tuples of Copy types are Copy
        TyKind::Tuple(elems) => elems.iter().all(is_copy),

        // Function pointers are Copy
        TyKind::Fn { .. } => true,

        // Named types need to check their fields
        TyKind::Named { name, .. } => {
            // Some built-in types are Copy
            matches!(name.as_str(), "Option" | "Result")
        }

        _ => false,
    }
}

/// Check if a type is Clone
pub fn is_clone(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    // Everything that's Copy is also Clone
    if is_copy(ty) {
        return true;
    }

    match &ty.kind {
        TyKind::Str => true,
        TyKind::Named { name, .. } => {
            matches!(name.as_str(), "String" | "Vec" | "Option" | "Result")
        }
        TyKind::Array { element, .. } => is_clone(element),
        TyKind::Tuple(elems) => elems.iter().all(is_clone),
        TyKind::Slice { element } => is_clone(element),
        _ => false,
    }
}

/// Check if a type implements Display
pub fn is_display(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_) => true,
        TyKind::Named { name, .. } => matches!(name.as_str(), "String"),
        _ => false,
    }
}

/// Check if a type implements Default
pub fn is_default(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_) => true,
        TyKind::Named { name, .. } => {
            matches!(name.as_str(), "String" | "Vec" | "Option")
        }
        TyKind::Tuple(elems) => elems.iter().all(is_default),
        TyKind::Array { element, .. } => is_default(element),
        _ => false,
    }
}

/// Check if a type implements Eq
pub fn is_eq(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Int(_)
        | TyKind::Uint(_) => true,
        // Floats are not Eq (due to NaN)
        TyKind::Float(_) => false,
        TyKind::Named { name, .. } => matches!(name.as_str(), "String"),
        TyKind::Tuple(elems) => elems.iter().all(is_eq),
        TyKind::Array { element, .. } => is_eq(element),
        TyKind::Ref { inner, .. } => is_eq(inner),
        _ => false,
    }
}

/// Check if a type implements PartialEq
pub fn is_partial_eq(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    // Everything that's Eq is also PartialEq
    if is_eq(ty) {
        return true;
    }

    match &ty.kind {
        TyKind::Float(_) => true,
        _ => false,
    }
}

/// Check if a type implements Ord
pub fn is_ord(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Int(_)
        | TyKind::Uint(_) => true,
        TyKind::Float(_) => false,
        TyKind::Named { name, .. } => matches!(name.as_str(), "String"),
        TyKind::Tuple(elems) => elems.iter().all(is_ord),
        _ => false,
    }
}

/// Check if a type implements PartialOrd
pub fn is_partial_ord(ty: &Ty) -> bool {
    if is_ord(ty) {
        return true;
    }

    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Float(_) => true,
        _ => false,
    }
}

/// Check if a type implements Hash
pub fn is_hash(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Int(_)
        | TyKind::Uint(_) => true,
        TyKind::Float(_) => false,
        TyKind::Named { name, .. } => matches!(name.as_str(), "String"),
        TyKind::Tuple(elems) => elems.iter().all(is_hash),
        TyKind::Array { element, .. } => is_hash(element),
        TyKind::Ref { inner, .. } => is_hash(inner),
        _ => false,
    }
}

/// Check if a type is Send
pub fn is_send(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        // Primitives are Send
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::Never => true,

        // References to Send types are Send
        TyKind::Ref { inner, .. } => is_send(inner),

        // Containers of Send types are Send
        TyKind::Array { element, .. } => is_send(element),
        TyKind::Tuple(elems) => elems.iter().all(is_send),
        TyKind::Slice { element } => is_send(element),

        // Most named types are Send
        TyKind::Named { .. } => true,

        // Actors are Send
        TyKind::Actor { .. } => true,

        _ => true,
    }
}

/// Check if a type is Sync
pub fn is_sync(ty: &Ty) -> bool {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        // Primitives are Sync
        TyKind::Unit
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::Never => true,

        // Immutable references to Sync types are Sync
        TyKind::Ref { inner, mutable } => !mutable && is_sync(inner),

        // Containers of Sync types are Sync
        TyKind::Array { element, .. } => is_sync(element),
        TyKind::Tuple(elems) => elems.iter().all(is_sync),

        _ => true,
    }
}

/// Get the size of a type (in bytes, approximate)
pub fn size_of(ty: &Ty) -> Option<usize> {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit | TyKind::Never => Some(0),
        TyKind::Bool => Some(1),
        TyKind::Char => Some(4),
        TyKind::Int(int_ty) => {
            use crate::typeck::ty::IntTy;
            Some(match int_ty {
                IntTy::I8 => 1,
                IntTy::I16 => 2,
                IntTy::I32 => 4,
                IntTy::I64 => 8,
                IntTy::I128 => 16,
                IntTy::Isize => 8,
            })
        }
        TyKind::Uint(uint_ty) => {
            use crate::typeck::ty::UintTy;
            Some(match uint_ty {
                UintTy::U8 => 1,
                UintTy::U16 => 2,
                UintTy::U32 => 4,
                UintTy::U64 => 8,
                UintTy::U128 => 16,
                UintTy::Usize => 8,
            })
        }
        TyKind::Float(float_ty) => {
            use crate::typeck::ty::FloatTy;
            Some(match float_ty {
                FloatTy::F32 => 4,
                FloatTy::F64 => 8,
            })
        }
        TyKind::Ref { .. } => Some(8), // Pointer size
        TyKind::Array { element, size } => size_of(element).map(|s| s * size),
        TyKind::Tuple(elems) => {
            elems.iter().map(size_of).try_fold(0, |acc, s| s.map(|s| acc + s))
        }
        _ => None,
    }
}

/// Get the alignment of a type (in bytes)
pub fn align_of(ty: &Ty) -> Option<usize> {
    use crate::typeck::ty::TyKind;
    match &ty.kind {
        TyKind::Unit | TyKind::Never => Some(1),
        TyKind::Bool => Some(1),
        TyKind::Char => Some(4),
        TyKind::Int(int_ty) => {
            use crate::typeck::ty::IntTy;
            Some(match int_ty {
                IntTy::I8 => 1,
                IntTy::I16 => 2,
                IntTy::I32 => 4,
                IntTy::I64 => 8,
                IntTy::I128 => 16,
                IntTy::Isize => 8,
            })
        }
        TyKind::Uint(uint_ty) => {
            use crate::typeck::ty::UintTy;
            Some(match uint_ty {
                UintTy::U8 => 1,
                UintTy::U16 => 2,
                UintTy::U32 => 4,
                UintTy::U64 => 8,
                UintTy::U128 => 16,
                UintTy::Usize => 8,
            })
        }
        TyKind::Float(float_ty) => {
            use crate::typeck::ty::FloatTy;
            Some(match float_ty {
                FloatTy::F32 => 4,
                FloatTy::F64 => 8,
            })
        }
        TyKind::Ref { .. } => Some(8),
        TyKind::Array { element, .. } => align_of(element),
        TyKind::Tuple(elems) => {
            elems.iter().filter_map(align_of).max().or(Some(1))
        }
        _ => None,
    }
}
