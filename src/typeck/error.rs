//! Type Error Definitions
//!
//! This module defines all error types that can occur during type checking.

use crate::span::Span;
use crate::typeck::Ty;
use std::fmt;

/// Result type for type checking operations
pub type TypeResult<T> = Result<T, TypeError>;

/// Type checking error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
}

impl TypeError {
    pub fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self { kind, span }
    }

    // ============ Error Constructors ============

    pub fn type_mismatch(expected: Ty, found: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::TypeMismatch { expected, found }, span)
    }

    pub fn undefined_variable(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::UndefinedVariable { name }, span)
    }

    pub fn undefined_type(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::UndefinedType { name }, span)
    }

    pub fn undefined_function(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::UndefinedFunction { name }, span)
    }

    pub fn undefined_field(ty: Ty, field: String, span: Span) -> Self {
        Self::new(TypeErrorKind::UndefinedField { ty, field }, span)
    }

    pub fn undefined_method(ty: Ty, method: String, span: Span) -> Self {
        Self::new(TypeErrorKind::UndefinedMethod { ty, method }, span)
    }

    pub fn not_callable(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::NotCallable { ty }, span)
    }

    pub fn wrong_arg_count(expected: usize, found: usize, span: Span) -> Self {
        Self::new(TypeErrorKind::WrongArgCount { expected, found }, span)
    }

    pub fn cannot_infer(span: Span) -> Self {
        Self::new(TypeErrorKind::CannotInfer, span)
    }

    pub fn infinite_type(var: String, ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::InfiniteType { var, ty }, span)
    }

    pub fn not_indexable(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::NotIndexable { ty }, span)
    }

    pub fn binary_op_mismatch(op: String, left: Ty, right: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::BinaryOpMismatch { op, left, right }, span)
    }

    pub fn unary_op_mismatch(op: String, ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::UnaryOpMismatch { op, ty }, span)
    }

    pub fn cannot_deref(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::CannotDeref { ty }, span)
    }

    pub fn missing_field(struct_name: String, field: String, span: Span) -> Self {
        Self::new(TypeErrorKind::MissingField { struct_name, field }, span)
    }

    pub fn duplicate_field(field: String, span: Span) -> Self {
        Self::new(TypeErrorKind::DuplicateField { field }, span)
    }

    pub fn extra_field(struct_name: String, field: String, span: Span) -> Self {
        Self::new(TypeErrorKind::ExtraField { struct_name, field }, span)
    }

    pub fn moved_value(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::MovedValue { name }, span)
    }

    pub fn borrowed_as_mutable(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::BorrowedAsMutable { name }, span)
    }

    pub fn already_borrowed(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::AlreadyBorrowed { name }, span)
    }

    pub fn not_mutable(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::NotMutable { name }, span)
    }

    pub fn return_type_mismatch(expected: Ty, found: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::ReturnTypeMismatch { expected, found }, span)
    }

    pub fn pattern_type_mismatch(expected: Ty, found: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::PatternTypeMismatch { expected, found }, span)
    }

    pub fn non_exhaustive_patterns(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::NonExhaustivePatterns { ty }, span)
    }

    pub fn non_exhaustive_match(ty: Ty, missing: String, span: Span) -> Self {
        Self::new(TypeErrorKind::NonExhaustiveMatch { ty, missing }, span)
    }

    pub fn duplicate_definition(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::DuplicateDefinition { name }, span)
    }

    pub fn wrong_generic_count(name: String, expected: usize, found: usize, span: Span) -> Self {
        Self::new(
            TypeErrorKind::WrongGenericCount {
                name,
                expected,
                found,
            },
            span,
        )
    }

    pub fn trait_not_implemented(ty: Ty, trait_name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::TraitNotImplemented { ty, trait_name }, span)
    }

    pub fn trait_not_found(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::TraitNotFound { name }, span)
    }

    pub fn missing_trait_method(trait_name: String, method_name: String, self_type: String, span: Span) -> Self {
        Self::new(TypeErrorKind::MissingTraitMethod { trait_name, method_name, self_type }, span)
    }

    pub fn missing_associated_type(trait_name: String, type_name: String, self_type: String, span: Span) -> Self {
        Self::new(TypeErrorKind::MissingAssociatedType { trait_name, type_name, self_type }, span)
    }

    pub fn trait_method_signature_mismatch(
        trait_name: String,
        method_name: String,
        expected: String,
        found: String,
        span: Span,
    ) -> Self {
        Self::new(
            TypeErrorKind::TraitMethodSignatureMismatch {
                trait_name,
                method_name,
                expected,
                found,
            },
            span,
        )
    }

    pub fn not_an_actor(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::NotAnActor { ty }, span)
    }

    pub fn invalid_message(actor: String, message: String, span: Span) -> Self {
        Self::new(TypeErrorKind::InvalidMessage { actor, message }, span)
    }

    pub fn invalid_cast(from: Ty, to: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::InvalidCast { from, to }, span)
    }

    pub fn condition_not_bool(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::ConditionNotBool { ty }, span)
    }

    pub fn break_outside_loop(span: Span) -> Self {
        Self::new(TypeErrorKind::BreakOutsideLoop, span)
    }

    pub fn continue_outside_loop(span: Span) -> Self {
        Self::new(TypeErrorKind::ContinueOutsideLoop, span)
    }

    pub fn return_outside_function(span: Span) -> Self {
        Self::new(TypeErrorKind::ReturnOutsideFunction, span)
    }

    pub fn await_on_non_future(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::AwaitOnNonFuture { ty }, span)
    }

    pub fn try_on_non_result(ty: Ty, span: Span) -> Self {
        Self::new(TypeErrorKind::TryOnNonResult { ty }, span)
    }

    pub fn private_access(name: String, module: String, span: Span) -> Self {
        Self::new(TypeErrorKind::PrivateAccess { name, module }, span)
    }

    pub fn undefined_macro(name: String, span: Span) -> Self {
        Self::new(TypeErrorKind::UndefinedMacro { name }, span)
    }

    pub fn macro_expansion_error(message: String, span: Span) -> Self {
        Self::new(TypeErrorKind::MacroExpansionError { message }, span)
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for TypeError {}

/// The kind of type error
#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    /// Type mismatch: expected X, found Y
    TypeMismatch { expected: Ty, found: Ty },

    /// Undefined variable
    UndefinedVariable { name: String },

    /// Undefined type
    UndefinedType { name: String },

    /// Undefined function
    UndefinedFunction { name: String },

    /// Undefined field on type
    UndefinedField { ty: Ty, field: String },

    /// Undefined method on type
    UndefinedMethod { ty: Ty, method: String },

    /// Trying to call a non-callable type
    NotCallable { ty: Ty },

    /// Wrong number of arguments
    WrongArgCount { expected: usize, found: usize },

    /// Cannot infer type
    CannotInfer,

    /// Infinite type (occurs check failure)
    InfiniteType { var: String, ty: Ty },

    /// Type is not indexable
    NotIndexable { ty: Ty },

    /// Binary operator type mismatch
    BinaryOpMismatch { op: String, left: Ty, right: Ty },

    /// Unary operator type mismatch
    UnaryOpMismatch { op: String, ty: Ty },

    /// Cannot dereference type
    CannotDeref { ty: Ty },

    /// Missing field in struct literal
    MissingField { struct_name: String, field: String },

    /// Duplicate field in struct literal
    DuplicateField { field: String },

    /// Extra field in struct literal
    ExtraField { struct_name: String, field: String },

    /// Value was moved
    MovedValue { name: String },

    /// Trying to mutably borrow something already borrowed
    BorrowedAsMutable { name: String },

    /// Already borrowed
    AlreadyBorrowed { name: String },

    /// Variable is not mutable
    NotMutable { name: String },

    /// Return type mismatch
    ReturnTypeMismatch { expected: Ty, found: Ty },

    /// Pattern doesn't match type
    PatternTypeMismatch { expected: Ty, found: Ty },

    /// Match is not exhaustive
    NonExhaustivePatterns { ty: Ty },

    /// Match is not exhaustive (with missing patterns)
    NonExhaustiveMatch { ty: Ty, missing: String },

    /// Duplicate definition
    DuplicateDefinition { name: String },

    /// Wrong number of generic arguments
    WrongGenericCount {
        name: String,
        expected: usize,
        found: usize,
    },

    /// Trait is not implemented for type
    TraitNotImplemented { ty: Ty, trait_name: String },

    /// Trait not found
    TraitNotFound { name: String },

    /// Missing required trait method in impl
    MissingTraitMethod { trait_name: String, method_name: String, self_type: String },

    /// Missing required associated type in impl
    MissingAssociatedType { trait_name: String, type_name: String, self_type: String },

    /// Trait method signature mismatch
    TraitMethodSignatureMismatch {
        trait_name: String,
        method_name: String,
        expected: String,
        found: String,
    },

    /// Expected an actor type
    NotAnActor { ty: Ty },

    /// Invalid message for actor
    InvalidMessage { actor: String, message: String },

    /// Invalid type cast
    InvalidCast { from: Ty, to: Ty },

    /// Condition must be bool
    ConditionNotBool { ty: Ty },

    /// Break outside of loop
    BreakOutsideLoop,

    /// Continue outside of loop
    ContinueOutsideLoop,

    /// Return outside of function
    ReturnOutsideFunction,

    /// Await on non-Future type
    AwaitOnNonFuture { ty: Ty },

    /// Try operator on non-Result/Option type
    TryOnNonResult { ty: Ty },

    /// Accessing private item from outside its module
    PrivateAccess { name: String, module: String },

    /// Undefined macro
    UndefinedMacro { name: String },

    /// Macro expansion error
    MacroExpansionError { message: String },
}

impl fmt::Display for TypeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeErrorKind::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected `{}`, found `{}`", expected, found)
            }
            TypeErrorKind::UndefinedVariable { name } => {
                write!(f, "undefined variable `{}`", name)
            }
            TypeErrorKind::UndefinedType { name } => {
                write!(f, "undefined type `{}`", name)
            }
            TypeErrorKind::UndefinedFunction { name } => {
                write!(f, "undefined function `{}`", name)
            }
            TypeErrorKind::UndefinedField { ty, field } => {
                write!(f, "type `{}` has no field `{}`", ty, field)
            }
            TypeErrorKind::UndefinedMethod { ty, method } => {
                write!(f, "type `{}` has no method `{}`", ty, method)
            }
            TypeErrorKind::NotCallable { ty } => {
                write!(f, "type `{}` is not callable", ty)
            }
            TypeErrorKind::WrongArgCount { expected, found } => {
                write!(
                    f,
                    "wrong number of arguments: expected {}, found {}",
                    expected, found
                )
            }
            TypeErrorKind::CannotInfer => {
                write!(f, "cannot infer type")
            }
            TypeErrorKind::InfiniteType { var, ty } => {
                write!(f, "infinite type: {} = {}", var, ty)
            }
            TypeErrorKind::NotIndexable { ty } => {
                write!(f, "type `{}` cannot be indexed", ty)
            }
            TypeErrorKind::BinaryOpMismatch { op, left, right } => {
                write!(
                    f,
                    "cannot apply `{}` to `{}` and `{}`",
                    op, left, right
                )
            }
            TypeErrorKind::UnaryOpMismatch { op, ty } => {
                write!(f, "cannot apply `{}` to `{}`", op, ty)
            }
            TypeErrorKind::CannotDeref { ty } => {
                write!(f, "cannot dereference type `{}`", ty)
            }
            TypeErrorKind::MissingField { struct_name, field } => {
                write!(f, "missing field `{}` in struct `{}`", field, struct_name)
            }
            TypeErrorKind::DuplicateField { field } => {
                write!(f, "duplicate field `{}`", field)
            }
            TypeErrorKind::ExtraField { struct_name, field } => {
                write!(
                    f,
                    "struct `{}` has no field `{}`",
                    struct_name, field
                )
            }
            TypeErrorKind::MovedValue { name } => {
                write!(f, "use of moved value `{}`", name)
            }
            TypeErrorKind::BorrowedAsMutable { name } => {
                write!(f, "cannot borrow `{}` as mutable", name)
            }
            TypeErrorKind::AlreadyBorrowed { name } => {
                write!(f, "`{}` is already borrowed", name)
            }
            TypeErrorKind::NotMutable { name } => {
                write!(f, "`{}` is not mutable", name)
            }
            TypeErrorKind::ReturnTypeMismatch { expected, found } => {
                write!(
                    f,
                    "return type mismatch: expected `{}`, found `{}`",
                    expected, found
                )
            }
            TypeErrorKind::PatternTypeMismatch { expected, found } => {
                write!(
                    f,
                    "pattern type mismatch: expected `{}`, found `{}`",
                    expected, found
                )
            }
            TypeErrorKind::NonExhaustivePatterns { ty } => {
                write!(f, "non-exhaustive patterns for type `{}`", ty)
            }
            TypeErrorKind::NonExhaustiveMatch { ty, missing } => {
                write!(f, "non-exhaustive patterns: {} not covered for type `{}`", missing, ty)
            }
            TypeErrorKind::DuplicateDefinition { name } => {
                write!(f, "duplicate definition of `{}`", name)
            }
            TypeErrorKind::WrongGenericCount {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "`{}` expects {} generic argument(s), found {}",
                    name, expected, found
                )
            }
            TypeErrorKind::TraitNotImplemented { ty, trait_name } => {
                write!(f, "trait `{}` is not implemented for `{}`", trait_name, ty)
            }
            TypeErrorKind::TraitNotFound { name } => {
                write!(f, "trait `{}` not found", name)
            }
            TypeErrorKind::MissingTraitMethod { trait_name, method_name, self_type } => {
                write!(
                    f,
                    "not all trait items implemented, missing: `{}` from trait `{}` for type `{}`",
                    method_name, trait_name, self_type
                )
            }
            TypeErrorKind::MissingAssociatedType { trait_name, type_name, self_type } => {
                write!(
                    f,
                    "not all trait items implemented, missing associated type `{}` from trait `{}` for type `{}`",
                    type_name, trait_name, self_type
                )
            }
            TypeErrorKind::TraitMethodSignatureMismatch {
                trait_name,
                method_name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "method `{}` has an incompatible signature for trait `{}`: expected `{}`, found `{}`",
                    method_name, trait_name, expected, found
                )
            }
            TypeErrorKind::NotAnActor { ty } => {
                write!(f, "expected actor, found `{}`", ty)
            }
            TypeErrorKind::InvalidMessage { actor, message } => {
                write!(f, "actor `{}` does not handle message `{}`", actor, message)
            }
            TypeErrorKind::InvalidCast { from, to } => {
                write!(f, "cannot cast `{}` to `{}`", from, to)
            }
            TypeErrorKind::ConditionNotBool { ty } => {
                write!(f, "condition must be `bool`, found `{}`", ty)
            }
            TypeErrorKind::BreakOutsideLoop => {
                write!(f, "`break` outside of loop")
            }
            TypeErrorKind::ContinueOutsideLoop => {
                write!(f, "`continue` outside of loop")
            }
            TypeErrorKind::ReturnOutsideFunction => {
                write!(f, "`return` outside of function")
            }
            TypeErrorKind::AwaitOnNonFuture { ty } => {
                write!(f, "`await` requires `Future`, found `{}`", ty)
            }
            TypeErrorKind::TryOnNonResult { ty } => {
                write!(f, "`?` operator requires `Result` or `Option`, found `{}`", ty)
            }
            TypeErrorKind::PrivateAccess { name, module } => {
                write!(f, "`{}` is private to module `{}`", name, module)
            }
            TypeErrorKind::UndefinedMacro { name } => {
                write!(f, "undefined macro `{}`", name)
            }
            TypeErrorKind::MacroExpansionError { message } => {
                write!(f, "macro expansion error: {}", message)
            }
        }
    }
}
