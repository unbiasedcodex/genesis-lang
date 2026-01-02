//! Pattern Exhaustiveness Checking
//!
//! This module implements exhaustiveness checking for pattern matching,
//! based on the algorithm described in:
//! - Maranget, Luc. "Compiling Pattern Matching to Good Decision Trees"
//! - Rust Compiler Development Guide: Pattern and Exhaustiveness Checking
//!
//! # Algorithm Overview
//!
//! The core question: "Given a list of patterns, is there any value that is NOT matched?"
//!
//! We decompose values into **constructors** and **fields**:
//! - `Some(x)` has constructor `Some` with 1 field
//! - `None` has constructor `None` with 0 fields
//! - `true` has constructor `true` with 0 fields
//!
//! For each type, we enumerate all possible constructors and check if
//! the patterns cover all of them.

use std::collections::HashSet;
use crate::ast::{Pattern, PatternKind, Literal};
use crate::typeck::{Ty, TyKind, TypeContext};
use crate::span::Span;

/// Result of exhaustiveness check
#[derive(Debug, Clone)]
pub struct ExhaustivenessResult {
    /// Whether the patterns are exhaustive
    pub is_exhaustive: bool,
    /// Missing patterns (witness of non-exhaustiveness)
    pub missing: Vec<MissingPattern>,
    /// Unreachable patterns (redundant arms)
    pub unreachable: Vec<Span>,
}

/// A missing pattern that would make the match exhaustive
#[derive(Debug, Clone)]
pub struct MissingPattern {
    pub description: String,
}

impl MissingPattern {
    fn new(desc: impl Into<String>) -> Self {
        Self { description: desc.into() }
    }
}

/// Constructor: a way to build a value of a type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constructor {
    /// Enum variant: (enum_name, variant_name, arity)
    Variant { enum_name: String, variant: String, arity: usize },
    /// Boolean literal
    Bool(bool),
    /// Integer literal (only for specific checks, not enumerable)
    Int(i128),
    /// String literal (not enumerable)
    Str(String),
    /// Tuple constructor with arity
    Tuple(usize),
    /// Struct constructor
    Struct { name: String },
    /// Wildcard (matches everything)
    Wildcard,
    /// Range pattern (not fully enumerable)
    Range,
}

impl Constructor {
    /// Get the arity (number of fields) of this constructor
    fn arity(&self) -> usize {
        match self {
            Constructor::Variant { arity, .. } => *arity,
            Constructor::Bool(_) => 0,
            Constructor::Int(_) => 0,
            Constructor::Str(_) => 0,
            Constructor::Tuple(n) => *n,
            Constructor::Struct { .. } => 0, // Simplified
            Constructor::Wildcard => 0,
            Constructor::Range => 0,
        }
    }
}

/// The exhaustiveness checker
pub struct ExhaustivenessChecker<'a> {
    ctx: &'a TypeContext,
}

impl<'a> ExhaustivenessChecker<'a> {
    pub fn new(ctx: &'a TypeContext) -> Self {
        Self { ctx }
    }

    /// Check if a set of patterns exhaustively covers a type
    pub fn check(&self, patterns: &[&Pattern], scrutinee_ty: &Ty) -> ExhaustivenessResult {
        let mut unreachable = Vec::new();
        let mut seen_constructors: HashSet<Constructor> = HashSet::new();
        let mut has_wildcard = false;

        // Track which patterns are useful
        for pattern in patterns.iter() {
            let ctor = self.pattern_to_constructor(pattern);

            match &ctor {
                Constructor::Wildcard => {
                    // Wildcard makes everything after unreachable
                    if has_wildcard {
                        unreachable.push(pattern.span);
                    } else {
                        has_wildcard = true;
                    }
                }
                _ => {
                    // Check if this constructor was already covered
                    if seen_constructors.contains(&ctor) {
                        unreachable.push(pattern.span);
                    } else if has_wildcard {
                        // Already have wildcard, this is unreachable
                        unreachable.push(pattern.span);
                    } else {
                        seen_constructors.insert(ctor.clone());
                    }
                }
            }
        }

        // If we have a wildcard, it's exhaustive
        if has_wildcard {
            return ExhaustivenessResult {
                is_exhaustive: true,
                missing: Vec::new(),
                unreachable,
            };
        }

        // Check for missing constructors based on the scrutinee type
        let missing = self.find_missing_constructors(scrutinee_ty, &seen_constructors);

        ExhaustivenessResult {
            is_exhaustive: missing.is_empty(),
            missing,
            unreachable,
        }
    }

    /// Convert a pattern to its constructor
    fn pattern_to_constructor(&self, pattern: &Pattern) -> Constructor {
        match &pattern.kind {
            PatternKind::Wildcard => Constructor::Wildcard,
            PatternKind::Ident { .. } => Constructor::Wildcard, // Binding is like wildcard
            PatternKind::Literal(lit) => match lit {
                Literal::Bool(b) => Constructor::Bool(*b),
                Literal::Int(i) => Constructor::Int(*i),
                Literal::String(s) => Constructor::Str(s.clone()),
                Literal::Float(_) => Constructor::Wildcard, // Float matching is problematic
                Literal::Char(c) => Constructor::Int(*c as i128),
            },
            PatternKind::Tuple(pats) => Constructor::Tuple(pats.len()),
            PatternKind::Struct { path, .. } => {
                let name = path.segments.last()
                    .map(|s| s.ident.name.clone())
                    .unwrap_or_default();
                Constructor::Struct { name }
            }
            PatternKind::Enum { path, fields } => {
                // Extract enum and variant names
                let segments: Vec<_> = path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect();

                let (enum_name, variant) = if segments.len() >= 2 {
                    (segments[segments.len() - 2].clone(), segments.last().unwrap().clone())
                } else {
                    // Single name, could be Option::Some -> just "Some"
                    // Try to infer the enum from context
                    let variant = segments.last().cloned().unwrap_or_default();
                    let enum_name = self.infer_enum_for_variant(&variant);
                    (enum_name, variant)
                };

                Constructor::Variant {
                    enum_name,
                    variant,
                    arity: fields.len(),
                }
            }
            PatternKind::Or(pats) => {
                // Or-pattern: take the first for constructor classification
                // (in full implementation, we'd merge all)
                if let Some(first) = pats.first() {
                    self.pattern_to_constructor(first)
                } else {
                    Constructor::Wildcard
                }
            }
            PatternKind::Range { .. } => Constructor::Range,
            PatternKind::Ref { pattern, .. } => {
                self.pattern_to_constructor(pattern)
            }
        }
    }

    /// Try to infer which enum a variant belongs to
    fn infer_enum_for_variant(&self, variant: &str) -> String {
        // Check built-in types first
        match variant {
            "Some" | "None" => "Option".to_string(),
            "Ok" | "Err" => "Result".to_string(),
            _ => {
                // Search through registered enums
                // This is a simplified approach
                "Unknown".to_string()
            }
        }
    }

    /// Find missing constructors for a type
    fn find_missing_constructors(
        &self,
        ty: &Ty,
        covered: &HashSet<Constructor>,
    ) -> Vec<MissingPattern> {
        let mut missing = Vec::new();

        match &ty.kind {
            TyKind::Bool => {
                let has_true = covered.contains(&Constructor::Bool(true));
                let has_false = covered.contains(&Constructor::Bool(false));

                if !has_true {
                    missing.push(MissingPattern::new("true"));
                }
                if !has_false {
                    missing.push(MissingPattern::new("false"));
                }
            }

            TyKind::Named { name, generics: _ } => {
                // Check if it's an enum
                if let Some(variants) = self.ctx.get_enum_variants(name) {
                    for (variant_name, fields) in variants {
                        // Check if this variant is covered
                        let is_covered = covered.iter().any(|c| {
                            match c {
                                Constructor::Variant { variant, .. } => variant == variant_name,
                                _ => false,
                            }
                        });

                        if !is_covered {
                            // Format the missing pattern nicely
                            let pattern_str = if fields.is_empty() {
                                variant_name.clone()
                            } else {
                                format!("{}(_)", variant_name)
                            };
                            missing.push(MissingPattern::new(pattern_str));
                        }
                    }
                }
                // Non-enum named types are considered exhaustive with any pattern
            }

            TyKind::Tuple(elements) => {
                // Tuples need all elements covered
                // For simplicity, if there's a tuple pattern, we consider it covered
                let tuple_covered = covered.iter().any(|c| {
                    matches!(c, Constructor::Tuple(n) if *n == elements.len())
                });

                if !tuple_covered && !elements.is_empty() {
                    let pattern = format!("({})", vec!["_"; elements.len()].join(", "));
                    missing.push(MissingPattern::new(pattern));
                }
            }

            // Integer types are not enumerable - must have wildcard
            TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) => {
                // Check if there's any coverage at all
                let has_int_pattern = covered.iter().any(|c| {
                    matches!(c, Constructor::Int(_) | Constructor::Range)
                });

                if has_int_pattern {
                    // Has some int patterns but not wildcard - not exhaustive
                    missing.push(MissingPattern::new("_ (not all integers covered)"));
                }
                // If no patterns at all, the check will fail elsewhere
            }

            // String is not enumerable
            TyKind::Str => {
                if covered.iter().any(|c| matches!(c, Constructor::Str(_))) {
                    missing.push(MissingPattern::new("_ (not all strings covered)"));
                }
            }

            // Unit type has only one value
            TyKind::Unit => {
                // Always exhaustive with any pattern
            }

            // Never type has no values
            TyKind::Never => {
                // Always exhaustive (no values to match)
            }

            // For other types, we can't enumerate constructors
            _ => {
                // No exhaustiveness checking possible
            }
        }

        missing
    }
}

/// Format missing patterns for error message
pub fn format_missing_patterns(missing: &[MissingPattern]) -> String {
    if missing.is_empty() {
        return String::new();
    }

    if missing.len() == 1 {
        format!("`{}`", missing[0].description)
    } else if missing.len() <= 3 {
        let patterns: Vec<_> = missing.iter()
            .map(|m| format!("`{}`", m.description))
            .collect();
        let last = patterns.last().unwrap().clone();
        let init = patterns[..patterns.len()-1].join(", ");
        format!("{} and {}", init, last)
    } else {
        let first_three: Vec<_> = missing.iter()
            .take(3)
            .map(|m| format!("`{}`", m.description))
            .collect();
        format!("{} and {} more", first_three.join(", "), missing.len() - 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typeck::TypeContext;
    use crate::ast::{Ident, Path, PathSegment};

    fn dummy_span() -> Span {
        Span::default()
    }

    fn make_ident(name: &str) -> Ident {
        Ident { name: name.to_string(), span: dummy_span() }
    }

    fn make_path(name: &str) -> Path {
        Path {
            segments: vec![PathSegment {
                ident: make_ident(name),
                generics: None,
            }],
            span: dummy_span(),
        }
    }

    fn make_enum_pattern(variant: &str, fields: Vec<Pattern>) -> Pattern {
        Pattern {
            kind: PatternKind::Enum {
                path: make_path(variant),
                fields,
            },
            span: dummy_span(),
        }
    }

    fn make_binding_pattern(name: &str) -> Pattern {
        Pattern {
            kind: PatternKind::Ident {
                name: make_ident(name),
                mutable: false,
            },
            span: dummy_span(),
        }
    }

    #[test]
    fn test_bool_exhaustive() {
        let ctx = TypeContext::new();
        let checker = ExhaustivenessChecker::new(&ctx);

        let true_pat = Pattern {
            kind: PatternKind::Literal(Literal::Bool(true)),
            span: dummy_span(),
        };
        let false_pat = Pattern {
            kind: PatternKind::Literal(Literal::Bool(false)),
            span: dummy_span(),
        };

        let patterns: Vec<&Pattern> = vec![&true_pat, &false_pat];
        let result = checker.check(&patterns, &Ty::bool());

        assert!(result.is_exhaustive, "true and false should be exhaustive for bool");
        assert!(result.missing.is_empty());
    }

    #[test]
    fn test_bool_missing_false() {
        let ctx = TypeContext::new();
        let checker = ExhaustivenessChecker::new(&ctx);

        let true_pat = Pattern {
            kind: PatternKind::Literal(Literal::Bool(true)),
            span: dummy_span(),
        };

        let patterns: Vec<&Pattern> = vec![&true_pat];
        let result = checker.check(&patterns, &Ty::bool());

        assert!(!result.is_exhaustive, "only true should not be exhaustive");
        assert_eq!(result.missing.len(), 1);
        assert_eq!(result.missing[0].description, "false");
    }

    #[test]
    fn test_wildcard_exhaustive() {
        let ctx = TypeContext::new();
        let checker = ExhaustivenessChecker::new(&ctx);

        let wildcard_pat = Pattern {
            kind: PatternKind::Wildcard,
            span: dummy_span(),
        };

        let patterns: Vec<&Pattern> = vec![&wildcard_pat];
        let result = checker.check(&patterns, &Ty::bool());

        assert!(result.is_exhaustive, "wildcard should be exhaustive");
    }

    #[test]
    fn test_binding_exhaustive() {
        let ctx = TypeContext::new();
        let checker = ExhaustivenessChecker::new(&ctx);

        let binding_pat = make_binding_pattern("x");

        let patterns: Vec<&Pattern> = vec![&binding_pat];
        let result = checker.check(&patterns, &Ty::bool());

        assert!(result.is_exhaustive, "binding should be exhaustive (like wildcard)");
    }

    #[test]
    fn test_option_exhaustive() {
        let ctx = TypeContext::new();
        let checker = ExhaustivenessChecker::new(&ctx);

        let some_pat = make_enum_pattern("Some", vec![make_binding_pattern("v")]);
        let none_pat = make_enum_pattern("None", vec![]);

        let option_ty = Ty::new(TyKind::Named {
            name: "Option".to_string(),
            generics: vec![Ty::i64()],
        });

        let patterns: Vec<&Pattern> = vec![&some_pat, &none_pat];
        let result = checker.check(&patterns, &option_ty);

        assert!(result.is_exhaustive, "Some and None should be exhaustive for Option");
    }

    #[test]
    fn test_option_missing_none() {
        let ctx = TypeContext::new();
        let checker = ExhaustivenessChecker::new(&ctx);

        let some_pat = make_enum_pattern("Some", vec![make_binding_pattern("v")]);

        let option_ty = Ty::new(TyKind::Named {
            name: "Option".to_string(),
            generics: vec![Ty::i64()],
        });

        let patterns: Vec<&Pattern> = vec![&some_pat];
        let result = checker.check(&patterns, &option_ty);

        assert!(!result.is_exhaustive, "only Some should not be exhaustive");
        assert!(!result.missing.is_empty());
        assert!(result.missing.iter().any(|m| m.description == "None"),
            "missing should include None, got: {:?}", result.missing);
    }
}
