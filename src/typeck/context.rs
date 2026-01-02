//! Type Context and Symbol Table
//!
//! This module manages the type environment during type checking.
//! It tracks:
//! - Variable bindings and their types
//! - Type definitions (structs, enums, type aliases)
//! - Function signatures
//! - Scopes and lexical nesting

use std::collections::HashMap;
use crate::typeck::{Ty, TyKind};
use crate::ast::Generics;

/// The type context manages all type information during type checking
#[derive(Debug)]
pub struct TypeContext {
    /// Stack of scopes (innermost last)
    scopes: Vec<Scope>,
    /// Type definitions (structs, enums, type aliases)
    types: HashMap<String, TypeDef>,
    /// Function signatures
    functions: HashMap<String, FnSig>,
    /// Trait definitions
    traits: HashMap<String, TraitDef>,
    /// Actor definitions
    actors: HashMap<String, ActorDef>,
    /// Impl blocks
    impls: Vec<ImplDef>,
    /// Generic parameters in current scope
    generics: Vec<HashMap<String, Ty>>,
    /// Module definitions (for inline modules)
    modules: HashMap<String, ModuleDef>,
    /// Import aliases (from use statements)
    imports: HashMap<String, ImportedSymbol>,
    /// Current module we're checking (None for top-level)
    current_module: Option<String>,
    /// Macro definitions
    macros: HashMap<String, MacroDefInfo>,
}

/// A module definition
#[derive(Debug, Clone)]
pub struct ModuleDef {
    pub name: String,
    pub types: Vec<String>,
    pub functions: Vec<String>,
    pub is_pub: bool,
}

/// Macro definition info for type checking
#[derive(Debug, Clone)]
pub struct MacroDefInfo {
    pub name: String,
    pub is_pub: bool,
    pub module: Option<String>,
    /// The full macro definition (rules, patterns, etc.)
    pub def: crate::ast::MacroDef,
}

/// An imported symbol from a use statement
#[derive(Debug, Clone)]
pub struct ImportedSymbol {
    pub original_path: Vec<String>,
    pub kind: ImportKind,
}

/// Kind of imported symbol
#[derive(Debug, Clone)]
pub enum ImportKind {
    Type,
    Function,
    Module,
}

impl TypeContext {
    /// Create a new type context with built-in types
    pub fn new() -> Self {
        let mut ctx = Self {
            scopes: vec![Scope::new()],
            types: HashMap::new(),
            functions: HashMap::new(),
            traits: HashMap::new(),
            actors: HashMap::new(),
            impls: Vec::new(),
            generics: Vec::new(),
            modules: HashMap::new(),
            imports: HashMap::new(),
            current_module: None,
            macros: HashMap::new(),
        };
        ctx.register_builtins();
        ctx
    }

    /// Set the current module we're checking
    pub fn set_current_module(&mut self, module: Option<String>) {
        self.current_module = module;
    }

    /// Get the current module
    pub fn current_module(&self) -> Option<&String> {
        self.current_module.as_ref()
    }

    /// Check if an item from a module is visible from the current context
    pub fn is_visible(&self, item_module: Option<&String>, is_pub: bool) -> bool {
        // Public items are always visible
        if is_pub {
            return true;
        }
        // Private items in same module are visible
        match (&self.current_module, item_module) {
            (None, None) => true, // Both top-level
            (Some(current), Some(item)) => current == item, // Same module
            (None, Some(_)) => false, // Accessing private item from module at top-level
            (Some(_), None) => true, // Top-level items visible to modules
        }
    }

    /// Register built-in types and functions
    fn register_builtins(&mut self) {
        // Register Option type
        self.types.insert(
            "Option".to_string(),
            TypeDef {
                name: "Option".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Enum {
                    variants: vec![
                        ("Some".to_string(), vec![Ty::generic("T".to_string())]),
                        ("None".to_string(), vec![]),
                    ],
                },
                is_pub: true,
                module: None,
            },
        );

        // Register Result type
        self.types.insert(
            "Result".to_string(),
            TypeDef {
                name: "Result".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Enum {
                    variants: vec![
                        ("Ok".to_string(), vec![Ty::generic("T".to_string())]),
                        ("Err".to_string(), vec![Ty::generic("E".to_string())]),
                    ],
                },
                is_pub: true,
                module: None,
            },
        );

        // Register Vec type
        self.types.insert(
            "Vec".to_string(),
            TypeDef {
                name: "Vec".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct {
                    fields: vec![], // Opaque built-in type
                },
                is_pub: true,
                module: None,
            },
        );

        // Register String type
        self.types.insert(
            "String".to_string(),
            TypeDef {
                name: "String".to_string(),
                generics: vec![],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] },
                is_pub: true,
                module: None,
            },
        );

        // Register Future type for async/await
        self.types.insert(
            "Future".to_string(),
            TypeDef {
                name: "Future".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque built-in
                is_pub: true,
                module: None,
            },
        );

        // ============ Option<T> Methods ============

        // Option::is_some(opt) - check if Option contains a value
        self.functions.insert(
            "Option::is_some".to_string(),
            FnSig {
                name: "Option::is_some".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::option(Ty::generic("T".to_string()))],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Option::is_none(opt) - check if Option is empty
        self.functions.insert(
            "Option::is_none".to_string(),
            FnSig {
                name: "Option::is_none".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::option(Ty::generic("T".to_string()))],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Option::unwrap(opt) - extract value, undefined behavior if None
        self.functions.insert(
            "Option::unwrap".to_string(),
            FnSig {
                name: "Option::unwrap".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::option(Ty::generic("T".to_string()))],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Option::unwrap_or(opt, default) - extract value or return default
        self.functions.insert(
            "Option::unwrap_or".to_string(),
            FnSig {
                name: "Option::unwrap_or".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::option(Ty::generic("T".to_string())),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Option::expect(opt, msg) - extract value or print message and abort
        self.functions.insert(
            "Option::expect".to_string(),
            FnSig {
                name: "Option::expect".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::option(Ty::generic("T".to_string())),
                    Ty::str(),
                ],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Option::map(opt, f) - transform the inner value if Some
        self.functions.insert(
            "Option::map".to_string(),
            FnSig {
                name: "Option::map".to_string(),
                generics: vec!["T".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::option(Ty::generic("T".to_string())),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::generic("U".to_string())),
                ],
                ret: Ty::option(Ty::generic("U".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Option::and_then(opt, f) - chain Option-returning operations
        self.functions.insert(
            "Option::and_then".to_string(),
            FnSig {
                name: "Option::and_then".to_string(),
                generics: vec!["T".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::option(Ty::generic("T".to_string())),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::option(Ty::generic("U".to_string()))),
                ],
                ret: Ty::option(Ty::generic("U".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Result<T, E> Methods ============

        // Result::is_ok(result) - check if Result is Ok
        self.functions.insert(
            "Result::is_ok".to_string(),
            FnSig {
                name: "Result::is_ok".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string()))],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::is_err(result) - check if Result is Err
        self.functions.insert(
            "Result::is_err".to_string(),
            FnSig {
                name: "Result::is_err".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string()))],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::unwrap(result) - extract Ok value, undefined behavior if Err
        self.functions.insert(
            "Result::unwrap".to_string(),
            FnSig {
                name: "Result::unwrap".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string()))],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::unwrap_err(result) - extract Err value, undefined behavior if Ok
        self.functions.insert(
            "Result::unwrap_err".to_string(),
            FnSig {
                name: "Result::unwrap_err".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string()))],
                ret: Ty::generic("E".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::unwrap_or(result, default) - extract Ok value or return default
        self.functions.insert(
            "Result::unwrap_or".to_string(),
            FnSig {
                name: "Result::unwrap_or".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string())),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::expect(result, msg) - extract Ok value or print message and abort
        self.functions.insert(
            "Result::expect".to_string(),
            FnSig {
                name: "Result::expect".to_string(),
                generics: vec!["T".to_string(), "E".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string())),
                    Ty::str(),
                ],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::map(result, f) - transform the Ok value
        self.functions.insert(
            "Result::map".to_string(),
            FnSig {
                name: "Result::map".to_string(),
                generics: vec!["T".to_string(), "E".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string())),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::generic("U".to_string())),
                ],
                ret: Ty::result(Ty::generic("U".to_string()), Ty::generic("E".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::map_err(result, f) - transform the Err value
        self.functions.insert(
            "Result::map_err".to_string(),
            FnSig {
                name: "Result::map_err".to_string(),
                generics: vec!["T".to_string(), "E".to_string(), "F".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string())),
                    Ty::function(vec![Ty::generic("E".to_string())], Ty::generic("F".to_string())),
                ],
                ret: Ty::result(Ty::generic("T".to_string()), Ty::generic("F".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Result::and_then(result, f) - chain Result-returning operations
        self.functions.insert(
            "Result::and_then".to_string(),
            FnSig {
                name: "Result::and_then".to_string(),
                generics: vec!["T".to_string(), "E".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::result(Ty::generic("T".to_string()), Ty::generic("E".to_string())),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::result(Ty::generic("U".to_string()), Ty::generic("E".to_string()))),
                ],
                ret: Ty::result(Ty::generic("U".to_string()), Ty::generic("E".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register built-in print function
        self.functions.insert(
            "print".to_string(),
            FnSig {
                name: "print".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::generic("T".to_string())],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register built-in println function
        self.functions.insert(
            "println".to_string(),
            FnSig {
                name: "println".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::generic("T".to_string())],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register built-in eprint function (prints to stderr)
        self.functions.insert(
            "eprint".to_string(),
            FnSig {
                name: "eprint".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::generic("T".to_string())],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register built-in eprintln function (prints to stderr with newline)
        self.functions.insert(
            "eprintln".to_string(),
            FnSig {
                name: "eprintln".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::generic("T".to_string())],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register built-in read_line function (reads a line from stdin)
        self.functions.insert(
            "read_line".to_string(),
            FnSig {
                name: "read_line".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register Box type (smart pointer for heap allocation)
        self.types.insert(
            "Box".to_string(),
            TypeDef {
                name: "Box".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque built-in
                is_pub: true,
                module: None,
            },
        );

        // Register Box::new function - allocates value on heap
        self.functions.insert(
            "Box::new".to_string(),
            FnSig {
                name: "Box::new".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::generic("T".to_string())],
                ret: Ty::named("Box".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Register drop function - deallocates value
        self.functions.insert(
            "drop".to_string(),
            FnSig {
                name: "drop".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::generic("T".to_string())],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Vec<T> Functions ============

        // Vec::new() - create empty vector
        self.functions.insert(
            "Vec::new".to_string(),
            FnSig {
                name: "Vec::new".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::with_capacity(cap) - create vector with initial capacity
        self.functions.insert(
            "Vec::with_capacity".to_string(),
            FnSig {
                name: "Vec::with_capacity".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::push(vec, elem) - add element to vector
        self.functions.insert(
            "Vec::push".to_string(),
            FnSig {
                name: "Vec::push".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::get(vec, index) - get element at index
        self.functions.insert(
            "Vec::get".to_string(),
            FnSig {
                name: "Vec::get".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::i64(),
                ],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::len(vec) - get length
        self.functions.insert(
            "Vec::len".to_string(),
            FnSig {
                name: "Vec::len".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::pop(vec) - remove and return last element
        self.functions.insert(
            "Vec::pop".to_string(),
            FnSig {
                name: "Vec::pop".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::set(vec, index, value) - set element at index
        self.functions.insert(
            "Vec::set".to_string(),
            FnSig {
                name: "Vec::set".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::i64(),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::is_empty(vec) - check if empty
        self.functions.insert(
            "Vec::is_empty".to_string(),
            FnSig {
                name: "Vec::is_empty".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::capacity(vec) - get capacity
        self.functions.insert(
            "Vec::capacity".to_string(),
            FnSig {
                name: "Vec::capacity".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::clear(vec) - remove all elements
        self.functions.insert(
            "Vec::clear".to_string(),
            FnSig {
                name: "Vec::clear".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::first(vec) - get first element
        self.functions.insert(
            "Vec::first".to_string(),
            FnSig {
                name: "Vec::first".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::last(vec) - get last element
        self.functions.insert(
            "Vec::last".to_string(),
            FnSig {
                name: "Vec::last".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Vec::iter(vec) - returns iterator over elements
        self.functions.insert(
            "Vec::iter".to_string(),
            FnSig {
                name: "Vec::iter".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ VecIter Methods ============

        // VecIter::next(iter) - returns Option<T> and advances
        self.functions.insert(
            "VecIter::next".to_string(),
            FnSig {
                name: "VecIter::next".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::map(iter, f) - transforms elements lazily
        self.functions.insert(
            "VecIter::map".to_string(),
            FnSig {
                name: "VecIter::map".to_string(),
                generics: vec!["T".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::generic("U".to_string())),
                ],
                ret: Ty::named("MapIter".to_string(), vec![Ty::generic("T".to_string()), Ty::generic("U".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::filter(iter, predicate) - filters elements lazily
        self.functions.insert(
            "VecIter::filter".to_string(),
            FnSig {
                name: "VecIter::filter".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::bool()),
                ],
                ret: Ty::named("FilterIter".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::fold(iter, init, f) - folds iterator into single value
        self.functions.insert(
            "VecIter::fold".to_string(),
            FnSig {
                name: "VecIter::fold".to_string(),
                generics: vec!["T".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("U".to_string()),
                    Ty::function(vec![Ty::generic("U".to_string()), Ty::generic("T".to_string())], Ty::generic("U".to_string())),
                ],
                ret: Ty::generic("U".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::collect(iter) - collects into Vec<T>
        self.functions.insert(
            "VecIter::collect".to_string(),
            FnSig {
                name: "VecIter::collect".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::find(iter, predicate) - finds first matching element
        self.functions.insert(
            "VecIter::find".to_string(),
            FnSig {
                name: "VecIter::find".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::bool()),
                ],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::any(iter, predicate) - checks if any element matches
        self.functions.insert(
            "VecIter::any".to_string(),
            FnSig {
                name: "VecIter::any".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::bool()),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::all(iter, predicate) - checks if all elements match
        self.functions.insert(
            "VecIter::all".to_string(),
            FnSig {
                name: "VecIter::all".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::bool()),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::count(iter) - counts elements
        self.functions.insert(
            "VecIter::count".to_string(),
            FnSig {
                name: "VecIter::count".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::sum(iter) - sums elements (for numeric types)
        self.functions.insert(
            "VecIter::sum".to_string(),
            FnSig {
                name: "VecIter::sum".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::enumerate(iter) - pairs elements with indices
        self.functions.insert(
            "VecIter::enumerate".to_string(),
            FnSig {
                name: "VecIter::enumerate".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::named("EnumerateIter".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::take(iter, n) - takes first n elements into a new Vec
        self.functions.insert(
            "VecIter::take".to_string(),
            FnSig {
                name: "VecIter::take".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::i64(),
                ],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::skip(iter, n) - skips first n elements into a new Vec
        self.functions.insert(
            "VecIter::skip".to_string(),
            FnSig {
                name: "VecIter::skip".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::i64(),
                ],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // VecIter::for_each(iter, f) - calls f on each element
        self.functions.insert(
            "VecIter::for_each".to_string(),
            FnSig {
                name: "VecIter::for_each".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("VecIter".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::function(vec![Ty::generic("T".to_string())], Ty::unit()),
                ],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ MapIter Methods ============

        // MapIter::next(iter) - returns Option<U>
        self.functions.insert(
            "MapIter::next".to_string(),
            FnSig {
                name: "MapIter::next".to_string(),
                generics: vec!["T".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("MapIter".to_string(), vec![Ty::generic("T".to_string()), Ty::generic("U".to_string())])],
                ret: Ty::option(Ty::generic("U".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // MapIter::collect(iter) - collects into Vec<U>
        self.functions.insert(
            "MapIter::collect".to_string(),
            FnSig {
                name: "MapIter::collect".to_string(),
                generics: vec!["T".to_string(), "U".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("MapIter".to_string(), vec![Ty::generic("T".to_string()), Ty::generic("U".to_string())])],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("U".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ FilterIter Methods ============

        // FilterIter::next(iter) - returns Option<T>
        self.functions.insert(
            "FilterIter::next".to_string(),
            FnSig {
                name: "FilterIter::next".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("FilterIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // FilterIter::collect(iter) - collects into Vec<T>
        self.functions.insert(
            "FilterIter::collect".to_string(),
            FnSig {
                name: "FilterIter::collect".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("FilterIter".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::named("Vec".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ String Functions ============

        // String::new() - create empty string
        self.functions.insert(
            "String::new".to_string(),
            FnSig {
                name: "String::new".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::from(literal) - create string from literal
        self.functions.insert(
            "String::from".to_string(),
            FnSig {
                name: "String::from".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::len(s) - get length
        self.functions.insert(
            "String::len".to_string(),
            FnSig {
                name: "String::len".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::push(s, char) - append character
        self.functions.insert(
            "String::push".to_string(),
            FnSig {
                name: "String::push".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![]), Ty::i64()],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::is_empty(s) - check if empty
        self.functions.insert(
            "String::is_empty".to_string(),
            FnSig {
                name: "String::is_empty".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::clear(s) - clear string (set len to 0)
        self.functions.insert(
            "String::clear".to_string(),
            FnSig {
                name: "String::clear".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::char_at(s, index) - get character at index
        self.functions.insert(
            "String::char_at".to_string(),
            FnSig {
                name: "String::char_at".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![]), Ty::i64()],
                ret: Ty::i64(), // Returns char as i64
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::push_str(s, str) - append string literal
        self.functions.insert(
            "String::push_str".to_string(),
            FnSig {
                name: "String::push_str".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![]), Ty::str()],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::concat(s1, s2) - concatenate two strings, returns new string
        self.functions.insert(
            "String::concat".to_string(),
            FnSig {
                name: "String::concat".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::named("String".to_string(), vec![]),
                ],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::substring(s, start, end) - get substring
        self.functions.insert(
            "String::substring".to_string(),
            FnSig {
                name: "String::substring".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::i64(),
                    Ty::i64(),
                ],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::capacity(s) - get capacity
        self.functions.insert(
            "String::capacity".to_string(),
            FnSig {
                name: "String::capacity".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::contains(s, pattern) - check if string contains substring
        self.functions.insert(
            "String::contains".to_string(),
            FnSig {
                name: "String::contains".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::starts_with(s, prefix) - check if string starts with prefix
        self.functions.insert(
            "String::starts_with".to_string(),
            FnSig {
                name: "String::starts_with".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::ends_with(s, suffix) - check if string ends with suffix
        self.functions.insert(
            "String::ends_with".to_string(),
            FnSig {
                name: "String::ends_with".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::find(s, pattern) - find first occurrence, returns Option<i64>
        self.functions.insert(
            "String::find".to_string(),
            FnSig {
                name: "String::find".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::option(Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::rfind(s, pattern) - find last occurrence, returns Option<i64>
        self.functions.insert(
            "String::rfind".to_string(),
            FnSig {
                name: "String::rfind".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::option(Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::to_uppercase(s) - convert to uppercase
        self.functions.insert(
            "String::to_uppercase".to_string(),
            FnSig {
                name: "String::to_uppercase".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::to_lowercase(s) - convert to lowercase
        self.functions.insert(
            "String::to_lowercase".to_string(),
            FnSig {
                name: "String::to_lowercase".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::trim(s) - remove leading and trailing whitespace
        self.functions.insert(
            "String::trim".to_string(),
            FnSig {
                name: "String::trim".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::trim_start(s) - remove leading whitespace
        self.functions.insert(
            "String::trim_start".to_string(),
            FnSig {
                name: "String::trim_start".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::trim_end(s) - remove trailing whitespace
        self.functions.insert(
            "String::trim_end".to_string(),
            FnSig {
                name: "String::trim_end".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::split(s, delimiter) - split string into Vec<String>
        self.functions.insert(
            "String::split".to_string(),
            FnSig {
                name: "String::split".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::named("Vec".to_string(), vec![Ty::named("String".to_string(), vec![])]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::lines(s) - split string by newlines into Vec<String>
        self.functions.insert(
            "String::lines".to_string(),
            FnSig {
                name: "String::lines".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("Vec".to_string(), vec![Ty::named("String".to_string(), vec![])]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::replace(s, from, to) - replace all occurrences
        self.functions.insert(
            "String::replace".to_string(),
            FnSig {
                name: "String::replace".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::str(),
                    Ty::str(),
                ],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::repeat(s, count) - repeat string n times
        self.functions.insert(
            "String::repeat".to_string(),
            FnSig {
                name: "String::repeat".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("String".to_string(), vec![]),
                    Ty::i64(),
                ],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::chars(s) - get Vec<i64> of character codes
        self.functions.insert(
            "String::chars".to_string(),
            FnSig {
                name: "String::chars".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("Vec".to_string(), vec![Ty::i64()]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::bytes(s) - get Vec<i64> of byte values
        self.functions.insert(
            "String::bytes".to_string(),
            FnSig {
                name: "String::bytes".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("Vec".to_string(), vec![Ty::i64()]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::split_whitespace(s) - split string by whitespace into Vec<String>
        self.functions.insert(
            "String::split_whitespace".to_string(),
            FnSig {
                name: "String::split_whitespace".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::named("Vec".to_string(), vec![Ty::named("String".to_string(), vec![])]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // String::concat_with(vec, separator) - join Vec<String> with separator
        self.functions.insert(
            "String::concat_with".to_string(),
            FnSig {
                name: "String::concat_with".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Vec".to_string(), vec![Ty::named("String".to_string(), vec![])]),
                    Ty::str(),
                ],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Integer Methods ============

        // i64::to_string(n) - convert integer to String
        self.functions.insert(
            "i64::to_string".to_string(),
            FnSig {
                name: "i64::to_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i32::to_string(n) - convert integer to String
        self.functions.insert(
            "i32::to_string".to_string(),
            FnSig {
                name: "i32::to_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i32()],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // bool::to_string(b) - convert bool to String
        self.functions.insert(
            "bool::to_string".to_string(),
            FnSig {
                name: "bool::to_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::bool()],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Parse Methods ============

        // i64::parse(s) - convert String to i64, returns Result<i64, i64>
        // Error codes: 1 = empty string, 2 = invalid characters
        self.functions.insert(
            "i64::parse".to_string(),
            FnSig {
                name: "i64::parse".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::result(Ty::i64(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i32::parse(s) - convert String to i32, returns Result<i32, i64>
        self.functions.insert(
            "i32::parse".to_string(),
            FnSig {
                name: "i32::parse".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::result(Ty::i32(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::parse(s) - convert String to f64, returns Result<f64, i64>
        // Error codes: 1 = empty string, 2 = invalid format
        self.functions.insert(
            "f64::parse".to_string(),
            FnSig {
                name: "f64::parse".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::result(Ty::f64(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::parse(s) - convert String to f32, returns Result<f32, i64>
        // Error codes: 1 = empty string, 2 = invalid format
        self.functions.insert(
            "f32::parse".to_string(),
            FnSig {
                name: "f32::parse".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::result(Ty::f32(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // bool::parse(s) - convert String to bool, returns Result<bool, i64>
        // Accepts "true"/"false" (case insensitive)
        // Error codes: 1 = empty string, 2 = invalid value
        self.functions.insert(
            "bool::parse".to_string(),
            FnSig {
                name: "bool::parse".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("String".to_string(), vec![])],
                ret: Ty::result(Ty::bool(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Math Functions (f64) ============

        // f64::abs(x) - absolute value
        self.functions.insert(
            "f64::abs".to_string(),
            FnSig {
                name: "f64::abs".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::sqrt(x) - square root
        self.functions.insert(
            "f64::sqrt".to_string(),
            FnSig {
                name: "f64::sqrt".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::cbrt(x) - cube root
        self.functions.insert(
            "f64::cbrt".to_string(),
            FnSig {
                name: "f64::cbrt".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::pow(base, exp) - power
        self.functions.insert(
            "f64::pow".to_string(),
            FnSig {
                name: "f64::pow".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::sin(x) - sine
        self.functions.insert(
            "f64::sin".to_string(),
            FnSig {
                name: "f64::sin".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::cos(x) - cosine
        self.functions.insert(
            "f64::cos".to_string(),
            FnSig {
                name: "f64::cos".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::tan(x) - tangent
        self.functions.insert(
            "f64::tan".to_string(),
            FnSig {
                name: "f64::tan".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::asin(x) - arc sine
        self.functions.insert(
            "f64::asin".to_string(),
            FnSig {
                name: "f64::asin".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::acos(x) - arc cosine
        self.functions.insert(
            "f64::acos".to_string(),
            FnSig {
                name: "f64::acos".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::atan(x) - arc tangent
        self.functions.insert(
            "f64::atan".to_string(),
            FnSig {
                name: "f64::atan".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::atan2(y, x) - arc tangent of y/x
        self.functions.insert(
            "f64::atan2".to_string(),
            FnSig {
                name: "f64::atan2".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::sinh(x) - hyperbolic sine
        self.functions.insert(
            "f64::sinh".to_string(),
            FnSig {
                name: "f64::sinh".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::cosh(x) - hyperbolic cosine
        self.functions.insert(
            "f64::cosh".to_string(),
            FnSig {
                name: "f64::cosh".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::tanh(x) - hyperbolic tangent
        self.functions.insert(
            "f64::tanh".to_string(),
            FnSig {
                name: "f64::tanh".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::exp(x) - e^x
        self.functions.insert(
            "f64::exp".to_string(),
            FnSig {
                name: "f64::exp".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::exp2(x) - 2^x
        self.functions.insert(
            "f64::exp2".to_string(),
            FnSig {
                name: "f64::exp2".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::ln(x) - natural logarithm
        self.functions.insert(
            "f64::ln".to_string(),
            FnSig {
                name: "f64::ln".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::log2(x) - base-2 logarithm
        self.functions.insert(
            "f64::log2".to_string(),
            FnSig {
                name: "f64::log2".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::log10(x) - base-10 logarithm
        self.functions.insert(
            "f64::log10".to_string(),
            FnSig {
                name: "f64::log10".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::log(x, base) - logarithm with arbitrary base
        self.functions.insert(
            "f64::log".to_string(),
            FnSig {
                name: "f64::log".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::floor(x) - round down
        self.functions.insert(
            "f64::floor".to_string(),
            FnSig {
                name: "f64::floor".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::ceil(x) - round up
        self.functions.insert(
            "f64::ceil".to_string(),
            FnSig {
                name: "f64::ceil".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::round(x) - round to nearest
        self.functions.insert(
            "f64::round".to_string(),
            FnSig {
                name: "f64::round".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::trunc(x) - truncate toward zero
        self.functions.insert(
            "f64::trunc".to_string(),
            FnSig {
                name: "f64::trunc".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::min(x, y) - minimum
        self.functions.insert(
            "f64::min".to_string(),
            FnSig {
                name: "f64::min".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::max(x, y) - maximum
        self.functions.insert(
            "f64::max".to_string(),
            FnSig {
                name: "f64::max".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::hypot(x, y) - hypotenuse sqrt(x^2 + y^2)
        self.functions.insert(
            "f64::hypot".to_string(),
            FnSig {
                name: "f64::hypot".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::fmod(x, y) - floating point remainder
        self.functions.insert(
            "f64::fmod".to_string(),
            FnSig {
                name: "f64::fmod".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::copysign(x, y) - copy sign of y to x
        self.functions.insert(
            "f64::copysign".to_string(),
            FnSig {
                name: "f64::copysign".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64(), Ty::f64()],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::is_nan(x) - check if NaN
        self.functions.insert(
            "f64::is_nan".to_string(),
            FnSig {
                name: "f64::is_nan".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::is_infinite(x) - check if infinite
        self.functions.insert(
            "f64::is_infinite".to_string(),
            FnSig {
                name: "f64::is_infinite".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::is_finite(x) - check if finite (not NaN or infinite)
        self.functions.insert(
            "f64::is_finite".to_string(),
            FnSig {
                name: "f64::is_finite".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f64()],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Math Functions (f32) ============

        // f32::abs(x) - absolute value
        self.functions.insert(
            "f32::abs".to_string(),
            FnSig {
                name: "f32::abs".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::sqrt(x) - square root
        self.functions.insert(
            "f32::sqrt".to_string(),
            FnSig {
                name: "f32::sqrt".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::sin(x) - sine
        self.functions.insert(
            "f32::sin".to_string(),
            FnSig {
                name: "f32::sin".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::cos(x) - cosine
        self.functions.insert(
            "f32::cos".to_string(),
            FnSig {
                name: "f32::cos".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::tan(x) - tangent
        self.functions.insert(
            "f32::tan".to_string(),
            FnSig {
                name: "f32::tan".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::pow(base, exp) - power
        self.functions.insert(
            "f32::pow".to_string(),
            FnSig {
                name: "f32::pow".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32(), Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::exp(x) - e^x
        self.functions.insert(
            "f32::exp".to_string(),
            FnSig {
                name: "f32::exp".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::ln(x) - natural logarithm
        self.functions.insert(
            "f32::ln".to_string(),
            FnSig {
                name: "f32::ln".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::floor(x) - round down
        self.functions.insert(
            "f32::floor".to_string(),
            FnSig {
                name: "f32::floor".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::ceil(x) - round up
        self.functions.insert(
            "f32::ceil".to_string(),
            FnSig {
                name: "f32::ceil".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::round(x) - round to nearest
        self.functions.insert(
            "f32::round".to_string(),
            FnSig {
                name: "f32::round".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::trunc(x) - truncate toward zero
        self.functions.insert(
            "f32::trunc".to_string(),
            FnSig {
                name: "f32::trunc".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::min(x, y) - minimum
        self.functions.insert(
            "f32::min".to_string(),
            FnSig {
                name: "f32::min".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32(), Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f32::max(x, y) - maximum
        self.functions.insert(
            "f32::max".to_string(),
            FnSig {
                name: "f32::max".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::f32(), Ty::f32()],
                ret: Ty::f32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Math Functions (i64) ============

        // i64::abs(x) - absolute value
        self.functions.insert(
            "i64::abs".to_string(),
            FnSig {
                name: "i64::abs".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i64::min(x, y) - minimum
        self.functions.insert(
            "i64::min".to_string(),
            FnSig {
                name: "i64::min".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64(), Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i64::max(x, y) - maximum
        self.functions.insert(
            "i64::max".to_string(),
            FnSig {
                name: "i64::max".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64(), Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i64::pow(base, exp) - integer power (exp must be non-negative)
        self.functions.insert(
            "i64::pow".to_string(),
            FnSig {
                name: "i64::pow".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64(), Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Math Functions (i32) ============

        // i32::abs(x) - absolute value
        self.functions.insert(
            "i32::abs".to_string(),
            FnSig {
                name: "i32::abs".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i32()],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i32::min(x, y) - minimum
        self.functions.insert(
            "i32::min".to_string(),
            FnSig {
                name: "i32::min".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i32(), Ty::i32()],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // i32::max(x, y) - maximum
        self.functions.insert(
            "i32::max".to_string(),
            FnSig {
                name: "i32::max".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i32(), Ty::i32()],
                ret: Ty::i32(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Math Constants ============

        // f64::PI - pi constant
        self.functions.insert(
            "f64::PI".to_string(),
            FnSig {
                name: "f64::PI".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::E - euler's number
        self.functions.insert(
            "f64::E".to_string(),
            FnSig {
                name: "f64::E".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::INFINITY - positive infinity
        self.functions.insert(
            "f64::INFINITY".to_string(),
            FnSig {
                name: "f64::INFINITY".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::NEG_INFINITY - negative infinity
        self.functions.insert(
            "f64::NEG_INFINITY".to_string(),
            FnSig {
                name: "f64::NEG_INFINITY".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // f64::NAN - not a number
        self.functions.insert(
            "f64::NAN".to_string(),
            FnSig {
                name: "f64::NAN".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ File I/O ============

        // File::open(path) - open file for reading, returns Result<File, i64>
        // Error codes: 1 = file not found, 2 = permission denied, 3 = other error
        self.functions.insert(
            "File::open".to_string(),
            FnSig {
                name: "File::open".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::named("File".to_string(), vec![]), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::create(path) - create/truncate file for writing, returns Result<File, i64>
        self.functions.insert(
            "File::create".to_string(),
            FnSig {
                name: "File::create".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::named("File".to_string(), vec![]), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::read_to_string(file) - read entire file to String, returns Result<String, i64>
        self.functions.insert(
            "File::read_to_string".to_string(),
            FnSig {
                name: "File::read_to_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("File".to_string(), vec![])],
                ret: Ty::result(Ty::named("String".to_string(), vec![]), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::write_string(file, data) - write string to file, returns Result<i64, i64> (bytes written)
        self.functions.insert(
            "File::write_string".to_string(),
            FnSig {
                name: "File::write_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("File".to_string(), vec![]),
                    Ty::str(),
                ],
                ret: Ty::result(Ty::i64(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::close(file) - close file
        self.functions.insert(
            "File::close".to_string(),
            FnSig {
                name: "File::close".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("File".to_string(), vec![])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Extended File System Operations ============

        // File::exists(path) -> bool - check if file or directory exists
        self.functions.insert(
            "File::exists".to_string(),
            FnSig {
                name: "File::exists".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::size(path) -> Result<i64, i64> - get file size in bytes
        self.functions.insert(
            "File::size".to_string(),
            FnSig {
                name: "File::size".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::i64(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::is_file(path) -> bool - check if path is a regular file
        self.functions.insert(
            "File::is_file".to_string(),
            FnSig {
                name: "File::is_file".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::is_dir(path) -> bool - check if path is a directory
        self.functions.insert(
            "File::is_dir".to_string(),
            FnSig {
                name: "File::is_dir".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // File::remove(path) -> Result<(), i64> - remove a file
        self.functions.insert(
            "File::remove".to_string(),
            FnSig {
                name: "File::remove".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::unit(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Directory Operations ============

        // Dir::create(path) -> Result<(), i64> - create a directory
        self.functions.insert(
            "Dir::create".to_string(),
            FnSig {
                name: "Dir::create".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::unit(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Dir::create_all(path) -> Result<(), i64> - create directory and all parent directories
        self.functions.insert(
            "Dir::create_all".to_string(),
            FnSig {
                name: "Dir::create_all".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::unit(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Dir::remove(path) -> Result<(), i64> - remove an empty directory
        self.functions.insert(
            "Dir::remove".to_string(),
            FnSig {
                name: "Dir::remove".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(Ty::unit(), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Dir::list(path) -> Result<Vec<String>, i64> - list directory contents
        self.functions.insert(
            "Dir::list".to_string(),
            FnSig {
                name: "Dir::list".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::result(
                    Ty::named("Vec".to_string(), vec![Ty::named("String".to_string(), vec![])]),
                    Ty::i64(),
                ),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Path Operations ============
        // Using Fs:: prefix to avoid conflict with macro $x:path capture kind

        // Fs::path_join(base, path) -> String - join two paths with separator
        self.functions.insert(
            "Fs::path_join".to_string(),
            FnSig {
                name: "Fs::path_join".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str(), Ty::str()],
                ret: Ty::named("String".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Fs::path_parent(path) -> Option<String> - get parent directory
        self.functions.insert(
            "Fs::path_parent".to_string(),
            FnSig {
                name: "Fs::path_parent".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::option(Ty::named("String".to_string(), vec![])),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Fs::path_filename(path) -> Option<String> - get filename component
        self.functions.insert(
            "Fs::path_filename".to_string(),
            FnSig {
                name: "Fs::path_filename".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::option(Ty::named("String".to_string(), vec![])),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Fs::path_extension(path) -> Option<String> - get file extension
        self.functions.insert(
            "Fs::path_extension".to_string(),
            FnSig {
                name: "Fs::path_extension".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str()],
                ret: Ty::option(Ty::named("String".to_string(), vec![])),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ HashMap<K, V> Methods ============
        // HashMap uses open addressing with linear probing
        // Internal structure: {entries: *Entry, count: i64, capacity: i64}
        // Entry structure: {key: K, value: V, occupied: i64} where 0=empty, 1=occupied, 2=deleted

        // HashMap::new() - create empty hash map
        self.functions.insert(
            "HashMap::new".to_string(),
            FnSig {
                name: "HashMap::new".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::with_capacity(cap) - create hash map with initial capacity
        self.functions.insert(
            "HashMap::with_capacity".to_string(),
            FnSig {
                name: "HashMap::with_capacity".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::insert(map, key, value) - insert key-value pair, returns old value if key existed
        self.functions.insert(
            "HashMap::insert".to_string(),
            FnSig {
                name: "HashMap::insert".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())]),
                    Ty::generic("K".to_string()),
                    Ty::generic("V".to_string()),
                ],
                ret: Ty::option(Ty::generic("V".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::get(map, key) - get value for key
        self.functions.insert(
            "HashMap::get".to_string(),
            FnSig {
                name: "HashMap::get".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())]),
                    Ty::generic("K".to_string()),
                ],
                ret: Ty::option(Ty::generic("V".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::contains_key(map, key) - check if key exists
        self.functions.insert(
            "HashMap::contains_key".to_string(),
            FnSig {
                name: "HashMap::contains_key".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())]),
                    Ty::generic("K".to_string()),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::remove(map, key) - remove key and return value if existed
        self.functions.insert(
            "HashMap::remove".to_string(),
            FnSig {
                name: "HashMap::remove".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())]),
                    Ty::generic("K".to_string()),
                ],
                ret: Ty::option(Ty::generic("V".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::len(map) - get number of entries
        self.functions.insert(
            "HashMap::len".to_string(),
            FnSig {
                name: "HashMap::len".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::is_empty(map) - check if map is empty
        self.functions.insert(
            "HashMap::is_empty".to_string(),
            FnSig {
                name: "HashMap::is_empty".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())])],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::clear(map) - remove all entries
        self.functions.insert(
            "HashMap::clear".to_string(),
            FnSig {
                name: "HashMap::clear".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashMap::capacity(map) - get current capacity
        self.functions.insert(
            "HashMap::capacity".to_string(),
            FnSig {
                name: "HashMap::capacity".to_string(),
                generics: vec!["K".to_string(), "V".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashMap".to_string(), vec![Ty::generic("K".to_string()), Ty::generic("V".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Channel Types and Functions (Async Phase 6) ============

        // Channel<T> - internal channel structure
        self.types.insert(
            "Channel".to_string(),
            TypeDef {
                name: "Channel".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque built-in
                is_pub: true,
                module: None,
            },
        );

        // Sender<T> - sending half of a channel
        self.types.insert(
            "Sender".to_string(),
            TypeDef {
                name: "Sender".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque built-in
                is_pub: true,
                module: None,
            },
        );

        // Receiver<T> - receiving half of a channel
        self.types.insert(
            "Receiver".to_string(),
            TypeDef {
                name: "Receiver".to_string(),
                generics: vec!["T".to_string()],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque built-in
                is_pub: true,
                module: None,
            },
        );

        // channel<T>(capacity: i64) -> Channel<T> - create a new channel
        // Returns the internal channel, use get_sender/get_receiver to get handles
        self.functions.insert(
            "channel".to_string(),
            FnSig {
                name: "channel".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("Channel".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Channel::sender(channel) -> Sender<T> - get sender handle from channel
        self.functions.insert(
            "Channel::sender".to_string(),
            FnSig {
                name: "Channel::sender".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Channel".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::named("Sender".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Channel::receiver(channel) -> Receiver<T> - get receiver handle from channel
        self.functions.insert(
            "Channel::receiver".to_string(),
            FnSig {
                name: "Channel::receiver".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Channel".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::named("Receiver".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Sender::send(sender, value) -> Future<()> - send value, suspends if buffer full
        self.functions.insert(
            "Sender::send".to_string(),
            FnSig {
                name: "Sender::send".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Sender".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::future(Ty::unit()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Receiver::recv(receiver) -> Future<Option<T>> - receive value, suspends if buffer empty
        self.functions.insert(
            "Receiver::recv".to_string(),
            FnSig {
                name: "Receiver::recv".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Receiver".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::future(Ty::option(Ty::generic("T".to_string()))),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Sender::try_send(sender, value) -> i64 - non-blocking send
        // Returns: 1 = success, 0 = buffer full, -1 = closed
        self.functions.insert(
            "Sender::try_send".to_string(),
            FnSig {
                name: "Sender::try_send".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Sender".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Receiver::try_recv(receiver) -> Option<T> - non-blocking receive
        self.functions.insert(
            "Receiver::try_recv".to_string(),
            FnSig {
                name: "Receiver::try_recv".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Receiver".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::option(Ty::generic("T".to_string())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Sender::is_closed(sender) -> bool - check if receiver has been dropped
        self.functions.insert(
            "Sender::is_closed".to_string(),
            FnSig {
                name: "Sender::is_closed".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Sender".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Receiver::is_closed(receiver) -> bool - check if all senders have been dropped
        self.functions.insert(
            "Receiver::is_closed".to_string(),
            FnSig {
                name: "Receiver::is_closed".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Receiver".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ HashSet Functions ============

        // HashSet::new() - create empty hash set
        self.functions.insert(
            "HashSet::new".to_string(),
            FnSig {
                name: "HashSet::new".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::with_capacity(cap) - create hash set with initial capacity
        self.functions.insert(
            "HashSet::with_capacity".to_string(),
            FnSig {
                name: "HashSet::with_capacity".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::insert(set, value) - insert value, returns true if new
        self.functions.insert(
            "HashSet::insert".to_string(),
            FnSig {
                name: "HashSet::insert".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::contains(set, value) - check if value exists
        self.functions.insert(
            "HashSet::contains".to_string(),
            FnSig {
                name: "HashSet::contains".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::remove(set, value) - remove value, returns true if existed
        self.functions.insert(
            "HashSet::remove".to_string(),
            FnSig {
                name: "HashSet::remove".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())]),
                    Ty::generic("T".to_string()),
                ],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::len(set) - get number of elements
        self.functions.insert(
            "HashSet::len".to_string(),
            FnSig {
                name: "HashSet::len".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::is_empty(set) - check if set is empty
        self.functions.insert(
            "HashSet::is_empty".to_string(),
            FnSig {
                name: "HashSet::is_empty".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::clear(set) - remove all elements
        self.functions.insert(
            "HashSet::clear".to_string(),
            FnSig {
                name: "HashSet::clear".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // HashSet::capacity(set) - get current capacity
        self.functions.insert(
            "HashSet::capacity".to_string(),
            FnSig {
                name: "HashSet::capacity".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("HashSet".to_string(), vec![Ty::generic("T".to_string())])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Future<T> Methods ============

        // Future::poll(future) -> bool - poll future, returns true if ready
        self.functions.insert(
            "Future::poll".to_string(),
            FnSig {
                name: "Future::poll".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::future(Ty::generic("T".to_string()))],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Future::is_ready(future) -> bool - check if future is ready
        self.functions.insert(
            "Future::is_ready".to_string(),
            FnSig {
                name: "Future::is_ready".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::future(Ty::generic("T".to_string()))],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Future::get(future) -> T - get value (undefined if not ready)
        self.functions.insert(
            "Future::get".to_string(),
            FnSig {
                name: "Future::get".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::future(Ty::generic("T".to_string()))],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Async Runtime Functions ============
        // Note: spawn is a keyword, not a function - handled via SpawnTask ExprKind

        // block_on(future) -> T - run future to completion
        self.functions.insert(
            "block_on".to_string(),
            FnSig {
                name: "block_on".to_string(),
                generics: vec!["T".to_string()],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::future(Ty::generic("T".to_string()))],
                ret: Ty::generic("T".to_string()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // yield_now() -> Future<()> - yield control to scheduler
        self.functions.insert(
            "yield_now".to_string(),
            FnSig {
                name: "yield_now".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::future(Ty::unit()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // sleep_ms(ms: i64) -> Future<()> - sleep for milliseconds
        self.functions.insert(
            "sleep_ms".to_string(),
            FnSig {
                name: "sleep_ms".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::future(Ty::unit()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // pending() -> Future<()> - return a pending future (for testing suspension)
        self.functions.insert(
            "pending".to_string(),
            FnSig {
                name: "pending".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::future(Ty::unit()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // wake(future_ptr: i64) -> () - wake a pending future
        self.functions.insert(
            "wake".to_string(),
            FnSig {
                name: "wake".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Async TCP I/O (Phase 4.1) ============

        // TcpListener - listening TCP socket
        self.types.insert(
            "TcpListener".to_string(),
            TypeDef {
                name: "TcpListener".to_string(),
                generics: vec![],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque - holds socket fd
                is_pub: true,
                module: None,
            },
        );

        // TcpStream - connected TCP socket
        self.types.insert(
            "TcpStream".to_string(),
            TypeDef {
                name: "TcpStream".to_string(),
                generics: vec![],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct { fields: vec![] }, // Opaque - holds socket fd
                is_pub: true,
                module: None,
            },
        );

        // TcpListener::bind(addr: String, port: i64) -> Result<TcpListener, i64>
        // Creates a listening socket bound to addr:port
        // Error codes: 1 = socket failed, 2 = bind failed, 3 = listen failed
        self.functions.insert(
            "TcpListener::bind".to_string(),
            FnSig {
                name: "TcpListener::bind".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::str(), Ty::i64()],
                ret: Ty::result(Ty::named("TcpListener".to_string(), vec![]), Ty::i64()),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // TcpListener::accept(listener) -> Future<Result<TcpStream, i64>>
        // Async accept - waits for incoming connection
        self.functions.insert(
            "TcpListener::accept".to_string(),
            FnSig {
                name: "TcpListener::accept".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("TcpListener".to_string(), vec![])],
                ret: Ty::future(Ty::result(Ty::named("TcpStream".to_string(), vec![]), Ty::i64())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // TcpListener::close(listener) -> ()
        self.functions.insert(
            "TcpListener::close".to_string(),
            FnSig {
                name: "TcpListener::close".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("TcpListener".to_string(), vec![])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // TcpStream::read_string(stream, max_len: i64) -> Future<Result<String, i64>>
        // Async read - reads up to max_len bytes into a String
        self.functions.insert(
            "TcpStream::read_string".to_string(),
            FnSig {
                name: "TcpStream::read_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("TcpStream".to_string(), vec![]), Ty::i64()],
                ret: Ty::future(Ty::result(Ty::str(), Ty::i64())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // TcpStream::write_string(stream, data: String) -> Future<Result<i64, i64>>
        // Async write - writes String to socket, returns bytes written
        self.functions.insert(
            "TcpStream::write_string".to_string(),
            FnSig {
                name: "TcpStream::write_string".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("TcpStream".to_string(), vec![]), Ty::str()],
                ret: Ty::future(Ty::result(Ty::i64(), Ty::i64())),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // TcpStream::close(stream) -> ()
        self.functions.insert(
            "TcpStream::close".to_string(),
            FnSig {
                name: "TcpStream::close".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("TcpStream".to_string(), vec![])],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Time Module ============

        // time::now_ms() -> i64 - current time in milliseconds since epoch
        self.functions.insert(
            "time::now_ms".to_string(),
            FnSig {
                name: "time::now_ms".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // time::now_ns() -> i64 - current time in nanoseconds since epoch
        self.functions.insert(
            "time::now_ns".to_string(),
            FnSig {
                name: "time::now_ns".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // time::now_us() -> i64 - current time in microseconds since epoch
        self.functions.insert(
            "time::now_us".to_string(),
            FnSig {
                name: "time::now_us".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // time::elapsed_ms(start: i64) -> i64 - milliseconds elapsed since start
        self.functions.insert(
            "time::elapsed_ms".to_string(),
            FnSig {
                name: "time::elapsed_ms".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // time::elapsed_us(start: i64) -> i64 - microseconds elapsed since start
        self.functions.insert(
            "time::elapsed_us".to_string(),
            FnSig {
                name: "time::elapsed_us".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // time::elapsed_ns(start: i64) -> i64 - nanoseconds elapsed since start
        self.functions.insert(
            "time::elapsed_ns".to_string(),
            FnSig {
                name: "time::elapsed_ns".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Duration Type ============

        // Duration struct - represents a span of time
        self.types.insert(
            "Duration".to_string(),
            TypeDef {
                name: "Duration".to_string(),
                generics: vec![],
                generic_defaults: HashMap::new(),
                kind: TypeDefKind::Struct {
                    fields: vec![
                        ("secs".to_string(), Ty::i64()),
                        ("nanos".to_string(), Ty::i64()),
                    ],
                },
                is_pub: true,
                module: None,
            },
        );

        // Duration::from_secs(secs: i64) -> Duration
        self.functions.insert(
            "Duration::from_secs".to_string(),
            FnSig {
                name: "Duration::from_secs".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("Duration".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::from_millis(millis: i64) -> Duration
        self.functions.insert(
            "Duration::from_millis".to_string(),
            FnSig {
                name: "Duration::from_millis".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("Duration".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::from_micros(micros: i64) -> Duration
        self.functions.insert(
            "Duration::from_micros".to_string(),
            FnSig {
                name: "Duration::from_micros".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("Duration".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::from_nanos(nanos: i64) -> Duration
        self.functions.insert(
            "Duration::from_nanos".to_string(),
            FnSig {
                name: "Duration::from_nanos".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::named("Duration".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::as_secs(d: Duration) -> i64
        self.functions.insert(
            "Duration::as_secs".to_string(),
            FnSig {
                name: "Duration::as_secs".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Duration".to_string(), vec![])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::as_millis(d: Duration) -> i64
        self.functions.insert(
            "Duration::as_millis".to_string(),
            FnSig {
                name: "Duration::as_millis".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Duration".to_string(), vec![])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::as_micros(d: Duration) -> i64
        self.functions.insert(
            "Duration::as_micros".to_string(),
            FnSig {
                name: "Duration::as_micros".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Duration".to_string(), vec![])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::as_nanos(d: Duration) -> i64
        self.functions.insert(
            "Duration::as_nanos".to_string(),
            FnSig {
                name: "Duration::as_nanos".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::named("Duration".to_string(), vec![])],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::add(a: Duration, b: Duration) -> Duration
        self.functions.insert(
            "Duration::add".to_string(),
            FnSig {
                name: "Duration::add".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Duration".to_string(), vec![]),
                    Ty::named("Duration".to_string(), vec![]),
                ],
                ret: Ty::named("Duration".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // Duration::sub(a: Duration, b: Duration) -> Duration
        self.functions.insert(
            "Duration::sub".to_string(),
            FnSig {
                name: "Duration::sub".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![
                    Ty::named("Duration".to_string(), vec![]),
                    Ty::named("Duration".to_string(), vec![]),
                ],
                ret: Ty::named("Duration".to_string(), vec![]),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // ============ Random Module ============

        // random::seed(seed: i64) - initialize the random number generator
        self.functions.insert(
            "random::seed".to_string(),
            FnSig {
                name: "random::seed".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64()],
                ret: Ty::unit(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // random::next_i64() -> i64 - get next random i64
        self.functions.insert(
            "random::next_i64".to_string(),
            FnSig {
                name: "random::next_i64".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // random::next_f64() -> f64 - get next random f64 in [0.0, 1.0)
        self.functions.insert(
            "random::next_f64".to_string(),
            FnSig {
                name: "random::next_f64".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::f64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // random::range(min: i64, max: i64) -> i64 - get random i64 in [min, max)
        self.functions.insert(
            "random::range".to_string(),
            FnSig {
                name: "random::range".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![Ty::i64(), Ty::i64()],
                ret: Ty::i64(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );

        // random::coin() -> bool - get random boolean (coin flip)
        self.functions.insert(
            "random::coin".to_string(),
            FnSig {
                name: "random::coin".to_string(),
                generics: vec![],
                generic_bounds: HashMap::new(),
                generic_defaults: HashMap::new(),
                params: vec![],
                ret: Ty::bool(),
                is_method: false,
                is_async: false,
                is_pub: true,
                module: None,
            },
        );
    }

    // ============ Scope Management ============

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Leave the current scope
    /// Note: Will not pop the global scope (keeps at least one scope)
    pub fn leave_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Get the current scope
    /// This is safe because TypeContext always maintains at least one scope
    fn current_scope(&self) -> &Scope {
        self.scopes.last().expect("BUG: TypeContext scope stack is empty - this should never happen")
    }

    /// Get the current scope mutably
    /// This is safe because TypeContext always maintains at least one scope
    fn current_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("BUG: TypeContext scope stack is empty - this should never happen")
    }

    // ============ Variable Bindings ============

    /// Define a variable in the current scope
    pub fn define_var(&mut self, name: &str, ty: Ty, mutable: bool) {
        let symbol = Symbol {
            name: name.to_string(),
            ty,
            kind: SymbolKind::Variable { mutable },
        };
        self.current_scope_mut().symbols.insert(name.to_string(), symbol);
    }

    /// Look up a variable by name
    pub fn lookup_var(&self, name: &str) -> Option<&Symbol> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.symbols.get(name) {
                return Some(symbol);
            }
        }
        None
    }

    /// Check if a variable exists in current scope
    pub fn var_exists_in_current_scope(&self, name: &str) -> bool {
        self.current_scope().symbols.contains_key(name)
    }

    // ============ Type Definitions ============

    /// Register a struct type
    pub fn register_struct(&mut self, name: &str, generics: Vec<String>, generic_defaults: HashMap<String, Ty>, fields: Vec<(String, Ty)>, is_pub: bool, module: Option<String>) {
        self.types.insert(
            name.to_string(),
            TypeDef {
                name: name.to_string(),
                generics,
                generic_defaults,
                kind: TypeDefKind::Struct { fields },
                is_pub,
                module,
            },
        );
    }

    /// Register an enum type
    pub fn register_enum(&mut self, name: &str, generics: Vec<String>, generic_defaults: HashMap<String, Ty>, variants: Vec<(String, Vec<Ty>)>, is_pub: bool, module: Option<String>) {
        self.types.insert(
            name.to_string(),
            TypeDef {
                name: name.to_string(),
                generics,
                generic_defaults,
                kind: TypeDefKind::Enum { variants },
                is_pub,
                module,
            },
        );
    }

    /// Register a type alias
    pub fn register_type_alias(&mut self, name: &str, generics: Vec<String>, generic_defaults: HashMap<String, Ty>, ty: Ty, is_pub: bool, module: Option<String>) {
        self.types.insert(
            name.to_string(),
            TypeDef {
                name: name.to_string(),
                generics,
                generic_defaults,
                kind: TypeDefKind::Alias { target: ty },
                is_pub,
                module,
            },
        );
    }

    /// Look up a type definition
    pub fn lookup_type(&self, name: &str) -> Option<&TypeDef> {
        // First check direct types
        if let Some(ty) = self.types.get(name) {
            return Some(ty);
        }
        // Then check imports
        if let Some(import) = self.imports.get(name) {
            if matches!(import.kind, ImportKind::Type) {
                let original_name = import.original_path.last().unwrap();
                return self.types.get(original_name);
            }
        }
        None
    }

    /// Expand type aliases recursively
    /// If the type is a named type that refers to an alias, expand it to its target type.
    /// Handles generic type aliases by substituting type parameters.
    /// Detects cycles to prevent infinite recursion.
    pub fn expand_type_alias(&self, ty: &Ty) -> Ty {
        self.expand_type_alias_impl(ty, &mut std::collections::HashSet::new(), 0)
    }

    fn expand_type_alias_impl(&self, ty: &Ty, visited: &mut std::collections::HashSet<String>, depth: usize) -> Ty {
        // Prevent runaway recursion
        if depth > 100 {
            return ty.clone();
        }
        match &ty.kind {
            TyKind::Named { name, generics } => {
                // Check for cycles
                if visited.contains(name) {
                    // Cycle detected, return as-is to avoid infinite loop
                    return ty.clone();
                }

                // Look up the type definition
                if let Some(def) = self.lookup_type(name) {
                    if let TypeDefKind::Alias { target } = &def.kind {
                        // Mark as visited to detect cycles
                        visited.insert(name.clone());

                        // Substitute generic parameters if any
                        let expanded = if !def.generics.is_empty() && !generics.is_empty() {
                            // Build substitution map: generic_name -> concrete_type
                            let subst: std::collections::HashMap<String, Ty> = def
                                .generics
                                .iter()
                                .zip(generics.iter())
                                .map(|(n, t)| (n.clone(), self.expand_type_alias_impl(t, visited, depth + 1)))
                                .collect();
                            self.substitute_type_params(target, &subst)
                        } else {
                            target.clone()
                        };

                        // Recursively expand in case the target is also an alias
                        let result = self.expand_type_alias_impl(&expanded, visited, depth + 1);
                        visited.remove(name);
                        return result;
                    }
                }

                // Not an alias, but recursively expand generic arguments
                let expanded_generics: Vec<Ty> = generics
                    .iter()
                    .map(|g| self.expand_type_alias_impl(g, visited, depth + 1))
                    .collect();
                Ty::named(name.clone(), expanded_generics)
            }
            // Recursively expand in compound types
            TyKind::Ref { inner, mutable } => {
                Ty::reference(self.expand_type_alias_impl(inner, visited, depth + 1), *mutable)
            }
            TyKind::Array { element, size } => {
                Ty::array(self.expand_type_alias_impl(element, visited, depth + 1), *size)
            }
            TyKind::Slice { element } => {
                Ty::slice(self.expand_type_alias_impl(element, visited, depth + 1))
            }
            TyKind::Tuple(elements) => {
                Ty::tuple(elements.iter().map(|e| self.expand_type_alias_impl(e, visited, depth + 1)).collect())
            }
            TyKind::Fn { params, ret } => {
                Ty::function(
                    params.iter().map(|p| self.expand_type_alias_impl(p, visited, depth + 1)).collect(),
                    self.expand_type_alias_impl(ret, visited, depth + 1),
                )
            }
            TyKind::Projection { base_ty, trait_name, assoc_name } => {
                Ty {
                    kind: TyKind::Projection {
                        base_ty: Box::new(self.expand_type_alias_impl(base_ty, visited, depth + 1)),
                        trait_name: trait_name.clone(),
                        assoc_name: assoc_name.clone(),
                    },
                }
            }
            // Other types don't need expansion
            _ => ty.clone(),
        }
    }

    /// Substitute type parameters in a type
    fn substitute_type_params(&self, ty: &Ty, subst: &std::collections::HashMap<String, Ty>) -> Ty {
        match &ty.kind {
            TyKind::Generic { name } => {
                subst.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            TyKind::Named { name, generics } => {
                let new_generics: Vec<Ty> = generics
                    .iter()
                    .map(|g| self.substitute_type_params(g, subst))
                    .collect();
                Ty::named(name.clone(), new_generics)
            }
            TyKind::Ref { inner, mutable } => {
                Ty::reference(self.substitute_type_params(inner, subst), *mutable)
            }
            TyKind::Array { element, size } => {
                Ty::array(self.substitute_type_params(element, subst), *size)
            }
            TyKind::Slice { element } => {
                Ty::slice(self.substitute_type_params(element, subst))
            }
            TyKind::Tuple(elements) => {
                Ty::tuple(elements.iter().map(|e| self.substitute_type_params(e, subst)).collect())
            }
            TyKind::Fn { params, ret } => {
                Ty::function(
                    params.iter().map(|p| self.substitute_type_params(p, subst)).collect(),
                    self.substitute_type_params(ret, subst),
                )
            }
            TyKind::Projection { base_ty, trait_name, assoc_name } => {
                Ty {
                    kind: TyKind::Projection {
                        base_ty: Box::new(self.substitute_type_params(base_ty, subst)),
                        trait_name: trait_name.clone(),
                        assoc_name: assoc_name.clone(),
                    },
                }
            }
            _ => ty.clone(),
        }
    }

    /// Check if a type exists
    pub fn type_exists(&self, name: &str) -> bool {
        self.types.contains_key(name) || self.is_primitive(name) || self.imports.contains_key(name)
    }

    /// Check if a name is a primitive type
    pub fn is_primitive(&self, name: &str) -> bool {
        matches!(
            name,
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
            "u8" | "u16" | "u32" | "u64" | "u128" | "usize" |
            "f32" | "f64" | "bool" | "char" | "str"
        )
    }

    // ============ Function Signatures ============

    /// Register a function signature
    pub fn register_function(&mut self, name: &str, mut sig: FnSig) {
        // Track the module if not already set
        if sig.module.is_none() {
            sig.module = self.current_module.clone();
        }
        self.functions.insert(name.to_string(), sig);
    }

    /// Look up a function signature
    pub fn lookup_function(&self, name: &str) -> Option<&FnSig> {
        // First check direct functions
        if let Some(sig) = self.functions.get(name) {
            return Some(sig);
        }
        // Then check imports
        if let Some(import) = self.imports.get(name) {
            if matches!(import.kind, ImportKind::Function) {
                let original_name = import.original_path.last().unwrap();
                return self.functions.get(original_name);
            }
        }
        None
    }

    // ============ Modules ============

    /// Register a module
    pub fn register_module(&mut self, name: &str, module: ModuleDef) {
        self.modules.insert(name.to_string(), module);
    }

    /// Look up a module
    pub fn lookup_module(&self, name: &str) -> Option<&ModuleDef> {
        self.modules.get(name)
    }

    /// Register an import from a use statement
    pub fn register_import(&mut self, alias: &str, path: Vec<String>, kind: ImportKind) {
        self.imports.insert(
            alias.to_string(),
            ImportedSymbol {
                original_path: path,
                kind,
            },
        );
    }

    /// Look up an import
    pub fn lookup_import(&self, name: &str) -> Option<&ImportedSymbol> {
        self.imports.get(name)
    }

    /// Resolve a path that may go through modules
    /// Returns the resolved type name if found
    pub fn resolve_path(&self, segments: &[String]) -> Option<String> {
        if segments.len() == 1 {
            // Simple name - check imports first
            let name = &segments[0];
            if let Some(import) = self.imports.get(name) {
                return Some(import.original_path.last().unwrap().clone());
            }
            // Otherwise return as-is
            return Some(name.clone());
        }

        // Multi-segment path like Module::Type
        // For now, just concatenate with ::
        let full_path = segments.join("::");

        // Check if first segment is a module
        if let Some(_module) = self.modules.get(&segments[0]) {
            // Return the last segment (the actual type name)
            return Some(segments.last().unwrap().clone());
        }

        Some(full_path)
    }

    // ============ Macros ============

    /// Register a macro definition
    pub fn register_macro(&mut self, def: &crate::ast::MacroDef) {
        let info = MacroDefInfo {
            name: def.name.name.clone(),
            is_pub: def.is_pub,
            module: self.current_module.clone(),
            def: def.clone(),
        };
        self.macros.insert(def.name.name.clone(), info);
    }

    /// Look up a macro
    pub fn lookup_macro(&self, name: &str) -> Option<&MacroDefInfo> {
        self.macros.get(name)
    }

    // ============ Traits ============

    /// Register a trait
    pub fn register_trait(&mut self, name: &str, def: TraitDef) {
        self.traits.insert(name.to_string(), def);
    }

    /// Look up a trait
    pub fn lookup_trait(&self, name: &str) -> Option<&TraitDef> {
        self.traits.get(name)
    }

    // ============ Actors ============

    /// Register an actor
    pub fn register_actor(&mut self, name: &str, def: ActorDef) {
        self.actors.insert(name.to_string(), def);
    }

    /// Look up an actor
    pub fn lookup_actor(&self, name: &str) -> Option<&ActorDef> {
        self.actors.get(name)
    }

    /// Look up a message handler for an actor
    pub fn lookup_actor_message(&self, actor_name: &str, message_name: &str) -> Option<&MessageDef> {
        self.actors.get(actor_name).and_then(|actor| {
            actor.messages.iter().find(|m| m.name == message_name)
        })
    }

    // ============ Impl Blocks ============

    /// Register an impl block
    pub fn register_impl(&mut self, impl_def: ImplDef) {
        self.impls.push(impl_def);
    }

    /// Find methods for a type
    pub fn find_methods(&self, ty: &Ty) -> Vec<&FnSig> {
        let mut methods = Vec::new();
        let type_name = match &ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            _ => return methods,
        };

        for impl_def in &self.impls {
            if impl_def.self_type == type_name {
                for method in &impl_def.methods {
                    methods.push(method);
                }
            }
        }

        methods
    }

    /// Find a specific method for a type (including trait methods)
    pub fn find_method(&self, ty: &Ty, method_name: &str) -> Option<&FnSig> {
        let type_name = match &ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            _ => return None,
        };

        // First, look in inherent impls (impl Type)
        for impl_def in &self.impls {
            if impl_def.self_type == type_name && impl_def.trait_name.is_none() {
                if let Some(method) = impl_def.methods.iter().find(|m| m.name == method_name) {
                    return Some(method);
                }
            }
        }

        // Then, look in trait impls (impl Trait for Type)
        for impl_def in &self.impls {
            if impl_def.self_type == type_name && impl_def.trait_name.is_some() {
                if let Some(method) = impl_def.methods.iter().find(|m| m.name == method_name) {
                    return Some(method);
                }
            }
        }

        None
    }

    /// Check if a type implements a trait
    pub fn implements_trait(&self, ty: &Ty, trait_name: &str) -> bool {
        let type_name = match &ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            _ => return false,
        };

        self.impls.iter().any(|impl_def| {
            impl_def.self_type == type_name &&
            impl_def.trait_name.as_ref().map(|t| t == trait_name).unwrap_or(false)
        })
    }

    /// Get all traits implemented by a type
    pub fn get_implemented_traits(&self, ty: &Ty) -> Vec<String> {
        let type_name = match &ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            _ => return Vec::new(),
        };

        self.impls.iter()
            .filter(|impl_def| impl_def.self_type == type_name)
            .filter_map(|impl_def| impl_def.trait_name.clone())
            .collect()
    }

    /// Find a method from a specific trait for a type
    pub fn find_trait_method(&self, ty: &Ty, trait_name: &str, method_name: &str) -> Option<&FnSig> {
        let type_name = match &ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            _ => return None,
        };

        for impl_def in &self.impls {
            if impl_def.self_type == type_name {
                if let Some(ref tn) = impl_def.trait_name {
                    if tn == trait_name {
                        if let Some(method) = impl_def.methods.iter().find(|m| m.name == method_name) {
                            return Some(method);
                        }
                    }
                }
            }
        }

        None
    }

    /// Resolve an associated type for a concrete type
    /// e.g., for `<Vec<i32> as Iterator>::Item`, returns `i32`
    pub fn resolve_associated_type(&self, ty: &Ty, trait_name: &str, assoc_type_name: &str) -> Option<Ty> {
        let type_name = match &ty.kind {
            TyKind::Named { name, .. } => name.clone(),
            _ => return None,
        };

        // Find the impl block for this type and trait
        for impl_def in &self.impls {
            if impl_def.self_type == type_name {
                if let Some(ref tn) = impl_def.trait_name {
                    if tn == trait_name {
                        // Look up the associated type implementation
                        if let Some(assoc_ty) = impl_def.associated_types.get(assoc_type_name) {
                            return Some(assoc_ty.clone());
                        }
                    }
                }
            }
        }

        // Check if the trait has a default type for this associated type
        if let Some(trait_def) = self.traits.get(trait_name) {
            if let Some(assoc_def) = trait_def.associated_types.get(assoc_type_name) {
                if let Some(ref default) = assoc_def.default {
                    return Some(default.clone());
                }
            }
        }

        None
    }

    /// Get all associated type names from a trait
    pub fn get_trait_associated_types(&self, trait_name: &str) -> Vec<String> {
        self.traits
            .get(trait_name)
            .map(|t| t.associated_types.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Resolve an associated type by type name (searches all impls)
    pub fn resolve_associated_type_by_name(&self, type_name: &str, assoc_type_name: &str) -> Option<Ty> {
        // Find any impl block for this type that has this associated type
        for impl_def in &self.impls {
            if impl_def.self_type == type_name {
                if let Some(assoc_ty) = impl_def.associated_types.get(assoc_type_name) {
                    return Some(assoc_ty.clone());
                }
            }
        }
        None
    }

    /// Check if an impl provides all required associated types
    pub fn impl_has_all_associated_types(&self, impl_def: &ImplDef, trait_name: &str) -> Vec<String> {
        let required = self.get_trait_associated_types(trait_name);
        let provided: std::collections::HashSet<_> = impl_def.associated_types.keys().cloned().collect();

        required.into_iter()
            .filter(|name| !provided.contains(name))
            .collect()
    }

    // ============ Generics ============

    /// Enter a generic scope
    pub fn enter_generic_scope(&mut self, params: &Option<Generics>) {
        let mut generic_map = HashMap::new();
        if let Some(generics) = params {
            for param in &generics.params {
                generic_map.insert(
                    param.name.name.clone(),
                    Ty::generic(param.name.name.clone()),
                );
            }
        }
        self.generics.push(generic_map);
    }

    /// Leave the generic scope
    pub fn leave_generic_scope(&mut self) {
        self.generics.pop();
    }

    /// Look up a generic parameter
    pub fn lookup_generic(&self, name: &str) -> Option<&Ty> {
        for scope in self.generics.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Check if a name is a generic parameter
    pub fn is_generic(&self, name: &str) -> bool {
        self.lookup_generic(name).is_some()
    }

    // ============ Struct Fields ============

    /// Get fields of a struct type
    pub fn get_struct_fields(&self, name: &str) -> Option<&[(String, Ty)]> {
        self.types.get(name).and_then(|def| {
            if let TypeDefKind::Struct { fields } = &def.kind {
                Some(fields.as_slice())
            } else {
                None
            }
        })
    }

    /// Get variants of an enum type
    pub fn get_enum_variants(&self, name: &str) -> Option<&[(String, Vec<Ty>)]> {
        self.types.get(name).and_then(|def| {
            if let TypeDefKind::Enum { variants } = &def.kind {
                Some(variants.as_slice())
            } else {
                None
            }
        })
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A scope contains symbol bindings
#[derive(Debug, Clone)]
pub struct Scope {
    pub symbols: HashMap<String, Symbol>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

/// A symbol in the symbol table
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub ty: Ty,
    pub kind: SymbolKind,
}

/// The kind of symbol
#[derive(Debug, Clone)]
pub enum SymbolKind {
    /// A variable binding
    Variable { mutable: bool },
    /// A function
    Function,
    /// A constant
    Constant,
    /// A type
    Type,
}

/// A type definition
#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    pub generics: Vec<String>,
    /// Default types for generic parameters: param_name -> default_type
    pub generic_defaults: HashMap<String, Ty>,
    pub kind: TypeDefKind,
    /// Whether the type is public
    pub is_pub: bool,
    /// The module this type belongs to (None for top-level)
    pub module: Option<String>,
}

/// The kind of type definition
#[derive(Debug, Clone)]
pub enum TypeDefKind {
    /// Struct: fields are (name, type)
    Struct { fields: Vec<(String, Ty)> },
    /// Enum: variants are (name, field types)
    Enum { variants: Vec<(String, Vec<Ty>)> },
    /// Type alias
    Alias { target: Ty },
}

/// A function signature
#[derive(Debug, Clone)]
pub struct FnSig {
    pub name: String,
    pub generics: Vec<String>,
    /// Trait bounds for each generic parameter: generic_name -> [bound_names]
    pub generic_bounds: HashMap<String, Vec<String>>,
    /// Default types for generic parameters: param_name -> default_type
    pub generic_defaults: HashMap<String, Ty>,
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub is_method: bool,
    pub is_async: bool,
    /// Whether the function is public
    pub is_pub: bool,
    /// The module this function belongs to (None for top-level)
    pub module: Option<String>,
}

/// A trait definition
#[derive(Debug, Clone)]
pub struct TraitDef {
    pub name: String,
    pub generics: Vec<String>,
    pub super_traits: Vec<String>,
    pub methods: Vec<FnSig>,
    /// Associated types: name -> (bounds, default_type)
    pub associated_types: HashMap<String, AssociatedTypeDef>,
}

/// Associated type definition in a trait
#[derive(Debug, Clone)]
pub struct AssociatedTypeDef {
    /// Bounds on the associated type (e.g., Clone + Debug)
    pub bounds: Vec<String>,
    /// Default type if any
    pub default: Option<Ty>,
}

/// An actor definition
#[derive(Debug, Clone)]
pub struct ActorDef {
    pub name: String,
    pub generics: Vec<String>,
    pub state: Vec<(String, Ty)>,
    pub messages: Vec<MessageDef>,
}

/// A message definition for actor
#[derive(Debug, Clone)]
pub struct MessageDef {
    pub name: String,
    pub fields: Vec<Ty>,
    /// Return type of the message (for <-? responses)
    pub response_ty: Option<Ty>,
}

/// An impl block
#[derive(Debug, Clone)]
pub struct ImplDef {
    pub self_type: String,
    pub trait_name: Option<String>,
    pub methods: Vec<FnSig>,
    /// Associated type implementations: name -> concrete_type
    pub associated_types: HashMap<String, Ty>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_management() {
        let mut ctx = TypeContext::new();

        ctx.define_var("x", Ty::i32(), false);
        assert!(ctx.lookup_var("x").is_some());

        ctx.enter_scope();
        ctx.define_var("y", Ty::bool(), false);
        assert!(ctx.lookup_var("x").is_some());
        assert!(ctx.lookup_var("y").is_some());

        ctx.leave_scope();
        assert!(ctx.lookup_var("x").is_some());
        assert!(ctx.lookup_var("y").is_none());
    }

    #[test]
    fn test_type_registration() {
        let mut ctx = TypeContext::new();

        ctx.register_struct(
            "Point",
            vec![],
            HashMap::new(),  // generic_defaults
            vec![("x".to_string(), Ty::f64()), ("y".to_string(), Ty::f64())],
            true,  // is_pub
            None,  // module
        );

        assert!(ctx.type_exists("Point"));
        let fields = ctx.get_struct_fields("Point").unwrap();
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn test_builtin_types() {
        let ctx = TypeContext::new();

        assert!(ctx.type_exists("Option"));
        assert!(ctx.type_exists("Result"));
        assert!(ctx.type_exists("Vec"));
        assert!(ctx.lookup_function("print").is_some());
    }

    #[test]
    fn test_option_methods() {
        let ctx = TypeContext::new();

        // Verify Option methods are registered
        assert!(ctx.lookup_function("Option::is_some").is_some(), "Option::is_some not found");
        assert!(ctx.lookup_function("Option::is_none").is_some(), "Option::is_none not found");
        assert!(ctx.lookup_function("Option::unwrap").is_some(), "Option::unwrap not found");
        assert!(ctx.lookup_function("Option::unwrap_or").is_some(), "Option::unwrap_or not found");
        assert!(ctx.lookup_function("Option::map").is_some(), "Option::map not found");
        assert!(ctx.lookup_function("Option::and_then").is_some(), "Option::and_then not found");
    }

    #[test]
    fn test_result_methods() {
        let ctx = TypeContext::new();

        // Verify Result methods are registered
        assert!(ctx.lookup_function("Result::is_ok").is_some(), "Result::is_ok not found");
        assert!(ctx.lookup_function("Result::is_err").is_some(), "Result::is_err not found");
        assert!(ctx.lookup_function("Result::unwrap").is_some(), "Result::unwrap not found");
        assert!(ctx.lookup_function("Result::unwrap_or").is_some(), "Result::unwrap_or not found");
        assert!(ctx.lookup_function("Result::map").is_some(), "Result::map not found");
        assert!(ctx.lookup_function("Result::and_then").is_some(), "Result::and_then not found");
    }
}
