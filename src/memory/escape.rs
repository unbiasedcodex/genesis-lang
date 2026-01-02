//! Escape Analysis for RC Optimization
//!
//! Escape analysis determines whether an allocation escapes its defining scope.
//! This enables several optimizations:
//!
//! 1. **Stack Promotion**: If a Box<T> never escapes, allocate on stack
//! 2. **RC Elision**: If a value never has aliases, skip retain/release
//! 3. **Region Assignment**: Values that don't escape can use task region
//!
//! # Escape Categories
//!
//! - **NoEscape**: Value stays in defining scope (can stack allocate)
//! - **LocalEscape**: Escapes to outer scope (needs heap, may skip RC)
//! - **GlobalEscape**: Escapes to global scope or returned (full RC)
//! - **Unknown**: Can't determine (conservative: full RC)

use std::collections::{HashMap, HashSet};

/// Escape classification for a value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeKind {
    /// Never escapes defining scope
    NoEscape,
    /// Escapes to outer local scope
    LocalEscape,
    /// Escapes to return value or global
    GlobalEscape,
    /// Unknown (treat as GlobalEscape)
    Unknown,
}

impl EscapeKind {
    /// Does this need heap allocation?
    pub fn needs_heap(&self) -> bool {
        !matches!(self, EscapeKind::NoEscape)
    }

    /// Does this need reference counting?
    pub fn needs_rc(&self) -> bool {
        matches!(self, EscapeKind::GlobalEscape | EscapeKind::Unknown)
    }

    /// Can be optimized to skip retain/release?
    pub fn can_elide_rc(&self) -> bool {
        matches!(self, EscapeKind::NoEscape | EscapeKind::LocalEscape)
    }
}

/// Tracks escape information for variables
#[derive(Debug, Clone)]
pub struct EscapeInfo {
    /// Variable name
    pub name: String,
    /// Escape classification
    pub kind: EscapeKind,
    /// Variables that may alias this one
    pub aliases: HashSet<String>,
    /// Whether this is stored in a container
    pub stored: bool,
    /// Whether this is returned from a function
    pub returned: bool,
    /// Whether this is passed to unknown function
    pub passed_out: bool,
}

impl EscapeInfo {
    pub fn new(name: String) -> Self {
        Self {
            name,
            kind: EscapeKind::NoEscape,
            aliases: HashSet::new(),
            stored: false,
            returned: false,
            passed_out: false,
        }
    }

    /// Mark as escaping due to return
    pub fn mark_returned(&mut self) {
        self.returned = true;
        self.kind = EscapeKind::GlobalEscape;
    }

    /// Mark as escaping due to being stored
    pub fn mark_stored(&mut self) {
        self.stored = true;
        if self.kind == EscapeKind::NoEscape {
            self.kind = EscapeKind::LocalEscape;
        }
    }

    /// Mark as escaping due to being passed out
    pub fn mark_passed(&mut self) {
        self.passed_out = true;
        self.kind = EscapeKind::GlobalEscape;
    }

    /// Add an alias
    pub fn add_alias(&mut self, alias: String) {
        self.aliases.insert(alias);
    }

    /// Recompute escape kind based on flags
    pub fn recompute(&mut self) {
        if self.returned || self.passed_out {
            self.kind = EscapeKind::GlobalEscape;
        } else if self.stored || !self.aliases.is_empty() {
            self.kind = EscapeKind::LocalEscape;
        } else {
            self.kind = EscapeKind::NoEscape;
        }
    }
}

/// Performs escape analysis on a function
#[derive(Debug)]
pub struct EscapeAnalyzer {
    /// Escape info for each variable
    vars: HashMap<String, EscapeInfo>,
    /// Current scope depth
    scope_depth: usize,
    /// Variables declared at each scope
    scope_vars: Vec<Vec<String>>,
    /// Parameters (start as GlobalEscape)
    parameters: HashSet<String>,
}

impl EscapeAnalyzer {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            scope_depth: 0,
            scope_vars: vec![Vec::new()],
            parameters: HashSet::new(),
        }
    }

    /// Register a function parameter (starts as potentially escaped)
    pub fn register_param(&mut self, name: String) {
        let mut info = EscapeInfo::new(name.clone());
        info.kind = EscapeKind::GlobalEscape; // Parameters may come from anywhere
        self.vars.insert(name.clone(), info);
        self.parameters.insert(name);
    }

    /// Register a new local variable
    pub fn register_local(&mut self, name: String) {
        self.vars.insert(name.clone(), EscapeInfo::new(name.clone()));
        if let Some(scope) = self.scope_vars.last_mut() {
            scope.push(name);
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scope_depth += 1;
        self.scope_vars.push(Vec::new());
    }

    /// Leave current scope
    pub fn leave_scope(&mut self) {
        if self.scope_depth > 0 {
            self.scope_depth -= 1;
            self.scope_vars.pop();
        }
    }

    /// Mark a variable as returned
    pub fn mark_returned(&mut self, name: &str) {
        if let Some(info) = self.vars.get_mut(name) {
            info.mark_returned();
        }
    }

    /// Mark a variable as stored in container
    pub fn mark_stored(&mut self, name: &str) {
        if let Some(info) = self.vars.get_mut(name) {
            info.mark_stored();
        }
    }

    /// Mark a variable as passed to function
    pub fn mark_passed(&mut self, name: &str) {
        if let Some(info) = self.vars.get_mut(name) {
            info.mark_passed();
        }
    }

    /// Record an alias relationship
    pub fn add_alias(&mut self, source: &str, alias: &str) {
        if let Some(info) = self.vars.get_mut(source) {
            info.add_alias(alias.to_string());
        }
        if let Some(info) = self.vars.get_mut(alias) {
            info.add_alias(source.to_string());
        }
    }

    /// Get escape kind for a variable
    pub fn get_escape(&self, name: &str) -> EscapeKind {
        self.vars.get(name)
            .map(|i| i.kind)
            .unwrap_or(EscapeKind::Unknown)
    }

    /// Check if RC can be elided for a variable
    pub fn can_elide_rc(&self, name: &str) -> bool {
        self.get_escape(name).can_elide_rc()
    }

    /// Check if stack allocation is possible
    pub fn can_stack_alloc(&self, name: &str) -> bool {
        self.get_escape(name) == EscapeKind::NoEscape
    }

    /// Finalize analysis and return results
    pub fn finalize(self) -> HashMap<String, EscapeInfo> {
        self.vars
    }
}

impl Default for EscapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization decisions based on escape analysis
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Variables that can skip RC
    pub elide_rc: HashSet<String>,
    /// Variables that can use stack
    pub stack_alloc: HashSet<String>,
    /// Variables that need full RC
    pub full_rc: HashSet<String>,
}

impl OptimizationHints {
    pub fn from_analysis(analysis: &HashMap<String, EscapeInfo>) -> Self {
        let mut hints = Self {
            elide_rc: HashSet::new(),
            stack_alloc: HashSet::new(),
            full_rc: HashSet::new(),
        };

        for (name, info) in analysis {
            match info.kind {
                EscapeKind::NoEscape => {
                    hints.stack_alloc.insert(name.clone());
                    hints.elide_rc.insert(name.clone());
                }
                EscapeKind::LocalEscape => {
                    hints.elide_rc.insert(name.clone());
                }
                EscapeKind::GlobalEscape | EscapeKind::Unknown => {
                    hints.full_rc.insert(name.clone());
                }
            }
        }

        hints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_no_escape() {
        let mut analyzer = EscapeAnalyzer::new();

        analyzer.register_local("x".into());
        // x is never returned or stored

        assert_eq!(analyzer.get_escape("x"), EscapeKind::NoEscape);
        assert!(analyzer.can_stack_alloc("x"));
        assert!(analyzer.can_elide_rc("x"));
    }

    #[test]
    fn test_returned_escapes() {
        let mut analyzer = EscapeAnalyzer::new();

        analyzer.register_local("x".into());
        analyzer.mark_returned("x");

        assert_eq!(analyzer.get_escape("x"), EscapeKind::GlobalEscape);
        assert!(!analyzer.can_stack_alloc("x"));
        assert!(!analyzer.can_elide_rc("x"));
    }

    #[test]
    fn test_stored_local_escape() {
        let mut analyzer = EscapeAnalyzer::new();

        analyzer.register_local("x".into());
        analyzer.mark_stored("x");

        assert_eq!(analyzer.get_escape("x"), EscapeKind::LocalEscape);
        assert!(!analyzer.can_stack_alloc("x"));
        assert!(analyzer.can_elide_rc("x")); // Still can elide!
    }

    #[test]
    fn test_param_global_escape() {
        let mut analyzer = EscapeAnalyzer::new();

        analyzer.register_param("x".into());

        // Parameters start as GlobalEscape
        assert_eq!(analyzer.get_escape("x"), EscapeKind::GlobalEscape);
    }
}
