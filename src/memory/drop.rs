//! Automatic Drop Insertion
//!
//! This module handles automatic insertion of drop/release calls at:
//! - End of variable scope
//! - Variable reassignment
//! - Function return
//! - Match arm exits
//!
//! # Drop Order
//!
//! Variables are dropped in reverse declaration order (LIFO),
//! matching Rust's behavior for predictable cleanup.

use std::collections::HashMap;
use crate::span::Span;

/// Represents a variable that needs to be dropped
#[derive(Debug, Clone)]
pub struct DropEntry {
    /// Variable name
    pub name: String,
    /// Type of the variable (for destructor dispatch)
    pub type_name: String,
    /// Where the variable was declared
    pub decl_span: Span,
    /// Scope depth when declared
    pub scope_depth: usize,
    /// Whether this variable is consumed (moved)
    pub is_moved: bool,
    /// Whether this is an RC type (vs inline container)
    pub is_rc: bool,
}

/// Tracks variables for automatic drop insertion
#[derive(Debug)]
pub struct DropTracker {
    /// Variables by scope depth
    scopes: Vec<Vec<DropEntry>>,
    /// Current scope depth
    current_depth: usize,
    /// Map from variable name to its current state
    var_states: HashMap<String, DropEntry>,
    /// Whether to track drops (disabled for primitives)
    enabled: bool,
}

impl DropTracker {
    pub fn new() -> Self {
        Self {
            scopes: vec![Vec::new()],
            current_depth: 0,
            var_states: HashMap::new(),
            enabled: true,
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.current_depth += 1;
        self.scopes.push(Vec::new());
    }

    /// Leave current scope, returning variables that need to be dropped
    pub fn leave_scope(&mut self) -> Vec<DropEntry> {
        if self.scopes.len() <= 1 {
            return Vec::new();
        }

        self.current_depth -= 1;
        let scope_vars = self.scopes.pop().unwrap_or_default();

        // Remove from var_states and return non-moved entries in reverse order
        let mut drops = Vec::new();
        for entry in scope_vars.into_iter().rev() {
            self.var_states.remove(&entry.name);
            if !entry.is_moved {
                drops.push(entry);
            }
        }

        drops
    }

    /// Register a new variable to track
    pub fn register(&mut self, name: String, type_name: String, decl_span: Span, is_rc: bool) {
        if !self.enabled {
            return;
        }

        let entry = DropEntry {
            name: name.clone(),
            type_name,
            decl_span,
            scope_depth: self.current_depth,
            is_moved: false,
            is_rc,
        };

        self.var_states.insert(name.clone(), entry.clone());

        if let Some(scope) = self.scopes.last_mut() {
            scope.push(entry);
        }
    }

    /// Mark a variable as moved (won't be dropped)
    pub fn mark_moved(&mut self, name: &str) {
        if let Some(entry) = self.var_states.get_mut(name) {
            entry.is_moved = true;
        }

        // Also update in scope list
        for scope in &mut self.scopes {
            for entry in scope.iter_mut() {
                if entry.name == name {
                    entry.is_moved = true;
                }
            }
        }
    }

    /// Mark a variable as valid again (after reassignment)
    pub fn mark_valid(&mut self, name: &str) {
        if let Some(entry) = self.var_states.get_mut(name) {
            entry.is_moved = false;
        }

        for scope in &mut self.scopes {
            for entry in scope.iter_mut() {
                if entry.name == name {
                    entry.is_moved = false;
                }
            }
        }
    }

    /// Check if a variable needs drop
    pub fn needs_drop(&self, name: &str) -> bool {
        self.var_states.get(name).map(|e| !e.is_moved && e.is_rc).unwrap_or(false)
    }

    /// Get the type of a tracked variable
    pub fn get_type(&self, name: &str) -> Option<&str> {
        self.var_states.get(name).map(|e| e.type_name.as_str())
    }

    /// Check if variable is moved
    pub fn is_moved(&self, name: &str) -> bool {
        self.var_states.get(name).map(|e| e.is_moved).unwrap_or(false)
    }

    /// Get all variables at current scope that need drop
    pub fn current_scope_drops(&self) -> Vec<&DropEntry> {
        self.scopes.last()
            .map(|s| s.iter().filter(|e| !e.is_moved && e.is_rc).collect())
            .unwrap_or_default()
    }

    /// Get all variables that need drop (for function return)
    pub fn all_drops(&self) -> Vec<&DropEntry> {
        self.scopes.iter()
            .flat_map(|s| s.iter())
            .filter(|e| !e.is_moved && e.is_rc)
            .collect()
    }

    /// Disable tracking (for scopes with only primitives)
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable tracking
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Current scope depth
    pub fn depth(&self) -> usize {
        self.current_depth
    }
}

impl Default for DropTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates drop code for a specific type
#[derive(Debug, Clone)]
pub struct DropGenerator {
    /// Custom drop functions registered
    custom_drops: HashMap<String, String>,
}

impl DropGenerator {
    pub fn new() -> Self {
        let mut custom_drops = HashMap::new();

        // Register built-in drop functions
        custom_drops.insert("Vec".to_string(), "__drop_vec".to_string());
        custom_drops.insert("String".to_string(), "__drop_string".to_string());
        custom_drops.insert("HashMap".to_string(), "__drop_hashmap".to_string());
        custom_drops.insert("HashSet".to_string(), "__drop_hashset".to_string());
        custom_drops.insert("Box".to_string(), "__drop_box".to_string());
        custom_drops.insert("File".to_string(), "__drop_file".to_string());
        custom_drops.insert("Channel".to_string(), "__drop_channel".to_string());
        custom_drops.insert("TcpStream".to_string(), "__drop_tcpstream".to_string());
        custom_drops.insert("TcpListener".to_string(), "__drop_tcplistener".to_string());

        Self { custom_drops }
    }

    /// Get the drop function name for a type
    pub fn drop_fn_for(&self, type_name: &str) -> Option<&str> {
        // Extract base type name
        let base = type_name.split('<').next().unwrap_or(type_name);
        self.custom_drops.get(base).map(|s| s.as_str())
    }

    /// Register a custom drop function
    pub fn register_drop(&mut self, type_name: String, drop_fn: String) {
        self.custom_drops.insert(type_name, drop_fn);
    }

    /// Check if a type has a custom drop
    pub fn has_custom_drop(&self, type_name: &str) -> bool {
        let base = type_name.split('<').next().unwrap_or(type_name);
        self.custom_drops.contains_key(base)
    }
}

impl Default for DropGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_tracking() {
        let mut tracker = DropTracker::new();

        // Enter first scope (simulates function body)
        tracker.enter_scope();
        tracker.register("x".into(), "Vec<i64>".into(), Span::new(0, 0), true);

        // Enter inner scope
        tracker.enter_scope();
        tracker.register("y".into(), "String".into(), Span::new(1, 1), true);

        // Leave inner scope - y should be dropped
        let drops = tracker.leave_scope();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].name, "y");

        // Leave outer scope - x should be dropped
        let drops = tracker.leave_scope();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].name, "x");
    }

    #[test]
    fn test_moved_not_dropped() {
        let mut tracker = DropTracker::new();

        // Enter a scope (simulates function body)
        tracker.enter_scope();
        tracker.register("x".into(), "Vec<i64>".into(), Span::new(0, 0), true);
        tracker.mark_moved("x");

        let drops = tracker.leave_scope();
        assert!(drops.is_empty()); // x was moved, not dropped
    }

    #[test]
    fn test_drop_generator() {
        let gen = DropGenerator::new();

        assert_eq!(gen.drop_fn_for("Vec<i64>"), Some("__drop_vec"));
        assert_eq!(gen.drop_fn_for("String"), Some("__drop_string"));
        assert_eq!(gen.drop_fn_for("HashMap<String, i64>"), Some("__drop_hashmap"));
        assert_eq!(gen.drop_fn_for("i64"), None);
    }
}
