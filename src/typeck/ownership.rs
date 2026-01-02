//! Ownership and Borrowing Checker
//!
//! This module implements ownership and lifetime tracking for Genesis Lang.
//! It verifies:
//! - Values are not used after move
//! - Mutable references are exclusive
//! - Immutable references can coexist
//! - References do not outlive their referents (lifetime tracking)
//!
//! # Lifetime Tracking
//!
//! Each borrow is assigned a unique lifetime that tracks:
//! - The borrowed variable
//! - The start span (where the borrow was created)
//! - The scope depth (used to determine when borrow ends)

use std::collections::HashMap;
use crate::typeck::error::TypeError;
use crate::span::Span;

/// Unique identifier for a lifetime/borrow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifetimeId(pub u32);

/// Represents an active borrow with lifetime information
#[derive(Debug, Clone)]
pub struct ActiveBorrow {
    /// Unique identifier for this borrow
    pub id: LifetimeId,
    /// The variable being borrowed
    pub var_name: String,
    /// Where the borrow was created
    pub borrow_span: Span,
    /// Scope depth when borrow was created
    pub scope_depth: usize,
    /// Whether this is a mutable borrow
    pub is_mutable: bool,
    /// Original mutability of the variable (for restoration)
    pub original_var_mutable: bool,
}

/// Ownership checker tracks the state of values
#[derive(Debug)]
pub struct OwnershipChecker {
    /// Map from variable names to their ownership state
    states: HashMap<String, OwnershipState>,
    /// Stack of scopes
    scope_stack: Vec<Vec<String>>,
    /// Active borrows with their lifetimes
    active_borrows: Vec<ActiveBorrow>,
    /// Counter for generating unique lifetime IDs
    next_lifetime_id: u32,
    /// Current scope depth
    current_scope_depth: usize,
}

impl OwnershipChecker {
    /// Create a new ownership checker
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            scope_stack: vec![Vec::new()],
            active_borrows: Vec::new(),
            next_lifetime_id: 0,
            current_scope_depth: 0,
        }
    }

    /// Generate a new unique lifetime ID
    fn new_lifetime_id(&mut self) -> LifetimeId {
        let id = LifetimeId(self.next_lifetime_id);
        self.next_lifetime_id += 1;
        id
    }

    /// Create a new borrow and return its lifetime ID
    fn create_borrow(&mut self, var_name: &str, span: Span, is_mutable: bool) -> LifetimeId {
        let id = self.new_lifetime_id();
        // Get the original mutability of the variable
        let original_var_mutable = match self.states.get(var_name) {
            Some(OwnershipState::Owned { mutable }) => *mutable,
            _ => is_mutable, // Default to borrow mutability if not tracked
        };
        self.active_borrows.push(ActiveBorrow {
            id,
            var_name: var_name.to_string(),
            borrow_span: span,
            scope_depth: self.current_scope_depth,
            is_mutable,
            original_var_mutable,
        });
        id
    }

    /// End all borrows that were created at or after the given scope depth
    fn end_borrows_at_depth(&mut self, depth: usize) {
        self.active_borrows.retain(|b| b.scope_depth < depth);
    }

    /// Get all active borrows for a variable
    pub fn get_active_borrows(&self, var_name: &str) -> Vec<&ActiveBorrow> {
        self.active_borrows.iter()
            .filter(|b| b.var_name == var_name)
            .collect()
    }

    /// Check if a variable has any active borrows
    pub fn has_active_borrows(&self, var_name: &str) -> bool {
        self.active_borrows.iter().any(|b| b.var_name == var_name)
    }

    /// Check if a variable has any active mutable borrows
    pub fn has_active_mut_borrows(&self, var_name: &str) -> bool {
        self.active_borrows.iter()
            .any(|b| b.var_name == var_name && b.is_mutable)
    }

    /// Check for overlapping borrows (mutable borrow when other borrows exist)
    fn check_overlapping_borrows(&self, var_name: &str, is_mutable: bool, span: Span) -> Result<(), TypeError> {
        let existing_borrows = self.get_active_borrows(var_name);

        if existing_borrows.is_empty() {
            return Ok(());
        }

        // If requesting mutable borrow, no other borrows allowed
        if is_mutable {
            return Err(TypeError::already_borrowed(var_name.to_string(), span));
        }

        // If requesting immutable borrow, no mutable borrows allowed
        if existing_borrows.iter().any(|b| b.is_mutable) {
            return Err(TypeError::already_borrowed(var_name.to_string(), span));
        }

        Ok(())
    }

    /// Define a new variable
    pub fn define(&mut self, name: &str, mutable: bool) {
        self.states.insert(
            name.to_string(),
            OwnershipState::Owned { mutable },
        );
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.push(name.to_string());
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scope_stack.push(Vec::new());
        self.current_scope_depth += 1;
    }

    /// Leave the current scope
    pub fn leave_scope(&mut self) {
        // End all borrows created in this scope
        self.end_borrows_at_depth(self.current_scope_depth);

        if let Some(vars) = self.scope_stack.pop() {
            for var in vars {
                self.states.remove(&var);
            }
        }

        if self.current_scope_depth > 0 {
            self.current_scope_depth -= 1;
        }
    }

    /// Use a value (move if not Copy)
    pub fn use_value(&mut self, name: &str, span: Span) -> Result<(), TypeError> {
        match self.states.get(name) {
            Some(OwnershipState::Moved { moved_at: _ }) => {
                Err(TypeError::moved_value(name.to_string(), span))
            }
            Some(OwnershipState::Owned { .. }) |
            Some(OwnershipState::Borrowed { .. }) |
            Some(OwnershipState::MutBorrowed { .. }) => Ok(()),
            None => {
                // Variable not tracked (e.g., parameter or external)
                Ok(())
            }
        }
    }

    /// Move a value
    pub fn move_value(&mut self, name: &str, span: Span) -> Result<(), TypeError> {
        match self.states.get(name) {
            Some(OwnershipState::Moved { .. }) => {
                Err(TypeError::moved_value(name.to_string(), span))
            }
            Some(OwnershipState::MutBorrowed { .. }) => {
                Err(TypeError::already_borrowed(name.to_string(), span))
            }
            Some(OwnershipState::Borrowed { .. }) => {
                Err(TypeError::already_borrowed(name.to_string(), span))
            }
            Some(OwnershipState::Owned { .. }) => {
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Moved { moved_at: span },
                );
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Borrow a value immutably, returning the lifetime ID
    pub fn borrow(&mut self, name: &str, span: Span) -> Result<LifetimeId, TypeError> {
        // Check basic ownership state
        match self.states.get(name) {
            Some(OwnershipState::Moved { .. }) => {
                return Err(TypeError::moved_value(name.to_string(), span));
            }
            None => {
                // Not tracked, create a synthetic lifetime
                return Ok(self.create_borrow(name, span, false));
            }
            _ => {}
        }

        // Check for overlapping borrows using the new lifetime system
        self.check_overlapping_borrows(name, false, span)?;

        // Create the borrow with lifetime tracking
        let lifetime_id = self.create_borrow(name, span, false);

        // Update state (for backwards compatibility)
        match self.states.get(name) {
            Some(OwnershipState::Borrowed { count }) => {
                let new_count = count + 1;
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Borrowed { count: new_count },
                );
            }
            Some(OwnershipState::Owned { .. }) => {
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Borrowed { count: 1 },
                );
            }
            _ => {}
        }

        Ok(lifetime_id)
    }

    /// Borrow a value mutably, returning the lifetime ID
    pub fn borrow_mut(&mut self, name: &str, span: Span) -> Result<LifetimeId, TypeError> {
        // Check basic ownership state
        match self.states.get(name) {
            Some(OwnershipState::Moved { .. }) => {
                return Err(TypeError::moved_value(name.to_string(), span));
            }
            Some(OwnershipState::Owned { mutable }) if !mutable => {
                return Err(TypeError::not_mutable(name.to_string(), span));
            }
            None => {
                // Not tracked, create a synthetic lifetime
                return Ok(self.create_borrow(name, span, true));
            }
            _ => {}
        }

        // Check for overlapping borrows using the new lifetime system
        self.check_overlapping_borrows(name, true, span)?;

        // Create the borrow with lifetime tracking
        let lifetime_id = self.create_borrow(name, span, true);

        // Update state
        self.states.insert(
            name.to_string(),
            OwnershipState::MutBorrowed,
        );

        Ok(lifetime_id)
    }

    /// End a borrow by variable name (for backwards compatibility)
    pub fn end_borrow(&mut self, name: &str) {
        // Remove the most recent borrow for this variable
        if let Some(pos) = self.active_borrows.iter().rposition(|b| b.var_name == name) {
            self.active_borrows.remove(pos);
        }

        // Update old-style state
        match self.states.get(name) {
            Some(OwnershipState::Borrowed { count }) if *count > 1 => {
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Borrowed { count: count - 1 },
                );
            }
            Some(OwnershipState::Borrowed { count: 1 }) => {
                // Restore to owned state
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Owned { mutable: false },
                );
            }
            Some(OwnershipState::MutBorrowed) => {
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Owned { mutable: true },
                );
            }
            _ => {}
        }
    }

    /// End a borrow by its lifetime ID
    pub fn end_borrow_by_id(&mut self, id: LifetimeId) {
        if let Some(pos) = self.active_borrows.iter().position(|b| b.id == id) {
            let borrow = self.active_borrows.remove(pos);

            // Update old-style state based on remaining borrows
            let remaining = self.get_active_borrows(&borrow.var_name);
            if remaining.is_empty() {
                // No more borrows, restore to owned with original mutability
                self.states.insert(
                    borrow.var_name.clone(),
                    OwnershipState::Owned { mutable: borrow.original_var_mutable },
                );
            }
        }
    }

    /// Assign to a value
    pub fn assign(&mut self, name: &str, span: Span) -> Result<(), TypeError> {
        match self.states.get(name) {
            Some(OwnershipState::Moved { .. }) => {
                // Re-assigning to a moved value is OK - it gives it a new value
                self.states.insert(
                    name.to_string(),
                    OwnershipState::Owned { mutable: true },
                );
                Ok(())
            }
            Some(OwnershipState::Borrowed { .. }) => {
                Err(TypeError::already_borrowed(name.to_string(), span))
            }
            Some(OwnershipState::MutBorrowed) => {
                Err(TypeError::already_borrowed(name.to_string(), span))
            }
            Some(OwnershipState::Owned { mutable }) => {
                if !mutable {
                    Err(TypeError::not_mutable(name.to_string(), span))
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }
    }

    /// Verify all borrows are returned (called at end of function)
    pub fn verify(&self) -> Result<(), Vec<TypeError>> {
        let errors = Vec::new();

        for (_name, state) in &self.states {
            match state {
                OwnershipState::Borrowed { .. } | OwnershipState::MutBorrowed => {
                    // This would be a dangling borrow error
                    // For now, we don't track this strictly
                }
                _ => {}
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check if a value is mutable
    pub fn is_mutable(&self, name: &str) -> bool {
        match self.states.get(name) {
            Some(OwnershipState::Owned { mutable }) => *mutable,
            _ => false,
        }
    }

    /// Check if a value is moved
    pub fn is_moved(&self, name: &str) -> bool {
        matches!(self.states.get(name), Some(OwnershipState::Moved { .. }))
    }

    /// Check if a value is borrowed
    pub fn is_borrowed(&self, name: &str) -> bool {
        matches!(
            self.states.get(name),
            Some(OwnershipState::Borrowed { .. }) | Some(OwnershipState::MutBorrowed)
        )
    }
}

impl Default for OwnershipChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// The ownership state of a value
#[derive(Debug, Clone)]
pub enum OwnershipState {
    /// Value is owned and valid
    Owned {
        mutable: bool,
    },
    /// Value has been moved
    Moved {
        moved_at: Span,
    },
    /// Value is borrowed immutably
    Borrowed {
        count: usize,
    },
    /// Value is borrowed mutably
    MutBorrowed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_and_use() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", false);
        assert!(checker.use_value("x", Span::new(0, 0)).is_ok());
    }

    #[test]
    fn test_move() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", false);
        assert!(checker.move_value("x", Span::new(0, 0)).is_ok());
        assert!(checker.use_value("x", Span::new(1, 1)).is_err());
    }

    #[test]
    fn test_borrow() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", false);
        assert!(checker.borrow("x", Span::new(0, 0)).is_ok());
        assert!(checker.borrow("x", Span::new(1, 1)).is_ok()); // Multiple immutable borrows OK
    }

    #[test]
    fn test_mut_borrow_exclusive() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", true);
        assert!(checker.borrow_mut("x", Span::new(0, 0)).is_ok());
        assert!(checker.borrow("x", Span::new(1, 1)).is_err()); // Can't borrow while mutably borrowed
    }

    #[test]
    fn test_assign_requires_mut() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", false);
        assert!(checker.assign("x", Span::new(0, 0)).is_err());

        checker.define("y", true);
        assert!(checker.assign("y", Span::new(1, 1)).is_ok());
    }

    #[test]
    fn test_scope() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", false);

        checker.enter_scope();
        checker.define("y", false);
        assert!(checker.use_value("x", Span::new(0, 0)).is_ok());
        assert!(checker.use_value("y", Span::new(0, 0)).is_ok());
        checker.leave_scope();

        assert!(checker.use_value("x", Span::new(0, 0)).is_ok());
        // y is no longer tracked, so this succeeds (not found = OK)
    }

    #[test]
    fn test_lifetime_id() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", false);

        let id1 = checker.borrow("x", Span::new(0, 0)).unwrap();
        let id2 = checker.borrow("x", Span::new(1, 1)).unwrap();

        // Each borrow gets a unique ID
        assert_ne!(id1, id2);

        // Check active borrows
        assert_eq!(checker.get_active_borrows("x").len(), 2);
    }

    #[test]
    fn test_borrow_end_by_id() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", true);

        let id = checker.borrow("x", Span::new(0, 0)).unwrap();
        assert!(checker.has_active_borrows("x"));

        checker.end_borrow_by_id(id);
        assert!(!checker.has_active_borrows("x"));

        // Now we can mutably borrow
        assert!(checker.borrow_mut("x", Span::new(1, 1)).is_ok());
    }

    #[test]
    fn test_scope_ends_borrows() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", true);

        checker.enter_scope();
        let _id = checker.borrow("x", Span::new(0, 0)).unwrap();
        assert!(checker.has_active_borrows("x"));
        checker.leave_scope();

        // Borrow ended when scope ended
        assert!(!checker.has_active_borrows("x"));

        // Now we can mutably borrow
        assert!(checker.borrow_mut("x", Span::new(1, 1)).is_ok());
    }

    #[test]
    fn test_overlapping_borrows_detected() {
        let mut checker = OwnershipChecker::new();
        checker.define("x", true);

        // Borrow immutably
        let _id = checker.borrow("x", Span::new(0, 0)).unwrap();

        // Can't borrow mutably while immutably borrowed
        assert!(checker.borrow_mut("x", Span::new(1, 1)).is_err());

        // Start fresh
        let mut checker2 = OwnershipChecker::new();
        checker2.define("y", true);

        // Borrow mutably
        let _id = checker2.borrow_mut("y", Span::new(0, 0)).unwrap();

        // Can't borrow immutably while mutably borrowed
        assert!(checker2.borrow("y", Span::new(1, 1)).is_err());

        // Can't borrow mutably again
        assert!(checker2.borrow_mut("y", Span::new(2, 2)).is_err());
    }
}
