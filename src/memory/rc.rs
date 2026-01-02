//! Reference Counting Core Implementation
//!
//! This module provides the core RC infrastructure for HARC.

use std::collections::HashSet;

/// RC Header size in bytes (refcount + type_id)
pub const RC_HEADER_SIZE: usize = 16;

/// Offset of refcount field in RC header
pub const RC_REFCOUNT_OFFSET: usize = 0;

/// Offset of type_id field in RC header
pub const RC_TYPE_ID_OFFSET: usize = 8;

/// Type IDs for built-in RC types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum RcTypeId {
    /// Box<T> - simple heap wrapper
    Box = 1,
    /// Vec<T> - dynamic array
    Vec = 2,
    /// String - dynamic string
    String = 3,
    /// HashMap<K,V> - hash map
    HashMap = 4,
    /// HashSet<T> - hash set
    HashSet = 5,
    /// Closure - captured environment
    Closure = 6,
    /// Channel - async channel
    Channel = 7,
    /// TcpStream - TCP connection
    TcpStream = 8,
    /// TcpListener - TCP listener
    TcpListener = 9,
    /// File - file handle
    File = 10,
    /// Future<T> - async future
    Future = 11,
    /// Custom user-defined type
    Custom = 100,
}

impl RcTypeId {
    pub fn from_type_name(name: &str) -> Option<Self> {
        match name {
            "Box" => Some(RcTypeId::Box),
            "Vec" => Some(RcTypeId::Vec),
            "String" => Some(RcTypeId::String),
            "HashMap" => Some(RcTypeId::HashMap),
            "HashSet" => Some(RcTypeId::HashSet),
            "Closure" => Some(RcTypeId::Closure),
            "Channel" => Some(RcTypeId::Channel),
            "TcpStream" => Some(RcTypeId::TcpStream),
            "TcpListener" => Some(RcTypeId::TcpListener),
            "File" => Some(RcTypeId::File),
            "Future" => Some(RcTypeId::Future),
            _ => None,
        }
    }
}

/// Tracks which types require reference counting
#[derive(Debug, Clone)]
pub struct RcTypeInfo {
    /// Types that are reference counted
    rc_types: HashSet<String>,
    /// Types that contain RC fields (need recursive drop)
    container_types: HashSet<String>,
}

impl RcTypeInfo {
    pub fn new() -> Self {
        let mut rc_types = HashSet::new();
        let mut container_types = HashSet::new();

        // Built-in RC types
        rc_types.insert("Box".to_string());
        rc_types.insert("Vec".to_string());
        rc_types.insert("String".to_string());
        rc_types.insert("HashMap".to_string());
        rc_types.insert("HashSet".to_string());
        rc_types.insert("Channel".to_string());
        rc_types.insert("Sender".to_string());
        rc_types.insert("Receiver".to_string());
        rc_types.insert("TcpStream".to_string());
        rc_types.insert("TcpListener".to_string());
        rc_types.insert("File".to_string());
        rc_types.insert("Future".to_string());

        // Container types that may hold RC types
        container_types.insert("Option".to_string());
        container_types.insert("Result".to_string());

        Self { rc_types, container_types }
    }

    /// Check if a type name requires reference counting
    pub fn is_rc_type(&self, type_name: &str) -> bool {
        // Extract base type name (e.g., "Vec" from "Vec<i64>")
        let base_name = type_name.split('<').next().unwrap_or(type_name);
        self.rc_types.contains(base_name)
    }

    /// Check if a type may contain RC fields
    pub fn is_container_type(&self, type_name: &str) -> bool {
        let base_name = type_name.split('<').next().unwrap_or(type_name);
        self.container_types.contains(base_name)
    }

    /// Check if a type needs drop handling (either RC or contains RC)
    pub fn needs_drop(&self, type_name: &str) -> bool {
        self.is_rc_type(type_name) || self.is_container_type(type_name)
    }

    /// Register a custom RC type
    pub fn register_rc_type(&mut self, name: String) {
        self.rc_types.insert(name);
    }

    /// Register a container type
    pub fn register_container(&mut self, name: String) {
        self.container_types.insert(name);
    }
}

impl Default for RcTypeInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for RC operations (for debugging/profiling)
#[derive(Debug, Default, Clone)]
pub struct RcStats {
    /// Total allocations
    pub allocs: u64,
    /// Total retains
    pub retains: u64,
    /// Total releases
    pub releases: u64,
    /// Total deallocations (when refcount hits 0)
    pub deallocs: u64,
    /// Peak memory usage
    pub peak_memory: u64,
    /// Current memory usage
    pub current_memory: u64,
}

/// Optimization hints for RC operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcOptimization {
    /// No optimization, always emit retain/release
    None,
    /// Remove redundant retain-release pairs
    ElidePairs,
    /// Move retain to first use, release to last use
    ScopeNarrowing,
    /// Full optimization with escape analysis
    Full,
}

impl Default for RcOptimization {
    fn default() -> Self {
        RcOptimization::ElidePairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc_type_detection() {
        let info = RcTypeInfo::new();

        assert!(info.is_rc_type("Vec"));
        assert!(info.is_rc_type("Vec<i64>"));
        assert!(info.is_rc_type("String"));
        assert!(info.is_rc_type("HashMap<String, i64>"));

        assert!(!info.is_rc_type("i64"));
        assert!(!info.is_rc_type("bool"));
        assert!(!info.is_rc_type("f64"));
    }

    #[test]
    fn test_container_detection() {
        let info = RcTypeInfo::new();

        assert!(info.is_container_type("Option"));
        assert!(info.is_container_type("Option<Vec<i64>>"));
        assert!(info.is_container_type("Result"));

        assert!(!info.is_container_type("Vec"));
        assert!(!info.is_container_type("i64"));
    }

    #[test]
    fn test_needs_drop() {
        let info = RcTypeInfo::new();

        // RC types need drop
        assert!(info.needs_drop("Vec<i64>"));
        assert!(info.needs_drop("String"));

        // Containers need drop (might contain RC)
        assert!(info.needs_drop("Option<String>"));

        // Primitives don't need drop
        assert!(!info.needs_drop("i64"));
        assert!(!info.needs_drop("bool"));
    }
}
