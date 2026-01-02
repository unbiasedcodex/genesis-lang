//! HARC - Hybrid Automatic Reference Counting
//!
//! A modern memory management system for Genesis Lang that combines:
//! - Automatic Reference Counting (like Swift's ARC)
//! - Region-based memory management (inspired by Microsoft's Verona)
//! - Compile-time optimizations (escape analysis)
//!
//! # Design Philosophy
//!
//! Genesis Lang targets OS development where:
//! - Deterministic deallocation is crucial (no GC pauses)
//! - Performance is critical
//! - Memory safety is non-negotiable
//!
//! HARC achieves this by:
//! 1. Using reference counting for heap-allocated types
//! 2. Inserting retain/release calls at compile time
//! 3. Optimizing away redundant RC operations via escape analysis
//! 4. Providing region-based memory for async tasks
//!
//! # Memory Layout
//!
//! All reference-counted objects have this header:
//! ```text
//! +----------+----------+------------------+
//! | refcount | type_id  |     data...      |
//! |   i64    |   i64    |    (varies)      |
//! +----------+----------+------------------+
//! ```
//!
//! # Type Categories
//!
//! - **Value Types** (stack allocated, no RC): i8-i64, f32, f64, bool
//! - **RC Types** (heap allocated, reference counted): Box<T>, Vec<T>, String, HashMap, HashSet
//! - **Inline Types** (stack struct, may contain RC fields): Option<T>, Result<T,E>, tuples
//!
//! # Automatic Drop Insertion
//!
//! The compiler automatically inserts `release` calls:
//! - At end of scope for local variables
//! - After last use (with optimization enabled)
//! - When overwriting a variable
//!
//! # Cycle Detection
//!
//! For data structures that may contain cycles (future feature):
//! - Weak references (`Weak<T>`)
//! - Explicit cycle breakers
//!
//! # Async Integration
//!
//! Each async task gets its own memory region:
//! - Allocations within a task are tracked
//! - When task completes, region can be bulk-freed
//! - Cross-task references use shared RC

pub mod rc;
pub mod drop;
pub mod region;
pub mod escape;

pub use rc::*;
pub use drop::*;
pub use region::*;
