//! Region-Based Memory Management
//!
//! Regions provide bulk allocation/deallocation for async tasks
//! and other scoped memory needs. Inspired by Microsoft's Verona.
//!
//! # Region Types
//!
//! - **Task Region**: Each async task gets its own region
//! - **Arena Region**: Manual arena for bulk allocations
//! - **Shared Region**: Reference counted, shared between tasks
//!
//! # Benefits
//!
//! 1. Fast allocation (bump pointer)
//! 2. Bulk deallocation (free entire region at once)
//! 3. Better cache locality
//! 4. Reduced fragmentation

use std::collections::HashMap;

/// Unique identifier for a memory region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u64);

/// Types of memory regions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Task-local region (freed when task completes)
    Task,
    /// Arena allocation (user-managed)
    Arena,
    /// Shared region (reference counted)
    Shared,
    /// Global region (never freed)
    Global,
}

/// Represents a memory region
#[derive(Debug, Clone)]
pub struct Region {
    /// Unique ID
    pub id: RegionId,
    /// Kind of region
    pub kind: RegionKind,
    /// Parent region (for nesting)
    pub parent: Option<RegionId>,
    /// Reference count (for shared regions)
    pub refcount: u64,
    /// Whether region is active
    pub active: bool,
}

impl Region {
    pub fn new(id: RegionId, kind: RegionKind) -> Self {
        Self {
            id,
            kind,
            parent: None,
            refcount: 1,
            active: true,
        }
    }

    pub fn with_parent(mut self, parent: RegionId) -> Self {
        self.parent = Some(parent);
        self
    }
}

/// Tracks memory regions at compile time
#[derive(Debug)]
pub struct RegionTracker {
    /// All known regions
    regions: HashMap<RegionId, Region>,
    /// Next region ID
    next_id: u64,
    /// Current active region stack
    region_stack: Vec<RegionId>,
    /// Global region ID
    global_region: RegionId,
}

impl RegionTracker {
    pub fn new() -> Self {
        let global_region = RegionId(0);
        let mut regions = HashMap::new();

        regions.insert(global_region, Region::new(global_region, RegionKind::Global));

        Self {
            regions,
            next_id: 1,
            region_stack: vec![global_region],
            global_region,
        }
    }

    /// Create a new task region
    pub fn create_task_region(&mut self) -> RegionId {
        let id = RegionId(self.next_id);
        self.next_id += 1;

        let parent = *self.region_stack.last().unwrap_or(&self.global_region);
        let region = Region::new(id, RegionKind::Task).with_parent(parent);

        self.regions.insert(id, region);
        id
    }

    /// Create an arena region
    pub fn create_arena(&mut self) -> RegionId {
        let id = RegionId(self.next_id);
        self.next_id += 1;

        let parent = *self.region_stack.last().unwrap_or(&self.global_region);
        let region = Region::new(id, RegionKind::Arena).with_parent(parent);

        self.regions.insert(id, region);
        id
    }

    /// Create a shared region
    pub fn create_shared(&mut self) -> RegionId {
        let id = RegionId(self.next_id);
        self.next_id += 1;

        let region = Region::new(id, RegionKind::Shared);
        self.regions.insert(id, region);
        id
    }

    /// Enter a region (push onto stack)
    pub fn enter(&mut self, id: RegionId) {
        self.region_stack.push(id);
    }

    /// Leave current region (pop from stack)
    pub fn leave(&mut self) -> Option<RegionId> {
        if self.region_stack.len() > 1 {
            self.region_stack.pop()
        } else {
            None // Can't leave global region
        }
    }

    /// Get current region
    pub fn current(&self) -> RegionId {
        *self.region_stack.last().unwrap_or(&self.global_region)
    }

    /// Mark a region as inactive (freed)
    pub fn free_region(&mut self, id: RegionId) {
        if id == self.global_region {
            return; // Can't free global region
        }

        if let Some(region) = self.regions.get_mut(&id) {
            region.active = false;
        }
    }

    /// Retain a shared region
    pub fn retain(&mut self, id: RegionId) {
        if let Some(region) = self.regions.get_mut(&id) {
            if region.kind == RegionKind::Shared {
                region.refcount += 1;
            }
        }
    }

    /// Release a shared region
    pub fn release(&mut self, id: RegionId) -> bool {
        if let Some(region) = self.regions.get_mut(&id) {
            if region.kind == RegionKind::Shared {
                region.refcount -= 1;
                if region.refcount == 0 {
                    region.active = false;
                    return true; // Should be freed
                }
            }
        }
        false
    }

    /// Check if a region is active
    pub fn is_active(&self, id: RegionId) -> bool {
        self.regions.get(&id).map(|r| r.active).unwrap_or(false)
    }

    /// Get the global region
    pub fn global(&self) -> RegionId {
        self.global_region
    }
}

impl Default for RegionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime region state (for codegen)
#[derive(Debug, Clone)]
pub struct RuntimeRegion {
    /// Region ID
    pub id: RegionId,
    /// Base pointer of region memory
    pub base_ptr: u64,
    /// Current allocation pointer
    pub alloc_ptr: u64,
    /// End of region memory
    pub end_ptr: u64,
    /// Size in bytes
    pub size: u64,
}

impl RuntimeRegion {
    /// Default region size: 64KB
    pub const DEFAULT_SIZE: u64 = 64 * 1024;

    /// Large region size: 1MB
    pub const LARGE_SIZE: u64 = 1024 * 1024;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_creation() {
        let mut tracker = RegionTracker::new();

        let task_region = tracker.create_task_region();
        assert!(tracker.is_active(task_region));

        let arena = tracker.create_arena();
        assert!(tracker.is_active(arena));

        tracker.free_region(task_region);
        assert!(!tracker.is_active(task_region));
    }

    #[test]
    fn test_region_stack() {
        let mut tracker = RegionTracker::new();

        let global = tracker.current();
        assert_eq!(global, tracker.global());

        let task = tracker.create_task_region();
        tracker.enter(task);
        assert_eq!(tracker.current(), task);

        tracker.leave();
        assert_eq!(tracker.current(), global);
    }

    #[test]
    fn test_shared_region_refcount() {
        let mut tracker = RegionTracker::new();

        let shared = tracker.create_shared();
        assert!(tracker.is_active(shared));

        tracker.retain(shared);
        tracker.retain(shared);

        // Refcount is now 3
        assert!(!tracker.release(shared)); // 2
        assert!(!tracker.release(shared)); // 1
        assert!(tracker.release(shared));  // 0 - should free
        assert!(!tracker.is_active(shared));
    }
}
