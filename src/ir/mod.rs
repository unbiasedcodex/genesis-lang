//! Genesis Intermediate Representation (GIR)
//!
//! A simple SSA-form IR for the Genesis Lang compiler.
//! This IR is designed to be:
//! - Easy to generate from the type-checked AST
//! - Easy to lower to LLVM IR or other backends
//! - Amenable to optimization passes

// Note: instr must come before types to avoid circular deps
mod instr;
mod types;
mod builder;
mod lower;
mod llvm;

// Re-export in logical order
pub use instr::*;
pub use types::*;
pub use builder::*;
pub use lower::*;
pub use llvm::*;
