//! Genesis Lang Compiler
//!
//! This is the bootstrap compiler for the Genesis programming language,
//! written in Rust. Once the language is mature enough, we will rewrite
//! the compiler in Genesis Lang itself (self-hosting).
//!
//! # Architecture
//!
//! ```text
//! Source Code (.gl)
//!       │
//!       ▼
//! ┌─────────────┐
//! │    Lexer    │  → Tokens
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │   Parser    │  → AST
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │  Type Check │  → Typed AST
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │  IR Lowering│  → Genesis IR
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │  Code Gen   │  → Machine Code / LLVM IR
//! └─────────────┘
//! ```

pub mod lexer;
pub mod token;
pub mod span;
pub mod ast;
pub mod parser;
pub mod typeck;
pub mod ir;
pub mod memory;
pub mod macro_expand;

// Re-exports for convenience
pub use lexer::Lexer;
pub use token::{Token, TokenKind};
pub use span::Span;

/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// File extension for Genesis source files
pub const FILE_EXTENSION: &str = "gl";
