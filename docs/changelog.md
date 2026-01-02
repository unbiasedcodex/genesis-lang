# Changelog

All notable changes to Genesis Lang.

---

## [0.1.0] - Initial Release

First public release of Genesis Lang.

### Language Features

- **Type System**: Static typing with inference, generics, traits, where clauses
- **Memory Management**: HARC (Hybrid Automatic Reference Counting)
- **Pattern Matching**: Exhaustive matching with guards
- **Async/Await**: First-class concurrency with spawn, channels, TCP
- **Macros**: Declarative macro system with pattern matching
- **Modules**: Inline and external modules with visibility control

### Standard Library

- **Collections**: Vec, HashMap, HashSet with full iterator support
- **String**: 21 methods (split, trim, replace, contains, etc.)
- **Option/Result**: Complete monadic error handling
- **File I/O**: Files, directories, path manipulation
- **Time**: Duration type, elapsed time measurement
- **Math**: 50+ functions via libm integration

### Tooling

- **Compiler** (`glc`): LLVM 18 backend, native executables
- **LSP Server**: IDE support with diagnostics, hover, go-to-definition
- **Format Macro**: String formatting with specifiers

### Stats

- ~25,000 lines of Rust
- 176 passing tests
- 99 example programs
