# Release Notes - v0.1.0

**Genesis Lang** - Systems programming with memory safety, without the complexity.

This is the first public release of Genesis Lang, a statically-typed language that compiles to native code via LLVM. Genesis brings memory safety to systems programming through automatic reference counting, offering an alternative to Rust's borrow checker with a gentler learning curve.

---

## Highlights

### Memory Safety Without Lifetimes

Genesis uses **HARC (Hybrid Automatic Reference Counting)** for memory management. No garbage collection pauses, no lifetime annotations, no borrow checker fights.

```genesis
fn process() -> String {
    let data: Vec<i64> = Vec::new()
    Vec::push(data, 1)
    Vec::push(data, 2)
    String::from("done")  // data automatically freed
}
```

### Complete Type System

- **Generics** with monomorphization
- **Traits** with bounds and where clauses
- **Pattern matching** with exhaustiveness checking
- **Type inference** throughout

```genesis
fn identity<T>(x: T) -> T { x }

trait Speak {
    fn speak(&self) -> i64
}
```

### First-Class Async

Built-in async/await with channels, timers, and TCP networking.

```genesis
async fn fetch() -> i64 {
    sleep_ms(100).await
    42
}

fn main() -> i64 {
    let handle = spawn(fetch())
    handle.await
}
```

### Rich Standard Library

| Module | Functions | Highlights |
|--------|-----------|------------|
| Collections | 30+ | Vec, HashMap, HashSet with iterators |
| String | 21 | split, trim, replace, contains, format! |
| Math | 50+ | Trigonometry, logarithms, powers |
| Async | 15+ | spawn, channels, timers, TCP |
| File I/O | 15 | Files, directories, paths |
| Time | 16 | Duration, elapsed, timestamps |

### Developer Experience

- **LSP Server** for IDE support (VSCode, Neovim, Helix, Emacs)
- **Declarative macros** for code generation
- **Clear error messages** with source locations

---

## What's Included

### Compiler (`glc`)

- Lexer with string interning
- Recursive descent parser
- Hindley-Milner type inference
- Macro expansion engine
- LLVM 18 backend
- Native executable generation

### Language Features

- Primitive types: i8-i64, u8-u64, f32, f64, bool, char, str
- Composite types: arrays, tuples, structs, enums
- Type aliases and global constants
- Functions, closures, impl blocks
- Generics, traits, associated types
- Pattern matching with guards
- Modules (inline and external)
- Async/await, spawn, channels

### Standard Library

- `Vec<T>` - Dynamic arrays with iterators
- `String` - Dynamic strings with 21 methods
- `HashMap<K, V>` - Key-value store (i64/String keys)
- `HashSet<T>` - Unique value sets
- `Option<T>` - Optional values
- `Result<T, E>` - Error handling
- `Box<T>` - Heap allocation
- `File`, `Dir`, `Fs` - Filesystem operations
- `time::`, `Duration` - Time measurement
- Math functions via libm (50+)

### Tooling

- `glc build` - Compile to native executable
- `glc check` - Type check without compiling
- `genesis-lsp` - Language server for IDEs

---

## Quick Start

### Build

```bash
# Requirements: Rust 1.70+, LLVM 18
apt-get install llvm-18 llvm-18-dev clang
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 cargo build --release
```

### Hello World

```genesis
fn main() -> i32 {
    println("Hello, World!")
    0
}
```

```bash
./target/release/glc build hello.gl --native -o hello
./hello
```

---

## Stats

- **Compiler**: ~25,000 lines of Rust
- **Tests**: 176 passing
- **Examples**: 99 files
- **Documentation**: Complete language reference

---

## What's Next

- Package manager
- More standard library modules
- Self-hosting compiler (rewrite in Genesis)
- Genesis OS kernel development

---

## Links

- [Language Reference](docs/reference.md)
- [Memory Management](docs/memory.md)
- [Macro System](docs/macros.md)
- [LSP Setup](docs/lsp.md)

---

## License

MIT
