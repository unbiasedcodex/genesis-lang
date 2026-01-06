# Genesis Lang

**Systems programming with memory safety, without the complexity.**

Genesis is a statically-typed, compiled language that brings memory safety to systems programming through automatic reference counting, not complex lifetime annotations. From applications to operating system kernels, Genesis provides the power you need with a familiar syntax.

```genesis
fn main() -> i32 {
    let message = String::from("Hello, World!")
    println(message)
    0
}
```

---

## Why Genesis?

| Challenge | Rust's Approach | Genesis Approach |
|-----------|-----------------|------------------|
| Memory safety | Ownership + Lifetimes | Automatic Reference Counting (HARC) |
| Learning curve | Steep (borrow checker) | Familiar (like Go/Swift) |
| Compile errors | Complex lifetime errors | Clear, actionable errors |
| Async code | Pin, lifetimes in futures | Simple async/await |
| Kernel development | Requires unsafe everywhere | First-class freestanding support |

**Genesis is not trying to replace Rust.** Rust is excellent for maximum performance with zero-cost safety. Genesis offers an alternative: **memory safety with a gentler learning curve**, ideal for developers who want systems-level control without fighting the borrow checker.

### Who is Genesis for?

- Developers who want **native performance** without garbage collection
- Teams transitioning from **Go, Python, or JavaScript** to systems programming
- Projects where **developer productivity** matters as much as raw speed
- **Kernel and embedded developers** who need freestanding compilation
- The **Genesis OS** project (our primary use case)

---

## Quick Start

### Installation

```bash
# Requirements: Rust 1.70+, LLVM 18
git clone https://github.com/user/genesis-lang
cd genesis-lang
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 cargo build --release
```

### Hello World

Create `hello.gl`:
```genesis
fn main() -> i32 {
    println("Hello, World!")
    0
}
```

Compile and run:
```bash
./target/release/glc build hello.gl --native -o hello
./hello
```

---

## Features

### Memory Safety Without Lifetimes

Genesis uses **HARC (Hybrid Automatic Reference Counting)** — automatic memory management without garbage collection pauses or explicit lifetime annotations.

```genesis
fn process_data() -> String {
    let items: Vec<i64> = Vec::new()
    Vec::push(items, 1)
    Vec::push(items, 2)

    let result = String::from("done")
    result  // items automatically freed here
}
```

### Trait Objects (Dynamic Dispatch)

Full support for trait objects with vtable-based dynamic dispatch:

```genesis
trait Drawable {
    fn draw(&self)
}

struct Circle { radius: f64 }
struct Square { side: f64 }

impl Drawable for Circle {
    fn draw(&self) { println("Drawing circle") }
}

impl Drawable for Square {
    fn draw(&self) { println("Drawing square") }
}

fn render(shape: &dyn Drawable) {
    shape.draw()  // Dynamic dispatch via vtable
}
```

### Modern Type System

Generics, traits, type inference, and pattern matching with exhaustiveness checking.

```genesis
trait Printable {
    fn display(&self) -> String
}

struct User { name: String, age: i64 }

impl Printable for User {
    fn display(&self) -> String {
        format!("{} ({})", self.name, self.age)
    }
}

fn print_all<T: Printable>(items: Vec<T>) {
    for item in Vec::iter(items) {
        println(item.display())
    }
}
```

### First-Class Async

Built-in async/await with channels and TCP networking.

```genesis
async fn fetch_data(url: str) -> String {
    let stream = TcpStream::connect(url, 80).await
    TcpStream::write(stream, "GET / HTTP/1.1\r\n\r\n")
    let response = TcpStream::read(stream).await
    TcpStream::close(stream)
    response
}

fn main() -> i64 {
    let data = fetch_data("example.com").await
    println(data)
    0
}
```

### Kernel & Embedded Development

Freestanding mode for OS kernels and bare-metal applications:

```genesis
#[repr(C, packed)]
struct GdtEntry {
    limit_low: u16,
    base_low: u16,
    base_middle: u8,
    access: u8,
    granularity: u8,
    base_high: u8,
}

fn cli() {
    unsafe { asm!("cli", options(nomem, nostack)); }
}

fn _start() {
    cli();
    let vga: *mut u8 = 0xB8000 as *mut u8;
    unsafe {
        volatile_write_u8(vga, 72);  // 'H'
    }
    loop {}
}
```

Build with:
```bash
glc build kernel.gl --freestanding --linker-script=kernel.ld -o kernel.elf
```

### Powerful Macros

Declarative macros for code generation.

```genesis
macro_rules! hashmap {
    ($($key:expr => $value:expr),*) => {{
        let map = HashMap::new();
        $(HashMap::insert(map, $key, $value);)*
        map
    }};
}

let scores = hashmap!{
    String::from("alice") => 100,
    String::from("bob") => 85
};
```

---

## Standard Library

| Module | Functions | Highlights |
|--------|-----------|------------|
| **Collections** | 30+ | Vec, HashMap, HashSet with iterators |
| **String** | 21 | split, trim, replace, contains, format! |
| **Math** | 50+ | Trigonometry, logarithms, rounding |
| **Async** | 15+ | spawn, channels, timers, TCP |
| **File I/O** | 15 | Files, directories, paths |
| **Time** | 16 | Duration, elapsed, timestamps |
| **Random** | 5 | random_i64, random_f64, random_range |

### Example: Working with Collections

```genesis
fn main() -> i64 {
    let numbers: Vec<i64> = Vec::new()
    Vec::push(numbers, 10)
    Vec::push(numbers, 20)
    Vec::push(numbers, 30)

    // Functional iteration
    let doubled = VecIter::map(Vec::iter(numbers), |x| x * 2)
    let sum = VecIter::sum(doubled)  // 120

    sum
}
```

---

## Freestanding Mode

Genesis supports compilation without libc for kernel and embedded development:

| Feature | Description |
|---------|-------------|
| `--freestanding` | Compile without libc dependency |
| `--linker-script` | Custom memory layout |
| `--emit-obj` | Generate object file only |
| Raw Pointers | `*T`, `*mut T`, `null`, `&raw` |
| Volatile Ops | `volatile_read_*`, `volatile_write_*` |
| Inline Assembly | `asm!` with full operand support |
| Memory Layout | `#[repr(C)]`, `#[repr(packed)]` |
| Intrinsics | `size_of::<T>()`, `align_of::<T>()` |

See [Freestanding Documentation](docs/freestanding.md) for details.

---

## Project Status

Genesis is functional and actively developed. The compiler is written in Rust (~25,000 lines) and compiles to native code via LLVM.

| Component | Status |
|-----------|--------|
| Compiler (lexer, parser, type checker) | Complete |
| LLVM backend | Complete |
| Standard library | Complete |
| Async runtime | Complete |
| Trait objects (dyn Trait) | Complete |
| Freestanding mode | Complete |
| LSP server (IDE support) | Complete |
| Package manager | Planned |
| Self-hosting compiler | Planned |

**Tests**: 184 passing | **Examples**: 102 files

---

## Comparison

### Genesis vs Rust

| Aspect | Rust | Genesis |
|--------|------|---------|
| Memory model | Ownership + borrowing | Reference counting |
| Lifetimes | Explicit (`'a`) | None |
| Performance | Maximum | Near-native (RC overhead) |
| Learning curve | Steep | Moderate |
| Kernel support | Via unsafe | First-class freestanding |
| Ecosystem | Massive | New |

**Choose Rust if**: Maximum performance is critical, you're comfortable with lifetimes, or you need the ecosystem.

**Choose Genesis if**: You want memory safety with simpler code, faster onboarding, or you're building on Genesis OS.

### Genesis vs Go

| Aspect | Go | Genesis |
|--------|-----|---------|
| Memory model | Garbage collection | Reference counting |
| Generics | Limited | Full (with traits) |
| Null safety | No | Yes (Option type) |
| Pattern matching | No | Yes |
| Kernel development | Not supported | Full support |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Language Reference](docs/reference.md) | Complete language and stdlib reference |
| [Memory Management](docs/memory.md) | HARC system details |
| [Freestanding Mode](docs/freestanding.md) | Kernel and embedded development |
| [Math Functions](docs/math.md) | Complete math reference |
| [Macro System](docs/macros.md) | Declarative macros |
| [LSP Setup](docs/lsp.md) | IDE configuration |
| [Changelog](docs/changelog.md) | Version history |

---

## Building from Source

### Requirements

- Rust 1.70+
- LLVM 18
- GCC or Clang (linker)

### Ubuntu/Debian

```bash
apt-get install llvm-18 llvm-18-dev clang
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 cargo build --release
```

### Running Tests

```bash
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 cargo test
```

---

## Part of Genesis OS

Genesis Lang is the core language for [Genesis OS](../README.md), a new operating system built from scratch. The language is designed to eventually be self-hosting, with the compiler rewritten in Genesis itself.

---

## License

MIT License

Copyright (c) 2026 UnbiasedCodex
