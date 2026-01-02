# Genesis Lang Macro System

## Overview

Genesis Lang features a powerful macro system inspired by Rust's `macro_rules!`. Macros enable compile-time code generation and metaprogramming, allowing developers to reduce boilerplate and create domain-specific abstractions.

## Architecture

The macro system consists of four main phases:

```
┌────────────────────────────────────────────────────┐
│               MACRO EXPANSION PIPELINE             │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. PATTERN MATCHING                               │
│     ┌─────────────┐     ┌─────────────┐            │
│     │ Invocation  │ ──► │   Matcher   │            │
│     │   Tokens    │     │             │            │
│     └─────────────┘     └──────┬──────┘            │
│                                │                   │
│  2. CAPTURE BINDING            ▼                   │
│     ┌─────────────────────────────────────┐        │
│     │  Extract values: $x:expr, $n:ident  │        │
│     │  Build bindings map                 │        │
│     └──────────────────┬──────────────────┘        │
│                        │                           │
│  3. TRANSCRIPTION      ▼                           │
│     ┌─────────────────────────────────────┐        │
│     │  Substitute captures into template  │        │
│     │  Expand repetitions                 │        │
│     └──────────────────┬──────────────────┘        │
│                        │                           │
│  4. TOKEN → AST        ▼                           │
│     ┌─────────────────────────────────────┐        │
│     │  Convert expanded tokens to AST     │        │
│     │  Ready for type checking            │        │
│     └─────────────────────────────────────┘        │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Defining Macros

### Basic Syntax

```genesis
macro_rules! macro_name {
    (pattern) => {
        expansion
    };
}
```

### Simple Example

```genesis
macro_rules! answer {
    () => {
        42
    };
}

fn main() -> i32 {
    answer!()  // Expands to: 42
}
```

## Capture Types

Macros can capture different types of syntax elements:

| Capture | Description | Example |
|---------|-------------|---------|
| `$x:expr` | Any expression | `1 + 2`, `foo()`, `x` |
| `$x:ident` | An identifier | `foo`, `my_var` |
| `$x:ty` | A type | `i32`, `Vec<String>` |
| `$x:literal` | A literal value | `42`, `"hello"`, `true` |
| `$x:tt` | A single token tree | Any token or `(...)`, `[...]`, `{...}` |
| `$x:stmt` | A statement | `let x = 1;` |
| `$x:pat` | A pattern | `Some(x)`, `(a, b)` |
| `$x:path` | A path | `std::io::Result` |
| `$x:block` | A block | `{ ... }` |

### Example with Captures

```genesis
macro_rules! greet {
    ($name:expr) => {
        println($name)
    };
}

fn main() -> i32 {
    greet!("World")  // Expands to: println("World")
    0
}
```

## Repetitions

Macros support repetition patterns for variadic arguments:

| Syntax | Meaning |
|--------|---------|
| `$(...)*` | Zero or more repetitions |
| `$(...)+` | One or more repetitions |
| `$(...)?` | Zero or one (optional) |

### Repetition Example

```genesis
macro_rules! my_vec {
    () => {
        Vec::new()
    };
    ($($x:expr),*) => {
        {
            let mut v = Vec::new();
            $(v.push($x);)*
            v
        }
    };
}

fn main() -> i32 {
    let v = my_vec![1, 2, 3];  // Creates vec with 1, 2, 3
    0
}
```

### With Separators

```genesis
macro_rules! list {
    ($($x:expr),+ $(,)?) => {  // Trailing comma optional
        {
            let mut v = Vec::new();
            $(v.push($x);)+
            v
        }
    };
}
```

## Multiple Rules

Macros can have multiple rules, matched in order:

```genesis
macro_rules! log {
    () => {
        println("Empty log")
    };
    ($msg:expr) => {
        println($msg)
    };
    ($fmt:expr, $($arg:expr),+) => {
        // Format with arguments
        print($fmt)
        $(print($arg))+
        println("")
    };
}
```

## Built-in Macros

Genesis provides several built-in macros:

### `vec!` - Create a Vector

```genesis
let v = vec![1, 2, 3];
let empty: Vec<i32> = vec![];
```

### `println!` / `print!` - Output

```genesis
println!("Hello, World!");
print!("No newline");
```

### `eprint!` / `eprintln!` - Error Output

```genesis
eprintln!("Error message");
```

### `format!` - String Formatting

```genesis
let s = format!("Value: {}", x);
```

### `panic!` - Abort Execution

```genesis
panic!("Something went wrong");
```

### `assert!` - Debug Assertions

```genesis
assert!(x > 0);
assert!(x == y, "x and y must be equal");
```

## Implementation Details

### MacroToken Types

The macro system uses specialized token types to preserve literal values:

```rust
pub enum MacroToken {
    Token(TokenKind, Span),      // Keywords, operators
    Ident(String, Span),          // Identifiers
    IntLit(i64, Span),            // Integer literals with value
    FloatLit(f64, Span),          // Float literals with value
    StrLit(String, Span),         // String literals with value
    CharLit(char, Span),          // Char literals with value
    Capture { name, kind, span }, // $x:kind captures
    Repetition { ... },           // $(...)*
    Group { delimiter, tokens },  // Grouped tokens
}
```

### Key Components

1. **MacroExpander** (`src/macro_expand.rs`):
   - Central engine for macro expansion
   - Pattern matching against macro rules
   - Capture binding and transcription
   - Token-to-AST conversion

2. **Type Checker Integration** (`src/typeck/infer.rs`):
   - Registers macros during type inference
   - Expands user-defined macros
   - Re-parses expanded tokens
   - Type-checks expanded expressions

3. **Parser** (`src/parser.rs`):
   - Parses `macro_rules!` definitions
   - Parses macro invocations
   - Creates MacroToken streams
   - Provides `parse_expr()` for re-parsing

4. **IR Lowering** (`src/ir/lower.rs`):
   - Collects macro definitions
   - Expands user-defined macros during code generation
   - Special optimizations for built-in macros

### Expansion Flow

```
Source Code
     │
     ▼
┌─────────┐
│  Parse  │ ──► MacroDef registered
└────┬────┘
     │
     ▼
┌─────────────┐
│ Type Check  │ ──► Macro expanded, tokens re-parsed, types inferred
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ IR Lowering │ ──► Macro expanded, code generated
└─────────────┘
```

## Best Practices

1. **Keep macros simple** - Complex macros are hard to debug
2. **Use descriptive capture names** - `$item:expr` over `$x:expr`
3. **Document your macros** - Explain what patterns are accepted
4. **Prefer functions when possible** - Macros are for metaprogramming
5. **Test edge cases** - Empty inputs, single items, many items

## Current Capabilities

- ✅ User-defined declarative macros (`macro_rules!`)
- ✅ Pattern matching with captures (`$x:expr`, `$name:ident`, etc.)
- ✅ Repetition patterns (`$(...)*`, `$(...)+`, `$(...)? `)
- ✅ Multiple rules per macro
- ✅ Built-in macros (`vec!`, `println!`, `panic!`, etc.)
- ✅ Full integration with type checker and code generator

## Limitations

- No procedural macros (yet)
- Limited hygiene - name collision possible in complex cases
- No recursive macro expansion (macros calling other user macros)
- Repetition patterns limited to simple cases

## Future Enhancements

- [ ] Full hygiene support (gensym for generated identifiers)
- [ ] Procedural macros (derive, attribute macros)
- [ ] Recursive macro expansion
- [ ] Macro debugging tools
- [ ] Better error messages with expansion traces
