# Changelog

All notable changes to Genesis Lang.

---

## [0.1.2] - 2026-01-07

Bug fixes for kernel development compiler issues.

### Bug Fixes

- **Inline ASM outputs**: Fixed output constraints being ignored (`out("eax")`, `lateout("eax")`)
- **ASM u64 outputs**: Fixed u64/u32 outputs always returning zero
- **ASM $ escaping**: Fixed AT&T immediate syntax (`$$0x12` now works correctly)
- **Type casting**: Fixed integer-to-pointer and smaller type conversions
- **Value storage**: Fixed storing values to smaller type slots (i32, u32, etc.)

### Tests

- Added `tests/codegen/asm_output.gl` - ASM output constraints test
- Added `tests/codegen/asm_dollar_escape.gl` - ASM $ escaping test
- Added `tests/codegen/int_ptr_cast.gl` - Integer to pointer casting test
- Added `tests/codegen/type_conversion.gl` - Type conversion test

---

## [0.1.1] - 2026-01-05

Second release with major new features for kernel and embedded development.

### New Features

- **Trait Objects (dyn Trait)**: Full dynamic dispatch via vtables
  - Fat pointers with data + vtable
  - Method dispatch through vtable lookup
  - Drop via vtable for proper cleanup

- **Freestanding Mode**: Compile without libc for kernel development
  - `--freestanding` flag for no-libc compilation
  - `--linker-script` for custom memory layout
  - `--emit-obj` for object file generation
  - Custom `_start` entry point support

- **Raw Pointers**: Direct memory access for systems programming
  - `*T` and `*mut T` pointer types
  - `null` pointer literal
  - `&raw` and `&raw mut` address-of operators
  - Pointer casting (`0xB8000 as *mut u8`)

- **Volatile Operations**: Non-optimizable memory access
  - `volatile_read_*` for hardware register reads
  - `volatile_write_*` for hardware register writes
  - All integer types (i8-i64, u8-u64)

- **Inline Assembly**: Embed raw CPU instructions
  - `asm!` macro with operands and options
  - Named registers (`in("rax")`, `out("dx")`)
  - Register classes (`in(reg)`, `out(reg)`)
  - Options: `nomem`, `nostack`, `pure`, `att_syntax`

- **Memory Layout Control**: Hardware structure compatibility
  - `#[repr(C)]` for C ABI layout
  - `#[repr(packed)]` for no padding
  - `#[repr(align(N))]` for custom alignment
  - `size_of::<T>()` and `align_of::<T>()` intrinsics

### Improvements

- Updated README with kernel development section
- Added freestanding mode documentation
- Integration test suite for kernel features

### Stats

- 184 passing tests (was 176)
- 102 example programs (was 99)

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
