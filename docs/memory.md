# HARC - Hybrid Automatic Reference Counting

An innovative memory management system for Genesis Lang.

**Status**: Implemented

---

## Overview

HARC combines the best characteristics of different memory management approaches:

| Feature | Rust | Swift | HARC |
|---------|------|-------|------|
| Explicit lifetimes | Yes | No | **No** |
| Reference Counting | Optional (Arc) | Yes (ARC) | **Yes** |
| Deterministic | Yes | Yes | **Yes** |
| GC Pauses | No | No | **No** |
| Complexity | High | Medium | **Low** |

HARC provides automatic memory management without garbage collection pauses, explicit lifetime annotations, or manual memory management.

---

## Architecture

### Memory Layout

All reference-counted objects have a header:

```
+------------+------------+------------------+
|  refcount  |  type_id   |      data        |
|    i64     |    i64     |    (varies)      |
+------------+------------+------------------+
    offset 0     offset 8      offset 16
```

- **refcount**: Reference counter (starts at 1)
- **type_id**: Type identifier (for destructor dispatch)
- **data**: Object data

### Type IDs

| ID | Type |
|----|------|
| 1 | Box<T> |
| 2 | Vec<T> |
| 3 | String |
| 4 | HashMap<K,V> |
| 5 | HashSet<T> |
| 6 | Closure |
| 7 | Channel |
| 8 | TcpStream |
| 9 | TcpListener |
| 10 | File |
| 11 | Future<T> |
| 100+ | Custom types |

---

## IR Instructions

### RcAlloc
```
%ptr = rc_alloc <type> (type_id=N)
```
Allocates memory with RC header. Returns pointer to data (after header).

### RcRetain
```
rc_retain %ptr
```
Atomically increments refcount.

### RcRelease
```
rc_release %ptr
```
Decrements refcount. If it reaches zero, calls destructor and frees memory.

### RcGetCount
```
%count = rc_getcount %ptr
```
Returns current refcount (for debugging).

### Drop
```
drop %ptr (type_id=N)
```
Calls the type-specific destructor.

---

## Drop Functions

Automatically generated destructor functions:

| Function | Responsibility |
|----------|----------------|
| `__drop_vec` | Frees Vec data buffer |
| `__drop_string` | Frees String character buffer |
| `__drop_hashmap` | Frees HashMap entries array |
| `__drop_hashset` | Frees HashSet entries array |
| `__drop_box` | Frees boxed value |
| `__drop_file` | Closes file descriptor |
| `__drop_generic` | Fallback (no-op) |

---

## Automatic Tracking

The system tracks variables automatically:

```genesis
fn example() {
    let v: Vec<i64> = Vec::new()  // Registered for drop
    let s = String::from("hello") // Registered for drop

    // ... use variables ...

}  // <- Here: rc_release(v), rc_release(s) are emitted
```

### Drop Scope

1. **Block entry**: `enter_drop_scope()`
2. **Let declaration**: `register_local_for_drop(name, type, span)`
3. **Block exit**: `leave_drop_scope()` emits releases

---

## Type Classification

### RC Types (need drop)

- `Vec<T>`
- `String`
- `HashMap<K,V>`
- `HashSet<T>`
- `Box<T>`
- `File`
- `Channel`
- `TcpStream`
- `TcpListener`
- `Future<T>`
- Closures with captures

### Value Types (no drop needed)

- `i8`, `i16`, `i32`, `i64`
- `u8`, `u16`, `u32`, `u64`
- `f32`, `f64`
- `bool`
- Tuples of value types
- Arrays of value types

### Container Types (recursive drop)

- `Option<T>` - drop if T is RC
- `Result<T,E>` - drop if T or E is RC

---

## Source Files

| File | Responsibility |
|------|----------------|
| `src/memory/mod.rs` | Main HARC module |
| `src/memory/rc.rs` | Reference counting core |
| `src/memory/drop.rs` | Drop tracking |
| `src/memory/region.rs` | Region-based memory (future) |
| `src/memory/escape.rs` | Escape analysis (future) |
| `src/ir/instr.rs` | RC IR instructions |
| `src/ir/builder.rs` | RC builder methods |
| `src/ir/llvm.rs` | RC codegen |
| `src/ir/lower.rs` | Drop tracking integration |

---

## Future Optimizations

### Escape Analysis

Values that don't escape can:
1. Be stack-allocated (no RC overhead)
2. Have RC elided completely

### Region-Based Memory

For async tasks:
1. Each task has its own region
2. Bump-pointer allocations (fast)
3. Bulk deallocation at task end

### Cycle Detection

For structures with cycles:
1. Weak references (`Weak<T>`)
2. Optional cycle collector

---

## Comparison with Other Languages

### vs Rust

- **Simpler**: No explicit lifetimes
- **Trade-off**: Runtime RC overhead (minimal)

### vs Swift

- **Similar**: Automatic ARC
- **Difference**: Regions for async

### vs Go/Java

- **Deterministic**: No GC pauses
- **Predictable**: Ideal for OS/systems programming

---

## Example

```genesis
fn process_data() -> String {
    // Vec is tracked for drop
    let numbers: Vec<i64> = Vec::new()
    Vec::push(numbers, 1)
    Vec::push(numbers, 2)
    Vec::push(numbers, 3)

    // String is tracked for drop
    let mut result = String::new()

    for n in Vec::iter(numbers) {
        let s = i64::to_string(n)  // s is tracked
        String::push_str(result, s)
        // s is released here (end of for scope)
    }

    // numbers is released here (end of function)
    // result is returned (not released)
    result
}
```

---

## Implementation Status

- [x] HARC system design
- [x] IR instructions (RcAlloc, RcRetain, RcRelease, Drop)
- [x] LLVM codegen for RC instructions
- [x] Scope-based drop tracking
- [x] Drop functions for built-in types
- [ ] Escape analysis (structure ready)
- [ ] Region-based memory (structure ready)
- [ ] Cycle detection

---

## References

- [Swift ARC](https://docs.swift.org/swift-book/LanguageGuide/AutomaticReferenceCounting.html)
- [Rust Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Microsoft Verona](https://github.com/microsoft/verona)
- [Vale Memory Safety](https://vale.dev/guide/generational-references)
