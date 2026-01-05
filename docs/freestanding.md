# Freestanding Mode

Genesis Lang supports freestanding compilation for kernel and embedded development.

---

## Overview

Freestanding mode produces binaries without libc dependency, suitable for:
- Operating system kernels
- Bootloaders
- Embedded systems
- Bare-metal applications

---

## Compiler Flags

| Flag | Description |
|------|-------------|
| `--freestanding` | Compile without libc, use ld directly |
| `--linker-script=FILE` | Custom linker script for memory layout |
| `--emit-obj` | Generate object file only (no linking) |

### Usage

```bash
# Full kernel build
glc build kernel.gl --freestanding --linker-script=kernel.ld -o kernel.elf

# Object file for manual linking
glc build kernel.gl --freestanding --emit-obj -o kernel.o
ld -T kernel.ld -o kernel.elf kernel.o --nostdlib -e _start
```

---

## Entry Point

In freestanding mode, use `_start` as the entry point instead of `main`:

```genesis
fn _start() {
    // Kernel entry point
    loop {}
}
```

---

## Raw Pointers

Raw pointers enable direct memory access for hardware I/O:

```genesis
// Pointer types
let ptr: *i64 = null;           // Immutable raw pointer
let mut_ptr: *mut i64 = null;   // Mutable raw pointer

// Pointer from integer (MMIO)
let vga: *mut u8 = 0xB8000 as *mut u8;

// Address-of operators
let value: i64 = 42;
let ptr: *i64 = &raw value;
let ptr_mut: *mut i64 = &raw mut value;

// Dereference (requires unsafe)
unsafe {
    let val = *ptr;
}
```

---

## Volatile Operations

Non-optimizable memory access for hardware registers:

```genesis
unsafe {
    // Read from hardware register
    let status = volatile_read_u8(port_addr);

    // Write to hardware register
    volatile_write_u8(port_addr, value);
}
```

Available variants: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`

---

## Inline Assembly

Embed raw CPU instructions:

```genesis
// Basic instruction
unsafe {
    asm!("cli", options(nomem, nostack));
    asm!("hlt", options(nomem, nostack));
}

// With operands
unsafe {
    asm!("out %al, %dx",
        in("dx") port,
        in("al") value,
        options(nomem, nostack, att_syntax));
}

// Output operands
let result: u8 = 0;
unsafe {
    asm!("in %dx, %al",
        in("dx") port,
        out("al") result,
        options(nomem, nostack, att_syntax));
}
```

### Assembly Options

| Option | Description |
|--------|-------------|
| `nomem` | No memory access |
| `nostack` | No stack usage |
| `pure` | No side effects |
| `preserves_flags` | Does not modify flags |
| `att_syntax` | Use AT&T syntax |

### Operand Types

| Operand | Description |
|---------|-------------|
| `in(reg) var` | Input from register class |
| `out(reg) var` | Output to register class |
| `inout(reg) var` | Input and output |
| `in("rax") var` | Named register input |
| `out("al") var` | Named register output |

---

## Memory Layout Control

Control struct layout for hardware structures:

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

let size = size_of::<GdtEntry>();   // 8 bytes
let align = align_of::<GdtEntry>(); // 1 (packed)
```

### Repr Options

| Option | Description |
|--------|-------------|
| `#[repr(C)]` | C ABI compatible layout |
| `#[repr(packed)]` | No padding (alignment 1) |
| `#[repr(align(N))]` | Custom alignment |
| `#[repr(C, packed)]` | Combined options |

---

## Known Limitations

Issues to be aware of in freestanding mode:

| Limitation | Workaround |
|------------|------------|
| Bitwise operators (`\|`, `<<`, `>>`) not as infix | Use `\|=`, `<<=`, `>>=` or multiplication |
| No pointer arithmetic | Use integer math then cast to pointer |
| No pointer-to-integer cast | Calculate with integers first |
| No static mut | Use local variables in _start |
| No string literals in freestanding | Use ASCII codes directly |

### Workaround Examples

```genesis
// Bitwise OR: use compound assignment
let mut result: u8 = bg * 16;  // bg << 4
result |= fg;                   // instead of: bg << 4 | fg

// Pointer arithmetic: use integer math
let offset = (y * 80 + x) * 2;
let addr = 0xB8000 + offset;
let ptr = addr as *mut u8;

// String output: use ASCII codes
vga_write(0, 0, 71);  // 'G' = 71
vga_write(1, 0, 101); // 'e' = 101
```

---

## Example: Mini-Kernel

Complete example demonstrating all freestanding features:

```genesis
#[repr(C, packed)]
struct VgaChar {
    character: u8,
    color: u8,
}

fn vga_color(fg: i64, bg: i64) -> u8 {
    let mut result: u8 = (bg * 16) as u8;
    result |= fg as u8;
    result
}

fn vga_write(x: i64, y: i64, ch: i64, color: u8) {
    let offset = (y * 80 + x) * 2;
    let addr = 0xB8000 + offset;
    let ptr = addr as *mut u8;

    unsafe {
        volatile_write_u8(ptr, ch as u8);
        let color_ptr = (addr + 1) as *mut u8;
        volatile_write_u8(color_ptr, color);
    }
}

fn cli() {
    unsafe { asm!("cli", options(nomem, nostack)); }
}

fn hlt() {
    unsafe { asm!("hlt", options(nomem, nostack)); }
}

fn _start() {
    cli();

    let green = vga_color(10, 0);
    vga_write(0, 0, 72, green);  // 'H'
    vga_write(1, 0, 105, green); // 'i'

    loop { hlt(); }
}
```

Build with:
```bash
glc build kernel.gl --freestanding --linker-script=kernel.ld -o kernel.elf
```

---

## Linker Script Example

```ld
OUTPUT_FORMAT(elf64-x86-64)
ENTRY(_start)

SECTIONS {
    . = 1M;

    .text BLOCK(4K) : ALIGN(4K) {
        *(.text .text.*)
    }

    .rodata BLOCK(4K) : ALIGN(4K) {
        *(.rodata .rodata.*)
    }

    .data BLOCK(4K) : ALIGN(4K) {
        *(.data .data.*)
    }

    .bss BLOCK(4K) : ALIGN(4K) {
        *(.bss .bss.*)
    }
}
```

---

## See Also

- `tests/kernel/mini_kernel.gl` - Integration test example
- `tests/kernel/kernel.ld` - Example linker script
