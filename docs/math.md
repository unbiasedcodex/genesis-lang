# Genesis Math Module

> Complete reference for mathematical functions in Genesis Lang

---

## Overview

Genesis provides a comprehensive set of mathematical functions through integration with the C math library (libm). All functions are available as associated functions on primitive types (`f64::`, `f32::`, `i64::`, `i32::`).

**Implementation:** Functions are declared as external symbols and linked with `-lm` during compilation.

---

## Constants

Mathematical constants are accessed as zero-argument functions:

```genesis
fn main() -> i64 {
    let pi: f64 = f64::PI();           // 3.141592653589793
    let e: f64 = f64::E();             // 2.718281828459045
    let inf: f64 = f64::INFINITY();    // Positive infinity
    let neg_inf: f64 = f64::NEG_INFINITY();  // Negative infinity
    let nan: f64 = f64::NAN();         // Not a Number

    println(pi);   // 3.14159
    println(e);    // 2.71828
    println(inf);  // inf
    0
}
```

| Constant | Value | Description |
|----------|-------|-------------|
| `f64::PI()` | 3.141592653589793 | Ratio of circumference to diameter |
| `f64::E()` | 2.718281828459045 | Euler's number, base of natural log |
| `f64::INFINITY()` | +∞ | Positive infinity |
| `f64::NEG_INFINITY()` | -∞ | Negative infinity |
| `f64::NAN()` | NaN | Not a Number (undefined result) |

---

## Basic Operations

### Absolute Value

```genesis
let a: f64 = f64::abs(-3.5);    // 3.5
let b: i64 = i64::abs(-42);     // 42
let c: i32 = i32::abs(-100);    // 100
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::abs(x)` | `fn(f64) -> f64` | Absolute value of float |
| `f32::abs(x)` | `fn(f32) -> f32` | Absolute value of float32 |
| `i64::abs(x)` | `fn(i64) -> i64` | Absolute value of integer |
| `i32::abs(x)` | `fn(i32) -> i32` | Absolute value of int32 |

### Minimum and Maximum

```genesis
let min_f: f64 = f64::min(3.0, 5.0);   // 3.0
let max_f: f64 = f64::max(3.0, 5.0);   // 5.0
let min_i: i64 = i64::min(10, 20);     // 10
let max_i: i64 = i64::max(10, 20);     // 20
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::min(a, b)` | `fn(f64, f64) -> f64` | Smaller of two floats |
| `f64::max(a, b)` | `fn(f64, f64) -> f64` | Larger of two floats |
| `i64::min(a, b)` | `fn(i64, i64) -> i64` | Smaller of two integers |
| `i64::max(a, b)` | `fn(i64, i64) -> i64` | Larger of two integers |
| `i32::min(a, b)` | `fn(i32, i32) -> i32` | Smaller of two int32s |
| `i32::max(a, b)` | `fn(i32, i32) -> i32` | Larger of two int32s |

---

## Powers and Roots

```genesis
fn main() -> i64 {
    // Square root
    let sqrt_16: f64 = f64::sqrt(16.0);    // 4.0

    // Cube root
    let cbrt_27: f64 = f64::cbrt(27.0);    // 3.0

    // Power (floating point)
    let pow_f: f64 = f64::pow(2.0, 10.0);  // 1024.0

    // Power (integer)
    let pow_i: i64 = i64::pow(2, 10);      // 1024

    // Hypotenuse: sqrt(x² + y²)
    let hyp: f64 = f64::hypot(3.0, 4.0);   // 5.0

    0
}
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::sqrt(x)` | `fn(f64) -> f64` | Square root √x |
| `f64::cbrt(x)` | `fn(f64) -> f64` | Cube root ∛x |
| `f64::pow(base, exp)` | `fn(f64, f64) -> f64` | base^exp |
| `i64::pow(base, exp)` | `fn(i64, i64) -> i64` | Integer power |
| `i32::pow(base, exp)` | `fn(i32, i32) -> i32` | Int32 power |
| `f64::hypot(x, y)` | `fn(f64, f64) -> f64` | √(x² + y²) without overflow |

**Note:** `f64::hypot` is preferred over `f64::sqrt(x*x + y*y)` as it handles overflow/underflow correctly.

---

## Trigonometry

All trigonometric functions use **radians**, not degrees.

```genesis
fn main() -> i64 {
    let pi: f64 = f64::PI();

    // Basic trig
    let sin_0: f64 = f64::sin(0.0);         // 0.0
    let cos_0: f64 = f64::cos(0.0);         // 1.0
    let sin_pi_2: f64 = f64::sin(pi / 2.0); // 1.0

    // Inverse trig
    let angle: f64 = f64::asin(1.0);        // π/2 ≈ 1.5708

    // Two-argument arctangent (preserves quadrant)
    let theta: f64 = f64::atan2(1.0, 1.0);  // π/4 ≈ 0.7854

    0
}
```

### Basic Trigonometric Functions

| Function | Signature | Description | Range |
|----------|-----------|-------------|-------|
| `f64::sin(x)` | `fn(f64) -> f64` | Sine | [-1, 1] |
| `f64::cos(x)` | `fn(f64) -> f64` | Cosine | [-1, 1] |
| `f64::tan(x)` | `fn(f64) -> f64` | Tangent | (-∞, +∞) |

### Inverse Trigonometric Functions

| Function | Signature | Description | Domain | Range |
|----------|-----------|-------------|--------|-------|
| `f64::asin(x)` | `fn(f64) -> f64` | Arc sine | [-1, 1] | [-π/2, π/2] |
| `f64::acos(x)` | `fn(f64) -> f64` | Arc cosine | [-1, 1] | [0, π] |
| `f64::atan(x)` | `fn(f64) -> f64` | Arc tangent | (-∞, +∞) | (-π/2, π/2) |
| `f64::atan2(y, x)` | `fn(f64, f64) -> f64` | Two-arg arc tangent | any | (-π, π] |

**Tip:** Use `atan2(y, x)` instead of `atan(y/x)` to get the correct quadrant and avoid division by zero.

### Converting Degrees ↔ Radians

```genesis
fn deg_to_rad(deg: f64) -> f64 {
    deg * f64::PI() / 180.0
}

fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / f64::PI()
}
```

---

## Hyperbolic Functions

```genesis
let sinh_1: f64 = f64::sinh(1.0);  // 1.1752...
let cosh_1: f64 = f64::cosh(1.0);  // 1.5430...
let tanh_1: f64 = f64::tanh(1.0);  // 0.7615...
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::sinh(x)` | `fn(f64) -> f64` | Hyperbolic sine: (e^x - e^-x) / 2 |
| `f64::cosh(x)` | `fn(f64) -> f64` | Hyperbolic cosine: (e^x + e^-x) / 2 |
| `f64::tanh(x)` | `fn(f64) -> f64` | Hyperbolic tangent: sinh(x) / cosh(x) |

---

## Exponential and Logarithmic

```genesis
fn main() -> i64 {
    let e: f64 = f64::E();

    // Exponential
    let exp_1: f64 = f64::exp(1.0);      // e ≈ 2.71828
    let exp2_10: f64 = f64::exp2(10.0);  // 2^10 = 1024.0

    // Natural logarithm (base e)
    let ln_e: f64 = f64::ln(e);          // 1.0

    // Common logarithms
    let log2_1024: f64 = f64::log2(1024.0);   // 10.0
    let log10_1000: f64 = f64::log10(1000.0); // 3.0

    // Logarithm with custom base
    let log_8_base_2: f64 = f64::log(8.0, 2.0);  // 3.0

    0
}
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::exp(x)` | `fn(f64) -> f64` | e^x |
| `f64::exp2(x)` | `fn(f64) -> f64` | 2^x |
| `f64::ln(x)` | `fn(f64) -> f64` | Natural logarithm (base e) |
| `f64::log2(x)` | `fn(f64) -> f64` | Base-2 logarithm |
| `f64::log10(x)` | `fn(f64) -> f64` | Base-10 logarithm |
| `f64::log(x, base)` | `fn(f64, f64) -> f64` | Logarithm with custom base |

**Note:** `f64::log(x, base)` is computed as `ln(x) / ln(base)`.

---

## Rounding

```genesis
fn main() -> i64 {
    let x: f64 = 3.7;
    let y: f64 = -3.7;

    // Floor: round toward -∞
    println(f64::floor(x));   // 3.0
    println(f64::floor(y));   // -4.0

    // Ceil: round toward +∞
    println(f64::ceil(x));    // 4.0
    println(f64::ceil(y));    // -3.0

    // Round: round to nearest (half away from zero)
    println(f64::round(3.5)); // 4.0
    println(f64::round(2.5)); // 3.0

    // Trunc: round toward zero
    println(f64::trunc(x));   // 3.0
    println(f64::trunc(y));   // -3.0

    0
}
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::floor(x)` | `fn(f64) -> f64` | Round toward negative infinity |
| `f64::ceil(x)` | `fn(f64) -> f64` | Round toward positive infinity |
| `f64::round(x)` | `fn(f64) -> f64` | Round to nearest integer |
| `f64::trunc(x)` | `fn(f64) -> f64` | Round toward zero |

### Comparison Table

| x | floor | ceil | round | trunc |
|---|-------|------|-------|-------|
| 3.7 | 3.0 | 4.0 | 4.0 | 3.0 |
| 3.2 | 3.0 | 4.0 | 3.0 | 3.0 |
| -3.7 | -4.0 | -3.0 | -4.0 | -3.0 |
| -3.2 | -4.0 | -3.0 | -3.0 | -3.0 |

---

## Special Value Checks

```genesis
fn main() -> i64 {
    let nan: f64 = f64::NAN();
    let inf: f64 = f64::INFINITY();
    let normal: f64 = 5.0;

    // Check for NaN
    println(f64::is_nan(nan));       // true
    println(f64::is_nan(normal));    // false

    // Check for infinity
    println(f64::is_infinite(inf));  // true
    println(f64::is_infinite(normal)); // false

    // Check for finite (not NaN and not infinite)
    println(f64::is_finite(normal)); // true
    println(f64::is_finite(nan));    // false
    println(f64::is_finite(inf));    // false

    0
}
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `f64::is_nan(x)` | `fn(f64) -> bool` | True if x is NaN |
| `f64::is_infinite(x)` | `fn(f64) -> bool` | True if x is ±∞ |
| `f64::is_finite(x)` | `fn(f64) -> bool` | True if x is neither NaN nor infinite |

**Important:** NaN has the property that `NaN != NaN`. Use `f64::is_nan()` to check for NaN, not equality comparison.

---

## Float Arithmetic

Genesis supports full floating-point arithmetic with proper type handling:

```genesis
fn main() -> i64 {
    let pi: f64 = f64::PI();
    let two: f64 = 2.0;

    // Basic arithmetic on floats
    let sum: f64 = pi + two;        // fadd
    let diff: f64 = pi - two;       // fsub
    let prod: f64 = pi * two;       // fmul
    let quot: f64 = pi / two;       // fdiv

    // Negation
    let neg: f64 = -pi;             // fneg

    // Comparisons
    let eq: bool = pi == two;       // fcmp oeq
    let lt: bool = two < pi;        // fcmp olt

    println(sum);   // 5.14159...
    println(quot);  // 1.57079...

    0
}
```

---

## Complete Example

```genesis
// Calculate the area and circumference of a circle
fn circle_area(radius: f64) -> f64 {
    f64::PI() * f64::pow(radius, 2.0)
}

fn circle_circumference(radius: f64) -> f64 {
    2.0 * f64::PI() * radius
}

// Calculate distance between two points
fn distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx: f64 = x2 - x1;
    let dy: f64 = y2 - y1;
    f64::hypot(dx, dy)
}

// Normalize an angle to [0, 2π)
fn normalize_angle(angle: f64) -> f64 {
    let two_pi: f64 = 2.0 * f64::PI();
    let normalized: f64 = angle - f64::floor(angle / two_pi) * two_pi;
    if normalized < 0.0 {
        normalized + two_pi
    } else {
        normalized
    }
}

fn main() -> i64 {
    println("=== Circle with radius 5 ===");
    let r: f64 = 5.0;
    println("Area:");
    println(circle_area(r));           // ~78.54
    println("Circumference:");
    println(circle_circumference(r));  // ~31.42

    println("");
    println("=== Distance (0,0) to (3,4) ===");
    println(distance(0.0, 0.0, 3.0, 4.0));  // 5.0

    println("");
    println("=== Normalized angles ===");
    let pi: f64 = f64::PI();
    println(normalize_angle(3.0 * pi));     // ~3.14 (π)
    println(normalize_angle(-pi / 2.0));    // ~4.71 (3π/2)

    0
}
```

---

## Implementation Notes

### Linkage

Math functions are implemented via libm and linked automatically with `-lm`. The compiler declares external functions:

```
// In generated LLVM IR
declare double @sin(double)
declare double @cos(double)
declare double @sqrt(double)
// ... etc
```

### Float Operations

Binary operations on floats use LLVM float instructions:
- `fadd` - Floating-point addition
- `fsub` - Floating-point subtraction
- `fmul` - Floating-point multiplication
- `fdiv` - Floating-point division
- `fneg` - Floating-point negation
- `fcmp` - Floating-point comparison

### NaN Detection

`f64::is_nan()` uses the LLVM `fcmp uno` (unordered) predicate, which returns true if either operand is NaN.

### Type Coercion

Integer literals in float contexts are automatically promoted:

```genesis
let x: f64 = 2.0;      // Float literal
let y: f64 = x + 1.0;  // OK: 1.0 is float
// Note: integer literals require explicit .0 suffix for float operations
```

---

## Function Reference

### f64 Functions (50+)

| Category | Functions |
|----------|-----------|
| Constants | `PI`, `E`, `INFINITY`, `NEG_INFINITY`, `NAN` |
| Basic | `abs`, `min`, `max` |
| Powers | `sqrt`, `cbrt`, `pow`, `hypot` |
| Trig | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Exp/Log | `exp`, `exp2`, `ln`, `log2`, `log10`, `log` |
| Rounding | `floor`, `ceil`, `round`, `trunc` |
| Checks | `is_nan`, `is_infinite`, `is_finite` |

### i64 Functions

| Function | Description |
|----------|-------------|
| `abs` | Absolute value |
| `min` | Minimum of two |
| `max` | Maximum of two |
| `pow` | Integer power |

### i32 Functions

| Function | Description |
|----------|-------------|
| `abs` | Absolute value |
| `min` | Minimum of two |
| `max` | Maximum of two |
| `pow` | Integer power |

---

*Documentation for Genesis Lang Math Module*

