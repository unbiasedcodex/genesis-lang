# Genesis Language Reference

Complete reference for the Genesis programming language.

---

## Table of Contents

- [Primitive Types](#primitive-types)
- [Composite Types](#composite-types)
- [Variables and Constants](#variables-and-constants)
- [Functions](#functions)
- [Control Flow](#control-flow)
- [Generics and Traits](#generics-and-traits)
- [Async Programming](#async-programming)
- [Standard Library](#standard-library)
  - [I/O](#io)
  - [Vec\<T\>](#vect)
  - [String](#string)
  - [Option\<T\>](#optiont)
  - [Result\<T, E\>](#resultt-e)
  - [HashMap\<K, V\>](#hashmapk-v)
  - [HashSet\<T\>](#hashsett)
  - [Box\<T\>](#boxt)
  - [File and Directory](#file-and-directory)
  - [Time and Duration](#time-and-duration)
  - [Math](#math)
  - [Conversions](#conversions)
- [Modules](#modules)
- [CLI Reference](#cli-reference)

---

## Primitive Types

### Integers

```genesis
// Signed integers
let a: i8 = 127
let b: i16 = 32767
let c: i32 = 2147483647
let d: i64 = 9223372036854775807

// Unsigned integers
let e: u8 = 255
let f: u16 = 65535
let g: u32 = 4294967295
let h: u64 = 18446744073709551615
```

### Floating Point

```genesis
let x: f32 = 3.14
let y: f64 = 3.141592653589793
```

### Boolean and Character

```genesis
let flag: bool = true
let letter: char = 'A'
```

### String Literal

```genesis
let s: str = "hello"  // Immutable string literal
```

---

## Composite Types

### Arrays

Fixed-size, stack-allocated arrays.

```genesis
let arr: [i64; 5] = [1, 2, 3, 4, 5]
let first = arr[0]  // 1
```

### Tuples

Fixed-size, heterogeneous collections.

```genesis
let tuple: (i64, bool, str) = (42, true, "hello")
let x = tuple.0  // 42
let y = tuple.1  // true
```

### Structs

Named product types with fields.

```genesis
struct Point {
    x: i64,
    y: i64
}

let p = Point { x: 10, y: 20 }
println(p.x)  // 10
```

### Enums

Algebraic data types with variants.

```genesis
enum Shape {
    Circle(i64),           // Variant with data
    Rectangle(i64, i64),   // Variant with multiple values
    Empty                  // Variant without data
}

let s = Shape::Circle(5)
```

### Type Aliases

Create synonyms for existing types.

```genesis
type Meters = i64
type Callback = fn(i64) -> bool
type IntVec = Vec<i64>
type StringMap<V> = HashMap<String, V>

let distance: Meters = 100
let raw: i64 = distance  // OK: interchangeable
```

---

## Variables and Constants

### Variables

```genesis
let x: i64 = 42         // Immutable with type
let y = 42              // Type inferred
let mut z: i64 = 0      // Mutable
z = 10
```

### Global Constants

Compile-time constants available throughout the program.

```genesis
const MAX_SIZE: i64 = 1024
const PI: f64 = 3.14159265359
const DEBUG: bool = true
const NEGATIVE: i64 = -42

fn main() -> i32 {
    let buffer = MAX_SIZE * 2
    0
}
```

**Supported types**: `i64`, `f64`, `f32`, `bool`

---

## Functions

### Basic Functions

```genesis
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn greet(name: str) {
    println(name)
}

fn main() -> i32 {
    let result = add(10, 20)
    greet("World")
    0
}
```

### Generic Functions

```genesis
fn identity<T>(x: T) -> T {
    x
}

fn swap<A, B>(pair: (A, B)) -> (B, A) {
    (pair.1, pair.0)
}

let n = identity(42)      // Inferred as identity$i64
let b = identity(true)    // Inferred as identity$bool
```

### Closures

Anonymous functions that capture their environment.

```genesis
let multiplier: i64 = 10
let times = |x: i64| -> i64 { x * multiplier }
times(5)  // 50
```

### Impl Blocks

Methods defined on types.

```genesis
struct Counter {
    value: i64,
}

impl Counter {
    // Static method (constructor)
    fn new(v: i64) -> Counter {
        Counter { value: v }
    }

    // Instance method
    fn get(self) -> i64 {
        self.value
    }

    fn increment(self) -> Counter {
        Counter { value: self.value + 1 }
    }
}

let c = Counter::new(42)
let v = c.get()           // 42
let c2 = c.increment()
```

---

## Control Flow

### Conditionals

```genesis
fn classify(n: i64) -> str {
    if n < 0 {
        "negative"
    } else if n == 0 {
        "zero"
    } else {
        "positive"
    }
}
```

### Pattern Matching

Exhaustive matching with the `match` expression.

```genesis
fn describe(opt: Option<i64>) -> str {
    match opt {
        Some(n) => {
            if n > 0 { "positive" }
            else { "non-positive" }
        }
        None => "nothing"
    }
}

fn area(shape: Shape) -> i64 {
    match shape {
        Shape::Circle(r) => r * r * 3,
        Shape::Rectangle(w, h) => w * h,
        Shape::Empty => 0
        // Compiler error if a variant is missing
    }
}
```

### Loops

```genesis
// While loop
let mut i: i64 = 0
while i < 10 {
    println(i64::to_string(i))
    i = i + 1
}

// For loop with range
for x in 0..10 {
    println(i64::to_string(x))
}

// For loop with iterator
for item in Vec::iter(v) {
    println(i64::to_string(item))
}

// Infinite loop
loop {
    break  // Exit loop
}
```

### Labeled Loops

```genesis
'outer: for x in 0..10 {
    for y in 0..10 {
        if x * y > 50 {
            break 'outer     // Exit outer loop
        }
        if y == 5 {
            continue 'outer  // Next iteration of outer
        }
    }
}
```

---

## Generics and Traits

### Generic Structs

```genesis
struct Wrapper<T> {
    value: T
}

struct Pair<A, B> {
    first: A,
    second: B
}

let w = Wrapper { value: 42 }        // Wrapper$i64
let p = Pair { first: 1, second: true }  // Pair$i64$bool
```

### Traits

Define shared behavior.

```genesis
trait Speak {
    fn speak(&self) -> i64
}

struct Dog {}
struct Cat {}

impl Speak for Dog {
    fn speak(&self) -> i64 { 1 }
}

impl Speak for Cat {
    fn speak(&self) -> i64 { 2 }
}
```

### Trait Bounds

Constrain generic parameters.

```genesis
fn make_speak<T: Speak>(item: T) -> i64 {
    item.speak()
}

struct Container<T: Clone> {
    value: T
}
```

### Where Clauses

Alternative syntax for bounds.

```genesis
fn process<T>(x: T) -> T where T: Copy {
    x
}

struct Container<T> where T: Clone {
    value: T,
}

impl<T> Container<T> where T: Clone + Copy {
    fn get(self) -> T {
        self.value
    }
}
```

### Associated Types

Types defined within traits.

```genesis
trait Iterator {
    type Item;
    fn next(&self) -> Option<Self::Item>;
}

impl Iterator for Counter {
    type Item = i64;

    fn next(&self) -> Option<Self::Item> {
        // ...
    }
}
```

---

## Async Programming

### Async Functions

```genesis
async fn fetch_data() -> i64 {
    sleep_ms(100).await
    42
}

async fn compute() -> i64 {
    let a = fetch_data().await
    let b = fetch_data().await
    a + b
}
```

### Spawn

Run tasks concurrently.

```genesis
fn main() -> i64 {
    let handle = spawn(fetch_data())
    // Do other work...
    handle.await
}
```

### Channels

Communicate between tasks.

```genesis
let ch = channel(10)  // Buffer size 10
let tx = Channel::sender(ch)
let rx = Channel::receiver(ch)

spawn(async {
    Sender::try_send(tx, 42)
})

let opt = Receiver::try_recv(rx)
Option::unwrap_or(opt, 0)
```

### Combinators

```genesis
// Wait for first ready
select! {
    a = fast_task() => a,
    b = slow_task() => b,
}

// Wait for all
let (a, b, c) = join!(task1(), task2(), task3())
```

### TCP Networking

```genesis
async fn handle_client(stream: TcpStream) {
    let buf: [u8; 1024] = [0; 1024]
    let n = TcpStream::read(stream, buf).await
    TcpStream::write(stream, buf)
    TcpStream::close(stream)
}

fn main() -> i64 {
    let listener = TcpListener::bind("127.0.0.1", 8080)
    loop {
        let stream = TcpListener::accept(listener).await
        spawn(handle_client(stream))
    }
}
```

---

## Standard Library

### I/O

```genesis
print(x)           // stdout without newline
println(x)         // stdout with newline
eprint(x)          // stderr without newline
eprintln(x)        // stderr with newline
read_line()        // Read line from stdin -> String
```

### format!

String formatting macro.

```genesis
// Basic usage
let s = format!("Value: {}", 42)
let s = format!("{} + {} = {}", 1, 2, 3)

// Format specifiers
format!("{:04}", 7)        // "0007" - zero-padded
format!("{:.2}", 3.14159)  // "3.14" - precision
format!("{:08.2}", 3.14)   // "00003.14" - width + precision
format!("{:x}", 255)       // "ff" - lowercase hex
format!("{:X}", 255)       // "FF" - uppercase hex
format!("{:o}", 8)         // "10" - octal

// Escaped braces
format!("Use {{}} for placeholders")  // "Use {} for placeholders"
```

---

### Vec\<T\>

Dynamic array.

```genesis
// Creation
Vec::new()                // Empty vector
Vec::with_capacity(n)     // With initial capacity

// Modification
Vec::push(v, item)        // Add item to end
Vec::pop(v)               // Remove last -> Option<T>
Vec::clear(v)             // Remove all items

// Access
Vec::get(v, i)            // Get item at index -> T
Vec::set(v, i, val)       // Set item at index
Vec::first(v)             // First item -> Option<T>
Vec::last(v)              // Last item -> Option<T>
v[i]                      // Index syntax

// Properties
Vec::len(v)               // Length
Vec::capacity(v)          // Capacity
Vec::is_empty(v)          // Check if empty

// Iteration
Vec::iter(v)              // Create iterator
```

**VecIter\<T\> Methods:**

```genesis
VecIter::next(iter)           // Next element -> Option<T>
VecIter::count(iter)          // Count elements -> i64
VecIter::sum(iter)            // Sum elements -> i64

// Searching
VecIter::find(iter, pred)     // First match -> Option<T>
VecIter::any(iter, pred)      // Any matches -> bool
VecIter::all(iter, pred)      // All match -> bool

// Transformations
VecIter::map(iter, fn)        // Transform -> MapIter
VecIter::filter(iter, fn)     // Filter -> FilterIter
VecIter::enumerate(iter)      // Add indices

// Reduction
VecIter::fold(iter, init, fn) // Reduce to value
VecIter::for_each(iter, fn)   // Apply to each

// Collecting
MapIter::collect(iter)        // Collect to Vec
FilterIter::collect(iter)     // Collect to Vec
```

---

### String

Dynamic string.

```genesis
// Creation
String::new()             // Empty string
String::from("text")      // From literal

// Properties
String::len(s)            // Length in bytes
String::capacity(s)       // Capacity
String::is_empty(s)       // Check if empty
String::char_at(s, i)     // Byte at index
s[i]                      // Index syntax

// Modification
String::push(s, byte)     // Add byte
String::push_str(s, "x")  // Add string
String::concat(a, b)      // Concatenate -> String
String::substring(s, i, j)// Substring -> String
String::clear(s)          // Clear contents

// Search
String::contains(s, "x")      // Contains substring -> bool
String::starts_with(s, "x")   // Check prefix -> bool
String::ends_with(s, "x")     // Check suffix -> bool
String::find(s, "x")          // First index -> Option<i64>
String::rfind(s, "x")         // Last index -> Option<i64>

// Transform
String::to_uppercase(s)       // -> String
String::to_lowercase(s)       // -> String
String::trim(s)               // Remove whitespace -> String
String::trim_start(s)         // Remove leading whitespace
String::trim_end(s)           // Remove trailing whitespace
String::replace(s, "a", "b")  // Replace all -> String
String::repeat(s, n)          // Repeat n times -> String

// Split
String::split(s, ",")     // Split by delimiter -> Vec<String>
String::lines(s)          // Split by newlines -> Vec<String>
String::chars(s)          // Characters -> Vec<String>
String::bytes(s)          // Byte values -> Vec<i64>
```

---

### Option\<T\>

Represents optional values.

```genesis
// Creation
Some(x)                   // Value present
None                      // No value

// Predicates
Option::is_some(opt)      // -> bool
Option::is_none(opt)      // -> bool

// Extraction
Option::unwrap(opt)       // Get value (panic if None)
Option::unwrap_or(o, def) // Get value or default
Option::expect(o, msg)    // Get value (panic with msg if None)

// Transformation
Option::map(o, fn)        // Transform value -> Option<U>
Option::and_then(o, fn)   // Chain operations -> Option<U>
```

---

### Result\<T, E\>

Represents success or failure.

```genesis
// Creation
Ok(x)                     // Success
Err(e)                    // Failure

// Predicates
Result::is_ok(res)        // -> bool
Result::is_err(res)       // -> bool

// Extraction
Result::unwrap(res)       // Get value (panic if Err)
Result::unwrap_err(res)   // Get error (panic if Ok)
Result::unwrap_or(r, def) // Get value or default
Result::expect(r, msg)    // Get value (panic with msg)

// Transformation
Result::map(r, fn)        // Transform value -> Result<U, E>
Result::map_err(r, fn)    // Transform error -> Result<T, F>
Result::and_then(r, fn)   // Chain operations
```

---

### HashMap\<K, V\>

Key-value store. Keys: `i64` or `String`. Auto-resize at 75% load.

```genesis
// Creation
HashMap::new()            // Capacity 16
HashMap::with_capacity(n) // Custom capacity

// Modification
HashMap::insert(m, k, v)  // Insert or update
HashMap::remove(m, k)     // Remove -> Option<V>
HashMap::clear(m)         // Remove all

// Access
HashMap::get(m, k)        // Get value -> Option<V>
HashMap::contains_key(m, k) // Check key -> bool

// Properties
HashMap::len(m)           // Number of entries
HashMap::capacity(m)      // Capacity
HashMap::is_empty(m)      // Check if empty
```

---

### HashSet\<T\>

Unique values. Types: `i64` or `String`. Auto-resize at 75% load.

```genesis
// Creation
HashSet::new()            // Capacity 16
HashSet::with_capacity(n) // Custom capacity

// Modification
HashSet::insert(s, val)   // Insert value
HashSet::remove(s, val)   // Remove value
HashSet::clear(s)         // Remove all

// Access
HashSet::contains(s, val) // Check if exists -> bool

// Properties
HashSet::len(s)           // Number of values
HashSet::capacity(s)      // Capacity
HashSet::is_empty(s)      // Check if empty
```

---

### Box\<T\>

Heap-allocated value.

```genesis
let boxed: Box<i64> = Box::new(42)
let value = Box::deref(boxed)  // 42
```

---

### File and Directory

```genesis
// File operations
File::open(path)          // Open for reading -> Result<File, i64>
File::create(path)        // Create for writing -> Result<File, i64>
File::read_to_string(f)   // Read contents -> Result<String, i64>
File::write_string(f, s)  // Write string -> Result<i64, i64>
File::close(f)            // Close handle

// File metadata
File::exists(path)        // Check existence -> bool
File::size(path)          // Size in bytes -> Result<i64, i64>
File::is_file(path)       // Is regular file -> bool
File::is_dir(path)        // Is directory -> bool
File::remove(path)        // Delete file -> Result<(), i64>

// Directory operations
Dir::create(path)         // Create directory -> Result<(), i64>
Dir::create_all(path)     // Create recursively -> Result<(), i64>
Dir::remove(path)         // Remove empty dir -> Result<(), i64>
Dir::list(path)           // List contents -> Result<Vec<String>, i64>

// Path manipulation
Fs::path_join(a, b)       // Join paths -> String
Fs::path_parent(path)     // Parent dir -> Option<String>
Fs::path_filename(path)   // Filename -> Option<String>
Fs::path_extension(path)  // Extension -> Option<String>
```

**Error codes**: 1 = not found / permission denied

---

### Time and Duration

```genesis
// Current time
time::now_ms()            // Milliseconds -> i64
time::now_us()            // Microseconds -> i64
time::now_ns()            // Nanoseconds -> i64

// Elapsed time
time::elapsed_ms(start)   // Milliseconds since start -> i64
time::elapsed_us(start)   // Microseconds since start -> i64
time::elapsed_ns(start)   // Nanoseconds since start -> i64

// Duration type: { secs: i64, nanos: i64 }
Duration::from_secs(n)    // Create from seconds
Duration::from_millis(n)  // Create from milliseconds
Duration::from_micros(n)  // Create from microseconds
Duration::from_nanos(n)   // Create from nanoseconds

Duration::as_secs(d)      // Get seconds -> i64
Duration::as_millis(d)    // Get milliseconds -> i64
Duration::as_micros(d)    // Get microseconds -> i64
Duration::as_nanos(d)     // Get nanoseconds -> i64

Duration::add(a, b)       // Add durations -> Duration
Duration::sub(a, b)       // Subtract durations -> Duration
```

---

### Math

See [math.md](math.md) for complete documentation.

```genesis
// Constants
f64::PI()                 // 3.14159265358979
f64::E()                  // 2.71828182845904
f64::INFINITY()           // Positive infinity
f64::NEG_INFINITY()       // Negative infinity
f64::NAN()                // Not a Number

// Basic
f64::abs(x)               // Absolute value
f64::min(a, b)            // Minimum
f64::max(a, b)            // Maximum
i64::abs(x)               // Integer absolute value
i64::min(a, b)            // Integer minimum
i64::max(a, b)            // Integer maximum

// Powers and roots
f64::sqrt(x)              // Square root
f64::cbrt(x)              // Cube root
f64::pow(base, exp)       // Power
i64::pow(base, exp)       // Integer power
f64::hypot(x, y)          // sqrt(x² + y²)

// Trigonometry (radians)
f64::sin(x)               // Sine
f64::cos(x)               // Cosine
f64::tan(x)               // Tangent
f64::asin(x)              // Arc sine
f64::acos(x)              // Arc cosine
f64::atan(x)              // Arc tangent
f64::atan2(y, x)          // Two-argument arc tangent

// Hyperbolic
f64::sinh(x)              // Hyperbolic sine
f64::cosh(x)              // Hyperbolic cosine
f64::tanh(x)              // Hyperbolic tangent

// Exponential and logarithmic
f64::exp(x)               // e^x
f64::exp2(x)              // 2^x
f64::ln(x)                // Natural logarithm
f64::log2(x)              // Base-2 logarithm
f64::log10(x)             // Base-10 logarithm
f64::log(x, base)         // Custom base logarithm

// Rounding
f64::floor(x)             // Round down
f64::ceil(x)              // Round up
f64::round(x)             // Round to nearest
f64::trunc(x)             // Truncate toward zero

// Special value checks
f64::is_nan(x)            // Is NaN -> bool
f64::is_infinite(x)       // Is ±infinity -> bool
f64::is_finite(x)         // Is finite -> bool
```

---

### Conversions

```genesis
// Integer to String
i64::to_string(n)         // -> String
i32::to_string(n)         // -> String
bool::to_string(b)        // -> "true" or "false"

// String to Integer
i64::parse(s)             // -> Result<i64, i64>
i32::parse(s)             // -> Result<i32, i64>
// Error: 0 = success, 1 = empty, 2 = invalid chars
```

---

## Modules

### Inline Modules

```genesis
mod math {
    pub fn square(x: i64) -> i64 {
        x * x
    }

    fn internal(x: i64) -> i64 {  // Private
        x + 1
    }
}

fn main() -> i32 {
    println(i64::to_string(math::square(5)))  // 25
    // math::internal(5)  // Error: private function
    0
}
```

### External Modules

```genesis
// main.gl
mod utils;  // Loads utils.gl from same directory

fn main() -> i32 {
    utils::helper()
    0
}
```

```genesis
// utils.gl
pub fn helper() -> i32 {
    println("Helper called")
    0
}
```

---

## CLI Reference

```bash
# Compile to native executable
glc build file.gl --native -o output

# Optimization level (0-3)
glc build file.gl --native -O3 -o output

# Emit intermediate representations
glc build file.gl --emit-ir      # Genesis IR
glc build file.gl --emit-llvm    # LLVM IR

# Check without compiling
glc check file.gl

# Parse only (show AST)
glc parse file.gl

# Tokenize only
glc tokenize file.gl
```
