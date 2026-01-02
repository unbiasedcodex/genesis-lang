//! Abstract Syntax Tree (AST) for Genesis Lang
//!
//! The AST represents the structure of a Genesis program after parsing.
//! Each node in the AST corresponds to a syntactic construct in the language.

use crate::span::Span;

/// A unique identifier for AST nodes
pub type NodeId = u32;

/// A complete Genesis program (compilation unit)
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
    pub span: Span,
}

/// Top-level items in a Genesis program
#[derive(Debug, Clone)]
pub enum Item {
    /// Function definition: `fn foo(x: i32) -> i32 { ... }`
    Function(FnDef),

    /// Struct definition: `struct Foo { x: i32, y: i32 }`
    Struct(StructDef),

    /// Enum definition: `enum Option<T> { Some(T), None }`
    Enum(EnumDef),

    /// Impl block: `impl Foo { ... }`
    Impl(ImplDef),

    /// Trait definition: `trait Comparable { ... }`
    Trait(TraitDef),

    /// Type alias: `type Result<T> = Result<T, Error>`
    TypeAlias(TypeAlias),

    /// Constant: `const MAX: i32 = 100`
    Const(ConstDef),

    /// Actor definition: `actor Counter { ... }`
    Actor(ActorDef),

    /// Use declaration: `use std::io`
    Use(UseDef),

    /// Module declaration: `mod foo`
    Mod(ModDef),

    /// Macro definition: `macro vec! { ... }`
    Macro(MacroDef),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Function(f) => f.span,
            Item::Struct(s) => s.span,
            Item::Enum(e) => e.span,
            Item::Impl(i) => i.span,
            Item::Trait(t) => t.span,
            Item::TypeAlias(t) => t.span,
            Item::Macro(m) => m.span,
            Item::Const(c) => c.span,
            Item::Actor(a) => a.span,
            Item::Use(u) => u.span,
            Item::Mod(m) => m.span,
        }
    }
}

// ============ Definitions ============

/// Function definition
#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub is_pub: bool,
    pub is_async: bool,
    pub span: Span,
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Param {
    pub name: Ident,
    pub ty: Type,
    pub is_mut: bool,
    pub span: Span,
}

/// Struct definition
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub fields: Vec<Field>,
    pub is_pub: bool,
    pub span: Span,
}

/// Struct field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Ident,
    pub ty: Type,
    pub is_pub: bool,
    pub span: Span,
}

/// Enum definition
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub variants: Vec<Variant>,
    pub is_pub: bool,
    pub span: Span,
}

/// Enum variant
#[derive(Debug, Clone)]
pub struct Variant {
    pub name: Ident,
    pub kind: VariantKind,
    pub span: Span,
}

/// Kind of enum variant
#[derive(Debug, Clone)]
pub enum VariantKind {
    /// Unit variant: `None`
    Unit,
    /// Tuple variant: `Some(T)`
    Tuple(Vec<Type>),
    /// Struct variant: `Point { x: i32, y: i32 }`
    Struct(Vec<Field>),
}

/// Impl block
#[derive(Debug, Clone)]
pub struct ImplDef {
    pub generics: Option<Generics>,
    pub trait_: Option<Type>,
    pub self_type: Type,
    pub items: Vec<ImplItem>,
    pub span: Span,
}

/// Item inside an impl block
#[derive(Debug, Clone)]
pub enum ImplItem {
    Function(FnDef),
    Const(ConstDef),
    TypeAlias(TypeAlias),
}

/// Trait definition
#[derive(Debug, Clone)]
pub struct TraitDef {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub super_traits: Vec<Type>,
    pub items: Vec<TraitItem>,
    pub is_pub: bool,
    pub span: Span,
}

/// Item inside a trait
#[derive(Debug, Clone)]
pub enum TraitItem {
    Function(FnSignature),
    Const(ConstDef),
    TypeAlias(TypeAlias),
    /// Associated type declaration: `type Item;` or `type Item: Bound;`
    AssociatedType(AssociatedTypeDef),
}

/// Associated type definition in a trait
/// e.g., `type Item;` or `type Item: Clone + Debug;`
#[derive(Debug, Clone)]
pub struct AssociatedTypeDef {
    pub name: Ident,
    /// Optional bounds on the associated type
    pub bounds: Vec<Type>,
    /// Default type (if any) - rarely used
    pub default: Option<Type>,
    pub span: Span,
}

/// Function signature (for trait methods)
#[derive(Debug, Clone)]
pub struct FnSignature {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub span: Span,
}

/// Type alias
#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub ty: Type,
    pub is_pub: bool,
    pub span: Span,
}

/// Constant definition
#[derive(Debug, Clone)]
pub struct ConstDef {
    pub name: Ident,
    pub ty: Option<Type>,
    pub value: Expr,
    pub is_pub: bool,
    pub span: Span,
}

/// Actor definition (Genesis-specific)
#[derive(Debug, Clone)]
pub struct ActorDef {
    pub name: Ident,
    pub generics: Option<Generics>,
    pub state: Vec<Field>,
    pub receive: Vec<MessageHandler>,
    pub is_pub: bool,
    pub span: Span,
}

/// Message handler in actor
#[derive(Debug, Clone)]
pub struct MessageHandler {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

/// Use declaration
#[derive(Debug, Clone)]
pub struct UseDef {
    pub path: Path,
    pub alias: Option<Ident>,
    pub span: Span,
}

/// Module declaration
#[derive(Debug, Clone)]
pub struct ModDef {
    pub name: Ident,
    pub items: Option<Vec<Item>>,
    pub is_pub: bool,
    pub span: Span,
}

// ============ Macros ============

/// Macro definition
/// ```genesis
/// macro vec {
///     () => { Vec::new() };
///     ($($x:expr),*) => { ... };
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: Ident,
    pub rules: Vec<MacroRule>,
    pub is_pub: bool,
    pub span: Span,
}

/// A single macro rule: pattern => expansion
#[derive(Debug, Clone)]
pub struct MacroRule {
    pub pattern: MacroPattern,
    pub expansion: MacroExpansion,
    pub span: Span,
}

/// Macro pattern (left side of =>)
#[derive(Debug, Clone)]
pub struct MacroPattern {
    pub tokens: Vec<MacroToken>,
    pub span: Span,
}

/// Macro expansion (right side of =>)
#[derive(Debug, Clone)]
pub struct MacroExpansion {
    pub tokens: Vec<MacroToken>,
    pub span: Span,
}

/// Token in a macro pattern or expansion
#[derive(Debug, Clone)]
pub enum MacroToken {
    /// Regular token (keyword, punctuation)
    Token(crate::token::TokenKind, Span),

    /// Identifier token
    Ident(String, Span),

    /// Integer literal with value
    IntLit(i64, Span),

    /// Float literal with value
    FloatLit(f64, Span),

    /// String literal with value
    StrLit(String, Span),

    /// Char literal with value
    CharLit(char, Span),

    /// Captured variable: $name:kind
    Capture {
        name: String,
        kind: MacroCaptureKind,
        span: Span,
    },

    /// Repetition: $(...)*  or $(...)+  or $(...)?
    Repetition {
        tokens: Vec<MacroToken>,
        separator: Option<Box<MacroToken>>,
        kind: MacroRepKind,
        span: Span,
    },

    /// Nested group: (...) or [...] or {...}
    Group {
        delimiter: MacroDelimiter,
        tokens: Vec<MacroToken>,
        span: Span,
    },
}

/// Kind of macro capture ($x:expr, $t:ty, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroCaptureKind {
    /// Expression: $e:expr
    Expr,
    /// Type: $t:ty
    Ty,
    /// Identifier: $i:ident
    Ident,
    /// Pattern: $p:pat
    Pat,
    /// Statement: $s:stmt
    Stmt,
    /// Block: $b:block
    Block,
    /// Item: $i:item
    Item,
    /// Literal: $l:literal
    Literal,
    /// Token tree: $tt:tt
    Tt,
    /// Path: $p:path (e.g., std::collections::HashMap)
    Path,
}

/// Repetition kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroRepKind {
    /// Zero or more: *
    ZeroOrMore,
    /// One or more: +
    OneOrMore,
    /// Zero or one: ?
    ZeroOrOne,
}

/// Delimiter type for groups
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroDelimiter {
    Paren,   // ( )
    Bracket, // [ ]
    Brace,   // { }
}

/// Macro invocation: `name!(args)`
#[derive(Debug, Clone)]
pub struct MacroInvocation {
    pub name: Ident,
    pub delimiter: MacroDelimiter,
    pub tokens: Vec<MacroToken>,
    pub span: Span,
}

// ============ Types ============

/// Type expression
#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

/// Kind of type
#[derive(Debug, Clone)]
pub enum TypeKind {
    /// Named type: `i32`, `String`, `Vec<T>`
    Path(Path),

    /// Reference: `&T`, `&mut T`
    Reference { mutable: bool, inner: Box<Type> },

    /// Array: `[T; N]`
    Array { element: Box<Type>, size: Box<Expr> },

    /// Slice: `[T]`
    Slice { element: Box<Type> },

    /// Tuple: `(T1, T2, T3)`
    Tuple(Vec<Type>),

    /// Function pointer: `fn(i32, i32) -> i32`
    FnPtr {
        params: Vec<Type>,
        return_type: Option<Box<Type>>,
    },

    /// Never type: `!`
    Never,

    /// Inferred type: `_`
    Infer,

    /// Option type: `Option<T>`
    Option(Box<Type>),

    /// Result type: `Result<T, E>`
    Result { ok: Box<Type>, err: Box<Type> },

    /// Self type reference
    SelfType,

    /// Associated type projection: `Self::Item` or `T::Item`
    Projection {
        base: Box<Type>,
        assoc_name: String,
    },
}

// ============ Expressions ============

/// Expression
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

/// Kind of expression
#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Literal: `42`, `"hello"`, `true`
    Literal(Literal),

    /// Variable/path: `x`, `std::io::read`
    Path(Path),

    /// Binary operation: `a + b`
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation: `-x`, `!flag`, `*ptr`
    Unary { op: UnaryOp, operand: Box<Expr> },

    /// Function call: `foo(x, y)`
    Call { func: Box<Expr>, args: Vec<Expr> },

    /// Method call: `x.foo(y)`
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        args: Vec<Expr>,
    },

    /// Field access: `x.field`
    Field { object: Box<Expr>, field: Ident },

    /// Index: `arr[i]`
    Index { object: Box<Expr>, index: Box<Expr> },

    /// Struct literal: `Point { x: 1, y: 2 }`
    Struct {
        path: Path,
        fields: Vec<(Ident, Expr)>,
    },

    /// Array literal: `[1, 2, 3]`
    Array(Vec<Expr>),

    /// Tuple: `(a, b, c)`
    Tuple(Vec<Expr>),

    /// Block: `{ stmt; stmt; expr }`
    Block(Block),

    /// If expression: `if cond { ... } else { ... }`
    If {
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
    },

    /// Match expression: `match x { ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// Loop: `loop { ... }`
    Loop { body: Block, label: Option<Ident> },

    /// While loop: `while cond { ... }`
    While {
        condition: Box<Expr>,
        body: Block,
        label: Option<Ident>,
    },

    /// For loop: `for x in iter { ... }`
    For {
        pattern: Pattern,
        iterable: Box<Expr>,
        body: Block,
        label: Option<Ident>,
    },

    /// Break: `break`, `break 'label`, `break value`
    Break {
        label: Option<Ident>,
        value: Option<Box<Expr>>,
    },

    /// Continue: `continue`, `continue 'label`
    Continue { label: Option<Ident> },

    /// Return: `return`, `return value`
    Return { value: Option<Box<Expr>> },

    /// Closure: `|x, y| x + y`
    Closure {
        params: Vec<Param>,
        body: Box<Expr>,
    },

    /// Reference: `&x`, `&mut x`
    Ref { mutable: bool, operand: Box<Expr> },

    /// Dereference: `*ptr`
    Deref { operand: Box<Expr> },

    /// Type cast: `x as i64`
    Cast { expr: Box<Expr>, ty: Type },

    /// Range: `a..b`, `a..=b`
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },

    /// Await: `future.await`
    Await { operand: Box<Expr> },

    /// Try: `result?`
    Try { operand: Box<Expr> },

    /// Spawn actor: `spawn Counter`
    Spawn { actor: Path, args: Vec<Expr> },

    /// Spawn async task: `spawn(future_expr)`
    SpawnTask { future: Box<Expr> },

    /// Send message: `actor <- Message`
    Send {
        target: Box<Expr>,
        message: Box<Expr>,
    },

    /// Send and receive reply: `actor <-? Message`
    SendRecv {
        target: Box<Expr>,
        message: Box<Expr>,
    },

    /// Reply in actor: `reply value`
    Reply { value: Box<Expr> },

    /// Select expression: `select! { x = future1 => body1, y = future2 => body2 }`
    /// Polls all futures, returns when first completes
    Select { arms: Vec<SelectArm> },

    /// Join expression: `join!(future1, future2, future3)`
    /// Polls all futures, waits for all to complete, returns tuple
    Join { futures: Vec<Expr> },

    /// Assignment: `x = value`
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },

    /// Compound assignment: `x += value`
    AssignOp {
        op: BinaryOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },

    /// Macro invocation: `vec![1, 2, 3]`
    MacroCall(MacroInvocation),
}

/// Block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Box<Expr>>,
    pub span: Span,
}

/// Statement
#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

/// Kind of statement
#[derive(Debug, Clone)]
pub enum StmtKind {
    /// Let binding: `let x = value`, `let mut x: Type = value`
    Let {
        pattern: Pattern,
        ty: Option<Type>,
        value: Option<Expr>,
    },

    /// Expression statement: `expr;`
    Expr(Expr),

    /// Item statement (local function, struct, etc)
    Item(Item),
}

/// Match arm
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

/// Select arm for async combinators
/// `binding = future => body`
#[derive(Debug, Clone)]
pub struct SelectArm {
    /// Variable to bind the result: `x` in `x = future => body`
    pub binding: Ident,
    /// The future expression to poll
    pub future: Expr,
    /// The body to execute when this future completes first
    pub body: Expr,
    pub span: Span,
}

// ============ Patterns ============

/// Pattern for matching
#[derive(Debug, Clone)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

/// Kind of pattern
#[derive(Debug, Clone)]
pub enum PatternKind {
    /// Wildcard: `_`
    Wildcard,

    /// Binding: `x`, `mut x`
    Ident { name: Ident, mutable: bool },

    /// Literal: `42`, `"hello"`
    Literal(Literal),

    /// Tuple: `(a, b, c)`
    Tuple(Vec<Pattern>),

    /// Struct: `Point { x, y }`
    Struct {
        path: Path,
        fields: Vec<(Ident, Pattern)>,
        rest: bool,
    },

    /// Enum variant: `Some(x)`, `None`
    Enum {
        path: Path,
        fields: Vec<Pattern>,
    },

    /// Or pattern: `A | B | C`
    Or(Vec<Pattern>),

    /// Range: `1..=10`
    Range {
        start: Option<Box<Pattern>>,
        end: Option<Box<Pattern>>,
        inclusive: bool,
    },

    /// Reference: `&x`, `&mut x`
    Ref { mutable: bool, pattern: Box<Pattern> },
}

// ============ Generics ============

/// Generic parameters
#[derive(Debug, Clone)]
pub struct Generics {
    pub params: Vec<GenericParam>,
    pub where_clause: Option<WhereClause>,
    pub span: Span,
}

/// Generic parameter
#[derive(Debug, Clone)]
pub struct GenericParam {
    pub name: Ident,
    pub bounds: Vec<Type>,
    pub default: Option<Type>,
    pub span: Span,
}

/// Where clause
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
    pub span: Span,
}

/// Where predicate
#[derive(Debug, Clone)]
pub struct WherePredicate {
    pub ty: Type,
    pub bounds: Vec<Type>,
    pub span: Span,
}

// ============ Primitives ============

/// Identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }
}

/// Path: `std::io::read`
#[derive(Debug, Clone)]
pub struct Path {
    pub segments: Vec<PathSegment>,
    pub span: Span,
}

/// Path segment
#[derive(Debug, Clone)]
pub struct PathSegment {
    pub ident: Ident,
    pub generics: Option<Vec<Type>>,
}

/// Literal value
#[derive(Debug, Clone)]
pub enum Literal {
    Int(i128),
    Float(f64),
    String(String),
    Char(char),
    Bool(bool),
}

/// Binary operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    Deref,
    Ref,
    RefMut,
}
