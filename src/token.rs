//! Token definitions for Genesis Lang
//!
//! This module defines all the tokens that the lexer can produce.

use crate::span::Span;
use logos::Logos;
use std::fmt;

/// A token produced by the lexer
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Get the text of this token from source
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        self.span.text(source)
    }
}

/// All possible token types in Genesis Lang
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]  // Skip whitespace
#[logos(skip r"//[^\n]*")]      // Skip line comments
pub enum TokenKind {
    // ============ Literals ============

    /// Integer literal: 42, 0xFF, 0b1010, 0o77
    #[regex(r"[0-9][0-9_]*", priority = 2)]
    #[regex(r"0[xX][0-9a-fA-F][0-9a-fA-F_]*")]
    #[regex(r"0[bB][01][01_]*")]
    #[regex(r"0[oO][0-7][0-7_]*")]
    IntLiteral,

    /// Float literal: 3.14, 1e10, 2.5e-3
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9][0-9_]*)?")]
    #[regex(r"[0-9][0-9_]*[eE][+-]?[0-9][0-9_]*")]
    FloatLiteral,

    /// String literal: "hello", "with \"escapes\""
    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,

    /// Character literal: 'a', '\n'
    #[regex(r"'([^'\\]|\\.)'")]
    CharLiteral,

    /// Label/Lifetime: 'outer, 'loop1 (used for labeled loops)
    #[regex(r"'[a-zA-Z_][a-zA-Z0-9_]*")]
    Label,

    /// Boolean literal
    #[token("true")]
    True,
    #[token("false")]
    False,

    // ============ Keywords ============

    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
    #[token("const")]
    Const,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("impl")]
    Impl,
    #[token("trait")]
    Trait,
    #[token("type")]
    Type,
    #[token("pub")]
    Pub,
    #[token("use")]
    Use,
    #[token("mod")]
    Mod,
    #[token("self")]
    SelfValue,
    #[token("Self")]
    SelfType,
    #[token("as")]
    As,
    #[token("in")]
    In,
    #[token("where")]
    Where,
    #[token("dyn")]
    Dyn,

    // Genesis-specific keywords
    #[token("actor")]
    Actor,
    #[token("receive")]
    Receive,
    #[token("spawn")]
    Spawn,
    #[token("reply")]
    Reply,
    #[token("await")]
    Await,
    #[token("async")]
    Async,
    #[token("select")]
    Select,
    #[token("join")]
    Join,

    // Macro keywords
    #[token("macro")]
    Macro,
    #[token("macro_rules")]
    MacroRules,

    // ============ Types ============

    #[token("i8")]
    I8,
    #[token("i16")]
    I16,
    #[token("i32")]
    I32,
    #[token("i64")]
    I64,
    #[token("i128")]
    I128,
    #[token("u8")]
    U8,
    #[token("u16")]
    U16,
    #[token("u32")]
    U32,
    #[token("u64")]
    U64,
    #[token("u128")]
    U128,
    #[token("f32")]
    F32,
    #[token("f64")]
    F64,
    #[token("bool")]
    Bool,
    #[token("char")]
    Char,
    #[token("str")]
    Str,

    // ============ Operators ============

    // Arithmetic
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,

    // Comparison
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,

    // Logical
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("!")]
    Not,

    // Bitwise
    #[token("&")]
    And,
    #[token("|")]
    Or,
    #[token("^")]
    Caret,
    #[token("~")]
    Tilde,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,

    // Assignment
    #[token("=")]
    Eq,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("&=")]
    AndEq,
    #[token("|=")]
    OrEq,
    #[token("^=")]
    CaretEq,
    #[token("<<=")]
    ShlEq,
    #[token(">>=")]
    ShrEq,

    // Macro-specific
    #[token("$")]
    Dollar,

    // Other
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("<-")]
    LeftArrow,
    #[token("<-?")]
    LeftArrowQuestion,
    #[token("..")]
    DotDot,
    #[token("..=")]
    DotDotEq,
    #[token("::")]
    ColonColon,
    #[token("?")]
    Question,

    // ============ Delimiters ============

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    // ============ Punctuation ============

    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(".")]
    Dot,

    // ============ Identifiers ============

    /// Identifier: foo, _bar, MyStruct
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,

    // ============ Special ============

    /// End of file
    Eof,
}

impl TokenKind {
    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Fn
                | TokenKind::Let
                | TokenKind::Mut
                | TokenKind::Const
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::Match
                | TokenKind::For
                | TokenKind::While
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Return
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Impl
                | TokenKind::Trait
                | TokenKind::Type
                | TokenKind::Pub
                | TokenKind::Use
                | TokenKind::Mod
                | TokenKind::SelfValue
                | TokenKind::SelfType
                | TokenKind::As
                | TokenKind::In
                | TokenKind::Where
                | TokenKind::Dyn
                | TokenKind::Actor
                | TokenKind::Receive
                | TokenKind::Spawn
                | TokenKind::Reply
                | TokenKind::Await
                | TokenKind::Async
                | TokenKind::Select
                | TokenKind::Join
                | TokenKind::Macro
                | TokenKind::MacroRules
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Check if this token is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            TokenKind::IntLiteral
                | TokenKind::FloatLiteral
                | TokenKind::StringLiteral
                | TokenKind::CharLiteral
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Check if this token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::EqEq
                | TokenKind::NotEq
                | TokenKind::Lt
                | TokenKind::Gt
                | TokenKind::LtEq
                | TokenKind::GtEq
                | TokenKind::AndAnd
                | TokenKind::OrOr
                | TokenKind::Not
                | TokenKind::And
                | TokenKind::Or
                | TokenKind::Caret
                | TokenKind::Tilde
                | TokenKind::Shl
                | TokenKind::Shr
        )
    }

    /// Check if this token is a primitive type
    pub fn is_primitive_type(&self) -> bool {
        matches!(
            self,
            TokenKind::I8
                | TokenKind::I16
                | TokenKind::I32
                | TokenKind::I64
                | TokenKind::I128
                | TokenKind::U8
                | TokenKind::U16
                | TokenKind::U32
                | TokenKind::U64
                | TokenKind::U128
                | TokenKind::F32
                | TokenKind::F64
                | TokenKind::Bool
                | TokenKind::Char
                | TokenKind::Str
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TokenKind::IntLiteral => "integer",
            TokenKind::FloatLiteral => "float",
            TokenKind::StringLiteral => "string",
            TokenKind::CharLiteral => "char",
            TokenKind::Label => "label",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::Fn => "fn",
            TokenKind::Let => "let",
            TokenKind::Mut => "mut",
            TokenKind::Const => "const",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::Match => "match",
            TokenKind::For => "for",
            TokenKind::While => "while",
            TokenKind::Loop => "loop",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Return => "return",
            TokenKind::Struct => "struct",
            TokenKind::Enum => "enum",
            TokenKind::Impl => "impl",
            TokenKind::Trait => "trait",
            TokenKind::Type => "type",
            TokenKind::Pub => "pub",
            TokenKind::Use => "use",
            TokenKind::Mod => "mod",
            TokenKind::SelfValue => "self",
            TokenKind::SelfType => "Self",
            TokenKind::As => "as",
            TokenKind::In => "in",
            TokenKind::Where => "where",
            TokenKind::Dyn => "dyn",
            TokenKind::Actor => "actor",
            TokenKind::Receive => "receive",
            TokenKind::Spawn => "spawn",
            TokenKind::Reply => "reply",
            TokenKind::Await => "await",
            TokenKind::Async => "async",
            TokenKind::Select => "select",
            TokenKind::Join => "join",
            TokenKind::Macro => "macro",
            TokenKind::MacroRules => "macro_rules",
            TokenKind::I8 => "i8",
            TokenKind::I16 => "i16",
            TokenKind::I32 => "i32",
            TokenKind::I64 => "i64",
            TokenKind::I128 => "i128",
            TokenKind::U8 => "u8",
            TokenKind::U16 => "u16",
            TokenKind::U32 => "u32",
            TokenKind::U64 => "u64",
            TokenKind::U128 => "u128",
            TokenKind::F32 => "f32",
            TokenKind::F64 => "f64",
            TokenKind::Bool => "bool",
            TokenKind::Char => "char",
            TokenKind::Str => "str",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::EqEq => "==",
            TokenKind::NotEq => "!=",
            TokenKind::Lt => "<",
            TokenKind::Gt => ">",
            TokenKind::LtEq => "<=",
            TokenKind::GtEq => ">=",
            TokenKind::AndAnd => "&&",
            TokenKind::OrOr => "||",
            TokenKind::Not => "!",
            TokenKind::And => "&",
            TokenKind::Or => "|",
            TokenKind::Caret => "^",
            TokenKind::Tilde => "~",
            TokenKind::Shl => "<<",
            TokenKind::Shr => ">>",
            TokenKind::Eq => "=",
            TokenKind::PlusEq => "+=",
            TokenKind::MinusEq => "-=",
            TokenKind::StarEq => "*=",
            TokenKind::SlashEq => "/=",
            TokenKind::PercentEq => "%=",
            TokenKind::AndEq => "&=",
            TokenKind::OrEq => "|=",
            TokenKind::CaretEq => "^=",
            TokenKind::ShlEq => "<<=",
            TokenKind::ShrEq => ">>=",
            TokenKind::Dollar => "$",
            TokenKind::Arrow => "->",
            TokenKind::FatArrow => "=>",
            TokenKind::LeftArrow => "<-",
            TokenKind::LeftArrowQuestion => "<-?",
            TokenKind::DotDot => "..",
            TokenKind::DotDotEq => "..=",
            TokenKind::ColonColon => "::",
            TokenKind::Question => "?",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Comma => ",",
            TokenKind::Semicolon => ";",
            TokenKind::Colon => ":",
            TokenKind::Dot => ".",
            TokenKind::Ident => "identifier",
            TokenKind::Eof => "end of file",
        };
        write!(f, "{}", s)
    }
}
