//! Lexer for Genesis Lang
//!
//! The lexer converts source code into a stream of tokens.
//! It uses the `logos` crate for efficient lexing.

use crate::span::Span;
use crate::token::{Token, TokenKind};
use logos::Logos;
use thiserror::Error;

/// Lexer errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LexerError {
    #[error("unexpected character at position {0}")]
    UnexpectedChar(usize),

    #[error("unterminated string literal")]
    UnterminatedString,

    #[error("unterminated character literal")]
    UnterminatedChar,

    #[error("invalid escape sequence")]
    InvalidEscape,
}

/// The lexer for Genesis Lang
pub struct Lexer<'src> {
    source: &'src str,
    inner: logos::Lexer<'src, TokenKind>,
    peeked: Option<Token>,
    errors: Vec<LexerError>,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source code
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            inner: TokenKind::lexer(source),
            peeked: None,
            errors: Vec::new(),
        }
    }

    /// Get the source code
    pub fn source(&self) -> &'src str {
        self.source
    }

    /// Get any errors that occurred during lexing
    pub fn errors(&self) -> &[LexerError] {
        &self.errors
    }

    /// Peek at the next token without consuming it
    pub fn peek(&mut self) -> Option<&Token> {
        if self.peeked.is_none() {
            self.peeked = self.next_token();
        }
        self.peeked.as_ref()
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Option<Token> {
        // Return peeked token if available
        if let Some(token) = self.peeked.take() {
            return Some(token);
        }

        loop {
            match self.inner.next() {
                Some(Ok(kind)) => {
                    let span = self.inner.span();
                    return Some(Token::new(kind, Span::new(span.start, span.end)));
                }
                Some(Err(())) => {
                    // Skip invalid tokens and record error
                    let span = self.inner.span();
                    self.errors.push(LexerError::UnexpectedChar(span.start));
                    continue;
                }
                None => {
                    // End of input - return EOF token
                    let pos = self.source.len();
                    return Some(Token::new(TokenKind::Eof, Span::new(pos, pos)));
                }
            }
        }
    }

    /// Collect all tokens into a vector
    pub fn tokenize(mut self) -> (Vec<Token>, Vec<LexerError>) {
        let mut tokens = Vec::new();

        loop {
            match self.next_token() {
                Some(token) if token.kind == TokenKind::Eof => {
                    tokens.push(token);
                    break;
                }
                Some(token) => tokens.push(token),
                None => break,
            }
        }

        (tokens, self.errors)
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let token = self.next_token()?;
        if token.kind == TokenKind::Eof {
            None
        } else {
            Some(token)
        }
    }
}

/// Helper function to lex source code
pub fn lex(source: &str) -> (Vec<Token>, Vec<LexerError>) {
    Lexer::new(source).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_kinds(source: &str) -> Vec<TokenKind> {
        let (tokens, _) = lex(source);
        tokens.into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty_source() {
        let kinds = token_kinds("");
        assert_eq!(kinds, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_whitespace_only() {
        let kinds = token_kinds("   \t\n  ");
        assert_eq!(kinds, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_integers() {
        let kinds = token_kinds("42 0xFF 0b1010 0o77");
        assert_eq!(
            kinds,
            vec![
                TokenKind::IntLiteral,
                TokenKind::IntLiteral,
                TokenKind::IntLiteral,
                TokenKind::IntLiteral,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_floats() {
        let kinds = token_kinds("3.14 1e10 2.5e-3");
        assert_eq!(
            kinds,
            vec![
                TokenKind::FloatLiteral,
                TokenKind::FloatLiteral,
                TokenKind::FloatLiteral,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_strings() {
        let kinds = token_kinds(r#""hello" "world""#);
        assert_eq!(
            kinds,
            vec![TokenKind::StringLiteral, TokenKind::StringLiteral, TokenKind::Eof]
        );
    }

    #[test]
    fn test_keywords() {
        let kinds = token_kinds("fn let mut if else struct enum impl actor spawn");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Fn,
                TokenKind::Let,
                TokenKind::Mut,
                TokenKind::If,
                TokenKind::Else,
                TokenKind::Struct,
                TokenKind::Enum,
                TokenKind::Impl,
                TokenKind::Actor,
                TokenKind::Spawn,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_primitive_types() {
        let kinds = token_kinds("i32 u64 f64 bool str");
        assert_eq!(
            kinds,
            vec![
                TokenKind::I32,
                TokenKind::U64,
                TokenKind::F64,
                TokenKind::Bool,
                TokenKind::Str,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_operators() {
        let kinds = token_kinds("+ - * / == != < > <= >= && ||");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::EqEq,
                TokenKind::NotEq,
                TokenKind::Lt,
                TokenKind::Gt,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::AndAnd,
                TokenKind::OrOr,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_arrows() {
        let kinds = token_kinds("-> => <- <-?");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::LeftArrow,
                TokenKind::LeftArrowQuestion,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_delimiters() {
        let kinds = token_kinds("( ) [ ] { }");
        assert_eq!(
            kinds,
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        let kinds = token_kinds("foo bar_baz MyStruct _private");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident,
                TokenKind::Ident,
                TokenKind::Ident,
                TokenKind::Ident,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_function_definition() {
        let source = r#"
            fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;
        let kinds = token_kinds(source);
        assert_eq!(
            kinds,
            vec![
                TokenKind::Fn,
                TokenKind::Ident,      // add
                TokenKind::LParen,
                TokenKind::Ident,      // a
                TokenKind::Colon,
                TokenKind::I32,
                TokenKind::Comma,
                TokenKind::Ident,      // b
                TokenKind::Colon,
                TokenKind::I32,
                TokenKind::RParen,
                TokenKind::Arrow,
                TokenKind::I32,
                TokenKind::LBrace,
                TokenKind::Ident,      // a
                TokenKind::Plus,
                TokenKind::Ident,      // b
                TokenKind::RBrace,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_actor_definition() {
        let source = r#"
            actor Counter {
                state: i64 = 0
                receive {
                    Increment => self.state += 1
                }
            }
        "#;
        let kinds = token_kinds(source);
        assert!(kinds.contains(&TokenKind::Actor));
        assert!(kinds.contains(&TokenKind::Receive));
        assert!(kinds.contains(&TokenKind::FatArrow));
    }

    #[test]
    fn test_comments() {
        let kinds = token_kinds(r#"
            // This is a comment
            let x = 42 // inline comment
        "#);
        assert_eq!(
            kinds,
            vec![
                TokenKind::Let,
                TokenKind::Ident,
                TokenKind::Eq,
                TokenKind::IntLiteral,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_span_tracking() {
        let source = "let x = 42";
        let (tokens, _) = lex(source);

        assert_eq!(tokens[0].span.text(source), "let");
        assert_eq!(tokens[1].span.text(source), "x");
        assert_eq!(tokens[2].span.text(source), "=");
        assert_eq!(tokens[3].span.text(source), "42");
    }
}
