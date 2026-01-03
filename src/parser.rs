//! Parser for Genesis Lang
//!
//! This is a recursive descent parser that converts tokens into an AST.
//! The parser handles precedence and associativity correctly.

use crate::ast::{self, *};
use crate::lexer::Lexer;
use crate::span::Span;
use crate::token::{Token, TokenKind};
use thiserror::Error;

/// Parser errors
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found}")]
    UnexpectedToken {
        expected: String,
        found: TokenKind,
        span: Span,
    },

    #[error("unexpected end of file")]
    UnexpectedEof { span: Span },

    #[error("{message}")]
    Custom { message: String, span: Span },
}

impl ParseError {
    pub fn span(&self) -> Span {
        match self {
            ParseError::UnexpectedToken { span, .. } => *span,
            ParseError::UnexpectedEof { span } => *span,
            ParseError::Custom { span, .. } => *span,
        }
    }
}

/// Parse result
pub type ParseResult<T> = Result<T, ParseError>;

/// The parser for Genesis Lang
pub struct Parser<'src> {
    lexer: Lexer<'src>,
    current: Token,
    previous: Token,
    errors: Vec<ParseError>,
    /// Lookahead buffer for multi-token lookahead
    lookahead: Vec<Token>,
    /// Path to the source file (for resolving external modules)
    source_path: Option<std::path::PathBuf>,
}

impl<'src> Parser<'src> {
    /// Create a new parser
    pub fn new(source: &'src str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token().unwrap_or(Token::new(
            TokenKind::Eof,
            Span::new(source.len(), source.len()),
        ));
        let previous = current.clone();

        Self {
            lexer,
            current,
            previous,
            errors: Vec::new(),
            lookahead: Vec::new(),
            source_path: None,
        }
    }

    /// Create a new parser with a source file path (for external module resolution)
    pub fn with_path(source: &'src str, path: std::path::PathBuf) -> Self {
        let mut parser = Self::new(source);
        parser.source_path = Some(path);
        parser
    }

    /// Get the source code
    pub fn source(&self) -> &'src str {
        self.lexer.source()
    }

    /// Get parse errors
    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Advance to next token
    fn advance(&mut self) -> Token {
        self.previous = self.current.clone();
        self.current = if !self.lookahead.is_empty() {
            self.lookahead.remove(0)
        } else {
            self.lexer.next_token().unwrap_or(Token::new(
                TokenKind::Eof,
                Span::new(self.source().len(), self.source().len()),
            ))
        };
        self.previous.clone()
    }

    /// Peek at the nth token ahead (0 = current, 1 = next, etc.)
    fn peek_nth(&mut self, n: usize) -> &Token {
        if n == 0 {
            return &self.current;
        }
        // Fill lookahead buffer if needed
        while self.lookahead.len() < n {
            let token = self.lexer.next_token().unwrap_or(Token::new(
                TokenKind::Eof,
                Span::new(self.source().len(), self.source().len()),
            ));
            self.lookahead.push(token);
        }
        &self.lookahead[n - 1]
    }

    /// Check if current token matches
    fn check(&self, kind: TokenKind) -> bool {
        self.current.kind == kind
    }

    /// Check if at end of file
    fn is_at_end(&self) -> bool {
        self.check(TokenKind::Eof)
    }

    /// Consume token if it matches, otherwise error
    fn expect(&mut self, kind: TokenKind) -> ParseResult<Token> {
        if self.check(kind.clone()) {
            Ok(self.advance())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: format!("{}", kind),
                found: self.current.kind.clone(),
                span: self.current.span,
            })
        }
    }

    /// Consume token if it matches
    fn consume(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Get text of a token
    fn text(&self, token: &Token) -> &'src str {
        token.text(self.source())
    }

    // ============ Top-level parsing ============

    /// Parse a complete program
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let start = self.current.span.start;
        let mut items = Vec::new();

        while !self.is_at_end() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
        }

        let end = self.previous.span.end;
        Ok(Program {
            items,
            span: Span::new(start, end),
        })
    }

    /// Synchronize after error
    fn synchronize(&mut self) {
        self.advance();
        while !self.is_at_end() {
            if self.previous.kind == TokenKind::Semicolon {
                return;
            }
            match self.current.kind {
                TokenKind::Fn
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Impl
                | TokenKind::Trait
                | TokenKind::Type
                | TokenKind::Const
                | TokenKind::Actor
                | TokenKind::Use
                | TokenKind::Mod
                | TokenKind::Macro
                | TokenKind::Pub => return,
                _ => {}
            }
            self.advance();
        }
    }

    /// Parse a top-level item
    fn parse_item(&mut self) -> ParseResult<Item> {
        let is_pub = self.consume(TokenKind::Pub);
        let is_async = self.consume(TokenKind::Async);

        match self.current.kind {
            TokenKind::Fn => self.parse_fn(is_pub, is_async).map(Item::Function),
            TokenKind::Struct => self.parse_struct(is_pub).map(Item::Struct),
            TokenKind::Enum => self.parse_enum(is_pub).map(Item::Enum),
            TokenKind::Impl => self.parse_impl().map(Item::Impl),
            TokenKind::Trait => self.parse_trait(is_pub).map(Item::Trait),
            TokenKind::Type => self.parse_type_alias(is_pub).map(Item::TypeAlias),
            TokenKind::Const => self.parse_const(is_pub).map(Item::Const),
            TokenKind::Actor => self.parse_actor(is_pub).map(Item::Actor),
            TokenKind::Use => self.parse_use().map(Item::Use),
            TokenKind::Mod => self.parse_mod(is_pub).map(Item::Mod),
            TokenKind::Macro => self.parse_macro_def(is_pub).map(Item::Macro),
            TokenKind::MacroRules => self.parse_macro_def(is_pub).map(Item::Macro),
            _ => Err(ParseError::UnexpectedToken {
                expected: "item".to_string(),
                found: self.current.kind.clone(),
                span: self.current.span,
            }),
        }
    }

    // ============ Function parsing ============

    fn parse_fn(&mut self, is_pub: bool, is_async: bool) -> ParseResult<FnDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Fn)?;

        let name = self.parse_ident()?;
        let mut generics = self.parse_generics_opt()?;

        self.expect(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen)?;

        let return_type = if self.consume(TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        // Parse optional where clause: fn foo<T>(x: T) -> i64 where T: Clone { }
        let where_clause = self.parse_where_clause_opt()?;
        if let Some(wc) = where_clause {
            if let Some(ref mut g) = generics {
                g.where_clause = Some(wc);
            } else {
                // Where clause without generics - create empty generics to hold it
                generics = Some(Generics {
                    params: Vec::new(),
                    where_clause: Some(wc.clone()),
                    span: wc.span,
                });
            }
        }

        let body = self.parse_block()?;

        Ok(FnDef {
            name,
            generics,
            params,
            return_type,
            body,
            is_pub,
            is_async,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        let mut params = Vec::new();

        if !self.check(TokenKind::RParen) {
            loop {
                params.push(self.parse_param()?);
                if !self.consume(TokenKind::Comma) {
                    break;
                }
                if self.check(TokenKind::RParen) {
                    break;
                }
            }
        }

        Ok(params)
    }

    fn parse_param(&mut self) -> ParseResult<Param> {
        let start = self.current.span.start;

        // Handle special self parameters: self, &self, &mut self
        if self.check(TokenKind::And) {
            self.advance(); // consume &
            let is_mut = self.consume(TokenKind::Mut);

            if self.check(TokenKind::SelfValue) {
                let token = self.advance();
                let self_type = Type {
                    kind: TypeKind::Reference {
                        mutable: is_mut,
                        inner: Box::new(Type {
                            kind: TypeKind::Path(Path {
                                segments: vec![PathSegment {
                                    ident: Ident::new("Self", token.span),
                                    generics: None,
                                }],
                                span: token.span,
                            }),
                            span: token.span,
                        }),
                    },
                    span: Span::new(start, token.span.end),
                };
                return Ok(Param {
                    name: Ident::new("self", token.span),
                    ty: self_type,
                    is_mut: false,
                    span: Span::new(start, self.previous.span.end),
                });
            }
        }

        // Handle bare 'self'
        if self.check(TokenKind::SelfValue) {
            let token = self.advance();
            let self_type = Type {
                kind: TypeKind::Path(Path {
                    segments: vec![PathSegment {
                        ident: Ident::new("Self", token.span),
                        generics: None,
                    }],
                    span: token.span,
                }),
                span: token.span,
            };
            return Ok(Param {
                name: Ident::new("self", token.span),
                ty: self_type,
                is_mut: false,
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Normal parameter: name: Type
        let is_mut = self.consume(TokenKind::Mut);
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(Param {
            name,
            ty,
            is_mut,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Struct parsing ============

    fn parse_struct(&mut self, is_pub: bool) -> ParseResult<StructDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Struct)?;
        let name = self.parse_ident()?;
        let mut generics = self.parse_generics_opt()?;

        // Parse optional where clause: struct Foo<T> where T: Clone { }
        let where_clause = self.parse_where_clause_opt()?;
        if let Some(wc) = where_clause {
            if let Some(ref mut g) = generics {
                g.where_clause = Some(wc);
            } else {
                generics = Some(Generics {
                    params: Vec::new(),
                    where_clause: Some(wc.clone()),
                    span: wc.span,
                });
            }
        }

        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_fields()?;
        self.expect(TokenKind::RBrace)?;

        Ok(StructDef {
            name,
            generics,
            fields,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_fields(&mut self) -> ParseResult<Vec<Field>> {
        let mut fields = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            fields.push(self.parse_field()?);
            if !self.consume(TokenKind::Comma) {
                break;
            }
        }

        Ok(fields)
    }

    fn parse_field(&mut self) -> ParseResult<Field> {
        let start = self.current.span.start;
        let is_pub = self.consume(TokenKind::Pub);
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(Field {
            name,
            ty,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Enum parsing ============

    fn parse_enum(&mut self, is_pub: bool) -> ParseResult<EnumDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Enum)?;
        let name = self.parse_ident()?;
        let mut generics = self.parse_generics_opt()?;

        // Parse optional where clause: enum Foo<T> where T: Clone { }
        let where_clause = self.parse_where_clause_opt()?;
        if let Some(wc) = where_clause {
            if let Some(ref mut g) = generics {
                g.where_clause = Some(wc);
            } else {
                generics = Some(Generics {
                    params: Vec::new(),
                    where_clause: Some(wc.clone()),
                    span: wc.span,
                });
            }
        }

        self.expect(TokenKind::LBrace)?;
        let variants = self.parse_variants()?;
        self.expect(TokenKind::RBrace)?;

        Ok(EnumDef {
            name,
            generics,
            variants,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_variants(&mut self) -> ParseResult<Vec<Variant>> {
        let mut variants = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            variants.push(self.parse_variant()?);
            if !self.consume(TokenKind::Comma) {
                break;
            }
        }

        Ok(variants)
    }

    fn parse_variant(&mut self) -> ParseResult<Variant> {
        let start = self.current.span.start;
        let name = self.parse_ident()?;

        let kind = if self.consume(TokenKind::LParen) {
            let mut types = Vec::new();
            if !self.check(TokenKind::RParen) {
                loop {
                    types.push(self.parse_type()?);
                    if !self.consume(TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;
            VariantKind::Tuple(types)
        } else if self.consume(TokenKind::LBrace) {
            let fields = self.parse_fields()?;
            self.expect(TokenKind::RBrace)?;
            VariantKind::Struct(fields)
        } else {
            VariantKind::Unit
        };

        Ok(Variant {
            name,
            kind,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Impl parsing ============

    fn parse_impl(&mut self) -> ParseResult<ImplDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Impl)?;
        let mut generics = self.parse_generics_opt()?;

        let first_type = self.parse_type()?;

        let (trait_, self_type) = if self.consume(TokenKind::For) {
            let self_type = self.parse_type()?;
            (Some(first_type), self_type)
        } else {
            (None, first_type)
        };

        // Parse optional where clause: impl<T> Foo<T> where T: Clone { }
        let where_clause = self.parse_where_clause_opt()?;
        if let Some(wc) = where_clause {
            if let Some(ref mut g) = generics {
                g.where_clause = Some(wc);
            } else {
                generics = Some(Generics {
                    params: Vec::new(),
                    where_clause: Some(wc.clone()),
                    span: wc.span,
                });
            }
        }

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let is_pub = self.consume(TokenKind::Pub);
            let is_async = self.consume(TokenKind::Async);
            match self.current.kind {
                TokenKind::Fn => items.push(ImplItem::Function(self.parse_fn(is_pub, is_async)?)),
                TokenKind::Const => items.push(ImplItem::Const(self.parse_const(is_pub)?)),
                TokenKind::Type => items.push(ImplItem::TypeAlias(self.parse_type_alias(is_pub)?)),
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "fn, const, or type".to_string(),
                        found: self.current.kind.clone(),
                        span: self.current.span,
                    });
                }
            }
        }
        self.expect(TokenKind::RBrace)?;

        Ok(ImplDef {
            generics,
            trait_,
            self_type,
            items,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Other items ============

    fn parse_trait(&mut self, is_pub: bool) -> ParseResult<TraitDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Trait)?;
        let name = self.parse_ident()?;
        let mut generics = self.parse_generics_opt()?;

        let super_traits = if self.consume(TokenKind::Colon) {
            let mut traits = vec![self.parse_type()?];
            while self.consume(TokenKind::Plus) {
                traits.push(self.parse_type()?);
            }
            traits
        } else {
            Vec::new()
        };

        // Parse optional where clause: trait Foo<T>: Clone where T: Debug { }
        let where_clause = self.parse_where_clause_opt()?;
        if let Some(wc) = where_clause {
            if let Some(ref mut g) = generics {
                g.where_clause = Some(wc);
            } else {
                generics = Some(Generics {
                    params: Vec::new(),
                    where_clause: Some(wc.clone()),
                    span: wc.span,
                });
            }
        }

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            items.push(self.parse_trait_item()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(TraitDef {
            name,
            generics,
            super_traits,
            items,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_trait_item(&mut self) -> ParseResult<TraitItem> {
        let start = self.current.span.start;
        let _is_async = self.consume(TokenKind::Async); // TODO: Use in FnSignature

        match self.current.kind {
            TokenKind::Fn => {
                self.advance(); // consume 'fn'
                let name = self.parse_ident()?;
                let generics = self.parse_generics_opt()?;

                self.expect(TokenKind::LParen)?;
                let params = self.parse_params()?;
                self.expect(TokenKind::RParen)?;

                let return_type = if self.consume(TokenKind::Arrow) {
                    Some(self.parse_type()?)
                } else {
                    None
                };

                // Trait methods can have optional body
                if self.check(TokenKind::LBrace) {
                    // Has default implementation - parse but don't store (FnSignature doesn't have body)
                    let _body = self.parse_block()?;
                    Ok(TraitItem::Function(FnSignature {
                        name,
                        generics,
                        params,
                        return_type,
                        span: Span::new(start, self.previous.span.end),
                    }))
                } else {
                    // Just a signature, consume optional semicolon
                    self.consume(TokenKind::Semicolon);
                    Ok(TraitItem::Function(FnSignature {
                        name,
                        generics,
                        params,
                        return_type,
                        span: Span::new(start, self.previous.span.end),
                    }))
                }
            }
            TokenKind::Const => {
                Ok(TraitItem::Const(self.parse_const(false)?))
            }
            TokenKind::Type => {
                // Could be associated type declaration (type Item;) or type alias (type Foo = Bar;)
                self.parse_trait_type_item()
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "fn, const, or type".to_string(),
                found: self.current.kind.clone(),
                span: self.current.span,
            }),
        }
    }

    fn parse_type_alias(&mut self, is_pub: bool) -> ParseResult<TypeAlias> {
        let start = self.current.span.start;
        self.expect(TokenKind::Type)?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;
        self.expect(TokenKind::Eq)?;
        let ty = self.parse_type()?;
        self.consume(TokenKind::Semicolon);

        Ok(TypeAlias {
            name,
            generics,
            ty,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse a type item in a trait: either an associated type or a type alias
    /// Associated type: `type Item;` or `type Item: Clone + Debug;`
    /// Type alias: `type Foo = Bar;`
    fn parse_trait_type_item(&mut self) -> ParseResult<TraitItem> {
        use crate::ast::{AssociatedTypeDef, TraitItem};

        let start = self.current.span.start;
        self.expect(TokenKind::Type)?;
        let name = self.parse_ident()?;

        // Check for bounds: `type Item: Clone + Debug`
        let bounds = if self.consume(TokenKind::Colon) {
            let mut bounds = vec![self.parse_type()?];
            while self.consume(TokenKind::Plus) {
                bounds.push(self.parse_type()?);
            }
            bounds
        } else {
            vec![]
        };

        // Check if this is a type alias (has `=`) or an associated type declaration (ends with `;`)
        if self.consume(TokenKind::Eq) {
            // Type alias with default: `type Item = DefaultType;`
            let ty = self.parse_type()?;
            self.consume(TokenKind::Semicolon);
            Ok(TraitItem::AssociatedType(AssociatedTypeDef {
                name,
                bounds,
                default: Some(ty),
                span: Span::new(start, self.previous.span.end),
            }))
        } else {
            // Associated type declaration: `type Item;` or `type Item: Bound;`
            self.consume(TokenKind::Semicolon);
            Ok(TraitItem::AssociatedType(AssociatedTypeDef {
                name,
                bounds,
                default: None,
                span: Span::new(start, self.previous.span.end),
            }))
        }
    }

    fn parse_const(&mut self, is_pub: bool) -> ParseResult<ConstDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Const)?;
        let name = self.parse_ident()?;

        let ty = if self.consume(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.consume(TokenKind::Semicolon);

        Ok(ConstDef {
            name,
            ty,
            value,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_actor(&mut self, is_pub: bool) -> ParseResult<ActorDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Actor)?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;

        self.expect(TokenKind::LBrace)?;

        let mut state = Vec::new();
        let mut receive = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.consume(TokenKind::Receive) {
                self.expect(TokenKind::LBrace)?;
                while !self.check(TokenKind::RBrace) && !self.is_at_end() {
                    receive.push(self.parse_message_handler()?);
                }
                self.expect(TokenKind::RBrace)?;
            } else {
                state.push(self.parse_field()?);
                self.consume(TokenKind::Comma);
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(ActorDef {
            name,
            generics,
            state,
            receive,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_message_handler(&mut self) -> ParseResult<MessageHandler> {
        let start = self.current.span.start;
        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;
        self.consume(TokenKind::Comma);

        Ok(MessageHandler {
            pattern,
            body,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_use(&mut self) -> ParseResult<UseDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Use)?;
        let path = self.parse_path()?;

        let alias = if self.consume(TokenKind::As) {
            Some(self.parse_ident()?)
        } else {
            None
        };

        self.consume(TokenKind::Semicolon);

        Ok(UseDef {
            path,
            alias,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_mod(&mut self, is_pub: bool) -> ParseResult<ModDef> {
        let start = self.current.span.start;
        self.expect(TokenKind::Mod)?;
        let name = self.parse_ident()?;

        let items = if self.consume(TokenKind::LBrace) {
            let mut items = Vec::new();
            while !self.check(TokenKind::RBrace) && !self.is_at_end() {
                items.push(self.parse_item()?);
            }
            self.expect(TokenKind::RBrace)?;
            Some(items)
        } else {
            self.consume(TokenKind::Semicolon);
            None
        };

        Ok(ModDef {
            name,
            items,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Macro parsing ============

    /// Parse a macro definition: `macro name { rules }` or `macro_rules! name { rules }`
    fn parse_macro_def(&mut self, is_pub: bool) -> ParseResult<ast::MacroDef> {
        let start = self.current.span.start;

        // Handle both `macro` and `macro_rules!` syntax
        if self.check(TokenKind::MacroRules) {
            self.advance(); // consume macro_rules
            self.expect(TokenKind::Not)?; // consume !
        } else {
            self.expect(TokenKind::Macro)?;
        }

        let name = self.parse_ident()?;

        self.expect(TokenKind::LBrace)?;

        let mut rules = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            rules.push(self.parse_macro_rule()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(ast::MacroDef {
            name,
            rules,
            is_pub,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse a single macro rule: `pattern => expansion;`
    fn parse_macro_rule(&mut self) -> ParseResult<ast::MacroRule> {
        let start = self.current.span.start;

        // Parse pattern (tokens inside parens/brackets/braces)
        let pattern = self.parse_macro_pattern()?;

        // Expect =>
        self.expect(TokenKind::FatArrow)?;

        // Parse expansion
        let expansion = self.parse_macro_expansion()?;

        // Optional semicolon
        self.consume(TokenKind::Semicolon);

        Ok(ast::MacroRule {
            pattern,
            expansion,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse macro pattern: `( tokens... )`
    fn parse_macro_pattern(&mut self) -> ParseResult<ast::MacroPattern> {
        let start = self.current.span.start;

        // Must start with ( or [ or {
        let (end_token, _delimiter) = match self.current.kind {
            TokenKind::LParen => (TokenKind::RParen, ast::MacroDelimiter::Paren),
            TokenKind::LBracket => (TokenKind::RBracket, ast::MacroDelimiter::Bracket),
            TokenKind::LBrace => (TokenKind::RBrace, ast::MacroDelimiter::Brace),
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "( or [ or {".to_string(),
                    found: self.current.kind.clone(),
                    span: self.current.span,
                });
            }
        };

        self.advance(); // consume opening delimiter

        let tokens = self.parse_macro_tokens(&end_token)?;

        self.expect(end_token)?;

        Ok(ast::MacroPattern {
            tokens,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse macro expansion: `{ tokens... }` or `( tokens... )`
    fn parse_macro_expansion(&mut self) -> ParseResult<ast::MacroExpansion> {
        let start = self.current.span.start;

        // Must start with ( or [ or {
        let (end_token, _delimiter) = match self.current.kind {
            TokenKind::LParen => (TokenKind::RParen, ast::MacroDelimiter::Paren),
            TokenKind::LBracket => (TokenKind::RBracket, ast::MacroDelimiter::Bracket),
            TokenKind::LBrace => (TokenKind::RBrace, ast::MacroDelimiter::Brace),
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "( or [ or {".to_string(),
                    found: self.current.kind.clone(),
                    span: self.current.span,
                });
            }
        };

        self.advance(); // consume opening delimiter

        let tokens = self.parse_macro_tokens(&end_token)?;

        self.expect(end_token)?;

        Ok(ast::MacroExpansion {
            tokens,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse tokens inside a macro pattern/expansion until end_token
    fn parse_macro_tokens(&mut self, end_token: &TokenKind) -> ParseResult<Vec<ast::MacroToken>> {
        let mut tokens = Vec::new();

        while !self.check(end_token.clone()) && !self.is_at_end() {
            tokens.push(self.parse_macro_token()?);
        }

        Ok(tokens)
    }

    /// Parse a single macro token
    fn parse_macro_token(&mut self) -> ParseResult<ast::MacroToken> {
        let span = self.current.span;

        // Check for $ (capture or repetition)
        if self.check(TokenKind::Dollar) {
            self.advance(); // consume $

            // Check for $( ... ) repetition
            if self.check(TokenKind::LParen) {
                return self.parse_macro_repetition();
            }

            // Otherwise it's a capture: $name or $name:kind
            let name_token = self.advance();
            let name = self.text(&name_token).to_string();

            // Check for :kind
            let kind = if self.consume(TokenKind::Colon) {
                let kind_token = self.advance();
                let kind_str = self.text(&kind_token);
                match kind_str {
                    "expr" => ast::MacroCaptureKind::Expr,
                    "ty" => ast::MacroCaptureKind::Ty,
                    "ident" => ast::MacroCaptureKind::Ident,
                    "pat" => ast::MacroCaptureKind::Pat,
                    "stmt" => ast::MacroCaptureKind::Stmt,
                    "block" => ast::MacroCaptureKind::Block,
                    "item" => ast::MacroCaptureKind::Item,
                    "literal" => ast::MacroCaptureKind::Literal,
                    "tt" => ast::MacroCaptureKind::Tt,
                    "path" => ast::MacroCaptureKind::Path,
                    _ => {
                        return Err(ParseError::Custom {
                            message: format!("unknown macro capture kind: {}", kind_str),
                            span: kind_token.span,
                        });
                    }
                }
            } else {
                // No kind specified, treat as token tree
                ast::MacroCaptureKind::Tt
            };

            return Ok(ast::MacroToken::Capture {
                name,
                kind,
                span: Span::new(span.start, self.previous.span.end),
            });
        }

        // Check for nested group
        match self.current.kind {
            TokenKind::LParen => {
                let start = self.current.span.start;
                self.advance();
                let inner = self.parse_macro_tokens(&TokenKind::RParen)?;
                self.expect(TokenKind::RParen)?;
                return Ok(ast::MacroToken::Group {
                    delimiter: ast::MacroDelimiter::Paren,
                    tokens: inner,
                    span: Span::new(start, self.previous.span.end),
                });
            }
            TokenKind::LBracket => {
                let start = self.current.span.start;
                self.advance();
                let inner = self.parse_macro_tokens(&TokenKind::RBracket)?;
                self.expect(TokenKind::RBracket)?;
                return Ok(ast::MacroToken::Group {
                    delimiter: ast::MacroDelimiter::Bracket,
                    tokens: inner,
                    span: Span::new(start, self.previous.span.end),
                });
            }
            TokenKind::LBrace => {
                let start = self.current.span.start;
                self.advance();
                let inner = self.parse_macro_tokens(&TokenKind::RBrace)?;
                self.expect(TokenKind::RBrace)?;
                return Ok(ast::MacroToken::Group {
                    delimiter: ast::MacroDelimiter::Brace,
                    tokens: inner,
                    span: Span::new(start, self.previous.span.end),
                });
            }
            _ => {}
        }

        // Handle different token types
        if self.check(TokenKind::Ident) {
            let token = self.advance();
            let text = self.text(&token).to_string();
            Ok(ast::MacroToken::Ident(text, span))
        } else if self.check(TokenKind::IntLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            let value = parse_int(text).unwrap_or(0) as i64;
            Ok(ast::MacroToken::IntLit(value, span))
        } else if self.check(TokenKind::FloatLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            let value: f64 = text.parse().unwrap_or(0.0);
            Ok(ast::MacroToken::FloatLit(value, span))
        } else if self.check(TokenKind::StringLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            // Remove quotes and handle escapes
            let value = parse_string(text);
            Ok(ast::MacroToken::StrLit(value, span))
        } else if self.check(TokenKind::CharLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            // Remove quotes and handle escapes
            let value = parse_char(text);
            Ok(ast::MacroToken::CharLit(value, span))
        } else {
            let token = self.advance();
            Ok(ast::MacroToken::Token(token.kind, span))
        }
    }

    /// Parse macro repetition: $( ... )* or $( ... )+ or $( ... )?
    fn parse_macro_repetition(&mut self) -> ParseResult<ast::MacroToken> {
        let start = self.current.span.start;
        self.expect(TokenKind::LParen)?;

        let inner_tokens = self.parse_macro_tokens(&TokenKind::RParen)?;
        self.expect(TokenKind::RParen)?;

        // Check for separator (e.g., comma)
        let separator = if !self.check(TokenKind::Star)
            && !self.check(TokenKind::Plus)
            && !self.check(TokenKind::Question)
        {
            // There's a separator before the repetition kind
            let sep_span = self.current.span;
            let sep_token = self.advance();
            Some(Box::new(if sep_token.kind == TokenKind::Ident {
                ast::MacroToken::Ident(self.text(&sep_token).to_string(), sep_span)
            } else {
                ast::MacroToken::Token(sep_token.kind, sep_span)
            }))
        } else {
            None
        };

        // Parse repetition kind
        let kind = match self.current.kind {
            TokenKind::Star => {
                self.advance();
                ast::MacroRepKind::ZeroOrMore
            }
            TokenKind::Plus => {
                self.advance();
                ast::MacroRepKind::OneOrMore
            }
            TokenKind::Question => {
                self.advance();
                ast::MacroRepKind::ZeroOrOne
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "*, +, or ?".to_string(),
                    found: self.current.kind.clone(),
                    span: self.current.span,
                });
            }
        };

        Ok(ast::MacroToken::Repetition {
            tokens: inner_tokens,
            separator,
            kind,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse a macro invocation in expression position: `name!(args)` or `name![args]` or `name!{args}`
    fn parse_macro_invocation(&mut self, name: Ident) -> ParseResult<ast::MacroInvocation> {
        let start = name.span.start;

        // Already consumed the name, expect !
        self.expect(TokenKind::Not)?;

        // Determine delimiter
        let (end_token, delimiter) = match self.current.kind {
            TokenKind::LParen => (TokenKind::RParen, ast::MacroDelimiter::Paren),
            TokenKind::LBracket => (TokenKind::RBracket, ast::MacroDelimiter::Bracket),
            TokenKind::LBrace => (TokenKind::RBrace, ast::MacroDelimiter::Brace),
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "( or [ or {".to_string(),
                    found: self.current.kind.clone(),
                    span: self.current.span,
                });
            }
        };

        self.advance(); // consume opening delimiter

        let tokens = self.parse_macro_tokens(&end_token)?;

        self.expect(end_token)?;

        Ok(ast::MacroInvocation {
            name,
            delimiter,
            tokens,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Expression parsing ============

    /// Parse a single expression (public for macro expansion)
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> ParseResult<Expr> {
        let expr = self.parse_range()?;

        if self.check(TokenKind::Eq) {
            let start = expr.span.start;
            self.advance();
            let value = self.parse_assignment()?;
            return Ok(Expr {
                kind: ExprKind::Assign {
                    target: Box::new(expr),
                    value: Box::new(value),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Compound assignment
        let op = match self.current.kind {
            TokenKind::PlusEq => Some(BinaryOp::Add),
            TokenKind::MinusEq => Some(BinaryOp::Sub),
            TokenKind::StarEq => Some(BinaryOp::Mul),
            TokenKind::SlashEq => Some(BinaryOp::Div),
            TokenKind::PercentEq => Some(BinaryOp::Rem),
            TokenKind::AndEq => Some(BinaryOp::BitAnd),
            TokenKind::OrEq => Some(BinaryOp::BitOr),
            TokenKind::CaretEq => Some(BinaryOp::BitXor),
            TokenKind::ShlEq => Some(BinaryOp::Shl),
            TokenKind::ShrEq => Some(BinaryOp::Shr),
            _ => None,
        };

        if let Some(op) = op {
            let start = expr.span.start;
            self.advance();
            let value = self.parse_range()?;
            return Ok(Expr {
                kind: ExprKind::AssignOp {
                    op,
                    target: Box::new(expr),
                    value: Box::new(value),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        Ok(expr)
    }

    fn parse_range(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;

        // Check for prefix range: ..end or ..=end
        if self.check(TokenKind::DotDot) || self.check(TokenKind::DotDotEq) {
            let inclusive = self.check(TokenKind::DotDotEq);
            self.advance();
            let end = if self.check(TokenKind::RBrace)
                || self.check(TokenKind::RParen)
                || self.check(TokenKind::RBracket)
                || self.check(TokenKind::Comma)
            {
                None
            } else {
                Some(Box::new(self.parse_send()?))
            };
            return Ok(Expr {
                kind: ExprKind::Range {
                    start: None,
                    end,
                    inclusive,
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        let expr = self.parse_send()?;

        // Check for postfix range: start.. or start..end or start..=end
        if self.check(TokenKind::DotDot) || self.check(TokenKind::DotDotEq) {
            let inclusive = self.check(TokenKind::DotDotEq);
            self.advance();
            let end = if self.check(TokenKind::RBrace)
                || self.check(TokenKind::RParen)
                || self.check(TokenKind::RBracket)
                || self.check(TokenKind::Comma)
                || self.check(TokenKind::LBrace)
            {
                None
            } else {
                Some(Box::new(self.parse_send()?))
            };
            return Ok(Expr {
                kind: ExprKind::Range {
                    start: Some(Box::new(expr)),
                    end,
                    inclusive,
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        Ok(expr)
    }

    fn parse_send(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_or()?;

        loop {
            let start = expr.span.start;
            if self.consume(TokenKind::LeftArrowQuestion) {
                // Send with reply: actor <-? Message
                let message = self.parse_or()?;
                expr = Expr {
                    kind: ExprKind::SendRecv {
                        target: Box::new(expr),
                        message: Box::new(message),
                    },
                    span: Span::new(start, self.previous.span.end),
                };
            } else if self.consume(TokenKind::LeftArrow) {
                // Fire-and-forget send: actor <- Message
                let message = self.parse_or()?;
                expr = Expr {
                    kind: ExprKind::Send {
                        target: Box::new(expr),
                        message: Box::new(message),
                    },
                    span: Span::new(start, self.previous.span.end),
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_or(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_and()?;

        while self.consume(TokenKind::OrOr) {
            let start = expr.span.start;
            let right = self.parse_and()?;
            expr = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                span: Span::new(start, self.previous.span.end),
            };
        }

        Ok(expr)
    }

    fn parse_and(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_equality()?;

        while self.consume(TokenKind::AndAnd) {
            let start = expr.span.start;
            let right = self.parse_equality()?;
            expr = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::And,
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                span: Span::new(start, self.previous.span.end),
            };
        }

        Ok(expr)
    }

    fn parse_equality(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_comparison()?;

        loop {
            let op = match self.current.kind {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::NotEq => BinaryOp::Ne,
                _ => break,
            };
            self.advance();
            let start = expr.span.start;
            let right = self.parse_comparison()?;
            expr = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                span: Span::new(start, self.previous.span.end),
            };
        }

        Ok(expr)
    }

    fn parse_comparison(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_term()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::LtEq => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::GtEq => BinaryOp::Ge,
                _ => break,
            };
            self.advance();
            let start = expr.span.start;
            let right = self.parse_term()?;
            expr = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                span: Span::new(start, self.previous.span.end),
            };
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_factor()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            let start = expr.span.start;
            let right = self.parse_factor()?;
            expr = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                span: Span::new(start, self.previous.span.end),
            };
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_unary()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Rem,
                _ => break,
            };
            self.advance();
            let start = expr.span.start;
            let right = self.parse_unary()?;
            expr = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(expr),
                    right: Box::new(right),
                },
                span: Span::new(start, self.previous.span.end),
            };
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;

        if self.consume(TokenKind::Not) {
            let operand = self.parse_unary()?;
            return Ok(Expr {
                kind: ExprKind::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        if self.consume(TokenKind::Minus) {
            let operand = self.parse_unary()?;
            return Ok(Expr {
                kind: ExprKind::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        if self.consume(TokenKind::Star) {
            let operand = self.parse_unary()?;
            return Ok(Expr {
                kind: ExprKind::Deref {
                    operand: Box::new(operand),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        if self.consume(TokenKind::And) {
            let mutable = self.consume(TokenKind::Mut);
            let operand = self.parse_unary()?;
            return Ok(Expr {
                kind: ExprKind::Ref {
                    mutable,
                    operand: Box::new(operand),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.consume(TokenKind::LParen) {
                let args = self.parse_args()?;
                self.expect(TokenKind::RParen)?;
                let start = expr.span.start;
                expr = Expr {
                    kind: ExprKind::Call {
                        func: Box::new(expr),
                        args,
                    },
                    span: Span::new(start, self.previous.span.end),
                };
            } else if self.consume(TokenKind::Dot) {
                // Check for .await
                if self.consume(TokenKind::Await) {
                    let start = expr.span.start;
                    expr = Expr {
                        kind: ExprKind::Await {
                            operand: Box::new(expr),
                        },
                        span: Span::new(start, self.previous.span.end),
                    };
                    continue;
                }

                let field = self.parse_ident()?;
                if self.consume(TokenKind::LParen) {
                    let args = self.parse_args()?;
                    self.expect(TokenKind::RParen)?;
                    let start = expr.span.start;
                    expr = Expr {
                        kind: ExprKind::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            args,
                        },
                        span: Span::new(start, self.previous.span.end),
                    };
                } else {
                    let start = expr.span.start;
                    expr = Expr {
                        kind: ExprKind::Field {
                            object: Box::new(expr),
                            field,
                        },
                        span: Span::new(start, self.previous.span.end),
                    };
                }
            } else if self.consume(TokenKind::LBracket) {
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                let start = expr.span.start;
                expr = Expr {
                    kind: ExprKind::Index {
                        object: Box::new(expr),
                        index: Box::new(index),
                    },
                    span: Span::new(start, self.previous.span.end),
                };
            } else if self.consume(TokenKind::Question) {
                let start = expr.span.start;
                expr = Expr {
                    kind: ExprKind::Try {
                        operand: Box::new(expr),
                    },
                    span: Span::new(start, self.previous.span.end),
                };
            } else if self.consume(TokenKind::As) {
                // Type cast: expr as Type
                let start = expr.span.start;
                let ty = self.parse_type()?;
                expr = Expr {
                    kind: ExprKind::Cast {
                        expr: Box::new(expr),
                        ty,
                    },
                    span: Span::new(start, self.previous.span.end),
                };
            } else if self.check(TokenKind::LBrace) {
                // Check for struct literal: Path { field: value, ... }
                // Only if the expression is a path and next tokens look like field: value
                if let ExprKind::Path(ref path) = expr.kind {
                    // Peek ahead to see if this looks like a struct literal
                    // It's a struct literal if we see { ident : or { ident , or { ident } or { }
                    if self.is_struct_literal_start() {
                        let struct_path = path.clone();
                        self.advance(); // consume {
                        let mut fields = Vec::new();

                        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
                            let field_name = self.parse_ident()?;
                            self.expect(TokenKind::Colon)?;
                            let field_value = self.parse_expr()?;
                            fields.push((field_name, field_value));

                            if !self.consume(TokenKind::Comma) {
                                break;
                            }
                        }

                        self.expect(TokenKind::RBrace)?;
                        let start = expr.span.start;
                        expr = Expr {
                            kind: ExprKind::Struct {
                                path: struct_path,
                                fields,
                            },
                            span: Span::new(start, self.previous.span.end),
                        };
                        continue;
                    }
                }
                break;
            } else {
                break;
            }
        }

        Ok(expr)
    }

    /// Check if current position looks like start of a struct literal
    /// Uses lookahead to distinguish struct literals from blocks.
    /// A struct literal looks like: `{ ident: expr, ... }` or `{ }`
    /// A block looks like: `{ stmt; ... }` or `{ expr }`
    fn is_struct_literal_start(&mut self) -> bool {
        // Current token should be LBrace
        if !self.check(TokenKind::LBrace) {
            return false;
        }

        // Peek at the token after `{`
        let first = self.peek_nth(1);

        // Empty struct literal: `{ }`
        if first.kind == TokenKind::RBrace {
            return true;
        }

        // If it's an identifier, check what follows
        if first.kind == TokenKind::Ident {
            let second = self.peek_nth(2);
            // `{ ident: ...` is definitely a struct literal
            if second.kind == TokenKind::Colon {
                return true;
            }
            // `{ ident, ...` could be struct shorthand like `Point { x, y }`
            // but be conservative - only treat as struct if followed by colon
        }

        false
    }

    fn parse_args(&mut self) -> ParseResult<Vec<Expr>> {
        let mut args = Vec::new();

        if !self.check(TokenKind::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if !self.consume(TokenKind::Comma) {
                    break;
                }
                if self.check(TokenKind::RParen) {
                    break;
                }
            }
        }

        Ok(args)
    }

    /// Parse closure parameters: x, y: i32, z
    fn parse_closure_params(&mut self) -> ParseResult<Vec<(Ident, Option<Type>)>> {
        let mut params = Vec::new();

        // Empty params: ||
        if self.check(TokenKind::Or) {
            return Ok(params);
        }

        loop {
            let name = self.parse_ident()?;
            let ty = if self.consume(TokenKind::Colon) {
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push((name, ty));

            if !self.consume(TokenKind::Comma) {
                break;
            }
            // Check if we're at the closing |
            if self.check(TokenKind::Or) {
                break;
            }
        }

        Ok(params)
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;

        // Closure: |params| expr or |params| -> Type { body }
        if self.consume(TokenKind::Or) {
            // Parse closure parameters
            let params = self.parse_closure_params()?;
            self.expect(TokenKind::Or)?;

            // Optional return type (parsed but not yet used in AST)
            let _return_type = if self.consume(TokenKind::Arrow) {
                Some(self.parse_type()?)
            } else {
                None
            };

            // Body - either a block or a single expression
            let body = if self.check(TokenKind::LBrace) {
                let block = self.parse_block()?;
                Box::new(Expr {
                    span: block.span,
                    kind: ExprKind::Block(block),
                })
            } else {
                Box::new(self.parse_expr()?)
            };

            // If we have a return type, we need to wrap params properly
            let params_with_types: Vec<Param> = params
                .into_iter()
                .map(|(name, ty)| {
                    let param_span = name.span;
                    Param {
                        span: param_span,
                        name,
                        ty: ty.unwrap_or(Type {
                            kind: TypeKind::Infer,
                            span: param_span, // Use parameter's span, not invalid (0,0)
                        }),
                        is_mut: false,
                    }
                })
                .collect();

            return Ok(Expr {
                kind: ExprKind::Closure {
                    params: params_with_types,
                    body,
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Empty closure: || expr
        if self.consume(TokenKind::OrOr) {
            let body = if self.check(TokenKind::LBrace) {
                let block = self.parse_block()?;
                Box::new(Expr {
                    span: block.span,
                    kind: ExprKind::Block(block),
                })
            } else {
                Box::new(self.parse_expr()?)
            };

            return Ok(Expr {
                kind: ExprKind::Closure {
                    params: Vec::new(),
                    body,
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Literals
        if self.check(TokenKind::IntLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            let value = parse_int(text).map_err(|_| ParseError::Custom {
                message: format!("invalid integer literal: {}", text),
                span: token.span,
            })?;
            return Ok(Expr {
                kind: ExprKind::Literal(Literal::Int(value)),
                span: token.span,
            });
        }

        if self.check(TokenKind::FloatLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            let value: f64 = text.parse().map_err(|_| ParseError::Custom {
                message: format!("invalid float literal: {}", text),
                span: token.span,
            })?;
            return Ok(Expr {
                kind: ExprKind::Literal(Literal::Float(value)),
                span: token.span,
            });
        }

        if self.check(TokenKind::StringLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            let value = parse_string(text);
            return Ok(Expr {
                kind: ExprKind::Literal(Literal::String(value)),
                span: token.span,
            });
        }

        if self.check(TokenKind::CharLiteral) {
            let token = self.advance();
            let text = self.text(&token);
            let value = parse_char(text);
            return Ok(Expr {
                kind: ExprKind::Literal(Literal::Char(value)),
                span: token.span,
            });
        }

        if self.consume(TokenKind::True) {
            return Ok(Expr {
                kind: ExprKind::Literal(Literal::Bool(true)),
                span: Span::new(start, self.previous.span.end),
            });
        }

        if self.consume(TokenKind::False) {
            return Ok(Expr {
                kind: ExprKind::Literal(Literal::Bool(false)),
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Control flow
        if self.check(TokenKind::If) {
            return self.parse_if();
        }

        if self.check(TokenKind::Match) {
            return self.parse_match();
        }

        // Check for labeled loops: 'label: loop/while/for
        if self.check(TokenKind::Label) {
            let label_token = self.current.clone();
            let label_text = label_token.span.text(self.source());
            // Remove the leading quote from the label
            let label_name = label_text[1..].to_string();
            self.advance();

            self.expect(TokenKind::Colon)?;

            if self.check(TokenKind::Loop) {
                return self.parse_loop_with_label(Some(label_name));
            } else if self.check(TokenKind::While) {
                return self.parse_while_with_label(Some(label_name));
            } else if self.check(TokenKind::For) {
                return self.parse_for_with_label(Some(label_name));
            } else {
                return Err(ParseError::Custom {
                    message: "expected 'loop', 'while', or 'for' after label".to_string(),
                    span: self.current.span,
                });
            }
        }

        if self.check(TokenKind::Loop) {
            return self.parse_loop_with_label(None);
        }

        if self.check(TokenKind::While) {
            return self.parse_while_with_label(None);
        }

        if self.check(TokenKind::For) {
            return self.parse_for_with_label(None);
        }

        if self.consume(TokenKind::Break) {
            // Check for optional label: break 'label
            let label = if self.check(TokenKind::Label) {
                let label_token = self.current.clone();
                let label_text = label_token.span.text(self.source());
                let label_name = label_text[1..].to_string();
                self.advance();
                Some(Ident {
                    name: label_name,
                    span: label_token.span,
                })
            } else {
                None
            };

            // Check for optional value (only if no label or after label)
            let value = if !self.check(TokenKind::Semicolon)
                && !self.check(TokenKind::RBrace)
                && !self.check(TokenKind::Comma)
                && label.is_none()  // Don't parse value if we have a label (ambiguous)
            {
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            return Ok(Expr {
                kind: ExprKind::Break { label, value },
                span: Span::new(start, self.previous.span.end),
            });
        }

        if self.consume(TokenKind::Continue) {
            // Check for optional label: continue 'label
            let label = if self.check(TokenKind::Label) {
                let label_token = self.current.clone();
                let label_text = label_token.span.text(self.source());
                let label_name = label_text[1..].to_string();
                self.advance();
                Some(Ident {
                    name: label_name,
                    span: label_token.span,
                })
            } else {
                None
            };
            return Ok(Expr {
                kind: ExprKind::Continue { label },
                span: Span::new(start, self.previous.span.end),
            });
        }

        if self.consume(TokenKind::Return) {
            let value = if !self.check(TokenKind::Semicolon) && !self.check(TokenKind::RBrace) {
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            return Ok(Expr {
                kind: ExprKind::Return { value },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Genesis-specific
        if self.consume(TokenKind::Spawn) {
            // Check if this is spawn(expr) for async tasks or spawn Actor for actor spawning
            if self.check(TokenKind::LParen) {
                // spawn(future_expr) - async task spawning
                self.expect(TokenKind::LParen)?;
                let future = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                return Ok(Expr {
                    kind: ExprKind::SpawnTask { future: Box::new(future) },
                    span: Span::new(start, self.previous.span.end),
                });
            } else {
                // spawn ActorPath(args) - actor spawning
                let path = self.parse_path()?;
                let args = if self.consume(TokenKind::LParen) {
                    let args = self.parse_args()?;
                    self.expect(TokenKind::RParen)?;
                    args
                } else {
                    Vec::new()
                };
                return Ok(Expr {
                    kind: ExprKind::Spawn { actor: path, args },
                    span: Span::new(start, self.previous.span.end),
                });
            }
        }

        if self.consume(TokenKind::Reply) {
            let value = self.parse_expr()?;
            return Ok(Expr {
                kind: ExprKind::Reply {
                    value: Box::new(value),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Select expression: select! { a = f1 => body1, b = f2 => body2 }
        if self.check(TokenKind::Select) {
            return self.parse_select();
        }

        // Join expression: join!(f1, f2, f3)
        if self.check(TokenKind::Join) {
            return self.parse_join();
        }

        // Grouping or tuple
        if self.consume(TokenKind::LParen) {
            if self.check(TokenKind::RParen) {
                self.advance();
                return Ok(Expr {
                    kind: ExprKind::Tuple(Vec::new()),
                    span: Span::new(start, self.previous.span.end),
                });
            }

            let first = self.parse_expr()?;

            if self.consume(TokenKind::Comma) {
                let mut elements = vec![first];
                if !self.check(TokenKind::RParen) {
                    loop {
                        elements.push(self.parse_expr()?);
                        if !self.consume(TokenKind::Comma) {
                            break;
                        }
                        if self.check(TokenKind::RParen) {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen)?;
                return Ok(Expr {
                    kind: ExprKind::Tuple(elements),
                    span: Span::new(start, self.previous.span.end),
                });
            }

            self.expect(TokenKind::RParen)?;
            return Ok(first);
        }

        // Array
        if self.consume(TokenKind::LBracket) {
            let mut elements = Vec::new();
            if !self.check(TokenKind::RBracket) {
                loop {
                    elements.push(self.parse_expr()?);
                    if !self.consume(TokenKind::Comma) {
                        break;
                    }
                    if self.check(TokenKind::RBracket) {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RBracket)?;
            return Ok(Expr {
                kind: ExprKind::Array(elements),
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Block
        if self.check(TokenKind::LBrace) {
            let block = self.parse_block()?;
            return Ok(Expr {
                span: block.span,
                kind: ExprKind::Block(block),
            });
        }

        // Primitive type path (e.g., i64::to_string, bool::to_string)
        if let Some(prim_name) = self.try_primitive_type() {
            if self.peek_is(TokenKind::ColonColon) {
                let path = self.parse_primitive_type_path(prim_name)?;
                return Ok(Expr {
                    span: path.span,
                    kind: ExprKind::Path(path),
                });
            }
        }

        // Path (variable or qualified name) or macro invocation
        if self.check(TokenKind::Ident) || self.check(TokenKind::SelfValue) || self.check(TokenKind::SelfType) {
            // Check if this is a macro invocation: name!(...)
            if self.check(TokenKind::Ident) {
                // Peek ahead to see if this is name!
                let name_token = self.current.clone();
                if self.peek_nth(1).kind == TokenKind::Not {
                    // This is a macro invocation
                    let name = Ident {
                        name: self.text(&name_token).to_string(),
                        span: name_token.span,
                    };
                    self.advance(); // consume name
                    let invocation = self.parse_macro_invocation(name)?;
                    return Ok(Expr {
                        span: invocation.span,
                        kind: ExprKind::MacroCall(invocation),
                    });
                }
            }

            let path = self.parse_path()?;
            return Ok(Expr {
                span: path.span,
                kind: ExprKind::Path(path),
            });
        }

        Err(ParseError::UnexpectedToken {
            expected: "expression".to_string(),
            found: self.current.kind.clone(),
            span: self.current.span,
        })
    }

    fn parse_if(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::If)?;

        let condition = self.parse_expr()?;
        let then_branch = self.parse_block()?;

        let else_branch = if self.consume(TokenKind::Else) {
            if self.check(TokenKind::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                let block = self.parse_block()?;
                Some(Box::new(Expr {
                    span: block.span,
                    kind: ExprKind::Block(block),
                }))
            }
        } else {
            None
        };

        Ok(Expr {
            kind: ExprKind::If {
                condition: Box::new(condition),
                then_branch,
                else_branch,
            },
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_match(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::Match)?;

        let scrutinee = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let arm_start = self.current.span.start;
            let pattern = self.parse_pattern()?;

            let guard = if self.consume(TokenKind::If) {
                Some(self.parse_expr()?)
            } else {
                None
            };

            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expr()?;
            self.consume(TokenKind::Comma);

            arms.push(MatchArm {
                pattern,
                guard,
                body,
                span: Span::new(arm_start, self.previous.span.end),
            });
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            },
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_loop_with_label(&mut self, label: Option<String>) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::Loop)?;
        let body = self.parse_block()?;

        let label_ident = label.map(|name| Ident {
            name,
            span: Span::new(start, start),
        });

        Ok(Expr {
            kind: ExprKind::Loop { body, label: label_ident },
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_while_with_label(&mut self, label: Option<String>) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::While)?;
        let condition = self.parse_expr()?;
        let body = self.parse_block()?;

        let label_ident = label.map(|name| Ident {
            name,
            span: Span::new(start, start),
        });

        Ok(Expr {
            kind: ExprKind::While {
                condition: Box::new(condition),
                body,
                label: label_ident,
            },
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_for_with_label(&mut self, label: Option<String>) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::For)?;
        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::In)?;
        let iterable = self.parse_expr()?;
        let body = self.parse_block()?;

        let label_ident = label.map(|name| Ident {
            name,
            span: Span::new(start, start),
        });

        Ok(Expr {
            kind: ExprKind::For {
                pattern,
                iterable: Box::new(iterable),
                body,
                label: label_ident,
            },
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse select expression: `select! { a = f1 => body1, b = f2 => body2 }`
    fn parse_select(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::Select)?;

        // Optional '!' for macro-like syntax
        self.consume(TokenKind::Not);

        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let arm_start = self.current.span.start;

            // Parse binding: `x = future_expr => body`
            let binding = self.parse_ident()?;
            self.expect(TokenKind::Eq)?;
            let future = self.parse_expr()?;
            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expr()?;

            self.consume(TokenKind::Comma);

            arms.push(ast::SelectArm {
                binding,
                future,
                body,
                span: Span::new(arm_start, self.previous.span.end),
            });
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr {
            kind: ExprKind::Select { arms },
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse join expression: `join!(f1, f2, f3)`
    fn parse_join(&mut self) -> ParseResult<Expr> {
        let start = self.current.span.start;
        self.expect(TokenKind::Join)?;

        // Optional '!' for macro-like syntax
        self.consume(TokenKind::Not);

        self.expect(TokenKind::LParen)?;

        let mut futures = Vec::new();
        if !self.check(TokenKind::RParen) {
            loop {
                futures.push(self.parse_expr()?);
                if !self.consume(TokenKind::Comma) {
                    break;
                }
                if self.check(TokenKind::RParen) {
                    break;
                }
            }
        }

        self.expect(TokenKind::RParen)?;

        Ok(Expr {
            kind: ExprKind::Join { futures },
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Block parsing ============

    fn parse_block(&mut self) -> ParseResult<Block> {
        let start = self.current.span.start;
        self.expect(TokenKind::LBrace)?;

        let mut stmts = Vec::new();
        let mut final_expr = None;

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.check(TokenKind::Let) {
                stmts.push(self.parse_let_stmt()?);
            } else {
                let expr = self.parse_expr()?;

                if self.consume(TokenKind::Semicolon) {
                    stmts.push(Stmt {
                        span: expr.span,
                        kind: StmtKind::Expr(expr),
                    });
                } else if self.check(TokenKind::RBrace) {
                    final_expr = Some(Box::new(expr));
                } else {
                    stmts.push(Stmt {
                        span: expr.span,
                        kind: StmtKind::Expr(expr),
                    });
                }
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Block {
            stmts,
            expr: final_expr,
            span: Span::new(start, self.previous.span.end),
        })
    }

    fn parse_let_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.current.span.start;
        self.expect(TokenKind::Let)?;

        let pattern = self.parse_pattern()?;

        let ty = if self.consume(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let value = if self.consume(TokenKind::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.consume(TokenKind::Semicolon);

        Ok(Stmt {
            kind: StmtKind::Let { pattern, ty, value },
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Type parsing ============

    fn parse_type(&mut self) -> ParseResult<Type> {
        let start = self.current.span.start;

        // Trait object: dyn Trait
        if self.consume(TokenKind::Dyn) {
            let trait_ident = self.parse_ident()?;
            return Ok(Type {
                kind: TypeKind::TraitObject {
                    trait_name: trait_ident.name,
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Reference types
        if self.consume(TokenKind::And) {
            let mutable = self.consume(TokenKind::Mut);
            let inner = self.parse_type()?;
            return Ok(Type {
                kind: TypeKind::Reference {
                    mutable,
                    inner: Box::new(inner),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Tuple or grouped type
        if self.consume(TokenKind::LParen) {
            if self.check(TokenKind::RParen) {
                self.advance();
                return Ok(Type {
                    kind: TypeKind::Tuple(Vec::new()),
                    span: Span::new(start, self.previous.span.end),
                });
            }

            let first = self.parse_type()?;
            if self.consume(TokenKind::Comma) {
                let mut types = vec![first];
                if !self.check(TokenKind::RParen) {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.consume(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen)?;
                return Ok(Type {
                    kind: TypeKind::Tuple(types),
                    span: Span::new(start, self.previous.span.end),
                });
            }
            self.expect(TokenKind::RParen)?;
            return Ok(first);
        }

        // Array or slice
        if self.consume(TokenKind::LBracket) {
            let element = self.parse_type()?;
            if self.consume(TokenKind::Semicolon) {
                let size = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                return Ok(Type {
                    kind: TypeKind::Array {
                        element: Box::new(element),
                        size: Box::new(size),
                    },
                    span: Span::new(start, self.previous.span.end),
                });
            }
            self.expect(TokenKind::RBracket)?;
            return Ok(Type {
                kind: TypeKind::Slice {
                    element: Box::new(element),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Never type
        if self.consume(TokenKind::Not) {
            return Ok(Type {
                kind: TypeKind::Never,
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Self type or Self::AssociatedType
        if self.consume(TokenKind::SelfType) {
            let self_span = self.previous.span;

            // Check for associated type: Self::Item
            if self.consume(TokenKind::ColonColon) {
                let assoc_ident = self.parse_ident()?;
                let base = Type {
                    kind: TypeKind::SelfType,
                    span: self_span,
                };
                return Ok(Type {
                    kind: TypeKind::Projection {
                        base: Box::new(base),
                        assoc_name: assoc_ident.name,
                    },
                    span: Span::new(start, self.previous.span.end),
                });
            }

            // Just Self type
            return Ok(Type {
                kind: TypeKind::SelfType,
                span: self_span,
            });
        }

        // Primitive types - handle them as paths with single segment
        if let Some(name) = self.try_primitive_type() {
            let token = self.advance();
            let ident = Ident {
                name: name.to_string(),
                span: token.span,
            };
            let path = Path {
                segments: vec![PathSegment {
                    ident,
                    generics: None,
                }],
                span: token.span,
            };
            return Ok(Type {
                span: path.span,
                kind: TypeKind::Path(path),
            });
        }

        // Path type (uses parse_type_path to allow generics like Vec<T>)
        let path = self.parse_type_path()?;
        Ok(Type {
            span: path.span,
            kind: TypeKind::Path(path),
        })
    }

    /// Check if current token is a primitive type and return its name
    fn try_primitive_type(&self) -> Option<&'static str> {
        match self.current.kind {
            TokenKind::I8 => Some("i8"),
            TokenKind::I16 => Some("i16"),
            TokenKind::I32 => Some("i32"),
            TokenKind::I64 => Some("i64"),
            TokenKind::I128 => Some("i128"),
            TokenKind::U8 => Some("u8"),
            TokenKind::U16 => Some("u16"),
            TokenKind::U32 => Some("u32"),
            TokenKind::U64 => Some("u64"),
            TokenKind::U128 => Some("u128"),
            TokenKind::F32 => Some("f32"),
            TokenKind::F64 => Some("f64"),
            TokenKind::Bool => Some("bool"),
            TokenKind::Char => Some("char"),
            TokenKind::Str => Some("str"),
            _ => None,
        }
    }

    /// Check if the next token (after current) is a specific kind
    fn peek_is(&mut self, kind: TokenKind) -> bool {
        self.peek_nth(1).kind == kind
    }

    /// Parse a path that starts with a primitive type (e.g., i64::to_string)
    fn parse_primitive_type_path(&mut self, prim_name: &str) -> ParseResult<Path> {
        let start = self.current.span.start;
        let prim_span = self.current.span;

        // Consume the primitive type token
        self.advance();

        // We know there's a :: (checked by caller)
        self.expect(TokenKind::ColonColon)?;

        // Parse the rest of the path
        let mut segments = vec![PathSegment {
            ident: Ident {
                name: prim_name.to_string(),
                span: prim_span,
            },
            generics: None,
        }];

        // Parse remaining segments
        loop {
            let ident = self.parse_ident()?;
            segments.push(PathSegment {
                ident,
                generics: None,
            });

            if !self.consume(TokenKind::ColonColon) {
                break;
            }
        }

        Ok(Path {
            segments,
            span: Span::new(start, self.previous.span.end),
        })
    }

    // ============ Pattern parsing ============

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        let start = self.current.span.start;

        // Wildcard
        if self.check(TokenKind::Ident) && self.text(&self.current) == "_" {
            self.advance();
            return Ok(Pattern {
                kind: PatternKind::Wildcard,
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Literal patterns
        if self.check(TokenKind::IntLiteral)
            || self.check(TokenKind::StringLiteral)
            || self.check(TokenKind::True)
            || self.check(TokenKind::False)
        {
            let expr = self.parse_primary()?;
            if let ExprKind::Literal(lit) = expr.kind {
                return Ok(Pattern {
                    kind: PatternKind::Literal(lit),
                    span: expr.span,
                });
            }
        }

        // Tuple pattern
        if self.consume(TokenKind::LParen) {
            let mut patterns = Vec::new();
            if !self.check(TokenKind::RParen) {
                loop {
                    patterns.push(self.parse_pattern()?);
                    if !self.consume(TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;
            return Ok(Pattern {
                kind: PatternKind::Tuple(patterns),
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Ref pattern
        if self.consume(TokenKind::And) {
            let mutable = self.consume(TokenKind::Mut);
            let inner = self.parse_pattern()?;
            return Ok(Pattern {
                kind: PatternKind::Ref {
                    mutable,
                    pattern: Box::new(inner),
                },
                span: Span::new(start, self.previous.span.end),
            });
        }

        // Binding pattern (possibly mutable) or path pattern (enum variant)
        let mutable = self.consume(TokenKind::Mut);

        // If mutable, it's definitely just an ident binding
        if mutable {
            let name = self.parse_ident()?;
            return Ok(Pattern {
                span: Span::new(start, self.previous.span.end),
                kind: PatternKind::Ident { name, mutable },
            });
        }

        // Try to parse as path - might be enum variant like Option::Some
        let name = self.parse_ident()?;

        // Check if this is a path (has ::)
        if self.check(TokenKind::ColonColon) {
            // It's a path pattern - parse the rest of the path
            let mut segments = vec![PathSegment {
                ident: name.clone(),
                generics: None,
            }];

            while self.consume(TokenKind::ColonColon) {
                let ident = self.parse_ident()?;
                segments.push(PathSegment {
                    ident,
                    generics: None,
                });
            }

            let path = Path {
                span: Span::new(start, self.previous.span.end),
                segments,
            };

            // Check for tuple variant: Path::Variant(fields)
            if self.consume(TokenKind::LParen) {
                let mut fields = Vec::new();
                if !self.check(TokenKind::RParen) {
                    loop {
                        fields.push(self.parse_pattern()?);
                        if !self.consume(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen)?;
                return Ok(Pattern {
                    span: Span::new(start, self.previous.span.end),
                    kind: PatternKind::Enum { path, fields },
                });
            }

            // Unit variant: Path::Variant
            return Ok(Pattern {
                span: Span::new(start, self.previous.span.end),
                kind: PatternKind::Enum {
                    path,
                    fields: Vec::new(),
                },
            });
        }

        // Check if simple ident is followed by tuple (unit enum variant with tuple)
        if self.consume(TokenKind::LParen) {
            let path = Path {
                span: name.span,
                segments: vec![PathSegment {
                    ident: name,
                    generics: None,
                }],
            };
            let mut fields = Vec::new();
            if !self.check(TokenKind::RParen) {
                loop {
                    fields.push(self.parse_pattern()?);
                    if !self.consume(TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;
            return Ok(Pattern {
                span: Span::new(start, self.previous.span.end),
                kind: PatternKind::Enum { path, fields },
            });
        }

        // Simple identifier binding
        Ok(Pattern {
            span: Span::new(start, self.previous.span.end),
            kind: PatternKind::Ident { name, mutable },
        })
    }

    // ============ Helper functions ============

    fn parse_ident(&mut self) -> ParseResult<Ident> {
        if self.check(TokenKind::Ident) {
            let token = self.advance();
            let name = self.text(&token).to_string();
            Ok(Ident {
                name,
                span: token.span,
            })
        } else if self.check(TokenKind::SelfValue) {
            let token = self.advance();
            Ok(Ident {
                name: "self".to_string(),
                span: token.span,
            })
        } else if self.check(TokenKind::SelfType) {
            let token = self.advance();
            Ok(Ident {
                name: "Self".to_string(),
                span: token.span,
            })
        } else {
            Err(ParseError::UnexpectedToken {
                expected: "identifier".to_string(),
                found: self.current.kind.clone(),
                span: self.current.span,
            })
        }
    }

    /// Parse a path (identifier or qualified path like std::io::read)
    /// Note: Generic arguments in expression context (like `foo::<T>()`) require
    /// the turbofish syntax `::< >` to disambiguate from comparison operators.
    fn parse_path(&mut self) -> ParseResult<Path> {
        let start = self.current.span.start;
        let mut segments = Vec::new();

        loop {
            let ident = self.parse_ident()?;

            // Only parse generics with turbofish syntax (::< >) to avoid
            // ambiguity with comparison operators like x < y
            // For type contexts, use parse_type_path instead
            segments.push(PathSegment {
                ident,
                generics: None,
            });

            if !self.consume(TokenKind::ColonColon) {
                break;
            }
        }

        Ok(Path {
            segments,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse a path in type context (allows < > for generics)
    fn parse_type_path(&mut self) -> ParseResult<Path> {
        let start = self.current.span.start;
        let mut segments = Vec::new();

        loop {
            let ident = self.parse_ident()?;
            let generics = if self.consume(TokenKind::Lt) {
                let mut types = Vec::new();
                if !self.check(TokenKind::Gt) {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.consume(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::Gt)?;
                Some(types)
            } else {
                None
            };

            segments.push(PathSegment { ident, generics });

            if !self.consume(TokenKind::ColonColon) {
                break;
            }
        }

        Ok(Path {
            segments,
            span: Span::new(start, self.previous.span.end),
        })
    }

    /// Parse an optional where clause: `where T: Clone, U: Debug + Display`
    fn parse_where_clause_opt(&mut self) -> ParseResult<Option<WhereClause>> {
        if !self.consume(TokenKind::Where) {
            return Ok(None);
        }

        let start = self.previous.span.start;
        let mut predicates = Vec::new();

        loop {
            let ty_start = self.current.span.start;
            let ty = self.parse_type()?;

            self.expect(TokenKind::Colon)?;

            // Parse bounds: Clone + Debug + Display
            let mut bounds = vec![self.parse_type()?];
            while self.consume(TokenKind::Plus) {
                bounds.push(self.parse_type()?);
            }

            predicates.push(WherePredicate {
                ty,
                bounds,
                span: Span::new(ty_start, self.previous.span.end),
            });

            // Continue if comma, stop at { or other terminators
            if !self.consume(TokenKind::Comma) {
                break;
            }
            // Stop if we hit the block start
            if self.check(TokenKind::LBrace) {
                break;
            }
        }

        Ok(Some(WhereClause {
            predicates,
            span: Span::new(start, self.previous.span.end),
        }))
    }

    fn parse_generics_opt(&mut self) -> ParseResult<Option<Generics>> {
        if !self.consume(TokenKind::Lt) {
            return Ok(None);
        }

        let start = self.previous.span.start;
        let mut params = Vec::new();

        loop {
            let name = self.parse_ident()?;
            let bounds = if self.consume(TokenKind::Colon) {
                let mut bounds = vec![self.parse_type()?];
                while self.consume(TokenKind::Plus) {
                    bounds.push(self.parse_type()?);
                }
                bounds
            } else {
                Vec::new()
            };

            let default = if self.consume(TokenKind::Eq) {
                Some(self.parse_type()?)
            } else {
                None
            };

            params.push(GenericParam {
                span: Span::new(name.span.start, self.previous.span.end),
                name,
                bounds,
                default,
            });

            if !self.consume(TokenKind::Comma) {
                break;
            }
            if self.check(TokenKind::Gt) {
                break;
            }
        }

        self.expect(TokenKind::Gt)?;

        Ok(Some(Generics {
            params,
            where_clause: None,
            span: Span::new(start, self.previous.span.end),
        }))
    }
}

// ============ Helper functions ============

fn parse_int(s: &str) -> Result<i128, ()> {
    let s = s.replace('_', "");
    if s.starts_with("0x") || s.starts_with("0X") {
        i128::from_str_radix(&s[2..], 16).map_err(|_| ())
    } else if s.starts_with("0b") || s.starts_with("0B") {
        i128::from_str_radix(&s[2..], 2).map_err(|_| ())
    } else if s.starts_with("0o") || s.starts_with("0O") {
        i128::from_str_radix(&s[2..], 8).map_err(|_| ())
    } else {
        s.parse().map_err(|_| ())
    }
}

fn parse_string(s: &str) -> String {
    // Remove quotes and handle escapes
    let s = &s[1..s.len() - 1];
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('0') => result.push('\0'),
                Some(c) => result.push(c),
                None => {}
            }
        } else {
            result.push(c);
        }
    }

    result
}

fn parse_char(s: &str) -> char {
    // Unicode replacement character used for invalid/malformed char literals
    const REPLACEMENT: char = '\u{FFFD}';

    if s.len() < 2 {
        return REPLACEMENT;
    }
    let s = &s[1..s.len() - 1];
    if s.is_empty() {
        return REPLACEMENT;
    }

    if s.starts_with('\\') {
        match s.chars().nth(1) {
            Some('n') => '\n',
            Some('r') => '\r',
            Some('t') => '\t',
            Some('\\') => '\\',
            Some('\'') => '\'',
            Some('0') => '\0',
            Some(c) => c, // Unknown escape: use literal character
            None => REPLACEMENT,
        }
    } else {
        s.chars().next().unwrap_or(REPLACEMENT)
    }
}

/// Parse source code into an AST
pub fn parse(source: &str) -> (Program, Vec<ParseError>) {
    let mut parser = Parser::new(source);
    let program = parser.parse_program().unwrap_or(Program {
        items: Vec::new(),
        span: Span::new(0, source.len()),
    });
    let errors = parser.errors.clone();
    (program, errors)
}

/// Parse source code from a file, with support for external module resolution
pub fn parse_file(path: &std::path::Path) -> Result<(Program, Vec<ParseError>), std::io::Error> {
    let source = std::fs::read_to_string(path)?;
    // We need to leak the string to satisfy the borrow checker
    // since the parser holds a reference to the source
    let source: &'static str = Box::leak(source.into_boxed_str());
    let mut parser = Parser::with_path(source, path.to_path_buf());
    let program = parser.parse_program().unwrap_or(Program {
        items: Vec::new(),
        span: Span::new(0, source.len()),
    });
    let errors = parser.errors.clone();
    Ok((program, errors))
}

/// Resolve external modules in a parsed program
/// This function takes a program and resolves any `mod foo;` declarations
/// by loading the corresponding .gl files
pub fn resolve_external_modules(program: &mut Program, base_path: &std::path::Path) -> Vec<ParseError> {
    let mut errors = Vec::new();

    for item in &mut program.items {
        if let Item::Mod(ref mut mod_def) = item {
            if mod_def.items.is_none() {
                // External module - need to resolve from file
                let mod_name = &mod_def.name.name;

                // Try foo.gl first
                let mut mod_path = base_path.to_path_buf();
                mod_path.push(format!("{}.gl", mod_name));

                if !mod_path.exists() {
                    // Try foo/mod.gl
                    mod_path = base_path.to_path_buf();
                    mod_path.push(mod_name);
                    mod_path.push("mod.gl");
                }

                if mod_path.exists() {
                    match std::fs::read_to_string(&mod_path) {
                        Ok(source) => {
                            let source: &'static str = Box::leak(source.into_boxed_str());
                            let (mod_program, parse_errors) = parse(source);
                            errors.extend(parse_errors);
                            // Set the items from the external file
                            mod_def.items = Some(mod_program.items);
                        }
                        Err(e) => {
                            errors.push(ParseError::Custom {
                                message: format!("Failed to read module file '{}': {}", mod_path.display(), e),
                                span: mod_def.span,
                            });
                        }
                    }
                } else {
                    errors.push(ParseError::Custom {
                        message: format!("Cannot find module '{}' - tried {}.gl and {}/mod.gl",
                            mod_name, mod_name, mod_name),
                        span: mod_def.span,
                    });
                }
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(source: &str) -> Program {
        let (program, errors) = parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        program
    }

    #[test]
    fn test_empty_program() {
        let program = parse_ok("");
        assert!(program.items.is_empty());
    }

    #[test]
    fn test_simple_function() {
        let program = parse_ok("fn main() {}");
        assert_eq!(program.items.len(), 1);
        if let Item::Function(f) = &program.items[0] {
            assert_eq!(f.name.name, "main");
            assert!(f.params.is_empty());
            assert!(f.return_type.is_none());
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_function_with_params() {
        let program = parse_ok("fn add(a: i32, b: i32) -> i32 { a + b }");
        if let Item::Function(f) = &program.items[0] {
            assert_eq!(f.name.name, "add");
            assert_eq!(f.params.len(), 2);
            assert!(f.return_type.is_some());
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_struct() {
        let program = parse_ok("struct Point { x: i32, y: i32 }");
        if let Item::Struct(s) = &program.items[0] {
            assert_eq!(s.name.name, "Point");
            assert_eq!(s.fields.len(), 2);
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_enum() {
        let program = parse_ok("enum Option<T> { Some(T), None }");
        if let Item::Enum(e) = &program.items[0] {
            assert_eq!(e.name.name, "Option");
            assert_eq!(e.variants.len(), 2);
        } else {
            panic!("Expected enum");
        }
    }

    #[test]
    fn test_actor() {
        let program = parse_ok(
            r#"
            actor Counter {
                count: i64,
                receive {
                    Increment => self.count += 1
                }
            }
            "#,
        );
        if let Item::Actor(a) = &program.items[0] {
            assert_eq!(a.name.name, "Counter");
            assert_eq!(a.state.len(), 1);
            assert_eq!(a.receive.len(), 1);
        } else {
            panic!("Expected actor");
        }
    }

    #[test]
    fn test_let_statement() {
        let program = parse_ok("fn main() { let x = 42; }");
        if let Item::Function(f) = &program.items[0] {
            assert_eq!(f.body.stmts.len(), 1);
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_if_expression() {
        let program = parse_ok("fn main() { if true { 1 } else { 2 } }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_binary_expressions() {
        let program = parse_ok("fn main() { 1 + 2 * 3 }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_method_call() {
        let program = parse_ok("fn main() { x.foo().bar(1, 2) }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_struct_literal() {
        let program = parse_ok("fn main() { Point { x: 1, y: 2 } }");
        assert_eq!(program.items.len(), 1);
        if let Item::Function(f) = &program.items[0] {
            if let Some(expr) = &f.body.expr {
                if let ExprKind::Struct { path, fields } = &expr.kind {
                    assert_eq!(path.segments[0].ident.name, "Point");
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].0.name, "x");
                    assert_eq!(fields[1].0.name, "y");
                } else {
                    panic!("Expected struct literal");
                }
            } else {
                panic!("Expected expression in block");
            }
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_struct_literal_in_return() {
        let program = parse_ok(
            r#"
            impl Point {
                fn new(x: f64, y: f64) -> Point {
                    Point { x: x, y: y }
                }
            }
            "#,
        );
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_empty_struct_literal() {
        let program = parse_ok("fn main() { Empty { } }");
        assert_eq!(program.items.len(), 1);
    }
}
