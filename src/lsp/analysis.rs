//! Document analysis for the LSP server
//!
//! Provides parsing and type checking for Genesis source files.

use std::collections::HashMap;
use genesis::ast::{self, Program, Item, Expr, ExprKind, Ident, FnDef, StructDef, EnumDef, TraitDef, ImplDef, ImplItem, StmtKind};
use genesis::parser::{self, ParseError};
use genesis::span::Span;
use genesis::typeck::{self, Ty, TypeError, TypedProgram};
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticSeverity, Position, Range, CompletionItem,
    CompletionItemKind, SymbolKind, DocumentSymbol, Location, Hover,
    HoverContents, MarkedString, SignatureHelp, SignatureInformation,
    ParameterInformation,
};

use crate::utils::LineIndex;

/// Result of analyzing a document
#[derive(Debug)]
pub struct AnalysisResult {
    /// The parsed AST
    pub program: Program,
    /// Parse errors
    pub parse_errors: Vec<ParseError>,
    /// Type checking result
    pub typed_program: Option<TypedProgram>,
    /// Type errors
    pub type_errors: Vec<TypeError>,
    /// Symbol definitions: name -> (span, kind, type)
    pub definitions: HashMap<String, SymbolDefinition>,
    /// Symbol references: span -> name
    pub references: HashMap<Span, String>,
}

/// A symbol definition
#[derive(Debug, Clone)]
pub struct SymbolDefinition {
    pub name: String,
    pub span: Span,
    pub kind: SymbolDefKind,
    pub ty: Option<Ty>,
    pub doc: Option<String>,
}

/// Kind of symbol definition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolDefKind {
    Function,
    Struct,
    Enum,
    Trait,
    Variable,
    Constant,
    Field,
    Method,
    Actor,
    Module,
    TypeAlias,
}

impl SymbolDefKind {
    pub fn to_symbol_kind(self) -> SymbolKind {
        match self {
            SymbolDefKind::Function => SymbolKind::FUNCTION,
            SymbolDefKind::Struct => SymbolKind::STRUCT,
            SymbolDefKind::Enum => SymbolKind::ENUM,
            SymbolDefKind::Trait => SymbolKind::INTERFACE,
            SymbolDefKind::Variable => SymbolKind::VARIABLE,
            SymbolDefKind::Constant => SymbolKind::CONSTANT,
            SymbolDefKind::Field => SymbolKind::FIELD,
            SymbolDefKind::Method => SymbolKind::METHOD,
            SymbolDefKind::Actor => SymbolKind::CLASS,
            SymbolDefKind::Module => SymbolKind::MODULE,
            SymbolDefKind::TypeAlias => SymbolKind::TYPE_PARAMETER,
        }
    }

    pub fn to_completion_kind(self) -> CompletionItemKind {
        match self {
            SymbolDefKind::Function => CompletionItemKind::FUNCTION,
            SymbolDefKind::Struct => CompletionItemKind::STRUCT,
            SymbolDefKind::Enum => CompletionItemKind::ENUM,
            SymbolDefKind::Trait => CompletionItemKind::INTERFACE,
            SymbolDefKind::Variable => CompletionItemKind::VARIABLE,
            SymbolDefKind::Constant => CompletionItemKind::CONSTANT,
            SymbolDefKind::Field => CompletionItemKind::FIELD,
            SymbolDefKind::Method => CompletionItemKind::METHOD,
            SymbolDefKind::Actor => CompletionItemKind::CLASS,
            SymbolDefKind::Module => CompletionItemKind::MODULE,
            SymbolDefKind::TypeAlias => CompletionItemKind::TYPE_PARAMETER,
        }
    }
}

/// Analyze a Genesis source file
pub fn analyze(source: &str) -> AnalysisResult {
    // Parse the source
    let (program, parse_errors) = parser::parse(source);

    // Collect definitions from AST
    let mut definitions = HashMap::new();
    let mut references = HashMap::new();
    collect_definitions(&program, &mut definitions);

    // Type check if parsing succeeded
    let (typed_program, type_errors) = if parse_errors.is_empty() {
        match typeck::check_program(&program) {
            Ok(typed) => (Some(typed), Vec::new()),
            Err(errors) => (None, errors),
        }
    } else {
        (None, Vec::new())
    };

    // Collect references
    collect_references(&program, &mut references);

    AnalysisResult {
        program,
        parse_errors,
        typed_program,
        type_errors,
        definitions,
        references,
    }
}

/// Collect symbol definitions from the AST
fn collect_definitions(program: &Program, definitions: &mut HashMap<String, SymbolDefinition>) {
    for item in &program.items {
        match item {
            Item::Function(f) => {
                definitions.insert(
                    f.name.name.clone(),
                    SymbolDefinition {
                        name: f.name.name.clone(),
                        span: f.name.span,
                        kind: SymbolDefKind::Function,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::Struct(s) => {
                definitions.insert(
                    s.name.name.clone(),
                    SymbolDefinition {
                        name: s.name.name.clone(),
                        span: s.name.span,
                        kind: SymbolDefKind::Struct,
                        ty: None,
                        doc: None,
                    },
                );
                // Also add fields
                for field in &s.fields {
                    let field_name = format!("{}::{}", s.name.name, field.name.name);
                    definitions.insert(
                        field_name,
                        SymbolDefinition {
                            name: field.name.name.clone(),
                            span: field.name.span,
                            kind: SymbolDefKind::Field,
                            ty: None,
                            doc: None,
                        },
                    );
                }
            }
            Item::Enum(e) => {
                definitions.insert(
                    e.name.name.clone(),
                    SymbolDefinition {
                        name: e.name.name.clone(),
                        span: e.name.span,
                        kind: SymbolDefKind::Enum,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::Trait(t) => {
                definitions.insert(
                    t.name.name.clone(),
                    SymbolDefinition {
                        name: t.name.name.clone(),
                        span: t.name.span,
                        kind: SymbolDefKind::Trait,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::Const(c) => {
                definitions.insert(
                    c.name.name.clone(),
                    SymbolDefinition {
                        name: c.name.name.clone(),
                        span: c.name.span,
                        kind: SymbolDefKind::Constant,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::Actor(a) => {
                definitions.insert(
                    a.name.name.clone(),
                    SymbolDefinition {
                        name: a.name.name.clone(),
                        span: a.name.span,
                        kind: SymbolDefKind::Actor,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::TypeAlias(t) => {
                definitions.insert(
                    t.name.name.clone(),
                    SymbolDefinition {
                        name: t.name.name.clone(),
                        span: t.name.span,
                        kind: SymbolDefKind::TypeAlias,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::Impl(i) => {
                // Add methods from impl block
                for impl_item in &i.items {
                    if let ImplItem::Function(method) = impl_item {
                        let method_name = format!("{}::{}", format_type(&i.self_type), method.name.name);
                        definitions.insert(
                            method_name,
                            SymbolDefinition {
                                name: method.name.name.clone(),
                                span: method.name.span,
                                kind: SymbolDefKind::Method,
                                ty: None,
                                doc: None,
                            },
                        );
                    }
                }
            }
            Item::Mod(m) => {
                definitions.insert(
                    m.name.name.clone(),
                    SymbolDefinition {
                        name: m.name.name.clone(),
                        span: m.name.span,
                        kind: SymbolDefKind::Module,
                        ty: None,
                        doc: None,
                    },
                );
            }
            Item::Use(_) => {}
            Item::Macro(m) => {
                definitions.insert(
                    m.name.name.clone(),
                    SymbolDefinition {
                        name: m.name.name.clone(),
                        span: m.name.span,
                        kind: SymbolDefKind::Function, // Treat macros like functions for LSP purposes
                        ty: None,
                        doc: None,
                    },
                );
            }
        }
    }
}

/// Collect symbol references from the AST
fn collect_references(program: &Program, references: &mut HashMap<Span, String>) {
    for item in &program.items {
        match item {
            Item::Function(f) => {
                collect_expr_references(&f.body, references);
            }
            Item::Impl(i) => {
                for impl_item in &i.items {
                    if let ImplItem::Function(method) = impl_item {
                        collect_expr_references(&method.body, references);
                    }
                }
            }
            _ => {}
        }
    }
}

fn collect_expr_references(block: &ast::Block, references: &mut HashMap<Span, String>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. } => {
                if let Some(ref expr) = value {
                    visit_expr(expr, references);
                }
            }
            StmtKind::Expr(e) => {
                visit_expr(e, references);
            }
            StmtKind::Item(_) => {}
        }
    }
    if let Some(ref expr) = block.expr {
        visit_expr(expr, references);
    }
}

fn visit_expr(expr: &Expr, references: &mut HashMap<Span, String>) {
    match &expr.kind {
        ExprKind::Path(path) => {
            // Record reference to the path
            let name = path.segments.iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("::");
            references.insert(expr.span, name);
        }
        ExprKind::Call { func, args } => {
            visit_expr(func, references);
            for arg in args {
                visit_expr(arg, references);
            }
        }
        ExprKind::MethodCall { receiver, method: _, args } => {
            visit_expr(receiver, references);
            for arg in args {
                visit_expr(arg, references);
            }
        }
        ExprKind::Field { object, field: _ } => {
            visit_expr(object, references);
        }
        ExprKind::Index { object, index } => {
            visit_expr(object, references);
            visit_expr(index, references);
        }
        ExprKind::Binary { left, op: _, right } => {
            visit_expr(left, references);
            visit_expr(right, references);
        }
        ExprKind::Unary { op: _, operand } => {
            visit_expr(operand, references);
        }
        ExprKind::If { condition, then_branch, else_branch } => {
            visit_expr(condition, references);
            collect_expr_references(then_branch, references);
            if let Some(ref else_b) = else_branch {
                visit_expr(else_b, references);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            visit_expr(scrutinee, references);
            for arm in arms {
                visit_expr(&arm.body, references);
            }
        }
        ExprKind::Block(block) => {
            collect_expr_references(block, references);
        }
        ExprKind::Loop { body, label: _ } => {
            collect_expr_references(body, references);
        }
        ExprKind::While { condition, body, label: _ } => {
            visit_expr(condition, references);
            collect_expr_references(body, references);
        }
        ExprKind::For { pattern: _, iterable, body, label: _ } => {
            visit_expr(iterable, references);
            collect_expr_references(body, references);
        }
        ExprKind::Return { value } => {
            if let Some(ref e) = value {
                visit_expr(e, references);
            }
        }
        ExprKind::Closure { params: _, body } => {
            visit_expr(body, references);
        }
        ExprKind::Struct { path: _, fields } => {
            for (_, expr) in fields {
                visit_expr(expr, references);
            }
        }
        ExprKind::Array(elements) => {
            for e in elements {
                visit_expr(e, references);
            }
        }
        ExprKind::Tuple(elements) => {
            for e in elements {
                visit_expr(e, references);
            }
        }
        ExprKind::Ref { mutable: _, operand } => {
            visit_expr(operand, references);
        }
        ExprKind::Deref { operand } => {
            visit_expr(operand, references);
        }
        ExprKind::Cast { expr, ty: _ } => {
            visit_expr(expr, references);
        }
        ExprKind::Await { operand } => {
            visit_expr(operand, references);
        }
        ExprKind::Try { operand } => {
            visit_expr(operand, references);
        }
        ExprKind::Assign { target, value } => {
            visit_expr(target, references);
            visit_expr(value, references);
        }
        _ => {}
    }
}

/// Format a type for display
fn format_type(ty: &ast::Type) -> String {
    match &ty.kind {
        ast::TypeKind::Path(path) => {
            path.segments.iter()
                .map(|s| {
                    if let Some(ref generics) = s.generics {
                        format!("{}<{}>", s.ident.name,
                            generics.iter().map(|t| format_type(t)).collect::<Vec<_>>().join(", "))
                    } else {
                        s.ident.name.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join("::")
        }
        ast::TypeKind::Reference { mutable, inner } => {
            if *mutable {
                format!("&mut {}", format_type(inner))
            } else {
                format!("&{}", format_type(inner))
            }
        }
        ast::TypeKind::Array { element, size: _ } => {
            format!("[{}; N]", format_type(element))
        }
        ast::TypeKind::Slice { element } => {
            format!("[{}]", format_type(element))
        }
        ast::TypeKind::Tuple(elements) => {
            format!("({})", elements.iter().map(|t| format_type(t)).collect::<Vec<_>>().join(", "))
        }
        ast::TypeKind::FnPtr { params, return_type } => {
            let ret = return_type.as_ref()
                .map(|t| format_type(t))
                .unwrap_or_else(|| "()".to_string());
            format!("fn({}) -> {}",
                params.iter().map(|t| format_type(t)).collect::<Vec<_>>().join(", "),
                ret)
        }
        ast::TypeKind::Never => "!".to_string(),
        ast::TypeKind::Infer => "_".to_string(),
        ast::TypeKind::Option(inner) => format!("Option<{}>", format_type(inner)),
        ast::TypeKind::Result { ok, err } => format!("Result<{}, {}>", format_type(ok), format_type(err)),
        ast::TypeKind::SelfType => "Self".to_string(),
        ast::TypeKind::Projection { base, assoc_name } => format!("{}::{}", format_type(base), assoc_name),
    }
}

/// Convert analysis results to LSP diagnostics
pub fn to_diagnostics(result: &AnalysisResult, line_index: &LineIndex) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Parse errors
    for error in &result.parse_errors {
        let span = error.span();
        diagnostics.push(Diagnostic {
            range: line_index.span_to_range(span),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("genesis".to_string()),
            message: error.to_string(),
            related_information: None,
            tags: None,
            data: None,
        });
    }

    // Type errors
    for error in &result.type_errors {
        diagnostics.push(Diagnostic {
            range: line_index.span_to_range(error.span),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("genesis".to_string()),
            message: error.to_string(),
            related_information: None,
            tags: None,
            data: None,
        });
    }

    // Warnings from type checking
    if let Some(ref typed) = result.typed_program {
        for warning in &typed.warnings {
            diagnostics.push(Diagnostic {
                range: line_index.span_to_range(warning.span),
                severity: Some(DiagnosticSeverity::WARNING),
                code: None,
                code_description: None,
                source: Some("genesis".to_string()),
                message: warning.message.clone(),
                related_information: None,
                tags: None,
                data: None,
            });
        }
    }

    diagnostics
}

/// Get hover information at a position
pub fn get_hover(result: &AnalysisResult, line_index: &LineIndex, position: Position) -> Option<Hover> {
    let offset = line_index.position_to_offset(position);

    // First check if we have type information for this position
    if let Some(ref typed) = result.typed_program {
        for (span, ty) in &typed.expr_types {
            if offset >= span.start && offset < span.end {
                let content = format!("```genesis\n{}\n```", ty);
                return Some(Hover {
                    contents: HoverContents::Scalar(MarkedString::String(content)),
                    range: Some(line_index.span_to_range(*span)),
                });
            }
        }
    }

    // Check definitions
    for (name, def) in &result.definitions {
        if offset >= def.span.start && offset < def.span.end {
            let kind_str = match def.kind {
                SymbolDefKind::Function => "function",
                SymbolDefKind::Struct => "struct",
                SymbolDefKind::Enum => "enum",
                SymbolDefKind::Trait => "trait",
                SymbolDefKind::Variable => "variable",
                SymbolDefKind::Constant => "constant",
                SymbolDefKind::Field => "field",
                SymbolDefKind::Method => "method",
                SymbolDefKind::Actor => "actor",
                SymbolDefKind::Module => "module",
                SymbolDefKind::TypeAlias => "type",
            };
            let content = format!("```genesis\n{} {}\n```", kind_str, name);
            return Some(Hover {
                contents: HoverContents::Scalar(MarkedString::String(content)),
                range: Some(line_index.span_to_range(def.span)),
            });
        }
    }

    None
}

/// Get go-to-definition location
pub fn get_definition(result: &AnalysisResult, line_index: &LineIndex, position: Position) -> Option<(Span, String)> {
    let offset = line_index.position_to_offset(position);

    // Check if cursor is on a reference
    for (span, name) in &result.references {
        if offset >= span.start && offset < span.end {
            // Find the definition
            if let Some(def) = result.definitions.get(name) {
                return Some((def.span, name.clone()));
            }
        }
    }

    None
}

/// Get completions at a position
pub fn get_completions(result: &AnalysisResult, source: &str, line_index: &LineIndex, position: Position) -> Vec<CompletionItem> {
    let mut completions = Vec::new();
    let offset = line_index.position_to_offset(position);

    // Get the word being typed
    let prefix = get_completion_prefix(source, offset);

    // Add matching definitions
    for (name, def) in &result.definitions {
        if name.to_lowercase().starts_with(&prefix.to_lowercase()) || prefix.is_empty() {
            completions.push(CompletionItem {
                label: name.clone(),
                kind: Some(def.kind.to_completion_kind()),
                detail: def.ty.as_ref().map(|t| t.to_string()),
                documentation: def.doc.clone().map(|d| {
                    tower_lsp::lsp_types::Documentation::String(d)
                }),
                ..Default::default()
            });
        }
    }

    // Add keywords
    let keywords = [
        "fn", "let", "mut", "if", "else", "match", "while", "for", "loop",
        "break", "continue", "return", "struct", "enum", "impl", "trait",
        "pub", "use", "mod", "const", "type", "async", "await", "actor",
        "spawn", "receive", "true", "false",
    ];

    for kw in keywords {
        if kw.starts_with(&prefix) || prefix.is_empty() {
            completions.push(CompletionItem {
                label: kw.to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                ..Default::default()
            });
        }
    }

    // Add built-in types
    let types = [
        "i8", "i16", "i32", "i64", "i128", "isize",
        "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64", "bool", "char", "str",
        "String", "Vec", "Option", "Result", "Future", "HashMap", "HashSet",
    ];

    for ty in types {
        if ty.to_lowercase().starts_with(&prefix.to_lowercase()) || prefix.is_empty() {
            completions.push(CompletionItem {
                label: ty.to_string(),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                ..Default::default()
            });
        }
    }

    completions
}

/// Get the prefix of the word being completed
fn get_completion_prefix(source: &str, offset: usize) -> String {
    let bytes = source.as_bytes();
    let mut start = offset;
    while start > 0 {
        let c = bytes[start - 1] as char;
        if c.is_alphanumeric() || c == '_' {
            start -= 1;
        } else {
            break;
        }
    }
    source[start..offset].to_string()
}

/// Get document symbols (outline)
pub fn get_document_symbols(result: &AnalysisResult, line_index: &LineIndex) -> Vec<DocumentSymbol> {
    let mut symbols = Vec::new();

    for item in &result.program.items {
        if let Some(symbol) = item_to_document_symbol(item, line_index) {
            symbols.push(symbol);
        }
    }

    symbols
}

#[allow(deprecated)]
fn item_to_document_symbol(item: &Item, line_index: &LineIndex) -> Option<DocumentSymbol> {
    match item {
        Item::Function(f) => {
            let range = line_index.span_to_range(f.span);
            let selection_range = line_index.span_to_range(f.name.span);
            Some(DocumentSymbol {
                name: f.name.name.clone(),
                detail: Some(format_function_signature(f)),
                kind: SymbolKind::FUNCTION,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        Item::Struct(s) => {
            let range = line_index.span_to_range(s.span);
            let selection_range = line_index.span_to_range(s.name.span);
            let children: Vec<DocumentSymbol> = s.fields.iter()
                .map(|f| {
                    let field_range = line_index.span_to_range(f.span);
                    let field_selection = line_index.span_to_range(f.name.span);
                    DocumentSymbol {
                        name: f.name.name.clone(),
                        detail: Some(format_type(&f.ty)),
                        kind: SymbolKind::FIELD,
                        tags: None,
                        deprecated: None,
                        range: field_range,
                        selection_range: field_selection,
                        children: None,
                    }
                })
                .collect();
            Some(DocumentSymbol {
                name: s.name.name.clone(),
                detail: None,
                kind: SymbolKind::STRUCT,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: if children.is_empty() { None } else { Some(children) },
            })
        }
        Item::Enum(e) => {
            let range = line_index.span_to_range(e.span);
            let selection_range = line_index.span_to_range(e.name.span);
            let children: Vec<DocumentSymbol> = e.variants.iter()
                .map(|v| {
                    let var_range = line_index.span_to_range(v.span);
                    let var_selection = line_index.span_to_range(v.name.span);
                    DocumentSymbol {
                        name: v.name.name.clone(),
                        detail: None,
                        kind: SymbolKind::ENUM_MEMBER,
                        tags: None,
                        deprecated: None,
                        range: var_range,
                        selection_range: var_selection,
                        children: None,
                    }
                })
                .collect();
            Some(DocumentSymbol {
                name: e.name.name.clone(),
                detail: None,
                kind: SymbolKind::ENUM,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: if children.is_empty() { None } else { Some(children) },
            })
        }
        Item::Trait(t) => {
            let range = line_index.span_to_range(t.span);
            let selection_range = line_index.span_to_range(t.name.span);
            Some(DocumentSymbol {
                name: t.name.name.clone(),
                detail: None,
                kind: SymbolKind::INTERFACE,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        Item::Impl(i) => {
            let range = line_index.span_to_range(i.span);
            let name = if let Some(ref trait_) = i.trait_ {
                format!("impl {} for {}", format_type(trait_), format_type(&i.self_type))
            } else {
                format!("impl {}", format_type(&i.self_type))
            };
            let children: Vec<DocumentSymbol> = i.items.iter()
                .filter_map(|impl_item| {
                    if let ImplItem::Function(m) = impl_item {
                        let method_range = line_index.span_to_range(m.span);
                        let method_selection = line_index.span_to_range(m.name.span);
                        Some(DocumentSymbol {
                            name: m.name.name.clone(),
                            detail: Some(format_function_signature(m)),
                            kind: SymbolKind::METHOD,
                            tags: None,
                            deprecated: None,
                            range: method_range,
                            selection_range: method_selection,
                            children: None,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            Some(DocumentSymbol {
                name,
                detail: None,
                kind: SymbolKind::CLASS,
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: if children.is_empty() { None } else { Some(children) },
            })
        }
        Item::Const(c) => {
            let range = line_index.span_to_range(c.span);
            let selection_range = line_index.span_to_range(c.name.span);
            Some(DocumentSymbol {
                name: c.name.name.clone(),
                detail: c.ty.as_ref().map(|t| format_type(t)),
                kind: SymbolKind::CONSTANT,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        Item::Actor(a) => {
            let range = line_index.span_to_range(a.span);
            let selection_range = line_index.span_to_range(a.name.span);
            Some(DocumentSymbol {
                name: a.name.name.clone(),
                detail: None,
                kind: SymbolKind::CLASS,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        Item::Mod(m) => {
            let range = line_index.span_to_range(m.span);
            let selection_range = line_index.span_to_range(m.name.span);
            Some(DocumentSymbol {
                name: m.name.name.clone(),
                detail: None,
                kind: SymbolKind::MODULE,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        Item::Macro(m) => {
            let range = line_index.span_to_range(m.span);
            let selection_range = line_index.span_to_range(m.name.span);
            Some(DocumentSymbol {
                name: format!("{}!", m.name.name),
                detail: Some("macro".to_string()),
                kind: SymbolKind::FUNCTION,
                tags: None,
                deprecated: None,
                range,
                selection_range,
                children: None,
            })
        }
        _ => None,
    }
}

fn format_function_signature(f: &FnDef) -> String {
    let params: Vec<String> = f.params.iter()
        .map(|p| format!("{}: {}", p.name.name, format_type(&p.ty)))
        .collect();

    let ret = f.return_type.as_ref()
        .map(|t| format!(" -> {}", format_type(t)))
        .unwrap_or_default();

    format!("fn {}({}){}", f.name.name, params.join(", "), ret)
}

/// Find all references to a symbol
pub fn find_references(result: &AnalysisResult, line_index: &LineIndex, position: Position) -> Vec<Span> {
    let offset = line_index.position_to_offset(position);
    let mut locations = Vec::new();

    // Find the symbol at cursor
    let target_name = result.references.iter()
        .find(|(span, _)| offset >= span.start && offset < span.end)
        .map(|(_, name)| name.clone())
        .or_else(|| {
            result.definitions.iter()
                .find(|(_, def)| offset >= def.span.start && offset < def.span.end)
                .map(|(name, _)| name.clone())
        });

    if let Some(name) = target_name {
        // Add definition
        if let Some(def) = result.definitions.get(&name) {
            locations.push(def.span);
        }

        // Add all references
        for (span, ref_name) in &result.references {
            if ref_name == &name {
                locations.push(*span);
            }
        }
    }

    locations
}
