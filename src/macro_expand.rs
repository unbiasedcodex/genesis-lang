//! Macro Expansion Engine for Genesis Lang
//!
//! This module handles the expansion of macro invocations into AST nodes,
//! enabling compile-time code generation and metaprogramming.
//!
//! # Overview
//!
//! The macro system supports Rust-style `macro_rules!` declarative macros with:
//! - Pattern matching against invocation tokens
//! - Capture bindings (`$x:expr`, `$name:ident`, etc.)
//! - Repetition patterns (`$(...)*`, `$(...)+`, `$(...)? `)
//! - Multiple rules per macro (matched in order)
//!
//! # Supported Capture Types
//!
//! | Capture     | Description                    |
//! |-------------|--------------------------------|
//! | `$x:expr`   | Any expression                 |
//! | `$x:ident`  | An identifier                  |
//! | `$x:ty`     | A type                         |
//! | `$x:literal`| A literal value                |
//! | `$x:tt`     | A single token tree            |
//! | `$x:stmt`   | A statement                    |
//! | `$x:pat`    | A pattern                      |
//! | `$x:path`   | A path                         |
//! | `$x:block`  | A block                        |
//!
//! # Architecture
//!
//! ```text
//! MacroInvocation + MacroDef
//!        │
//!        ▼
//! ┌─────────────────┐
//! │ Pattern Matcher │  → Match invocation tokens against macro rules
//! └─────────────────┘
//!        │
//!        ▼
//! ┌─────────────────┐
//! │ Capture Binding │  → Extract captured values ($x:expr, etc.)
//! └─────────────────┘
//!        │
//!        ▼
//! ┌─────────────────┐
//! │  Transcription  │  → Substitute captures into expansion template
//! └─────────────────┘
//!        │
//!        ▼
//! ┌─────────────────┐
//! │  Token → AST    │  → Convert expanded tokens back to AST
//! └─────────────────┘
//! ```
//!
//! # Example
//!
//! ```genesis
//! macro_rules! my_vec {
//!     () => { Vec::new() };
//!     ($($x:expr),*) => {{
//!         let mut v = Vec::new();
//!         $(v.push($x);)*
//!         v
//!     }};
//! }
//!
//! let v = my_vec![1, 2, 3];
//! ```
//!
//! # Module Scope & Visibility
//!
//! Macros respect module boundaries and visibility modifiers:
//!
//! - **Private macros** (default) are only visible within the defining module
//! - **Public macros** (`pub macro_rules!`) are visible from other modules
//! - Use `register_in_module()` to register macros with scope information
//! - Use `set_current_module()` to set the resolution context
//! - Use `import_macro()` to import macros between modules
//!
//! # Hygiene
//!
//! The expander implements macro hygiene to prevent name collisions:
//!
//! - Local variables in macro bodies are renamed (`temp` → `__macro_temp_1`)
//! - Captured variables from the call site are preserved
//! - Keywords, builtins, and macro names are never renamed
//!
//! Hygiene can be disabled for testing with `new_without_hygiene()`.

use std::collections::HashMap;
use crate::ast::{
    Expr, ExprKind, Ident, MacroCaptureKind, MacroDef, MacroDelimiter,
    MacroExpansion, MacroInvocation, MacroPattern, MacroRepKind, MacroToken,
};
use crate::span::Span;
use crate::token::TokenKind;

/// Result type for macro expansion
pub type MacroResult<T> = Result<T, MacroError>;

/// Errors that can occur during macro expansion
#[derive(Debug, Clone)]
pub struct MacroError {
    pub kind: MacroErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum MacroErrorKind {
    /// No rule matched the invocation
    NoMatchingRule { macro_name: String },
    /// Undefined macro
    UndefinedMacro { name: String },
    /// Invalid token in expansion
    InvalidToken { message: String },
    /// Repetition mismatch
    RepetitionMismatch { expected: usize, got: usize },
    /// Missing capture
    MissingCapture { name: String },
    /// Invalid capture kind
    InvalidCaptureKind { name: String, expected: MacroCaptureKind },
    /// Recursion limit exceeded during nested macro expansion
    RecursionLimitExceeded { depth: usize, macro_name: String },
}

impl MacroError {
    pub fn no_matching_rule(macro_name: String, span: Span) -> Self {
        Self {
            kind: MacroErrorKind::NoMatchingRule { macro_name },
            span,
        }
    }

    pub fn undefined_macro(name: String, span: Span) -> Self {
        Self {
            kind: MacroErrorKind::UndefinedMacro { name },
            span,
        }
    }

    pub fn message(&self) -> String {
        match &self.kind {
            MacroErrorKind::NoMatchingRule { macro_name } => {
                format!("no rules matched for macro `{}`", macro_name)
            }
            MacroErrorKind::UndefinedMacro { name } => {
                format!("undefined macro `{}`", name)
            }
            MacroErrorKind::InvalidToken { message } => message.clone(),
            MacroErrorKind::RepetitionMismatch { expected, got } => {
                format!("repetition mismatch: expected {} elements, got {}", expected, got)
            }
            MacroErrorKind::MissingCapture { name } => {
                format!("missing capture `${}`", name)
            }
            MacroErrorKind::InvalidCaptureKind { name, expected } => {
                format!("invalid capture kind for `${}`, expected {:?}", name, expected)
            }
            MacroErrorKind::RecursionLimitExceeded { depth, macro_name } => {
                format!(
                    "recursion limit ({}) exceeded while expanding macro `{}`",
                    depth, macro_name
                )
            }
        }
    }
}

/// Maximum depth for nested macro expansion to prevent infinite recursion
pub const MAX_MACRO_RECURSION_DEPTH: usize = 128;

/// A captured value from pattern matching
#[derive(Debug, Clone)]
pub enum CapturedValue {
    /// Single captured expression/token
    Single(Vec<MacroToken>),
    /// Repeated captures (from $(...)*  patterns)
    Repeated(Vec<Vec<MacroToken>>),
}

/// Bindings from pattern matching
pub type Bindings = HashMap<String, CapturedValue>;

/// Hygiene context for preventing name collisions in macros
///
/// When a macro defines local variables (like `let x = 42;`), those variables
/// need to be renamed to avoid shadowing user variables at the call site.
/// This is called "macro hygiene".
///
/// # Example
/// ```genesis
/// macro_rules! make_temp {
///     ($val:expr) => {
///         let temp = $val;  // 'temp' gets renamed to '__macro_temp_1'
///         temp * 2
///     }
/// }
///
/// fn main() {
///     let temp = 10;      // User's 'temp'
///     let x = make_temp!(5);  // Doesn't shadow user's 'temp'
///     println(temp);      // Still prints 10
/// }
/// ```
#[derive(Debug, Clone)]
pub struct HygieneContext {
    /// Counter for generating unique symbol names
    gensym_counter: u64,
    /// Current expansion depth (for debugging)
    expansion_depth: u32,
    /// Set of names that should be hygienically renamed
    /// These are identifiers that appear in the macro definition body
    /// but are NOT capture variables
    hygiene_enabled: bool,
}

impl Default for HygieneContext {
    fn default() -> Self {
        Self::new()
    }
}

impl HygieneContext {
    /// Create a new hygiene context
    pub fn new() -> Self {
        Self {
            gensym_counter: 0,
            expansion_depth: 0,
            hygiene_enabled: true,
        }
    }

    /// Generate a unique symbol name
    pub fn gensym(&mut self, base: &str) -> String {
        self.gensym_counter += 1;
        format!("__macro_{}_{}", base, self.gensym_counter)
    }

    /// Check if hygiene is enabled
    pub fn is_enabled(&self) -> bool {
        self.hygiene_enabled
    }

    /// Enable or disable hygiene
    pub fn set_enabled(&mut self, enabled: bool) {
        self.hygiene_enabled = enabled;
    }
}

/// Information about a macro stored in the expander
#[derive(Debug, Clone)]
struct MacroInfo {
    /// The macro definition
    def: MacroDef,
    /// The module where this macro is defined (None for root/global)
    module: Option<String>,
    /// Whether this macro is public
    is_pub: bool,
}

/// Macros grouped by module
#[derive(Debug, Default, Clone)]
struct ModuleMacros {
    /// Macros defined in this module (name -> MacroInfo)
    local: HashMap<String, MacroInfo>,
    /// Macros imported into this module from other modules
    /// Maps local name -> (source_module, original_name)
    imported: HashMap<String, (String, String)>,
}

/// The macro expander
pub struct MacroExpander {
    /// Macros organized by module
    modules: HashMap<String, ModuleMacros>,
    /// Global/root module macros (for backward compatibility)
    global_macros: HashMap<String, MacroDef>,
    /// Current module being processed
    current_module: Option<String>,
    /// Hygiene context for preventing name collisions
    hygiene: HygieneContext,
    /// Mapping of original names to hygienized names for current expansion
    hygiene_map: HashMap<String, String>,
}

impl MacroExpander {
    /// Create a new macro expander
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            global_macros: HashMap::new(),
            current_module: None,
            hygiene: HygieneContext::new(),
            hygiene_map: HashMap::new(),
        }
    }

    /// Create a macro expander with hygiene disabled (for testing)
    pub fn new_without_hygiene() -> Self {
        let mut expander = Self::new();
        expander.hygiene.set_enabled(false);
        expander
    }

    /// Enable or disable hygiene
    pub fn set_hygiene_enabled(&mut self, enabled: bool) {
        self.hygiene.set_enabled(enabled);
    }

    /// Set the current module context for macro resolution
    pub fn set_current_module(&mut self, module: Option<String>) {
        self.current_module = module;
    }

    /// Get the current module context
    pub fn current_module(&self) -> Option<&str> {
        self.current_module.as_deref()
    }

    /// Register a macro definition (backward compatible - registers globally)
    pub fn register(&mut self, def: MacroDef) {
        self.global_macros.insert(def.name.name.clone(), def);
    }

    /// Register a macro with module and visibility information
    pub fn register_in_module(&mut self, def: MacroDef, module: Option<String>, is_pub: bool) {
        let name = def.name.name.clone();
        let info = MacroInfo {
            def,
            module: module.clone(),
            is_pub,
        };

        if let Some(mod_name) = module {
            let mod_macros = self.modules.entry(mod_name).or_default();
            mod_macros.local.insert(name, info);
        } else {
            // Global macro
            self.global_macros.insert(name, info.def);
        }
    }

    /// Import a macro from one module to another
    pub fn import_macro(&mut self, from_module: &str, macro_name: &str, to_module: &str, alias: Option<&str>) {
        let local_name = alias.unwrap_or(macro_name).to_string();
        let mod_macros = self.modules.entry(to_module.to_string()).or_default();
        mod_macros.imported.insert(local_name, (from_module.to_string(), macro_name.to_string()));
    }

    /// Lookup a macro by name, respecting module scope and visibility
    fn lookup_macro(&self, name: &str) -> Option<&MacroDef> {
        // 1. First check global macros (for backward compatibility)
        if let Some(def) = self.global_macros.get(name) {
            return Some(def);
        }

        // 2. Check current module's local macros
        if let Some(current) = &self.current_module {
            if let Some(mod_macros) = self.modules.get(current) {
                // Check local definitions
                if let Some(info) = mod_macros.local.get(name) {
                    return Some(&info.def);
                }

                // Check imports
                if let Some((src_mod, orig_name)) = mod_macros.imported.get(name) {
                    if let Some(src_macros) = self.modules.get(src_mod) {
                        if let Some(info) = src_macros.local.get(orig_name) {
                            // Must be public to access from another module
                            if info.is_pub {
                                return Some(&info.def);
                            }
                        }
                    }
                }
            }
        }

        // 3. Check all modules for public macros
        for (mod_name, mod_macros) in &self.modules {
            // Skip if this is the current module (already checked)
            if Some(mod_name.as_str()) == self.current_module.as_deref() {
                continue;
            }

            // For macros from other modules, they must be public
            if let Some(info) = mod_macros.local.get(name) {
                if info.is_pub {
                    return Some(&info.def);
                }
            }
        }

        None
    }

    /// Check if a macro is defined (considering scope)
    pub fn has_macro(&self, name: &str) -> bool {
        self.lookup_macro(name).is_some()
    }

    /// Check if a macro is visible from the current module
    pub fn is_macro_visible(&self, name: &str, from_module: Option<&str>) -> bool {
        // Check global macros first
        if self.global_macros.contains_key(name) {
            return true;
        }

        // Check module-specific visibility
        for (mod_name, mod_macros) in &self.modules {
            if let Some(info) = mod_macros.local.get(name) {
                // If macro is in the same module, it's visible
                if from_module == Some(mod_name.as_str()) || from_module == info.module.as_deref() {
                    return true;
                }
                // If macro is public, it's visible from anywhere
                if info.is_pub {
                    return true;
                }
            }
        }

        false
    }

    /// Expand a macro invocation
    pub fn expand(&mut self, invocation: &MacroInvocation) -> MacroResult<Vec<MacroToken>> {
        let macro_name = &invocation.name.name;

        // Find the macro definition using the new lookup that respects scope
        let def = self.lookup_macro(macro_name).ok_or_else(|| {
            MacroError::undefined_macro(macro_name.clone(), invocation.span)
        })?.clone();

        // Clear hygiene map for this expansion
        self.hygiene_map.clear();

        // Try each rule until one matches
        for rule in &def.rules {
            if let Some(bindings) = self.match_pattern(&rule.pattern, &invocation.tokens) {
                // Collect identifiers that should be hygienized
                // (identifiers in expansion that are NOT capture names)
                let capture_names: std::collections::HashSet<_> = bindings.keys().cloned().collect();

                // Pattern matched, perform transcription with hygiene
                return self.transcribe(&rule.expansion, &bindings, invocation.span, &capture_names);
            }
        }

        // No rule matched
        Err(MacroError::no_matching_rule(macro_name.clone(), invocation.span))
    }

    /// Expand a macro invocation with recursive expansion of nested macros
    ///
    /// This method expands the macro and then scans the result for any nested
    /// macro invocations, expanding them recursively up to MAX_MACRO_RECURSION_DEPTH.
    ///
    /// # Example
    /// ```genesis
    /// macro_rules! inner {
    ///     ($x:expr) => { $x + 1 }
    /// }
    ///
    /// macro_rules! outer {
    ///     ($x:expr) => { inner!($x) * 2 }
    /// }
    ///
    /// // outer!(5) expands to inner!(5) * 2, then to (5 + 1) * 2
    /// ```
    pub fn expand_recursive(&mut self, invocation: &MacroInvocation) -> MacroResult<Vec<MacroToken>> {
        self.expand_recursive_inner(invocation, 0)
    }

    /// Internal recursive expansion with depth tracking
    fn expand_recursive_inner(
        &mut self,
        invocation: &MacroInvocation,
        depth: usize,
    ) -> MacroResult<Vec<MacroToken>> {
        if depth > MAX_MACRO_RECURSION_DEPTH {
            return Err(MacroError {
                kind: MacroErrorKind::RecursionLimitExceeded {
                    depth,
                    macro_name: invocation.name.name.clone(),
                },
                span: invocation.span,
            });
        }

        // First, expand the macro
        let expanded = self.expand(invocation)?;

        // Then, scan for and expand any nested macro invocations
        self.expand_tokens_recursive(&expanded, depth, invocation.span)
    }

    /// Expand any macro invocations found in a token stream
    ///
    /// Scans the token stream for patterns like `ident!(...)`  and expands them.
    fn expand_tokens_recursive(
        &mut self,
        tokens: &[MacroToken],
        depth: usize,
        span: Span,
    ) -> MacroResult<Vec<MacroToken>> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            // Check if we have a macro invocation pattern: Ident + ! + Group
            if let Some((macro_name, invocation, consumed)) = self.try_parse_macro_invocation(tokens, i) {
                if self.has_macro(&macro_name) {
                    // This is a macro we know - expand it recursively
                    let expanded = self.expand_recursive_inner(&invocation, depth + 1)?;
                    result.extend(expanded);
                    i += consumed;
                    continue;
                } else {
                    // Unknown macro - might be a builtin, keep as-is
                    // We need to preserve the tokens as they are
                    result.push(tokens[i].clone());
                    i += 1;
                }
            } else if let MacroToken::Group { delimiter, tokens: inner, span: group_span } = &tokens[i] {
                // Recursively process tokens inside groups
                let expanded_inner = self.expand_tokens_recursive(inner, depth, span)?;
                result.push(MacroToken::Group {
                    delimiter: *delimiter,
                    tokens: expanded_inner,
                    span: *group_span,
                });
                i += 1;
            } else {
                // Regular token, just copy it
                result.push(tokens[i].clone());
                i += 1;
            }
        }

        Ok(result)
    }

    /// Try to parse a macro invocation starting at the given index
    ///
    /// Looks for the pattern: Ident + Token(Not) + Group
    /// Returns (macro_name, invocation, tokens_consumed) if found
    fn try_parse_macro_invocation(
        &self,
        tokens: &[MacroToken],
        start: usize,
    ) -> Option<(String, MacroInvocation, usize)> {
        // Need at least 3 tokens: ident, !, group
        if start + 2 >= tokens.len() {
            // Check for ident + ! + group (exactly 3 tokens remaining)
            if start + 2 > tokens.len() {
                return None;
            }
        }

        // First token must be an identifier
        let macro_name = match &tokens[start] {
            MacroToken::Ident(name, _) => name.clone(),
            _ => return None,
        };

        // Second token must be !
        let has_bang = match &tokens[start + 1] {
            MacroToken::Token(TokenKind::Not, _) => true,
            _ => false,
        };

        if !has_bang {
            return None;
        }

        // Third token must be a group (the macro arguments)
        let (delimiter, inner_tokens, invocation_span) = match &tokens[start + 2] {
            MacroToken::Group { delimiter, tokens, span } => (*delimiter, tokens.clone(), *span),
            _ => return None,
        };

        // Build the MacroInvocation
        let name_span = match &tokens[start] {
            MacroToken::Ident(_, span) => *span,
            _ => invocation_span,
        };

        let invocation = MacroInvocation {
            name: Ident {
                name: macro_name.clone(),
                span: name_span,
            },
            delimiter,
            tokens: inner_tokens,
            span: invocation_span,
        };

        Some((macro_name, invocation, 3))
    }

    /// Match invocation tokens against a pattern
    /// Returns bindings if matched, None otherwise
    fn match_pattern(&self, pattern: &MacroPattern, tokens: &[MacroToken]) -> Option<Bindings> {
        let mut bindings = Bindings::new();
        let mut token_idx = 0;

        if self.match_tokens(&pattern.tokens, tokens, &mut token_idx, &mut bindings) {
            // Check that we consumed all tokens
            if token_idx == tokens.len() {
                Some(bindings)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Match a sequence of pattern tokens against input tokens
    fn match_tokens(
        &self,
        pattern: &[MacroToken],
        input: &[MacroToken],
        input_idx: &mut usize,
        bindings: &mut Bindings,
    ) -> bool {
        let mut pattern_idx = 0;

        while pattern_idx < pattern.len() {
            let pat_token = &pattern[pattern_idx];

            match pat_token {
                MacroToken::Capture { name, kind, .. } => {
                    // Capture a single element based on kind
                    if *input_idx >= input.len() {
                        return false;
                    }

                    let captured = self.capture_by_kind(input, input_idx, *kind);
                    if captured.is_empty() {
                        return false;
                    }

                    bindings.insert(name.clone(), CapturedValue::Single(captured));
                    pattern_idx += 1;
                }

                MacroToken::Repetition { tokens: rep_tokens, separator, kind, .. } => {
                    // Handle repetition patterns
                    let mut all_captures: HashMap<String, Vec<Vec<MacroToken>>> = HashMap::new();
                    let mut count = 0;

                    loop {
                        // Try to match the repetition body
                        let start_idx = *input_idx;
                        let mut local_bindings = Bindings::new();

                        if self.match_tokens(rep_tokens, input, input_idx, &mut local_bindings) {
                            // Collect captures from this iteration
                            for (name, value) in local_bindings {
                                if let CapturedValue::Single(tokens) = value {
                                    all_captures
                                        .entry(name)
                                        .or_insert_with(Vec::new)
                                        .push(tokens);
                                }
                            }
                            count += 1;

                            // Check for separator
                            if let Some(sep) = separator {
                                if *input_idx < input.len() {
                                    if self.tokens_match_single(sep, &input[*input_idx]) {
                                        *input_idx += 1;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                        } else {
                            *input_idx = start_idx;
                            break;
                        }
                    }

                    // Check repetition constraints
                    match kind {
                        MacroRepKind::OneOrMore if count == 0 => return false,
                        MacroRepKind::ZeroOrOne if count > 1 => return false,
                        _ => {}
                    }

                    // Store all captures as Repeated
                    for (name, values) in all_captures {
                        bindings.insert(name, CapturedValue::Repeated(values));
                    }

                    pattern_idx += 1;
                }

                MacroToken::Group { delimiter, tokens: group_tokens, .. } => {
                    // Match a grouped pattern
                    if *input_idx >= input.len() {
                        return false;
                    }

                    if let MacroToken::Group {
                        delimiter: input_delim,
                        tokens: input_tokens,
                        ..
                    } = &input[*input_idx]
                    {
                        if delimiter != input_delim {
                            return false;
                        }

                        let mut inner_idx = 0;
                        if !self.match_tokens(group_tokens, input_tokens, &mut inner_idx, bindings) {
                            return false;
                        }
                        if inner_idx != input_tokens.len() {
                            return false;
                        }

                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }

                MacroToken::Token(pat_kind, _) => {
                    // Match a literal token
                    if *input_idx >= input.len() {
                        return false;
                    }

                    if let MacroToken::Token(input_kind, _) = &input[*input_idx] {
                        if !self.token_kinds_match(pat_kind, input_kind) {
                            return false;
                        }
                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }

                MacroToken::Ident(pat_name, _) => {
                    // Match an identifier
                    if *input_idx >= input.len() {
                        return false;
                    }

                    if let MacroToken::Ident(input_name, _) = &input[*input_idx] {
                        if pat_name != input_name {
                            return false;
                        }
                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }

                // Handle literal tokens with values
                MacroToken::IntLit(pat_val, _) => {
                    if *input_idx >= input.len() {
                        return false;
                    }
                    if let MacroToken::IntLit(input_val, _) = &input[*input_idx] {
                        if pat_val != input_val {
                            return false;
                        }
                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }

                MacroToken::FloatLit(pat_val, _) => {
                    if *input_idx >= input.len() {
                        return false;
                    }
                    if let MacroToken::FloatLit(input_val, _) = &input[*input_idx] {
                        if (pat_val - input_val).abs() > f64::EPSILON {
                            return false;
                        }
                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }

                MacroToken::StrLit(pat_val, _) => {
                    if *input_idx >= input.len() {
                        return false;
                    }
                    if let MacroToken::StrLit(input_val, _) = &input[*input_idx] {
                        if pat_val != input_val {
                            return false;
                        }
                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }

                MacroToken::CharLit(pat_val, _) => {
                    if *input_idx >= input.len() {
                        return false;
                    }
                    if let MacroToken::CharLit(input_val, _) = &input[*input_idx] {
                        if pat_val != input_val {
                            return false;
                        }
                        *input_idx += 1;
                        pattern_idx += 1;
                    } else {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Capture tokens based on the capture kind
    fn capture_by_kind(
        &self,
        input: &[MacroToken],
        idx: &mut usize,
        kind: MacroCaptureKind,
    ) -> Vec<MacroToken> {
        if *idx >= input.len() {
            return vec![];
        }

        match kind {
            MacroCaptureKind::Expr => {
                // Capture a complete expression
                self.capture_expression(input, idx)
            }
            MacroCaptureKind::Ident => {
                // Capture a single identifier
                if let MacroToken::Ident(_, _) = &input[*idx] {
                    let token = input[*idx].clone();
                    *idx += 1;
                    vec![token]
                } else {
                    vec![]
                }
            }
            MacroCaptureKind::Literal => {
                // Capture a literal
                match &input[*idx] {
                    MacroToken::IntLit(_, _) |
                    MacroToken::FloatLit(_, _) |
                    MacroToken::StrLit(_, _) |
                    MacroToken::CharLit(_, _) => {
                        let token = input[*idx].clone();
                        *idx += 1;
                        vec![token]
                    }
                    MacroToken::Token(kind, _) => {
                        match kind {
                            TokenKind::True | TokenKind::False |
                            TokenKind::IntLiteral | TokenKind::FloatLiteral |
                            TokenKind::StringLiteral | TokenKind::CharLiteral => {
                                let token = input[*idx].clone();
                                *idx += 1;
                                vec![token]
                            }
                            _ => vec![]
                        }
                    }
                    _ => vec![]
                }
            }
            MacroCaptureKind::Tt => {
                // Capture any single token tree
                let token = input[*idx].clone();
                *idx += 1;
                vec![token]
            }
            MacroCaptureKind::Ty => {
                // Capture a type (with support for complex types)
                self.capture_type(input, idx)
            }
            MacroCaptureKind::Path => {
                // Capture a path (e.g., std::collections::HashMap)
                self.capture_path(input, idx)
            }
            MacroCaptureKind::Pat => {
                // Capture a pattern
                self.capture_pattern(input, idx)
            }
            MacroCaptureKind::Stmt => {
                // Capture a statement
                self.capture_statement(input, idx)
            }
            MacroCaptureKind::Block => {
                // Capture a block { ... }
                self.capture_block(input, idx)
            }
            MacroCaptureKind::Item => {
                // Capture an item (fn, struct, enum, etc.)
                self.capture_item(input, idx)
            }
        }
    }

    /// Capture a path: ident (:: ident)* with optional generics at the end
    /// Examples: foo, foo::bar, std::collections::HashMap<K, V>
    fn capture_path(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        let mut result = vec![];
        let start = *idx;

        // Must start with an identifier
        if !matches!(input.get(*idx), Some(MacroToken::Ident(_, _))) {
            return vec![];
        }

        loop {
            // Capture identifier
            if let Some(MacroToken::Ident(_, _)) = input.get(*idx) {
                result.push(input[*idx].clone());
                *idx += 1;
            } else {
                break;
            }

            // Check for :: to continue path
            if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::ColonColon, _))) {
                result.push(input[*idx].clone());
                *idx += 1;
            } else {
                break;
            }
        }

        // Check for generic parameters <...> at the end
        if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Lt, _))) {
            result.push(input[*idx].clone());
            *idx += 1;

            let mut depth = 1;
            while *idx < input.len() && depth > 0 {
                let token = &input[*idx];
                result.push(token.clone());
                *idx += 1;

                match token {
                    MacroToken::Token(TokenKind::Lt, _) => depth += 1,
                    MacroToken::Token(TokenKind::Gt, _) => depth -= 1,
                    _ => {}
                }
            }
        }

        // Must have captured at least one identifier
        if result.is_empty() {
            *idx = start;
        }

        result
    }

    /// Capture a pattern
    /// Examples: x, _, (a, b), Some(x), Foo { x, y }, 1..=10
    fn capture_pattern(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        let mut result = vec![];
        let start = *idx;

        if *idx >= input.len() {
            return vec![];
        }

        match &input[*idx] {
            // Wildcard pattern: _
            MacroToken::Ident(name, _) if name == "_" => {
                result.push(input[*idx].clone());
                *idx += 1;
            }

            // Identifier pattern (may be followed by @ for binding, or struct fields)
            MacroToken::Ident(_, _) => {
                result.push(input[*idx].clone());
                *idx += 1;

                // Check for path continuation (e.g., Some::None or MyEnum::Variant)
                while matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::ColonColon, _))) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                    if let Some(MacroToken::Ident(_, _)) = input.get(*idx) {
                        result.push(input[*idx].clone());
                        *idx += 1;
                    } else {
                        break;
                    }
                }

                // Check for tuple pattern: Variant(x, y)
                if let Some(MacroToken::Group { delimiter: MacroDelimiter::Paren, .. }) = input.get(*idx) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }
                // Check for struct pattern: Struct { x, y }
                else if let Some(MacroToken::Group { delimiter: MacroDelimiter::Brace, .. }) = input.get(*idx) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }
                // Note: @ binding support would require TokenKind::At to be defined
            }

            // Literal patterns
            MacroToken::IntLit(_, _) |
            MacroToken::FloatLit(_, _) |
            MacroToken::StrLit(_, _) |
            MacroToken::CharLit(_, _) => {
                result.push(input[*idx].clone());
                *idx += 1;

                // Check for range pattern: 1..=10 or 1..10
                if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::DotDot, _))) ||
                   matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::DotDotEq, _))) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                    // Capture the end of range
                    if let Some(MacroToken::IntLit(_, _)) = input.get(*idx) {
                        result.push(input[*idx].clone());
                        *idx += 1;
                    }
                }
            }

            // true/false literals
            MacroToken::Token(TokenKind::True, _) | MacroToken::Token(TokenKind::False, _) => {
                result.push(input[*idx].clone());
                *idx += 1;
            }

            // Tuple pattern: (a, b, c)
            MacroToken::Group { delimiter: MacroDelimiter::Paren, .. } => {
                result.push(input[*idx].clone());
                *idx += 1;
            }

            // Slice pattern: [a, b, c]
            MacroToken::Group { delimiter: MacroDelimiter::Bracket, .. } => {
                result.push(input[*idx].clone());
                *idx += 1;
            }

            // Reference pattern: &x or &mut x
            MacroToken::Token(TokenKind::And, _) => {
                result.push(input[*idx].clone());
                *idx += 1;
                // Check for mut
                if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Mut, _))) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }
                // Capture inner pattern
                let inner = self.capture_pattern(input, idx);
                result.extend(inner);
            }

            // Rest pattern: ..
            MacroToken::Token(TokenKind::DotDot, _) => {
                result.push(input[*idx].clone());
                *idx += 1;
            }

            _ => {}
        }

        if result.is_empty() {
            *idx = start;
        }

        result
    }

    /// Capture a statement
    /// Examples: let x = 1;, x + y;, return x;, if x { y } else { z }
    fn capture_statement(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        let mut result = vec![];
        let start = *idx;

        if *idx >= input.len() {
            return vec![];
        }

        match &input[*idx] {
            // let statement: let [mut] ident [: ty] = expr;
            MacroToken::Token(TokenKind::Let, _) => {
                result.push(input[*idx].clone());
                *idx += 1;

                // Consume until semicolon, tracking brace depth
                let mut depth = 0;
                while *idx < input.len() {
                    let token = &input[*idx];
                    result.push(token.clone());

                    match token {
                        MacroToken::Token(TokenKind::Semicolon, _) if depth == 0 => {
                            *idx += 1;
                            break;
                        }
                        MacroToken::Token(TokenKind::LBrace, _) |
                        MacroToken::Group { delimiter: MacroDelimiter::Brace, .. } => {
                            depth += 1;
                        }
                        MacroToken::Token(TokenKind::RBrace, _) => {
                            if depth > 0 { depth -= 1; }
                        }
                        _ => {}
                    }
                    *idx += 1;
                }
            }

            // return statement
            MacroToken::Token(TokenKind::Return, _) => {
                result.push(input[*idx].clone());
                *idx += 1;

                // Capture expression if present
                if !matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Semicolon, _))) {
                    let expr = self.capture_expression(input, idx);
                    result.extend(expr);
                }

                // Capture semicolon if present
                if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Semicolon, _))) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }
            }

            // break/continue statements
            MacroToken::Token(TokenKind::Break, _) | MacroToken::Token(TokenKind::Continue, _) => {
                result.push(input[*idx].clone());
                *idx += 1;

                // Optional label
                if let Some(MacroToken::Ident(_, _)) = input.get(*idx) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }

                // Optional expression (for break)
                if !matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Semicolon, _))) &&
                   !matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::RBrace, _))) {
                    let expr = self.capture_expression(input, idx);
                    result.extend(expr);
                }

                // Capture semicolon
                if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Semicolon, _))) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }
            }

            // Expression statement (anything else)
            _ => {
                let expr = self.capture_expression(input, idx);
                if expr.is_empty() {
                    *idx = start;
                    return vec![];
                }
                result.extend(expr);

                // Capture trailing semicolon if present
                if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Semicolon, _))) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                }
            }
        }

        if result.is_empty() {
            *idx = start;
        }

        result
    }

    /// Capture a block: { ... }
    fn capture_block(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        if *idx >= input.len() {
            return vec![];
        }

        // Must start with { or be a Group with Brace delimiter
        match &input[*idx] {
            MacroToken::Group { delimiter: MacroDelimiter::Brace, .. } => {
                let token = input[*idx].clone();
                *idx += 1;
                vec![token]
            }
            MacroToken::Token(TokenKind::LBrace, _) => {
                let mut result = vec![];
                result.push(input[*idx].clone());
                *idx += 1;

                let mut depth = 1;
                while *idx < input.len() && depth > 0 {
                    let token = &input[*idx];
                    result.push(token.clone());
                    *idx += 1;

                    match token {
                        MacroToken::Token(TokenKind::LBrace, _) => depth += 1,
                        MacroToken::Token(TokenKind::RBrace, _) => depth -= 1,
                        MacroToken::Group { delimiter: MacroDelimiter::Brace, .. } => {
                            // Group already balanced, don't change depth
                        }
                        _ => {}
                    }
                }

                result
            }
            _ => vec![]
        }
    }

    /// Capture an item: fn, struct, enum, impl, trait, mod, use, type, const
    fn capture_item(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        let mut result = vec![];
        let start = *idx;

        if *idx >= input.len() {
            return vec![];
        }

        // Check for visibility modifier (pub)
        if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Pub, _))) {
            result.push(input[*idx].clone());
            *idx += 1;

            // Check for pub(crate), pub(super), etc.
            if let Some(MacroToken::Group { delimiter: MacroDelimiter::Paren, .. }) = input.get(*idx) {
                result.push(input[*idx].clone());
                *idx += 1;
            }
        }

        // Check for item keyword
        let is_item_keyword = match input.get(*idx) {
            Some(MacroToken::Token(TokenKind::Fn, _)) |
            Some(MacroToken::Token(TokenKind::Struct, _)) |
            Some(MacroToken::Token(TokenKind::Enum, _)) |
            Some(MacroToken::Token(TokenKind::Impl, _)) |
            Some(MacroToken::Token(TokenKind::Trait, _)) |
            Some(MacroToken::Token(TokenKind::Mod, _)) |
            Some(MacroToken::Token(TokenKind::Use, _)) |
            Some(MacroToken::Token(TokenKind::Const, _)) => true,
            Some(MacroToken::Ident(name, _)) if name == "type" || name == "static" => true,
            _ => false,
        };

        if !is_item_keyword {
            *idx = start;
            return vec![];
        }

        // Capture the item keyword
        result.push(input[*idx].clone());
        *idx += 1;

        // Consume until we find the terminating ; or the closing }
        let mut brace_depth = 0;

        while *idx < input.len() {
            let token = &input[*idx];

            match token {
                MacroToken::Token(TokenKind::LBrace, _) => {
                    brace_depth += 1;
                    result.push(token.clone());
                    *idx += 1;
                }
                MacroToken::Token(TokenKind::RBrace, _) => {
                    result.push(token.clone());
                    *idx += 1;
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        break;
                    }
                }
                MacroToken::Token(TokenKind::Semicolon, _) if brace_depth == 0 => {
                    result.push(token.clone());
                    *idx += 1;
                    break;
                }
                MacroToken::Group { delimiter: MacroDelimiter::Brace, .. } if brace_depth == 0 => {
                    // This is the item body as a pre-parsed group
                    result.push(token.clone());
                    *idx += 1;
                    break;
                }
                _ => {
                    result.push(token.clone());
                    *idx += 1;
                }
            }
        }

        if result.len() <= 1 {
            // Only captured keyword, no body - revert
            *idx = start;
            return vec![];
        }

        result
    }

    /// Capture a complete expression
    fn capture_expression(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        let mut result = vec![];
        let mut depth = 0;

        while *idx < input.len() {
            let token = &input[*idx];

            match token {
                MacroToken::Token(TokenKind::Comma, _) if depth == 0 => break,
                MacroToken::Token(TokenKind::Semicolon, _) if depth == 0 => break,
                MacroToken::Group { .. } => {
                    result.push(token.clone());
                    *idx += 1;
                }
                MacroToken::Token(TokenKind::LParen, _) |
                MacroToken::Token(TokenKind::LBracket, _) |
                MacroToken::Token(TokenKind::LBrace, _) => {
                    depth += 1;
                    result.push(token.clone());
                    *idx += 1;
                }
                MacroToken::Token(TokenKind::RParen, _) |
                MacroToken::Token(TokenKind::RBracket, _) |
                MacroToken::Token(TokenKind::RBrace, _) => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    result.push(token.clone());
                    *idx += 1;
                }
                _ => {
                    result.push(token.clone());
                    *idx += 1;
                }
            }
        }

        result
    }

    /// Capture a type expression
    /// Supports: ident, path::ident, &T, &mut T, [T], [T; N], (T, U), fn(T) -> U, Box<T>
    fn capture_type(&self, input: &[MacroToken], idx: &mut usize) -> Vec<MacroToken> {
        let mut result = vec![];
        let start = *idx;

        if *idx >= input.len() {
            return vec![];
        }

        // Handle reference types: &T, &mut T
        if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::And, _))) {
            result.push(input[*idx].clone());
            *idx += 1;

            // Check for 'mut'
            if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Mut, _))) {
                result.push(input[*idx].clone());
                *idx += 1;
            }

            // Recursively capture the inner type
            let inner = self.capture_type(input, idx);
            if inner.is_empty() {
                *idx = start;
                return vec![];
            }
            result.extend(inner);
            return result;
        }

        // Handle tuple types: (T, U, V)
        if let Some(MacroToken::Group { delimiter: MacroDelimiter::Paren, .. }) = input.get(*idx) {
            result.push(input[*idx].clone());
            *idx += 1;
            return result;
        }

        // Handle array/slice types: [T] or [T; N]
        if let Some(MacroToken::Group { delimiter: MacroDelimiter::Bracket, .. }) = input.get(*idx) {
            result.push(input[*idx].clone());
            *idx += 1;
            return result;
        }

        // Handle array types starting with [
        if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::LBracket, _))) {
            result.push(input[*idx].clone());
            *idx += 1;

            let mut depth = 1;
            while *idx < input.len() && depth > 0 {
                let token = &input[*idx];
                result.push(token.clone());
                *idx += 1;

                match token {
                    MacroToken::Token(TokenKind::LBracket, _) => depth += 1,
                    MacroToken::Token(TokenKind::RBracket, _) => depth -= 1,
                    _ => {}
                }
            }
            return result;
        }

        // Handle function pointer types: fn(T) -> U
        if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Fn, _))) {
            result.push(input[*idx].clone());
            *idx += 1;

            // Capture parameters (...)
            if let Some(MacroToken::Group { delimiter: MacroDelimiter::Paren, .. }) = input.get(*idx) {
                result.push(input[*idx].clone());
                *idx += 1;
            } else if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::LParen, _))) {
                result.push(input[*idx].clone());
                *idx += 1;

                let mut depth = 1;
                while *idx < input.len() && depth > 0 {
                    let token = &input[*idx];
                    result.push(token.clone());
                    *idx += 1;

                    match token {
                        MacroToken::Token(TokenKind::LParen, _) => depth += 1,
                        MacroToken::Token(TokenKind::RParen, _) => depth -= 1,
                        _ => {}
                    }
                }
            }

            // Check for return type: -> T
            if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Arrow, _))) {
                result.push(input[*idx].clone());
                *idx += 1;

                // Capture return type
                let ret_type = self.capture_type(input, idx);
                result.extend(ret_type);
            }

            return result;
        }

        // Handle impl Trait (simplified)
        if let Some(MacroToken::Ident(name, _)) = input.get(*idx) {
            if name == "impl" {
                result.push(input[*idx].clone());
                *idx += 1;
                // Capture the trait bound
                let trait_type = self.capture_type(input, idx);
                result.extend(trait_type);
                return result;
            }
        }

        // Handle dyn Trait (simplified)
        if let Some(MacroToken::Ident(name, _)) = input.get(*idx) {
            if name == "dyn" {
                result.push(input[*idx].clone());
                *idx += 1;
                // Capture the trait bound
                let trait_type = self.capture_type(input, idx);
                result.extend(trait_type);
                return result;
            }
        }

        // Handle identifier-based types (most common case)
        // path::to::Type<T, U>
        if let Some(MacroToken::Ident(_, _)) = input.get(*idx) {
            result.push(input[*idx].clone());
            *idx += 1;

            // Check for path continuation ::
            while matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::ColonColon, _))) {
                result.push(input[*idx].clone());
                *idx += 1;

                if let Some(MacroToken::Ident(_, _)) = input.get(*idx) {
                    result.push(input[*idx].clone());
                    *idx += 1;
                } else {
                    break;
                }
            }

            // Check for generic parameters <...>
            if matches!(input.get(*idx), Some(MacroToken::Token(TokenKind::Lt, _))) {
                result.push(input[*idx].clone());
                *idx += 1;

                let mut depth = 1;
                while *idx < input.len() && depth > 0 {
                    let token = &input[*idx];
                    result.push(token.clone());
                    *idx += 1;

                    match token {
                        MacroToken::Token(TokenKind::Lt, _) => depth += 1,
                        MacroToken::Token(TokenKind::Gt, _) => depth -= 1,
                        _ => {}
                    }
                }
            }

            return result;
        }

        // Nothing matched
        *idx = start;
        vec![]
    }

    /// Check if a single pattern token matches an input token
    fn tokens_match_single(&self, pattern: &MacroToken, input: &MacroToken) -> bool {
        match (pattern, input) {
            (MacroToken::Token(pk, _), MacroToken::Token(ik, _)) => {
                self.token_kinds_match(pk, ik)
            }
            (MacroToken::Ident(pn, _), MacroToken::Ident(in_, _)) => pn == in_,
            (MacroToken::IntLit(pv, _), MacroToken::IntLit(iv, _)) => pv == iv,
            (MacroToken::FloatLit(pv, _), MacroToken::FloatLit(iv, _)) => (pv - iv).abs() < f64::EPSILON,
            (MacroToken::StrLit(pv, _), MacroToken::StrLit(iv, _)) => pv == iv,
            (MacroToken::CharLit(pv, _), MacroToken::CharLit(iv, _)) => pv == iv,
            _ => false,
        }
    }

    /// Check if two token kinds match
    fn token_kinds_match(&self, pat: &TokenKind, input: &TokenKind) -> bool {
        std::mem::discriminant(pat) == std::mem::discriminant(input)
    }

    /// Transcribe the expansion template with captured values
    fn transcribe(
        &mut self,
        expansion: &MacroExpansion,
        bindings: &Bindings,
        span: Span,
        capture_names: &std::collections::HashSet<String>,
    ) -> MacroResult<Vec<MacroToken>> {
        self.transcribe_tokens(&expansion.tokens, bindings, span, None, capture_names)
    }

    /// Check if an identifier should be hygienized
    ///
    /// An identifier should be hygienized if:
    /// - Hygiene is enabled
    /// - It's not a capture variable name
    /// - It's not a keyword or builtin
    /// - It's not a macro name (for recursive/nested macro calls)
    fn should_hygienize(&self, name: &str, capture_names: &std::collections::HashSet<String>) -> bool {
        if !self.hygiene.is_enabled() {
            return false;
        }

        // Don't hygienize capture variables (they come from call site)
        if capture_names.contains(name) {
            return false;
        }

        // Don't hygienize macro names (needed for recursive/nested macros)
        if self.global_macros.contains_key(name) || self.lookup_macro(name).is_some() {
            return false;
        }

        // Don't hygienize keywords
        let keywords = [
            "let", "mut", "fn", "if", "else", "while", "for", "in", "loop",
            "break", "continue", "return", "struct", "enum", "impl", "trait",
            "pub", "use", "mod", "const", "static", "type", "true", "false",
            "self", "Self", "super", "crate", "async", "await", "match", "where",
        ];
        if keywords.contains(&name) {
            return false;
        }

        // Don't hygienize builtins and common types
        let builtins = [
            "println", "print", "eprintln", "eprint", "format", "vec", "assert",
            "assert_eq", "assert_ne", "panic", "todo", "unreachable", "dbg",
            "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64",
            "bool", "char", "str", "String", "Vec", "Option", "Result", "Some",
            "None", "Ok", "Err", "Box", "HashMap", "HashSet",
        ];
        if builtins.contains(&name) {
            return false;
        }

        true
    }

    /// Get or create a hygienized name for an identifier
    fn get_hygienized_name(&mut self, name: &str) -> String {
        if let Some(existing) = self.hygiene_map.get(name) {
            existing.clone()
        } else {
            let new_name = self.hygiene.gensym(name);
            self.hygiene_map.insert(name.to_string(), new_name.clone());
            new_name
        }
    }

    /// Transcribe a sequence of tokens with hygiene applied
    fn transcribe_tokens(
        &mut self,
        tokens: &[MacroToken],
        bindings: &Bindings,
        span: Span,
        rep_idx: Option<usize>,
        capture_names: &std::collections::HashSet<String>,
    ) -> MacroResult<Vec<MacroToken>> {
        let mut result = vec![];

        for token in tokens {
            match token {
                MacroToken::Capture { name, .. } => {
                    // Substitute captured value (no hygiene - comes from call site)
                    if let Some(value) = bindings.get(name) {
                        match value {
                            CapturedValue::Single(tokens) => {
                                result.extend(tokens.clone());
                            }
                            CapturedValue::Repeated(all_tokens) => {
                                if let Some(idx) = rep_idx {
                                    if idx < all_tokens.len() {
                                        result.extend(all_tokens[idx].clone());
                                    }
                                } else {
                                    // Outside repetition, use first if available
                                    if !all_tokens.is_empty() {
                                        result.extend(all_tokens[0].clone());
                                    }
                                }
                            }
                        }
                    } else {
                        return Err(MacroError {
                            kind: MacroErrorKind::MissingCapture { name: name.clone() },
                            span,
                        });
                    }
                }

                MacroToken::Repetition { tokens: rep_tokens, separator, kind, .. } => {
                    // Find the repetition count from bindings
                    let count = self.find_repetition_count(rep_tokens, bindings);

                    // Validate count
                    match kind {
                        MacroRepKind::OneOrMore if count == 0 => {
                            return Err(MacroError {
                                kind: MacroErrorKind::RepetitionMismatch { expected: 1, got: 0 },
                                span,
                            });
                        }
                        MacroRepKind::ZeroOrOne if count > 1 => {
                            return Err(MacroError {
                                kind: MacroErrorKind::RepetitionMismatch { expected: 1, got: count },
                                span,
                            });
                        }
                        _ => {}
                    }

                    // Transcribe for each iteration
                    for i in 0..count {
                        if i > 0 {
                            if let Some(sep) = separator {
                                result.push((**sep).clone());
                            }
                        }
                        let transcribed = self.transcribe_tokens(rep_tokens, bindings, span, Some(i), capture_names)?;
                        result.extend(transcribed);
                    }
                }

                MacroToken::Group { delimiter, tokens: group_tokens, span: group_span } => {
                    // Recursively transcribe group contents
                    let transcribed = self.transcribe_tokens(group_tokens, bindings, span, rep_idx, capture_names)?;
                    result.push(MacroToken::Group {
                        delimiter: *delimiter,
                        tokens: transcribed,
                        span: *group_span,
                    });
                }

                MacroToken::Ident(name, ident_span) => {
                    // Apply hygiene to identifiers from macro body
                    if self.should_hygienize(name, capture_names) {
                        let new_name = self.get_hygienized_name(name);
                        result.push(MacroToken::Ident(new_name, *ident_span));
                    } else {
                        result.push(token.clone());
                    }
                }

                MacroToken::Token(_, _) |
                MacroToken::IntLit(_, _) | MacroToken::FloatLit(_, _) |
                MacroToken::StrLit(_, _) | MacroToken::CharLit(_, _) => {
                    result.push(token.clone());
                }
            }
        }

        Ok(result)
    }

    /// Find how many iterations a repetition should have
    fn find_repetition_count(&self, tokens: &[MacroToken], bindings: &Bindings) -> usize {
        for token in tokens {
            match token {
                MacroToken::Capture { name, .. } => {
                    if let Some(CapturedValue::Repeated(all)) = bindings.get(name) {
                        return all.len();
                    }
                }
                MacroToken::Repetition { tokens: inner, .. } => {
                    let count = self.find_repetition_count(inner, bindings);
                    if count > 0 {
                        return count;
                    }
                }
                MacroToken::Group { tokens: inner, .. } => {
                    let count = self.find_repetition_count(inner, bindings);
                    if count > 0 {
                        return count;
                    }
                }
                _ => {}
            }
        }
        0
    }
}

/// Convert expanded macro tokens to an expression AST
pub fn tokens_to_expr(tokens: &[MacroToken], span: Span) -> Option<Expr> {
    if tokens.is_empty() {
        return None;
    }

    // Try to parse as a simple expression
    // For now, handle common cases

    if tokens.len() == 1 {
        match &tokens[0] {
            MacroToken::IntLit(n, s) => {
                return Some(Expr {
                    kind: ExprKind::Literal(crate::ast::Literal::Int(*n as i128)),
                    span: *s,
                });
            }
            MacroToken::FloatLit(f, s) => {
                return Some(Expr {
                    kind: ExprKind::Literal(crate::ast::Literal::Float(*f)),
                    span: *s,
                });
            }
            MacroToken::StrLit(st, s) => {
                return Some(Expr {
                    kind: ExprKind::Literal(crate::ast::Literal::String(st.clone())),
                    span: *s,
                });
            }
            MacroToken::CharLit(c, s) => {
                return Some(Expr {
                    kind: ExprKind::Literal(crate::ast::Literal::Char(*c)),
                    span: *s,
                });
            }
            MacroToken::Token(TokenKind::True, s) => {
                return Some(Expr {
                    kind: ExprKind::Literal(crate::ast::Literal::Bool(true)),
                    span: *s,
                });
            }
            MacroToken::Token(TokenKind::False, s) => {
                return Some(Expr {
                    kind: ExprKind::Literal(crate::ast::Literal::Bool(false)),
                    span: *s,
                });
            }
            MacroToken::Ident(name, s) => {
                return Some(Expr {
                    kind: ExprKind::Path(crate::ast::Path {
                        segments: vec![crate::ast::PathSegment {
                            ident: Ident { name: name.clone(), span: *s },
                            generics: None,
                        }],
                        span: *s,
                    }),
                    span: *s,
                });
            }
            _ => {}
        }
    }

    // For more complex expressions, we need to re-parse
    // This will be handled by converting tokens back to source and parsing
    None
}

/// Convert tokens back to source string for re-parsing
pub fn tokens_to_source(tokens: &[MacroToken]) -> String {
    let mut result = String::new();

    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(&token_to_string(token));
    }

    result
}

fn token_to_string(token: &MacroToken) -> String {
    match token {
        MacroToken::Token(kind, _) => token_kind_to_string(kind),
        MacroToken::Ident(name, _) => name.clone(),
        MacroToken::IntLit(n, _) => n.to_string(),
        MacroToken::FloatLit(f, _) => f.to_string(),
        MacroToken::StrLit(s, _) => format!("\"{}\"", s),
        MacroToken::CharLit(c, _) => format!("'{}'", c),
        MacroToken::Capture { name, .. } => format!("${}", name),
        MacroToken::Repetition { .. } => String::new(), // Should be expanded
        MacroToken::Group { delimiter, tokens, .. } => {
            let inner = tokens_to_source(tokens);
            match delimiter {
                MacroDelimiter::Paren => format!("({})", inner),
                MacroDelimiter::Bracket => format!("[{}]", inner),
                MacroDelimiter::Brace => format!("{{{}}}", inner),
            }
        }
    }
}

fn token_kind_to_string(kind: &TokenKind) -> String {
    match kind {
        // Literals
        TokenKind::IntLiteral => "<int>".to_string(),
        TokenKind::FloatLiteral => "<float>".to_string(),
        TokenKind::StringLiteral => "<string>".to_string(),
        TokenKind::CharLiteral => "<char>".to_string(),
        TokenKind::Label => "<label>".to_string(),
        TokenKind::True => "true".to_string(),
        TokenKind::False => "false".to_string(),

        // Arithmetic operators
        TokenKind::Plus => "+".to_string(),
        TokenKind::Minus => "-".to_string(),
        TokenKind::Star => "*".to_string(),
        TokenKind::Slash => "/".to_string(),
        TokenKind::Percent => "%".to_string(),

        // Comparison operators
        TokenKind::Eq => "=".to_string(),
        TokenKind::EqEq => "==".to_string(),
        TokenKind::NotEq => "!=".to_string(),
        TokenKind::Lt => "<".to_string(),
        TokenKind::LtEq => "<=".to_string(),
        TokenKind::Gt => ">".to_string(),
        TokenKind::GtEq => ">=".to_string(),

        // Logical operators
        TokenKind::AndAnd => "&&".to_string(),
        TokenKind::OrOr => "||".to_string(),
        TokenKind::Not => "!".to_string(),

        // Bitwise operators
        TokenKind::And => "&".to_string(),
        TokenKind::Or => "|".to_string(),
        TokenKind::Caret => "^".to_string(),
        TokenKind::Tilde => "~".to_string(),
        TokenKind::Shl => "<<".to_string(),
        TokenKind::Shr => ">>".to_string(),

        // Assignment operators (compound)
        TokenKind::PlusEq => "+=".to_string(),
        TokenKind::MinusEq => "-=".to_string(),
        TokenKind::StarEq => "*=".to_string(),
        TokenKind::SlashEq => "/=".to_string(),
        TokenKind::PercentEq => "%=".to_string(),
        TokenKind::AndEq => "&=".to_string(),
        TokenKind::OrEq => "|=".to_string(),
        TokenKind::CaretEq => "^=".to_string(),
        TokenKind::ShlEq => "<<=".to_string(),
        TokenKind::ShrEq => ">>=".to_string(),

        // Punctuation
        TokenKind::Comma => ",".to_string(),
        TokenKind::Semicolon => ";".to_string(),
        TokenKind::Colon => ":".to_string(),
        TokenKind::ColonColon => "::".to_string(),
        TokenKind::Dot => ".".to_string(),
        TokenKind::DotDot => "..".to_string(),
        TokenKind::DotDotEq => "..=".to_string(),

        // Arrows and special
        TokenKind::Arrow => "->".to_string(),
        TokenKind::FatArrow => "=>".to_string(),
        TokenKind::LeftArrow => "<-".to_string(),
        TokenKind::LeftArrowQuestion => "<-?".to_string(),
        TokenKind::Dollar => "$".to_string(),
        TokenKind::Question => "?".to_string(),

        // Delimiters
        TokenKind::LParen => "(".to_string(),
        TokenKind::RParen => ")".to_string(),
        TokenKind::LBracket => "[".to_string(),
        TokenKind::RBracket => "]".to_string(),
        TokenKind::LBrace => "{".to_string(),
        TokenKind::RBrace => "}".to_string(),

        // Core keywords
        TokenKind::Let => "let".to_string(),
        TokenKind::Mut => "mut".to_string(),
        TokenKind::Const => "const".to_string(),
        TokenKind::Fn => "fn".to_string(),
        TokenKind::If => "if".to_string(),
        TokenKind::Else => "else".to_string(),
        TokenKind::While => "while".to_string(),
        TokenKind::For => "for".to_string(),
        TokenKind::In => "in".to_string(),
        TokenKind::Return => "return".to_string(),
        TokenKind::Break => "break".to_string(),
        TokenKind::Continue => "continue".to_string(),
        TokenKind::Loop => "loop".to_string(),
        TokenKind::Match => "match".to_string(),

        // Type definition keywords
        TokenKind::Struct => "struct".to_string(),
        TokenKind::Enum => "enum".to_string(),
        TokenKind::Impl => "impl".to_string(),
        TokenKind::Trait => "trait".to_string(),
        TokenKind::Type => "type".to_string(),
        TokenKind::Pub => "pub".to_string(),
        TokenKind::Mod => "mod".to_string(),
        TokenKind::Use => "use".to_string(),
        TokenKind::As => "as".to_string(),

        // Self keywords
        TokenKind::SelfValue => "self".to_string(),
        TokenKind::SelfType => "Self".to_string(),

        // Where keyword
        TokenKind::Where => "where".to_string(),

        // Async keywords
        TokenKind::Async => "async".to_string(),
        TokenKind::Await => "await".to_string(),

        // Genesis-specific keywords (actor system)
        TokenKind::Actor => "actor".to_string(),
        TokenKind::Receive => "receive".to_string(),
        TokenKind::Spawn => "spawn".to_string(),
        TokenKind::Reply => "reply".to_string(),
        TokenKind::Select => "select".to_string(),
        TokenKind::Join => "join".to_string(),

        // Macro keywords
        TokenKind::Macro => "macro".to_string(),
        TokenKind::MacroRules => "macro_rules".to_string(),

        // Primitive type keywords
        TokenKind::I8 => "i8".to_string(),
        TokenKind::I16 => "i16".to_string(),
        TokenKind::I32 => "i32".to_string(),
        TokenKind::I64 => "i64".to_string(),
        TokenKind::I128 => "i128".to_string(),
        TokenKind::U8 => "u8".to_string(),
        TokenKind::U16 => "u16".to_string(),
        TokenKind::U32 => "u32".to_string(),
        TokenKind::U64 => "u64".to_string(),
        TokenKind::U128 => "u128".to_string(),
        TokenKind::F32 => "f32".to_string(),
        TokenKind::F64 => "f64".to_string(),
        TokenKind::Bool => "bool".to_string(),
        TokenKind::Char => "char".to_string(),
        TokenKind::Str => "str".to_string(),

        // Special
        TokenKind::Ident => "<ident>".to_string(),
        TokenKind::Eof => "".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span() -> Span {
        Span::new(0, 0)
    }

    fn ident(name: &str) -> MacroToken {
        MacroToken::Ident(name.to_string(), make_span())
    }

    fn token(kind: TokenKind) -> MacroToken {
        MacroToken::Token(kind, make_span())
    }

    fn int_lit(n: i64) -> MacroToken {
        MacroToken::IntLit(n, make_span())
    }

    fn group(delimiter: MacroDelimiter, tokens: Vec<MacroToken>) -> MacroToken {
        MacroToken::Group { delimiter, tokens, span: make_span() }
    }

    #[test]
    fn test_empty_pattern_match() {
        let expander = MacroExpander::new();
        let pattern = MacroPattern {
            tokens: vec![],
            span: make_span(),
        };
        let input: Vec<MacroToken> = vec![];

        let result = expander.match_pattern(&pattern, &input);
        assert!(result.is_some());
    }

    #[test]
    fn test_single_capture() {
        let expander = MacroExpander::new();
        let pattern = MacroPattern {
            tokens: vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            span: make_span(),
        };
        let input = vec![MacroToken::IntLit(42, make_span())];

        let result = expander.match_pattern(&pattern, &input);
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert!(bindings.contains_key("x"));
    }

    // ==================== PATH CAPTURE TESTS ====================

    #[test]
    fn test_capture_path_simple() {
        let expander = MacroExpander::new();
        let input = vec![ident("foo")];
        let mut idx = 0;

        let result = expander.capture_path(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_path_multi_segment() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("std"),
            token(TokenKind::ColonColon),
            ident("collections"),
            token(TokenKind::ColonColon),
            ident("HashMap"),
        ];
        let mut idx = 0;

        let result = expander.capture_path(&input, &mut idx);
        assert_eq!(result.len(), 5);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_capture_path_with_generics() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("Vec"),
            token(TokenKind::Lt),
            ident("i64"),
            token(TokenKind::Gt),
        ];
        let mut idx = 0;

        let result = expander.capture_path(&input, &mut idx);
        assert_eq!(result.len(), 4);
        assert_eq!(idx, 4);
    }

    // ==================== PATTERN CAPTURE TESTS ====================

    #[test]
    fn test_capture_pattern_identifier() {
        let expander = MacroExpander::new();
        let input = vec![ident("x")];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_pattern_wildcard() {
        let expander = MacroExpander::new();
        let input = vec![ident("_")];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_pattern_literal() {
        let expander = MacroExpander::new();
        let input = vec![int_lit(42)];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_pattern_tuple() {
        let expander = MacroExpander::new();
        let input = vec![group(MacroDelimiter::Paren, vec![ident("a"), token(TokenKind::Comma), ident("b")])];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_pattern_variant_with_tuple() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("Some"),
            group(MacroDelimiter::Paren, vec![ident("x")]),
        ];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 2);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_capture_pattern_struct() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("Point"),
            group(MacroDelimiter::Brace, vec![ident("x"), token(TokenKind::Comma), ident("y")]),
        ];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 2);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_capture_pattern_reference() {
        let expander = MacroExpander::new();
        let input = vec![token(TokenKind::And), ident("x")];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 2);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_capture_pattern_ref_mut() {
        let expander = MacroExpander::new();
        let input = vec![token(TokenKind::And), token(TokenKind::Mut), ident("x")];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 3);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_capture_pattern_range() {
        let expander = MacroExpander::new();
        let input = vec![int_lit(1), token(TokenKind::DotDotEq), int_lit(10)];
        let mut idx = 0;

        let result = expander.capture_pattern(&input, &mut idx);
        assert_eq!(result.len(), 3);
        assert_eq!(idx, 3);
    }

    // ==================== STATEMENT CAPTURE TESTS ====================

    #[test]
    fn test_capture_statement_let() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Let),
            ident("x"),
            token(TokenKind::Eq),
            int_lit(42),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_statement(&input, &mut idx);
        assert_eq!(result.len(), 5);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_capture_statement_return() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Return),
            int_lit(42),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_statement(&input, &mut idx);
        assert_eq!(result.len(), 3);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_capture_statement_break() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Break),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_statement(&input, &mut idx);
        assert_eq!(result.len(), 2);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_capture_statement_expr() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("x"),
            token(TokenKind::Plus),
            int_lit(1),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_statement(&input, &mut idx);
        assert_eq!(result.len(), 4);
        assert_eq!(idx, 4);
    }

    // ==================== BLOCK CAPTURE TESTS ====================

    #[test]
    fn test_capture_block_group() {
        let expander = MacroExpander::new();
        let input = vec![group(MacroDelimiter::Brace, vec![int_lit(42)])];
        let mut idx = 0;

        let result = expander.capture_block(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_block_tokens() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::LBrace),
            int_lit(42),
            token(TokenKind::RBrace),
        ];
        let mut idx = 0;

        let result = expander.capture_block(&input, &mut idx);
        assert_eq!(result.len(), 3);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_capture_block_nested() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::LBrace),
            token(TokenKind::LBrace),
            int_lit(42),
            token(TokenKind::RBrace),
            token(TokenKind::RBrace),
        ];
        let mut idx = 0;

        let result = expander.capture_block(&input, &mut idx);
        assert_eq!(result.len(), 5);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_capture_block_not_a_block() {
        let expander = MacroExpander::new();
        let input = vec![int_lit(42)];
        let mut idx = 0;

        let result = expander.capture_block(&input, &mut idx);
        assert!(result.is_empty());
        assert_eq!(idx, 0);
    }

    // ==================== ITEM CAPTURE TESTS ====================

    #[test]
    fn test_capture_item_fn() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Fn),
            ident("foo"),
            group(MacroDelimiter::Paren, vec![]),
            group(MacroDelimiter::Brace, vec![]),
        ];
        let mut idx = 0;

        let result = expander.capture_item(&input, &mut idx);
        assert_eq!(result.len(), 4);
        assert_eq!(idx, 4);
    }

    #[test]
    fn test_capture_item_struct() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Struct),
            ident("Foo"),
            group(MacroDelimiter::Brace, vec![ident("x"), token(TokenKind::Colon), ident("i64")]),
        ];
        let mut idx = 0;

        let result = expander.capture_item(&input, &mut idx);
        assert_eq!(result.len(), 3);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_capture_item_pub_fn() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Pub),
            token(TokenKind::Fn),
            ident("foo"),
            group(MacroDelimiter::Paren, vec![]),
            group(MacroDelimiter::Brace, vec![]),
        ];
        let mut idx = 0;

        let result = expander.capture_item(&input, &mut idx);
        assert_eq!(result.len(), 5);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_capture_item_use() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Use),
            ident("std"),
            token(TokenKind::ColonColon),
            ident("io"),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_item(&input, &mut idx);
        assert_eq!(result.len(), 5);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_capture_item_not_an_item() {
        let expander = MacroExpander::new();
        let input = vec![int_lit(42)];
        let mut idx = 0;

        let result = expander.capture_item(&input, &mut idx);
        assert!(result.is_empty());
        assert_eq!(idx, 0);
    }

    // ==================== TYPE CAPTURE TESTS ====================

    #[test]
    fn test_capture_type_simple() {
        let expander = MacroExpander::new();
        let input = vec![ident("i64")];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_type_generic() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("Vec"),
            token(TokenKind::Lt),
            ident("i64"),
            token(TokenKind::Gt),
        ];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 4);
        assert_eq!(idx, 4);
    }

    #[test]
    fn test_capture_type_reference() {
        let expander = MacroExpander::new();
        let input = vec![token(TokenKind::And), ident("str")];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 2);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_capture_type_ref_mut() {
        let expander = MacroExpander::new();
        let input = vec![token(TokenKind::And), token(TokenKind::Mut), ident("i64")];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 3);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_capture_type_tuple() {
        let expander = MacroExpander::new();
        let input = vec![group(MacroDelimiter::Paren, vec![ident("i64"), token(TokenKind::Comma), ident("String")])];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_type_array() {
        let expander = MacroExpander::new();
        let input = vec![group(MacroDelimiter::Bracket, vec![ident("i64")])];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 1);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_capture_type_fn_pointer() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Fn),
            group(MacroDelimiter::Paren, vec![ident("i64")]),
            token(TokenKind::Arrow),
            ident("bool"),
        ];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 4);
        assert_eq!(idx, 4);
    }

    #[test]
    fn test_capture_type_path() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("std"),
            token(TokenKind::ColonColon),
            ident("collections"),
            token(TokenKind::ColonColon),
            ident("HashMap"),
            token(TokenKind::Lt),
            ident("String"),
            token(TokenKind::Comma),
            ident("i64"),
            token(TokenKind::Gt),
        ];
        let mut idx = 0;

        let result = expander.capture_type(&input, &mut idx);
        assert_eq!(result.len(), 10);
        assert_eq!(idx, 10);
    }

    // ==================== CAPTURE BY KIND INTEGRATION TESTS ====================

    #[test]
    fn test_capture_by_kind_path() {
        let expander = MacroExpander::new();
        let input = vec![
            ident("foo"),
            token(TokenKind::ColonColon),
            ident("bar"),
        ];
        let mut idx = 0;

        let result = expander.capture_by_kind(&input, &mut idx, MacroCaptureKind::Path);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_capture_by_kind_pat() {
        let expander = MacroExpander::new();
        let input = vec![ident("Some"), group(MacroDelimiter::Paren, vec![ident("x")])];
        let mut idx = 0;

        let result = expander.capture_by_kind(&input, &mut idx, MacroCaptureKind::Pat);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_capture_by_kind_stmt() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Let),
            ident("x"),
            token(TokenKind::Eq),
            int_lit(1),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_by_kind(&input, &mut idx, MacroCaptureKind::Stmt);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_capture_by_kind_block() {
        let expander = MacroExpander::new();
        let input = vec![group(MacroDelimiter::Brace, vec![int_lit(42)])];
        let mut idx = 0;

        let result = expander.capture_by_kind(&input, &mut idx, MacroCaptureKind::Block);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_capture_by_kind_item() {
        let expander = MacroExpander::new();
        let input = vec![
            token(TokenKind::Struct),
            ident("Foo"),
            token(TokenKind::Semicolon),
        ];
        let mut idx = 0;

        let result = expander.capture_by_kind(&input, &mut idx, MacroCaptureKind::Item);
        assert_eq!(result.len(), 3);
    }

    // ==================== MACRO PATTERN MATCHING WITH NEW KINDS ====================

    #[test]
    fn test_pattern_match_path_capture() {
        let expander = MacroExpander::new();
        let pattern = MacroPattern {
            tokens: vec![MacroToken::Capture {
                name: "p".to_string(),
                kind: MacroCaptureKind::Path,
                span: make_span(),
            }],
            span: make_span(),
        };
        let input = vec![
            ident("std"),
            token(TokenKind::ColonColon),
            ident("io"),
        ];

        let result = expander.match_pattern(&pattern, &input);
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert!(bindings.contains_key("p"));
        if let Some(CapturedValue::Single(tokens)) = bindings.get("p") {
            assert_eq!(tokens.len(), 3);
        } else {
            panic!("Expected Single capture");
        }
    }

    #[test]
    fn test_pattern_match_block_capture() {
        let expander = MacroExpander::new();
        let pattern = MacroPattern {
            tokens: vec![MacroToken::Capture {
                name: "b".to_string(),
                kind: MacroCaptureKind::Block,
                span: make_span(),
            }],
            span: make_span(),
        };
        let input = vec![group(MacroDelimiter::Brace, vec![int_lit(1), token(TokenKind::Plus), int_lit(2)])];

        let result = expander.match_pattern(&pattern, &input);
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert!(bindings.contains_key("b"));
    }

    // ==================== NESTED MACRO EXPANSION TESTS ====================

    /// Helper to create a MacroDef for testing
    fn make_macro_def(name: &str, pattern: Vec<MacroToken>, expansion: Vec<MacroToken>) -> MacroDef {
        use crate::ast::MacroRule;
        MacroDef {
            name: Ident { name: name.to_string(), span: make_span() },
            rules: vec![MacroRule {
                pattern: MacroPattern { tokens: pattern, span: make_span() },
                expansion: MacroExpansion { tokens: expansion, span: make_span() },
                span: make_span(),
            }],
            is_pub: false,
            span: make_span(),
        }
    }

    /// Helper to create a MacroInvocation for testing
    fn make_invocation(name: &str, tokens: Vec<MacroToken>) -> MacroInvocation {
        MacroInvocation {
            name: Ident { name: name.to_string(), span: make_span() },
            delimiter: MacroDelimiter::Paren,
            tokens,
            span: make_span(),
        }
    }

    #[test]
    fn test_try_parse_macro_invocation_valid() {
        let expander = MacroExpander::new();
        let tokens = vec![
            ident("my_macro"),
            token(TokenKind::Not),
            group(MacroDelimiter::Paren, vec![int_lit(42)]),
        ];

        let result = expander.try_parse_macro_invocation(&tokens, 0);
        assert!(result.is_some());
        let (name, invocation, consumed) = result.unwrap();
        assert_eq!(name, "my_macro");
        assert_eq!(consumed, 3);
        assert_eq!(invocation.tokens.len(), 1);
    }

    #[test]
    fn test_try_parse_macro_invocation_no_bang() {
        let expander = MacroExpander::new();
        let tokens = vec![
            ident("my_macro"),
            token(TokenKind::Plus),  // Not a bang
            group(MacroDelimiter::Paren, vec![]),
        ];

        let result = expander.try_parse_macro_invocation(&tokens, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_parse_macro_invocation_no_group() {
        let expander = MacroExpander::new();
        let tokens = vec![
            ident("my_macro"),
            token(TokenKind::Not),
            int_lit(42),  // Not a group
        ];

        let result = expander.try_parse_macro_invocation(&tokens, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_nested_macro_expansion_simple() {
        let mut expander = MacroExpander::new();

        // Define inner macro: inner!($x:expr) => $x + 1
        let inner_def = make_macro_def(
            "inner",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                MacroToken::Capture {
                    name: "x".to_string(),
                    kind: MacroCaptureKind::Expr,
                    span: make_span(),
                },
                token(TokenKind::Plus),
                int_lit(1),
            ],
        );
        expander.register(inner_def);

        // Define outer macro: outer!($x:expr) => inner!($x) * 2
        let outer_def = make_macro_def(
            "outer",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                ident("inner"),
                token(TokenKind::Not),
                group(MacroDelimiter::Paren, vec![
                    MacroToken::Capture {
                        name: "x".to_string(),
                        kind: MacroCaptureKind::Expr,
                        span: make_span(),
                    },
                ]),
                token(TokenKind::Star),
                int_lit(2),
            ],
        );
        expander.register(outer_def);

        // Invoke outer!(5) - should expand to (5 + 1) * 2
        let invocation = make_invocation("outer", vec![int_lit(5)]);
        let result = expander.expand_recursive(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();
        // Should be: 5 + 1 * 2
        assert!(tokens.len() >= 5);
    }

    #[test]
    fn test_nested_macro_expansion_preserves_unknown_macros() {
        let mut expander = MacroExpander::new();

        // Define a macro that produces an unknown macro call
        let def = make_macro_def(
            "wrapper",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                ident("unknown_builtin"),
                token(TokenKind::Not),
                group(MacroDelimiter::Paren, vec![
                    MacroToken::Capture {
                        name: "x".to_string(),
                        kind: MacroCaptureKind::Expr,
                        span: make_span(),
                    },
                ]),
            ],
        );
        expander.register(def);

        let invocation = make_invocation("wrapper", vec![int_lit(42)]);
        let result = expander.expand_recursive(&invocation);

        // Should succeed - unknown macros are preserved
        assert!(result.is_ok());
        let tokens = result.unwrap();
        // Should contain: unknown_builtin ! ( 42 )
        assert!(tokens.len() >= 3);
    }

    #[test]
    fn test_nested_macro_expansion_in_groups() {
        let mut expander = MacroExpander::new();

        // Define inner macro
        let inner_def = make_macro_def(
            "double",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                MacroToken::Capture {
                    name: "x".to_string(),
                    kind: MacroCaptureKind::Expr,
                    span: make_span(),
                },
                token(TokenKind::Star),
                int_lit(2),
            ],
        );
        expander.register(inner_def);

        // Define outer macro that puts inner in a group: outer!($x) => { double!($x) }
        let outer_def = make_macro_def(
            "wrapped",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                group(MacroDelimiter::Brace, vec![
                    ident("double"),
                    token(TokenKind::Not),
                    group(MacroDelimiter::Paren, vec![
                        MacroToken::Capture {
                            name: "x".to_string(),
                            kind: MacroCaptureKind::Expr,
                            span: make_span(),
                        },
                    ]),
                ]),
            ],
        );
        expander.register(outer_def);

        let invocation = make_invocation("wrapped", vec![int_lit(5)]);
        let result = expander.expand_recursive(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();
        // Should be: { 5 * 2 }
        assert_eq!(tokens.len(), 1); // One group
        if let MacroToken::Group { tokens: inner, .. } = &tokens[0] {
            assert!(inner.len() >= 3); // 5 * 2
        } else {
            panic!("Expected a group");
        }
    }

    #[test]
    fn test_nested_macro_recursion_limit() {
        let mut expander = MacroExpander::new();

        // Define a self-recursive macro: recurse!() => recurse!()
        let def = make_macro_def(
            "recurse",
            vec![],  // Empty pattern
            vec![
                ident("recurse"),
                token(TokenKind::Not),
                group(MacroDelimiter::Paren, vec![]),
            ],
        );
        expander.register(def);

        let invocation = make_invocation("recurse", vec![]);
        let result = expander.expand_recursive(&invocation);

        // Should fail with recursion limit error
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err.kind {
            MacroErrorKind::RecursionLimitExceeded { macro_name, .. } => {
                assert_eq!(macro_name, "recurse");
            }
            _ => panic!("Expected RecursionLimitExceeded error, got {:?}", err.kind),
        }
    }

    #[test]
    fn test_nested_macro_chain_three_levels() {
        let mut expander = MacroExpander::new();

        // level3!($x) => $x + 3
        let level3 = make_macro_def(
            "level3",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                MacroToken::Capture {
                    name: "x".to_string(),
                    kind: MacroCaptureKind::Expr,
                    span: make_span(),
                },
                token(TokenKind::Plus),
                int_lit(3),
            ],
        );
        expander.register(level3);

        // level2!($x) => level3!($x) + 2
        let level2 = make_macro_def(
            "level2",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                ident("level3"),
                token(TokenKind::Not),
                group(MacroDelimiter::Paren, vec![
                    MacroToken::Capture {
                        name: "x".to_string(),
                        kind: MacroCaptureKind::Expr,
                        span: make_span(),
                    },
                ]),
                token(TokenKind::Plus),
                int_lit(2),
            ],
        );
        expander.register(level2);

        // level1!($x) => level2!($x) + 1
        let level1 = make_macro_def(
            "level1",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                ident("level2"),
                token(TokenKind::Not),
                group(MacroDelimiter::Paren, vec![
                    MacroToken::Capture {
                        name: "x".to_string(),
                        kind: MacroCaptureKind::Expr,
                        span: make_span(),
                    },
                ]),
                token(TokenKind::Plus),
                int_lit(1),
            ],
        );
        expander.register(level1);

        // Invoke level1!(10) - should expand fully
        let invocation = make_invocation("level1", vec![int_lit(10)]);
        let result = expander.expand_recursive(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();
        // Should produce: 10 + 3 + 2 + 1
        // That's 7 tokens: 10 + 3 + 2 + 1
        assert!(tokens.len() >= 7);
    }

    #[test]
    fn test_expand_tokens_recursive_no_macros() {
        let mut expander = MacroExpander::new();
        let tokens = vec![
            int_lit(1),
            token(TokenKind::Plus),
            int_lit(2),
        ];

        let result = expander.expand_tokens_recursive(&tokens, 0, make_span());
        assert!(result.is_ok());
        let expanded = result.unwrap();
        assert_eq!(expanded.len(), 3);
    }

    #[test]
    fn test_nested_macro_with_bracket_delimiter() {
        let mut expander = MacroExpander::new();

        // Define a macro that uses brackets
        let def = make_macro_def(
            "vec_like",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                ident("Vec"),
                token(TokenKind::ColonColon),
                ident("from"),
                group(MacroDelimiter::Paren, vec![
                    group(MacroDelimiter::Bracket, vec![
                        MacroToken::Capture {
                            name: "x".to_string(),
                            kind: MacroCaptureKind::Expr,
                            span: make_span(),
                        },
                    ]),
                ]),
            ],
        );
        expander.register(def);

        let invocation = make_invocation("vec_like", vec![int_lit(42)]);
        let result = expander.expand_recursive(&invocation);

        assert!(result.is_ok());
    }

    // ==================== MACRO HYGIENE TESTS ====================

    #[test]
    fn test_hygiene_context_gensym() {
        let mut ctx = HygieneContext::new();
        let name1 = ctx.gensym("temp");
        let name2 = ctx.gensym("temp");
        let name3 = ctx.gensym("x");

        // Each gensym should be unique
        assert_ne!(name1, name2);
        assert_ne!(name2, name3);
        assert_ne!(name1, name3);

        // Names should have the expected format
        assert!(name1.starts_with("__macro_temp_"));
        assert!(name2.starts_with("__macro_temp_"));
        assert!(name3.starts_with("__macro_x_"));
    }

    #[test]
    fn test_hygiene_enabled_by_default() {
        let expander = MacroExpander::new();
        assert!(expander.hygiene.is_enabled());
    }

    #[test]
    fn test_hygiene_can_be_disabled() {
        let expander = MacroExpander::new_without_hygiene();
        assert!(!expander.hygiene.is_enabled());
    }

    #[test]
    fn test_should_hygienize_user_identifiers() {
        let expander = MacroExpander::new();
        let captures: std::collections::HashSet<String> = std::collections::HashSet::new();

        // User-defined names should be hygienized
        assert!(expander.should_hygienize("temp", &captures));
        assert!(expander.should_hygienize("x", &captures));
        assert!(expander.should_hygienize("my_var", &captures));
    }

    #[test]
    fn test_should_not_hygienize_keywords() {
        let expander = MacroExpander::new();
        let captures: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Keywords should not be hygienized
        assert!(!expander.should_hygienize("let", &captures));
        assert!(!expander.should_hygienize("mut", &captures));
        assert!(!expander.should_hygienize("fn", &captures));
        assert!(!expander.should_hygienize("if", &captures));
        assert!(!expander.should_hygienize("return", &captures));
    }

    #[test]
    fn test_should_not_hygienize_builtins() {
        let expander = MacroExpander::new();
        let captures: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Builtins should not be hygienized
        assert!(!expander.should_hygienize("println", &captures));
        assert!(!expander.should_hygienize("Vec", &captures));
        assert!(!expander.should_hygienize("Option", &captures));
        assert!(!expander.should_hygienize("Some", &captures));
        assert!(!expander.should_hygienize("None", &captures));
    }

    #[test]
    fn test_should_not_hygienize_captures() {
        let expander = MacroExpander::new();
        let mut captures: std::collections::HashSet<String> = std::collections::HashSet::new();
        captures.insert("x".to_string());
        captures.insert("val".to_string());

        // Captured variables should not be hygienized
        assert!(!expander.should_hygienize("x", &captures));
        assert!(!expander.should_hygienize("val", &captures));

        // Non-captured should still be hygienized
        assert!(expander.should_hygienize("temp", &captures));
    }

    #[test]
    fn test_hygiene_renames_local_variables() {
        let mut expander = MacroExpander::new();

        // macro make_temp!($val:expr) => { let temp = $val; temp * 2 }
        let def = make_macro_def(
            "make_temp",
            vec![MacroToken::Capture {
                name: "val".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                token(TokenKind::Let),
                ident("temp"),  // Should be hygienized
                token(TokenKind::Eq),
                MacroToken::Capture {
                    name: "val".to_string(),
                    kind: MacroCaptureKind::Expr,
                    span: make_span(),
                },
                token(TokenKind::Semicolon),
                ident("temp"),  // Should be hygienized (same as above)
                token(TokenKind::Star),
                int_lit(2),
            ],
        );
        expander.register(def);

        let invocation = make_invocation("make_temp", vec![int_lit(5)]);
        let result = expander.expand(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();

        // Find the hygienized identifier
        let hygienized_names: Vec<_> = tokens.iter()
            .filter_map(|t| match t {
                MacroToken::Ident(name, _) if name.starts_with("__macro_temp_") => Some(name.clone()),
                _ => None,
            })
            .collect();

        // Should have exactly 2 occurrences of the same hygienized name
        assert_eq!(hygienized_names.len(), 2);
        assert_eq!(hygienized_names[0], hygienized_names[1]);
    }

    #[test]
    fn test_hygiene_preserves_captured_values() {
        let mut expander = MacroExpander::new();

        // macro wrap!($x:expr) => { $x }
        let def = make_macro_def(
            "wrap",
            vec![MacroToken::Capture {
                name: "x".to_string(),
                kind: MacroCaptureKind::Expr,
                span: make_span(),
            }],
            vec![
                MacroToken::Capture {
                    name: "x".to_string(),
                    kind: MacroCaptureKind::Expr,
                    span: make_span(),
                },
            ],
        );
        expander.register(def);

        // Call with an identifier - should NOT be renamed
        let invocation = make_invocation("wrap", vec![ident("user_var")]);
        let result = expander.expand(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();

        // The identifier should be preserved as-is
        assert_eq!(tokens.len(), 1);
        if let MacroToken::Ident(name, _) = &tokens[0] {
            assert_eq!(name, "user_var");  // Not hygienized!
        } else {
            panic!("Expected Ident token");
        }
    }

    #[test]
    fn test_hygiene_disabled_preserves_names() {
        let mut expander = MacroExpander::new_without_hygiene();

        // macro make_x!() => { let x = 42; x }
        let def = make_macro_def(
            "make_x",
            vec![],  // No captures
            vec![
                token(TokenKind::Let),
                ident("x"),
                token(TokenKind::Eq),
                int_lit(42),
                token(TokenKind::Semicolon),
                ident("x"),
            ],
        );
        expander.register(def);

        let invocation = make_invocation("make_x", vec![]);
        let result = expander.expand(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();

        // With hygiene disabled, 'x' should remain 'x'
        let x_tokens: Vec<_> = tokens.iter()
            .filter_map(|t| match t {
                MacroToken::Ident(name, _) if name == "x" => Some(name.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(x_tokens.len(), 2);
    }

    #[test]
    fn test_hygiene_different_macros_different_gensyms() {
        let mut expander = MacroExpander::new();

        // First macro with 'temp'
        let def1 = make_macro_def(
            "macro1",
            vec![],
            vec![ident("temp")],
        );
        expander.register(def1);

        // Second macro with 'temp'
        let def2 = make_macro_def(
            "macro2",
            vec![],
            vec![ident("temp")],
        );
        expander.register(def2);

        let invocation1 = make_invocation("macro1", vec![]);
        let result1 = expander.expand(&invocation1).unwrap();

        let invocation2 = make_invocation("macro2", vec![]);
        let result2 = expander.expand(&invocation2).unwrap();

        // Each expansion should get a different gensym
        let name1 = match &result1[0] {
            MacroToken::Ident(n, _) => n.clone(),
            _ => panic!("Expected Ident"),
        };
        let name2 = match &result2[0] {
            MacroToken::Ident(n, _) => n.clone(),
            _ => panic!("Expected Ident"),
        };

        assert_ne!(name1, name2);
        assert!(name1.starts_with("__macro_temp_"));
        assert!(name2.starts_with("__macro_temp_"));
    }

    #[test]
    fn test_hygiene_in_nested_groups() {
        let mut expander = MacroExpander::new();

        // macro block!() => { { let inner = 1; inner } }
        let def = make_macro_def(
            "block",
            vec![],
            vec![
                group(MacroDelimiter::Brace, vec![
                    token(TokenKind::Let),
                    ident("inner"),
                    token(TokenKind::Eq),
                    int_lit(1),
                    token(TokenKind::Semicolon),
                    ident("inner"),
                ]),
            ],
        );
        expander.register(def);

        let invocation = make_invocation("block", vec![]);
        let result = expander.expand(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();

        // Check that hygiene was applied inside the group
        if let MacroToken::Group { tokens: inner, .. } = &tokens[0] {
            let hygienized: Vec<_> = inner.iter()
                .filter_map(|t| match t {
                    MacroToken::Ident(name, _) if name.starts_with("__macro_inner_") => Some(name.clone()),
                    _ => None,
                })
                .collect();

            assert_eq!(hygienized.len(), 2);
            assert_eq!(hygienized[0], hygienized[1]);
        } else {
            panic!("Expected Group token");
        }
    }

    // ==================== MODULE SCOPE TESTS ====================

    #[test]
    fn test_register_global_macro() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("test_macro", vec![], vec![int_lit(42)]);

        expander.register(def);

        assert!(expander.has_macro("test_macro"));
    }

    #[test]
    fn test_register_in_module() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("my_macro", vec![], vec![int_lit(42)]);

        expander.register_in_module(def, Some("mymod".to_string()), true);

        // Should be visible because it's public
        assert!(expander.has_macro("my_macro"));
    }

    #[test]
    fn test_private_macro_not_visible_from_other_module() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("priv_macro", vec![], vec![int_lit(42)]);

        // Register as private in "mod_a"
        expander.register_in_module(def, Some("mod_a".to_string()), false);

        // Set current module to "mod_b"
        expander.set_current_module(Some("mod_b".to_string()));

        // Should NOT be visible from mod_b because it's private
        assert!(!expander.has_macro("priv_macro"));
    }

    #[test]
    fn test_private_macro_visible_from_same_module() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("priv_macro", vec![], vec![int_lit(42)]);

        // Register as private in "mod_a"
        expander.register_in_module(def, Some("mod_a".to_string()), false);

        // Set current module to "mod_a"
        expander.set_current_module(Some("mod_a".to_string()));

        // Should be visible from mod_a (same module)
        assert!(expander.has_macro("priv_macro"));
    }

    #[test]
    fn test_public_macro_visible_from_other_module() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("pub_macro", vec![], vec![int_lit(42)]);

        // Register as public in "mod_a"
        expander.register_in_module(def, Some("mod_a".to_string()), true);

        // Set current module to "mod_b"
        expander.set_current_module(Some("mod_b".to_string()));

        // Should be visible from mod_b because it's public
        assert!(expander.has_macro("pub_macro"));
    }

    #[test]
    fn test_import_macro() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("exported", vec![], vec![int_lit(42)]);

        // Register as public in "lib"
        expander.register_in_module(def, Some("lib".to_string()), true);

        // Import into "main" module
        expander.import_macro("lib", "exported", "main", None);

        // Set current module to "main"
        expander.set_current_module(Some("main".to_string()));

        // Should be visible via import
        assert!(expander.has_macro("exported"));
    }

    #[test]
    fn test_import_with_alias() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("original_name", vec![], vec![int_lit(42)]);

        // Register as public in "lib"
        expander.register_in_module(def, Some("lib".to_string()), true);

        // Import with alias into "main" module
        expander.import_macro("lib", "original_name", "main", Some("aliased"));

        // Set current module to "main"
        expander.set_current_module(Some("main".to_string()));

        // Should be visible via aliased name
        assert!(expander.has_macro("aliased"));
        // Original name should not be directly visible (need to use path)
    }

    #[test]
    fn test_global_macros_always_visible() {
        let mut expander = MacroExpander::new();
        let def = make_macro_def("global_macro", vec![], vec![int_lit(42)]);

        // Register globally
        expander.register(def);

        // Set current module to anything
        expander.set_current_module(Some("some_module".to_string()));

        // Should still be visible
        assert!(expander.has_macro("global_macro"));
    }

    #[test]
    fn test_is_macro_visible() {
        let mut expander = MacroExpander::new();

        // Register public macro
        let pub_def = make_macro_def("pub_mac", vec![], vec![int_lit(1)]);
        expander.register_in_module(pub_def, Some("mod_a".to_string()), true);

        // Register private macro
        let priv_def = make_macro_def("priv_mac", vec![], vec![int_lit(2)]);
        expander.register_in_module(priv_def, Some("mod_a".to_string()), false);

        // Public should be visible from anywhere
        assert!(expander.is_macro_visible("pub_mac", Some("mod_b")));
        assert!(expander.is_macro_visible("pub_mac", Some("mod_a")));
        assert!(expander.is_macro_visible("pub_mac", None));

        // Private should only be visible from same module
        assert!(expander.is_macro_visible("priv_mac", Some("mod_a")));
        assert!(!expander.is_macro_visible("priv_mac", Some("mod_b")));
    }

    #[test]
    fn test_expand_macro_from_module() {
        let mut expander = MacroExpander::new();

        // Create a simple macro that returns 42
        let def = make_macro_def("answer", vec![], vec![int_lit(42)]);

        // Register in module
        expander.register_in_module(def, Some("mymod".to_string()), true);

        // Set current module
        expander.set_current_module(Some("mymod".to_string()));

        // Expand
        let invocation = make_invocation("answer", vec![]);
        let result = expander.expand(&invocation);

        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0], MacroToken::IntLit(42, _)));
    }
}
