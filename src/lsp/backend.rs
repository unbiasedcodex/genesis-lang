//! LSP Backend implementation for Genesis Lang
//!
//! Implements the Language Server Protocol for Genesis.

use std::sync::Arc;
use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::analysis::{self, AnalysisResult};
use crate::document::{Document, DocumentStore};
use crate::utils::LineIndex;

/// The Genesis Language Server backend
pub struct GenesisBackend {
    /// LSP client for sending notifications
    client: Client,
    /// Open documents
    documents: DocumentStore,
    /// Cached analysis results
    analysis_cache: DashMap<Url, AnalysisResult>,
}

impl GenesisBackend {
    /// Create a new backend
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DocumentStore::new(),
            analysis_cache: DashMap::new(),
        }
    }

    /// Analyze a document and publish diagnostics
    async fn analyze_and_publish(&self, uri: Url, content: &str, line_index: &LineIndex) {
        // Run analysis
        let result = analysis::analyze(content);

        // Convert to diagnostics
        let diagnostics = analysis::to_diagnostics(&result, line_index);

        // Publish diagnostics
        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;

        // Cache analysis result
        self.analysis_cache.insert(uri, result);
    }

    /// Get cached analysis for a document
    fn get_analysis(&self, uri: &Url) -> Option<dashmap::mapref::one::Ref<'_, Url, AnalysisResult>> {
        self.analysis_cache.get(uri)
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for GenesisBackend {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "genesis-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
            capabilities: ServerCapabilities {
                // Full document sync
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                // Hover support
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                // Completion support
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                    ..Default::default()
                }),
                // Go-to-definition
                definition_provider: Some(OneOf::Left(true)),
                // Find references
                references_provider: Some(OneOf::Left(true)),
                // Document symbols (outline)
                document_symbol_provider: Some(OneOf::Left(true)),
                // Signature help
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                // Rename support
                rename_provider: Some(OneOf::Left(true)),
                // Workspace symbol search
                workspace_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Genesis LSP initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let content = params.text_document.text;
        let version = params.text_document.version;

        let doc = self.documents.open(uri.clone(), content.clone(), version);
        self.analyze_and_publish(uri, &content, &doc.line_index).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        // Get the new content (we use full sync, so there's only one change)
        if let Some(change) = params.content_changes.into_iter().next() {
            if let Some(doc) = self.documents.update(&uri, change.text.clone(), version) {
                self.analyze_and_publish(uri, &change.text, &doc.line_index).await;
            }
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.close(&uri);
        self.analysis_cache.remove(&uri);

        // Clear diagnostics
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        // Re-analyze on save if we have new content
        if let Some(text) = params.text {
            let uri = params.text_document.uri;
            if let Some(doc) = self.documents.get(&uri) {
                self.analyze_and_publish(uri, &text, &doc.line_index).await;
            }
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => return Ok(None),
        };

        Ok(analysis::get_hover(&analysis, &doc.line_index, position))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => return Ok(None),
        };

        if let Some((span, _)) = analysis::get_definition(&analysis, &doc.line_index, position) {
            let range = doc.line_index.span_to_range(span);
            return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                uri: uri.clone(),
                range,
            })));
        }

        Ok(None)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => return Ok(None),
        };

        let spans = analysis::find_references(&analysis, &doc.line_index, position);
        if spans.is_empty() {
            return Ok(None);
        }

        let locations: Vec<Location> = spans
            .into_iter()
            .map(|span| Location {
                uri: uri.clone(),
                range: doc.line_index.span_to_range(span),
            })
            .collect();

        Ok(Some(locations))
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => {
                // Even without analysis, provide keyword completions
                let result = analysis::analyze(&doc.content);
                let items = analysis::get_completions(&result, &doc.content, &doc.line_index, position);
                return Ok(Some(CompletionResponse::Array(items)));
            }
        };

        let items = analysis::get_completions(&analysis, &doc.content, &doc.line_index, position);
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => return Ok(None),
        };

        let symbols = analysis::get_document_symbols(&analysis, &doc.line_index);
        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => return Ok(None),
        };

        // Find the function being called at this position
        let offset = doc.line_index.position_to_offset(position);

        // Look backwards for the function name
        let content = &doc.content;
        let mut paren_depth = 0;
        let mut func_end = offset;

        for (i, c) in content[..offset].char_indices().rev() {
            match c {
                ')' => paren_depth += 1,
                '(' => {
                    if paren_depth == 0 {
                        func_end = i;
                        break;
                    }
                    paren_depth -= 1;
                }
                _ => {}
            }
        }

        // Get the function name
        let mut func_start = func_end;
        for (i, c) in content[..func_end].char_indices().rev() {
            if c.is_alphanumeric() || c == '_' {
                func_start = i;
            } else {
                break;
            }
        }

        if func_start >= func_end {
            return Ok(None);
        }

        let func_name = &content[func_start..func_end];

        // Find the function definition
        if let Some(def) = analysis.definitions.get(func_name) {
            if def.kind == analysis::SymbolDefKind::Function {
                // Get the function signature from the AST
                for item in &analysis.program.items {
                    if let genesis::ast::Item::Function(f) = item {
                        if f.name.name == func_name {
                            let params: Vec<ParameterInformation> = f.params.iter()
                                .map(|p| {
                                    let label = format!("{}: {}", p.name.name, format_ast_type(&p.ty));
                                    ParameterInformation {
                                        label: ParameterLabel::Simple(label),
                                        documentation: None,
                                    }
                                })
                                .collect();

                            let signature = SignatureInformation {
                                label: format_fn_signature(f),
                                documentation: None,
                                parameters: Some(params),
                                active_parameter: None,
                            };

                            return Ok(Some(SignatureHelp {
                                signatures: vec![signature],
                                active_signature: Some(0),
                                active_parameter: None,
                            }));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = params.new_name;

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return Ok(None),
        };

        let analysis = match self.get_analysis(uri) {
            Some(a) => a,
            None => return Ok(None),
        };

        // Find all references
        let spans = analysis::find_references(&analysis, &doc.line_index, position);
        if spans.is_empty() {
            return Ok(None);
        }

        // Create text edits
        let edits: Vec<TextEdit> = spans
            .into_iter()
            .map(|span| TextEdit {
                range: doc.line_index.span_to_range(span),
                new_text: new_name.clone(),
            })
            .collect();

        let mut changes = std::collections::HashMap::new();
        changes.insert(uri.clone(), edits);

        Ok(Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }))
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = params.query.to_lowercase();
        let mut symbols = Vec::new();

        for entry in self.analysis_cache.iter() {
            let uri = entry.key();
            let analysis = entry.value();

            if let Some(doc) = self.documents.get(uri) {
                for (name, def) in &analysis.definitions {
                    if query.is_empty() || name.to_lowercase().contains(&query) {
                        #[allow(deprecated)]
                        symbols.push(SymbolInformation {
                            name: name.clone(),
                            kind: def.kind.to_symbol_kind(),
                            tags: None,
                            deprecated: None,
                            location: Location {
                                uri: uri.clone(),
                                range: doc.line_index.span_to_range(def.span),
                            },
                            container_name: None,
                        });
                    }
                }
            }
        }

        if symbols.is_empty() {
            Ok(None)
        } else {
            Ok(Some(symbols))
        }
    }
}

fn format_ast_type(ty: &genesis::ast::Type) -> String {
    match &ty.kind {
        genesis::ast::TypeKind::Path(path) => {
            path.segments.iter()
                .map(|s| {
                    if let Some(ref generics) = s.generics {
                        format!("{}<{}>", s.ident.name,
                            generics.iter().map(|t| format_ast_type(t)).collect::<Vec<_>>().join(", "))
                    } else {
                        s.ident.name.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join("::")
        }
        genesis::ast::TypeKind::Reference { mutable, inner } => {
            if *mutable {
                format!("&mut {}", format_ast_type(inner))
            } else {
                format!("&{}", format_ast_type(inner))
            }
        }
        genesis::ast::TypeKind::Array { element, size: _ } => {
            format!("[{}; N]", format_ast_type(element))
        }
        genesis::ast::TypeKind::Slice { element } => {
            format!("[{}]", format_ast_type(element))
        }
        genesis::ast::TypeKind::Tuple(elements) => {
            format!("({})", elements.iter().map(|t| format_ast_type(t)).collect::<Vec<_>>().join(", "))
        }
        genesis::ast::TypeKind::FnPtr { params, return_type } => {
            let ret = return_type.as_ref()
                .map(|t| format_ast_type(t))
                .unwrap_or_else(|| "()".to_string());
            format!("fn({}) -> {}",
                params.iter().map(|t| format_ast_type(t)).collect::<Vec<_>>().join(", "),
                ret)
        }
        genesis::ast::TypeKind::Never => "!".to_string(),
        genesis::ast::TypeKind::Infer => "_".to_string(),
        genesis::ast::TypeKind::Option(inner) => format!("Option<{}>", format_ast_type(inner)),
        genesis::ast::TypeKind::Result { ok, err } => format!("Result<{}, {}>", format_ast_type(ok), format_ast_type(err)),
        genesis::ast::TypeKind::SelfType => "Self".to_string(),
        genesis::ast::TypeKind::Projection { base, assoc_name } => format!("{}::{}", format_ast_type(base), assoc_name),
        genesis::ast::TypeKind::TraitObject { trait_name } => format!("dyn {}", trait_name),
    }
}

fn format_fn_signature(f: &genesis::ast::FnDef) -> String {
    let params: Vec<String> = f.params.iter()
        .map(|p| format!("{}: {}", p.name.name, format_ast_type(&p.ty)))
        .collect();

    let ret = f.return_type.as_ref()
        .map(|t| format!(" -> {}", format_ast_type(t)))
        .unwrap_or_default();

    let async_prefix = if f.is_async { "async " } else { "" };

    format!("{}fn {}({}){}", async_prefix, f.name.name, params.join(", "), ret)
}
