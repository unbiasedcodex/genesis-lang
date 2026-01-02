//! Document management for the LSP server
//!
//! Tracks open documents and their analysis state.

use dashmap::DashMap;
use std::sync::Arc;
use tower_lsp::lsp_types::Url;

use crate::analysis::AnalysisResult;
use crate::utils::LineIndex;

/// A document being edited
#[derive(Debug)]
pub struct Document {
    /// The document URI
    pub uri: Url,
    /// Current document content
    pub content: String,
    /// Line index for position conversion
    pub line_index: LineIndex,
    /// Version for incremental sync
    pub version: i32,
    /// Cached analysis result
    pub analysis: Option<AnalysisResult>,
}

impl Document {
    /// Create a new document
    pub fn new(uri: Url, content: String, version: i32) -> Self {
        let line_index = LineIndex::new(&content);
        Self {
            uri,
            content,
            line_index,
            version,
            analysis: None,
        }
    }

    /// Update document content
    pub fn update(&mut self, content: String, version: i32) {
        self.content = content;
        self.line_index = LineIndex::new(&self.content);
        self.version = version;
        self.analysis = None; // Invalidate analysis
    }

    /// Set analysis result
    pub fn set_analysis(&mut self, analysis: AnalysisResult) {
        self.analysis = Some(analysis);
    }
}

/// Store for all open documents
#[derive(Debug, Default)]
pub struct DocumentStore {
    documents: DashMap<Url, Arc<Document>>,
}

impl DocumentStore {
    /// Create a new document store
    pub fn new() -> Self {
        Self {
            documents: DashMap::new(),
        }
    }

    /// Open a new document
    pub fn open(&self, uri: Url, content: String, version: i32) -> Arc<Document> {
        let doc = Arc::new(Document::new(uri.clone(), content, version));
        self.documents.insert(uri, Arc::clone(&doc));
        doc
    }

    /// Update a document
    pub fn update(&self, uri: &Url, content: String, version: i32) -> Option<Arc<Document>> {
        if let Some(mut entry) = self.documents.get_mut(uri) {
            // Create new document with updated content
            let new_doc = Arc::new(Document::new(uri.clone(), content, version));
            *entry = Arc::clone(&new_doc);
            Some(new_doc)
        } else {
            None
        }
    }

    /// Get a document
    pub fn get(&self, uri: &Url) -> Option<Arc<Document>> {
        self.documents.get(uri).map(|r| Arc::clone(&*r))
    }

    /// Close a document
    pub fn close(&self, uri: &Url) {
        self.documents.remove(uri);
    }

    /// Get all document URIs
    pub fn uris(&self) -> Vec<Url> {
        self.documents.iter().map(|r| r.key().clone()).collect()
    }
}
