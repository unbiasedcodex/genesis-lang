//! Genesis Language Server Protocol (LSP) Server
//!
//! This binary provides LSP support for Genesis Lang, enabling:
//! - Real-time diagnostics (errors and warnings)
//! - Hover information (types and documentation)
//! - Go-to-definition
//! - Autocomplete
//! - Find references
//! - Document symbols (outline)

use tower_lsp::{LspService, Server};

mod backend;
mod document;
mod analysis;
mod utils;

use backend::GenesisBackend;

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| GenesisBackend::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}
