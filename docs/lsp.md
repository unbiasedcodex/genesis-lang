# Genesis Language Server Setup

This guide explains how to configure the Genesis LSP server with popular editors.

---

## Features

The Genesis LSP server provides:

| Feature | Description |
|---------|-------------|
| Diagnostics | Real-time error and warning messages |
| Hover | Type information on hover |
| Go-to-Definition | Jump to symbol definitions |
| Autocomplete | Code completion suggestions |
| Find References | Find all references to a symbol |
| Document Symbols | Outline view of the file |
| Signature Help | Function parameter hints |
| Rename | Rename symbols across the file |
| Workspace Symbols | Search symbols across files |

---

## Installation

Build the LSP server:

```bash
cd genesis-lang
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 cargo build --release --bin genesis-lsp
```

The binary will be at `target/release/genesis-lsp`.

---

## Editor Configuration

### VSCode

1. Create a custom extension or use "Custom Local Formatters"

2. Add to `settings.json`:

```json
{
    "genesis.lsp.path": "/path/to/genesis-lsp",
    "[genesis]": {
        "editor.formatOnSave": true
    }
}
```

3. Create extension `package.json`:

```json
{
    "name": "genesis-lang",
    "displayName": "Genesis Language",
    "version": "0.1.0",
    "engines": { "vscode": "^1.75.0" },
    "categories": ["Programming Languages"],
    "contributes": {
        "languages": [{
            "id": "genesis",
            "aliases": ["Genesis", "genesis"],
            "extensions": [".gl"],
            "configuration": "./language-configuration.json"
        }],
        "grammars": [{
            "language": "genesis",
            "scopeName": "source.genesis",
            "path": "./syntaxes/genesis.tmLanguage.json"
        }]
    },
    "main": "./out/extension.js"
}
```

4. Create `extension.ts`:

```typescript
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    const serverOptions: ServerOptions = {
        command: '/path/to/genesis-lsp',
        args: [],
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'genesis' }],
    };

    client = new LanguageClient(
        'genesis-lsp',
        'Genesis Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) return undefined;
    return client.stop();
}
```

---

### Neovim (with nvim-lspconfig)

Add to `init.lua`:

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Register Genesis filetype
vim.filetype.add({
    extension = { gl = 'genesis' }
})

-- Define Genesis LSP config
if not configs.genesis_lsp then
    configs.genesis_lsp = {
        default_config = {
            cmd = { '/path/to/genesis-lsp' },
            filetypes = { 'genesis' },
            root_dir = function(fname)
                return lspconfig.util.find_git_ancestor(fname) or vim.fn.getcwd()
            end,
            settings = {},
        },
    }
end

-- Setup the LSP
lspconfig.genesis_lsp.setup({
    on_attach = function(client, bufnr)
        vim.bo[bufnr].omnifunc = 'v:lua.vim.lsp.omnifunc'

        local opts = { buffer = bufnr }
        vim.keymap.set('n', 'gd', vim.lsp.buf.definition, opts)
        vim.keymap.set('n', 'K', vim.lsp.buf.hover, opts)
        vim.keymap.set('n', 'gr', vim.lsp.buf.references, opts)
        vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, opts)
        vim.keymap.set('n', '<leader>ca', vim.lsp.buf.code_action, opts)
    end,
})
```

---

### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "genesis"
scope = "source.genesis"
injection-regex = "genesis"
file-types = ["gl"]
roots = ["Cargo.toml", ".git"]
comment-token = "//"
language-server = { command = "/path/to/genesis-lsp" }
indent = { tab-width = 4, unit = "    " }
```

---

### Emacs (with lsp-mode)

Add to your config:

```elisp
;; Define Genesis mode
(define-derived-mode genesis-mode prog-mode "Genesis"
  "Major mode for Genesis programming language."
  (setq-local comment-start "// ")
  (setq-local comment-end ""))

(add-to-list 'auto-mode-alist '("\\.gl\\'" . genesis-mode))

;; LSP configuration
(require 'lsp-mode)

(add-to-list 'lsp-language-id-configuration '(genesis-mode . "genesis"))

(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection '("/path/to/genesis-lsp"))
  :major-modes '(genesis-mode)
  :server-id 'genesis-lsp))

(add-hook 'genesis-mode-hook #'lsp)
```

---

### Zed

Add to `~/.config/zed/settings.json`:

```json
{
    "lsp": {
        "genesis-lsp": {
            "binary": { "path": "/path/to/genesis-lsp" }
        }
    },
    "languages": {
        "Genesis": {
            "language_servers": ["genesis-lsp"]
        }
    }
}
```

---

### Sublime Text (with LSP package)

1. Install the LSP package via Package Control

2. Add to `LSP.sublime-settings`:

```json
{
    "clients": {
        "genesis-lsp": {
            "enabled": true,
            "command": ["/path/to/genesis-lsp"],
            "selector": "source.genesis"
        }
    }
}
```

---

## Troubleshooting

### LSP not starting

1. Check execute permissions:
   ```bash
   chmod +x /path/to/genesis-lsp
   ```

2. Test the binary:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | /path/to/genesis-lsp
   ```

### No diagnostics

- Ensure file has `.gl` extension
- Check that file is saved (LSP analyzes saved content)

### Slow performance

- LSP analyzes entire file on each change
- Large files may experience delays
- Enable incremental sync in your editor

---

## Debug Mode

Run with logging:

```bash
RUST_LOG=debug genesis-lsp 2> /tmp/genesis-lsp.log
```
