"""Headless Language Server Protocol integration.

Tree-sitter and regex are great for cheap structural extraction, but
they don't know about types, inheritance, or cross-file resolution.
The same identifier ``processData`` in two files might refer to two
unrelated functions; tree-sitter can't tell. A real language server
can — and the Language Server Protocol makes that capability available
to anything that can speak JSON-RPC over stdio.

This subsystem:

  - launches a per-language LSP server as a subprocess
    (pyright-langserver, tsserver, gopls, rust-analyzer, jdtls, clangd,
    omnisharp, …) on demand
  - performs the LSP handshake (initialize, initialized) per repo
  - exposes ``definition``, ``references``, ``hover``, ``call_hierarchy``,
    and ``document_symbol`` as Python calls
  - feeds extracted ground-truth references into the multi-graph
    (replacing the heuristic CALLS edges from the regex/tree-sitter
    extractor)

All operations are best-effort. If a language server isn't installed,
the system silently falls back to tree-sitter. The heuristic graph is
always built; LSP only adds higher-fidelity edges on top.
"""
from src.lsp.manager import LSPManager, get_manager
from src.lsp.servers import SERVERS, detect_server_for

__all__ = ["LSPManager", "get_manager", "SERVERS", "detect_server_for"]
