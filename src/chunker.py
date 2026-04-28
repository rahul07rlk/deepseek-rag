"""Code chunking.

Per-file routing:
  - `.py` → stdlib `ast` (Python AST chunker, lowest overhead)
  - TS/JS/Go/Rust/Java/C/C++/C#/PHP/Ruby/Swift/Kotlin → tree-sitter
    via src.tree_sitter_chunker
  - everything else → overlapping line windows (fallback)

All paths emit dicts with the same shape:

    {"text": str, "start_line": int, "end_line": int, "symbol": str | None}
"""
from __future__ import annotations

import ast
from pathlib import Path

from src.config import CHUNK_LINES, CHUNK_OVERLAP, MAX_CHUNK_CHARS, MIN_CHUNK_LEN
from src.tree_sitter_chunker import chunk_with_tree_sitter, is_supported as _ts_supported

# AST nodes are semantic units - even a 1-line function is worth indexing.
# The bigger MIN_CHUNK_LEN only applies to line-window fragments.
_AST_MIN_CHARS = 10


def _window_chunks(lines: list[str]) -> list[dict]:
    """Overlapping line-window chunks. Used as fallback and for non-Python."""
    out: list[dict] = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for i in range(0, len(lines), step):
        window = lines[i : i + CHUNK_LINES]
        text = "".join(window).strip()
        if len(text) < MIN_CHUNK_LEN:
            continue
        out.append({
            "text": text,
            "start_line": i + 1,
            "end_line": i + len(window),
            "symbol": None,
        })
        if i + CHUNK_LINES >= len(lines):
            break
    return out


def _split_large(text: str, start_line: int, symbol: str | None) -> list[dict]:
    """If an AST node is too big, split by line windows but keep the symbol label."""
    lines = text.splitlines(keepends=True)
    if len(text) <= MAX_CHUNK_CHARS:
        return [{
            "text": text,
            "start_line": start_line,
            "end_line": start_line + max(1, len(lines)) - 1,
            "symbol": symbol,
        }]

    sub: list[dict] = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for i in range(0, len(lines), step):
        window = lines[i : i + CHUNK_LINES]
        chunk_text = "".join(window).strip()
        if len(chunk_text) < _AST_MIN_CHARS:
            continue
        sub.append({
            "text": chunk_text,
            "start_line": start_line + i,
            "end_line": start_line + i + len(window) - 1,
            "symbol": symbol,
        })
        if i + CHUNK_LINES >= len(lines):
            break
    return sub


def _python_ast_chunks(source: str) -> list[dict] | None:
    """Try AST-based chunking. Returns None if the file can't be parsed."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines(keepends=True)
    chunks: list[dict] = []
    module_nodes = list(ast.iter_child_nodes(tree))

    # Top-of-file "header": imports + module docstring + constants, grouped
    # together so they stay retrievable as one piece of context.
    header_end = 0
    for node in module_nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)):
            header_end = max(header_end, node.end_lineno or header_end)
        else:
            break
    if header_end > 0:
        header_text = "".join(lines[:header_end]).strip()
        if len(header_text) >= _AST_MIN_CHARS:
            chunks.extend(_split_large(header_text, 1, "<module header>"))

    for node in module_nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = node.end_lineno or start
        body = "".join(lines[start - 1 : end]).rstrip()
        if len(body) < _AST_MIN_CHARS:
            continue

        if isinstance(node, ast.ClassDef):
            # Emit the class as one chunk AND each method separately. The
            # class-level chunk gives structural context; method chunks give
            # targeted retrieval.
            chunks.extend(_split_large(body, start, f"class {node.name}"))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    s, e = sub.lineno, sub.end_lineno or sub.lineno
                    sub_body = "".join(lines[s - 1 : e]).rstrip()
                    if len(sub_body) >= _AST_MIN_CHARS:
                        chunks.extend(
                            _split_large(sub_body, s, f"{node.name}.{sub.name}")
                        )
        else:
            chunks.extend(_split_large(body, start, node.name))

    # If AST yielded nothing useful (e.g. script with only statements),
    # fall back to windowed chunking so the file still gets indexed.
    if not chunks:
        return None
    return chunks


def chunk_file(filepath: Path) -> list[dict]:
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    if not source.strip():
        return []

    suffix = filepath.suffix
    if suffix == ".py":
        ast_chunks = _python_ast_chunks(source)
        if ast_chunks:
            return ast_chunks
    elif _ts_supported(suffix):
        ts_chunks = chunk_with_tree_sitter(source, suffix)
        if ts_chunks:
            return ts_chunks

    return _window_chunks(source.splitlines(keepends=True))
