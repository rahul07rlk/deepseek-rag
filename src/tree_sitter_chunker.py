"""Multi-language AST chunking via tree-sitter-language-pack.

Produces chunks at function / class / interface / struct / impl boundaries
for TypeScript, JavaScript, Go, Rust, Java, C, C++, C#, PHP, Ruby, Swift,
and Kotlin. Same chunk shape as src/chunker.py's Python AST path so the
rest of the pipeline is identical.

Falls back to None on:
  - language not supported / not configured
  - tree-sitter import failure (package missing)
  - parse failure
The caller (src/chunker.py) handles fallback to line-window chunking.

The underlying package downloads each language's grammar lazily on first
parse (to ~%LOCALAPPDATA%/tree-sitter-language-pack-cache).
"""
from __future__ import annotations

from src.config import CHUNK_LINES, CHUNK_OVERLAP, MAX_CHUNK_CHARS

try:
    import _native as _ts  # tree-sitter-language-pack exposes itself as `_native`
    _AVAILABLE = True
except ImportError:
    _ts = None  # type: ignore
    _AVAILABLE = False

# Top-level chunks: even tiny ones are semantic units worth indexing.
_AST_MIN_CHARS = 10
# Class/interface children (methods): require a real size so that inline
# arrow functions inside React components don't pollute the index.
_CHILD_MIN_CHARS = 50
# How many lines to scan backwards for decorators (@Injectable, @Get, etc.)
# and immediately-preceding doc comments. Captures NestJS / Angular /
# Spring / Java attribute patterns the structure parser doesn't span.
_DECORATOR_LOOKBACK_LINES = 8

# Map our file extensions to tree-sitter language names.
# .jsx and .tsx use their dedicated grammars; .h is treated as C
# (most C++ headers parse fine under the C grammar for chunking purposes,
# and a misclassified header still falls back to line windows on parse fail).
_EXT_TO_LANG: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
}


def is_supported(extension: str) -> bool:
    return _AVAILABLE and extension in _EXT_TO_LANG


def _split_large(chunk: dict) -> list[dict]:
    """Sub-window large AST chunks so no single chunk exceeds MAX_CHUNK_CHARS.
    Mirrors the Python AST chunker's behavior; preserves the symbol label."""
    text = chunk["text"]
    if len(text) <= MAX_CHUNK_CHARS:
        return [chunk]

    lines = text.splitlines(keepends=True)
    out: list[dict] = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    base_start = chunk["start_line"]
    for i in range(0, len(lines), step):
        window = lines[i : i + CHUNK_LINES]
        chunk_text = "".join(window).strip()
        if len(chunk_text) < _AST_MIN_CHARS:
            continue
        out.append({
            "text": chunk_text,
            "start_line": base_start + i,
            "end_line": base_start + i + len(window) - 1,
            "symbol": chunk["symbol"],
        })
        if i + CHUNK_LINES >= len(lines):
            break
    return out


def _expand_to_decorators_and_comments(
    span_start_line: int, source_lines: list[str]
) -> int:
    """Walk backwards from `span_start_line` (1-indexed) through immediately-
    preceding decorators (`@Foo`) and doc comments. Returns the new effective
    start line. Decorators are followed even across blank lines (they're
    syntactic); comments are only followed without crossing a blank.
    """
    new_start = span_start_line
    crossed_blank = False
    max_back = min(_DECORATOR_LOOKBACK_LINES, span_start_line - 1)
    for i in range(max_back):
        candidate = span_start_line - 1 - i  # 1-indexed line number
        if candidate < 1:
            break
        line = source_lines[candidate - 1].strip()
        if not line:
            crossed_blank = True
            continue
        is_decorator = line.startswith("@")
        is_comment = (
            line.startswith("//") or line.startswith("/*") or line.startswith("*")
            or line.startswith("#") or line.startswith("--")
            or line.startswith("///")
        )
        if is_decorator:
            new_start = candidate
        elif is_comment and not crossed_blank:
            new_start = candidate
        else:
            break
    return new_start


def _line_to_byte(source_bytes: bytes, line_1_indexed: int) -> int:
    """Byte offset of the START of `line_1_indexed`. Line 1 → 0."""
    if line_1_indexed <= 1:
        return 0
    nl_count = 0
    for i, b in enumerate(source_bytes):
        if b == 0x0A:  # '\n'
            nl_count += 1
            if nl_count == line_1_indexed - 1:
                return i + 1
    return len(source_bytes)


def _structure_to_chunk(
    item,
    source_bytes: bytes,
    source_lines: list[str],
    parent_name: str | None,
) -> dict | None:
    """Convert a StructureItem into our chunk dict shape, or None if too small.
    Expands the chunk start backwards to capture decorators and immediately-
    preceding doc comments — important for NestJS / Angular / Spring code."""
    span = item.span
    # tree-sitter spans are 0-indexed; convert to 1-indexed throughout.
    raw_start_line = span.start_line + 1
    end_line = span.end_line + 1

    expanded_start = _expand_to_decorators_and_comments(raw_start_line, source_lines)
    if expanded_start < raw_start_line:
        start_byte = _line_to_byte(source_bytes, expanded_start)
    else:
        start_byte = span.start_byte

    text = source_bytes[start_byte : span.end_byte].decode("utf-8", errors="ignore").rstrip()
    if len(text) < _AST_MIN_CHARS:
        return None
    name = item.name or "<anonymous>"
    symbol = f"{parent_name}.{name}" if parent_name else name
    return {
        "text": text,
        "start_line": expanded_start,
        "end_line": end_line,
        "symbol": symbol,
    }


def _build_header_chunk(result, source_bytes: bytes) -> dict | None:
    """Imports-only header chunk, capped at the start of the first structure
    item so it never overlaps with downstream chunks. Without the cap, an
    `export interface` near the top of a TS file gets pulled into both the
    header AND its own chunk, doubling the tokens."""
    if not result.imports:
        return None
    end_byte = max(imp.span.end_byte for imp in result.imports)
    if result.structure:
        first_struct_byte = min(item.span.start_byte for item in result.structure)
        end_byte = min(end_byte, first_struct_byte)
    text = source_bytes[:end_byte].decode("utf-8", errors="ignore").rstrip()
    if len(text) < _AST_MIN_CHARS:
        return None
    end_line = source_bytes[:end_byte].count(b"\n") + 1
    return {
        "text": text,
        "start_line": 1,
        "end_line": end_line,
        "symbol": "<module header>",
    }


def chunk_with_tree_sitter(source: str, extension: str) -> list[dict] | None:
    """Parse `source` and return AST-aware chunks. Returns None on any failure
    so the caller can fall back to line-window chunking."""
    if not _AVAILABLE:
        return None
    lang = _EXT_TO_LANG.get(extension)
    if lang is None:
        return None

    try:
        cfg = _ts.ProcessConfig(
            language=lang,
            structure=True,
            imports=True,
            exports=True,
        )
        result = _ts.process(source, cfg)
    except Exception:
        # Any error (missing grammar, parse failure, etc.) → fall back.
        return None

    source_bytes = source.encode("utf-8")
    source_lines = source.splitlines(keepends=True)
    chunks: list[dict] = []

    header = _build_header_chunk(result, source_bytes)
    if header:
        chunks.extend(_split_large(header))

    # Walk the structure tree. Top-level items become chunks; classes /
    # interfaces / impls / namespaces also get their direct children
    # (methods) emitted as separate chunks for precise retrieval.
    for item in (result.structure or []):
        top_chunk = _structure_to_chunk(item, source_bytes, source_lines, parent_name=None)
        if top_chunk:
            chunks.extend(_split_large(top_chunk))
        # Only emit children that are meaningful named units. The library
        # surfaces inline arrow functions inside React components as
        # children too — those are noise, not retrievable units.
        for child in (item.children or []):
            if not child.name:
                continue
            child_chunk = _structure_to_chunk(
                child, source_bytes, source_lines, parent_name=item.name or None
            )
            if child_chunk and len(child_chunk["text"]) >= _CHILD_MIN_CHARS:
                chunks.extend(_split_large(child_chunk))

    return chunks if chunks else None
