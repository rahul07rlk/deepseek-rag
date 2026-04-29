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


# ── AST-based reference extraction ────────────────────────────────────────────
# Tree-sitter naturally splits identifier-like tokens by role:
#   identifier               — variables, functions, args (most languages)
#   property_identifier      — `.foo` in member access (TS/JS)
#   type_identifier          — class/type names (TS/Java/Rust/Go)
#   field_identifier         — struct field names (Go/Rust/C)
#   shorthand_property_id…   — `{foo}` in object literals (TS/JS)
# Comments and string literals are atomic — tree-sitter does NOT emit child
# identifier nodes inside them, so identifier-set extraction implicitly
# excludes references inside comments/strings (which is the #1 source of
# false-positive callers in the regex extractor it replaces).
# Extension → tree-sitter language for the references pass. Different from
# _EXT_TO_LANG (which controls chunking — Python chunks via stdlib ast, not
# tree-sitter). The references pass uses tree-sitter for ALL supported
# languages including Python, since it gives us comment/string-aware extraction.
_REF_EXT_TO_LANG: dict[str, str] = {
    **_EXT_TO_LANG,  # TS/JS/Go/Rust/Java/C/C++/C#/PHP/Ruby/Swift/Kotlin
    ".py": "python",
}

_REF_KINDS_BY_LANG: dict[str, tuple[str, ...]] = {
    "typescript": (
        "identifier", "property_identifier", "type_identifier",
        "shorthand_property_identifier",
    ),
    "tsx": (
        "identifier", "property_identifier", "type_identifier",
        "shorthand_property_identifier",
    ),
    "javascript": (
        "identifier", "property_identifier",
        "shorthand_property_identifier",
    ),
    "python": ("identifier",),
    "go": ("identifier", "field_identifier", "type_identifier"),
    "rust": ("identifier", "field_identifier", "type_identifier"),
    "java": ("identifier", "type_identifier"),
    "c": ("identifier", "field_identifier", "type_identifier"),
    "cpp": ("identifier", "field_identifier", "type_identifier"),
    "csharp": ("identifier",),
    "ruby": ("identifier", "constant"),
    "php": ("name",),
    "swift": ("simple_identifier",),
    "kotlin": ("simple_identifier",),
}

# Tokens that are syntactically identifiers but semantically noise. Keeps the
# refs table from being spammed by language built-ins on every line.
_REF_STOPWORDS = frozenset({
    "self", "this", "super", "cls", "None", "True", "False", "null",
    "true", "false", "undefined", "void", "nil", "new", "delete",
})

# Minimum identifier length. Single-char vars (i, j, x, y, n, e, f) are too
# common to be useful for caller lookups; 2-char names like ``id``, ``db``,
# ``os``, ``vm``, ``rx`` ARE distinctive enough so we keep them.
_REF_MIN_LEN = 2


def extract_references_with_tree_sitter(
    source: str,
    extension: str,
) -> list[tuple[str, int]] | None:
    """Return ``[(symbol, line_1_indexed), ...]`` of every meaningful identifier
    reference in ``source``. None on any failure (caller falls back).

    "Meaningful" excludes:
      - tokens inside comments / string literals (tree-sitter doesn't emit
        identifier nodes inside those — implicit filtering)
      - reserved words / built-ins (parsed as keywords, not identifiers)
      - tokens shorter than ``_REF_MIN_LEN``
      - language built-ins like ``self``, ``this``, ``super`` (semantic noise)

    The caller (symbol_graph) further filters references against the
    definition table to surface true cross-file callers.
    """
    if not _AVAILABLE:
        return None
    lang = _REF_EXT_TO_LANG.get(extension)
    if lang is None:
        return None
    ref_kinds = _REF_KINDS_BY_LANG.get(lang)
    if not ref_kinds:
        return None
    try:
        source_bytes = source.encode("utf-8")
        tree = _ts.parse_string(lang, source_bytes)
    except Exception:
        return None

    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for kind in ref_kinds:
        try:
            nodes = _ts.find_nodes_by_type(tree, kind)
        except Exception:
            continue
        for node in nodes:
            try:
                text = source_bytes[node.start_byte : node.end_byte].decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                continue
            if len(text) < _REF_MIN_LEN:
                continue
            if text in _REF_STOPWORDS:
                continue
            line = node.start_row + 1  # 0-indexed → 1-indexed
            key = (text, line)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out
