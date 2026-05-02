"""Query analysis: extract retrieval signals before searching.

For code retrieval, the optimal hybrid weighting depends on the query type:

  - Symbol queries  ("where is resolveHandle defined")     → favor BM25
  - Semantic queries ("how does the cache eviction work") → favor vector
  - Path queries    ("show me Auth.tsx")                  → boost matching files

This module returns a structured analysis used by rag_engine.retrieve() to:

  1. Set HYBRID_ALPHA dynamically per query (symbol-heavy → low alpha, prose-heavy → high alpha)
  2. Apply additive RRF score boosts to chunks whose stored symbol or file
     path matches identifiers / paths extracted from the query
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Identifier-like tokens (CamelCase, snake_case, UPPER_CONST, plain_word)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

# File-path / filename mentions: foo.ts, src/utils/bar.py, etc.
_PATH_RE = re.compile(r"[\w./\\-]+\.[A-Za-z]{1,5}\b")

# CamelCase or camelCase detection (i.e., not plain lowercase or UPPERCASE)
_MIXED_CASE_RE = re.compile(r"[A-Z][a-z]+[A-Z]|[a-z]+[A-Z][a-z]")

# Well-known files that have no extension but are common query targets.
# Detected as paths even though _PATH_RE requires an extension suffix.
_WELL_KNOWN_FILES = frozenset({
    "readme", "makefile", "dockerfile", "procfile", "gemfile",
    "cargo", "brewfile", "jenkinsfile", "vagrantfile", "pipfile",
    "changelog", "contributing", "license", "authors", "codeowners",
})

# Stop-words we should never treat as code identifiers.
_PROSE_WORDS = frozenset({
    "how", "why", "what", "when", "where", "which", "who", "whose",
    "does", "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "and", "or", "but", "not", "no", "yes",
    "do", "did", "done", "doing", "have", "has", "had",
    "should", "would", "could", "can", "will", "may", "might", "must",
    "this", "that", "these", "those", "such",
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they", "their",
    "find", "show", "tell", "explain", "describe", "list", "give", "make",
    "fix", "add", "remove", "change", "update", "create", "delete", "implement",
    "bug", "error", "issue", "problem", "feature", "function",
    "file", "code", "line", "lines",
    "from", "into", "out", "up", "down", "over", "under",
    "use", "uses", "used", "using",
    "work", "works", "working", "worked",
    "if", "then", "else", "while", "until",
    "between", "across", "through", "during", "before", "after",
    "really", "actually", "currently", "previously",
    "module", "modules", "package", "packages",
    "class", "classes", "method", "methods", "type", "types",
    "import", "imports", "export", "exports",
})


@dataclass
class QueryAnalysis:
    raw_query: str
    symbols: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    alpha: float = 0.5
    is_symbol_dominant: bool = False
    note: str = ""  # human-readable summary for logs


def _looks_like_symbol(tok: str) -> bool:
    """A token is symbol-like if it has CamelCase, snake_case, or UPPER_CONST shape."""
    if len(tok) < 3:
        return False
    if tok.lower() in _PROSE_WORDS:
        return False
    # snake_case (with underscore, not just leading)
    if "_" in tok.strip("_"):
        return True
    # ALL_UPPER constants of length ≥3
    if tok.isupper():
        return True
    # CamelCase / camelCase
    if _MIXED_CASE_RE.search(tok):
        return True
    return False


def analyze(query: str) -> QueryAnalysis:
    """Return retrieval signals for a query."""
    q = (query or "").strip()
    if not q:
        return QueryAnalysis(raw_query=query, alpha=0.5, note="empty")

    # File / path mentions first (they consume tokens that look like idents).
    paths = list({m for m in _PATH_RE.findall(q) if not m.startswith(".")})
    # Also catch well-known extensionless filenames (README, Dockerfile, etc.)
    # that _PATH_RE misses because they have no dot-extension.
    q_words = set(re.split(r"\W+", q.lower()))
    for wkf in _WELL_KNOWN_FILES:
        if wkf in q_words and wkf not in paths:
            paths.append(wkf)

    # Strip path mentions from the query before identifier extraction so we
    # don't double-count `Auth.tsx` as both a path and the symbol `Auth`.
    q_for_idents = q
    for p in paths:
        q_for_idents = q_for_idents.replace(p, " ")

    idents = _IDENT_RE.findall(q_for_idents)
    symbols: list[str] = []
    prose_count = 0
    for tok in idents:
        if _looks_like_symbol(tok):
            symbols.append(tok)
        elif tok.lower() in _PROSE_WORDS:
            prose_count += 1
        else:
            # Plain lowercase word that isn't a common stop-word: weight as half.
            prose_count += 0.5

    symbols = list(dict.fromkeys(symbols))  # preserve order, dedupe
    code_count = len(symbols)
    is_symbol_dominant = code_count > 0 and code_count >= prose_count * 0.7

    # Pick alpha. RRF uses alpha as the *vector* weight; (1-alpha) is BM25.
    if code_count > 0 and prose_count <= 1:
        # Pure symbol query — almost no prose. Lean hard on BM25.
        alpha = 0.25
        note = f"symbol-heavy ({code_count} sym, {prose_count:.1f} prose)"
    elif is_symbol_dominant:
        # Symbol embedded in a sentence ("where is resolveHandle defined")
        alpha = 0.4
        note = f"symbol-in-prose ({code_count} sym, {prose_count:.1f} prose)"
    elif code_count == 0:
        # Pure prose. Vector recall handles paraphrase well.
        alpha = 0.7
        note = f"prose-only ({prose_count:.1f} prose)"
    else:
        alpha = 0.5
        note = f"mixed ({code_count} sym, {prose_count:.1f} prose)"

    return QueryAnalysis(
        raw_query=query,
        symbols=symbols,
        paths=paths,
        alpha=alpha,
        is_symbol_dominant=is_symbol_dominant,
        note=note,
    )
