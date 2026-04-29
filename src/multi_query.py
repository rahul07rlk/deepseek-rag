"""Rule-based multi-query expansion.

Generate N paraphrases of the user query that emphasize different facets:

  Q  : "how does the cache eviction work"
  Q1 : "cache eviction implementation"        ← noun-phrase, code-y
  Q2 : "evict cache function"                 ← imperative + plain noun
  Q3 : "LRU cache TTL expiration"             ← keyword-only

Run hybrid retrieval against each variant and RRF-fuse the resulting
ranked lists. Each variant catches chunks the others miss because they
each emphasize different vocabulary slices.

Why rule-based and not LLM
--------------------------
LLM-based query expansion (RAG-Fusion) gives ~1-2% better recall but
costs another API round-trip (~500ms). HyDE already covers the prose-
to-code vocabulary gap. We run rule-based variants here so multi-query
is essentially free — no extra latency, no extra spend.

Each variant is built from primitives that already exist in the query:
extracted symbols, content words, prefixes/suffixes — so the variants
stay grounded in what the user actually asked.
"""
from __future__ import annotations

import re

from src.query_analyzer import QueryAnalysis


# Verbs we strip from prose when building noun-phrase variants.
_STRIP_VERBS = {
    "how", "does", "do", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "for", "with", "in",
    "on", "at", "and", "or", "what", "where", "when",
    "why", "who", "this", "that", "these", "those",
    "show", "tell", "explain", "describe", "find",
    "me", "us", "you", "i", "we",
    "really", "actually",
    "?", "!",
}

# Common code-y suffixes a question might map to.
_SUFFIX_HINTS = ("implementation", "function", "logic", "definition", "handler")


def _content_words(query: str) -> list[str]:
    """Strip filler words and punctuation; preserve identifier-like tokens."""
    out: list[str] = []
    for raw in re.split(r"[\s,;:?!]+", query.strip()):
        if not raw:
            continue
        tok = raw.strip("()[]{}<>\"'`").lower()
        if not tok or tok in _STRIP_VERBS:
            continue
        out.append(tok)
    # Dedupe, preserve order.
    return list(dict.fromkeys(out))


def expand(query: str, analysis: QueryAnalysis, variants: int = 3) -> list[str]:
    """Return up to ``variants`` paraphrases (always includes the original
    as the first entry). Variants are rule-based and grounded in the query
    so we never invent unrelated terms.

    The original query is index 0 — callers that want exactly N *new*
    variants should request ``variants + 1``.
    """
    q = (query or "").strip()
    if not q:
        return [q]

    out: list[str] = [q]
    if variants <= 1:
        return out

    words = _content_words(q)
    symbols = analysis.symbols or []

    # ── Variant 1: noun-phrase form ("X implementation") ──────────────────────
    if words:
        topic = " ".join(w for w in words if not w.isdigit())[:120]
        if topic and topic.lower() != q.lower():
            out.append(f"{topic} implementation")

    # ── Variant 2: symbol-only ("resolveHandle login flow") ───────────────────
    if symbols:
        # Pair symbols with the most informative content word for context.
        ctx = next(
            (w for w in words if w not in (s.lower() for s in symbols) and len(w) > 3),
            "",
        )
        sym_str = " ".join(symbols[:3])
        candidate = f"{sym_str} {ctx}".strip()
        if candidate and candidate.lower() != q.lower():
            out.append(candidate)

    # ── Variant 3: keyword-only ("cache evict TTL") ───────────────────────────
    if words and len(words) >= 2:
        kw = " ".join(words[:5])
        if kw and kw.lower() != q.lower() and kw not in out:
            out.append(kw)

    # ── Variant 4 (fallback): suffix-augmented prose ──────────────────────────
    if len(out) < variants and words:
        suffix = _SUFFIX_HINTS[len(out) % len(_SUFFIX_HINTS)]
        cand = f"{' '.join(words[:6])} {suffix}"
        if cand not in out:
            out.append(cand)

    # Truncate, dedupe (case-insensitive).
    seen: set[str] = set()
    deduped: list[str] = []
    for v in out:
        key = v.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped[:variants]
