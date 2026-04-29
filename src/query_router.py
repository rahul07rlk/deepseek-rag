"""Query routing: classify intent, pick a retrieval strategy.

Pure rule-based — no LLM call. Reuses ``query_analyzer`` for symbol/path
detection and adds intent heuristics on top.

Why this exists
---------------
The current pipeline runs the same retrieval flow for every query:
hybrid recall → rerank → neighbor expand → seed agentic loop. That's
overkill for "what does this repo do" (overview), wasteful for "find
resolveHandle" (single symbol), and slow for "show me Auth.tsx" (file
lookup). The log shows "content of this repo?" burning 14 tool turns
and ~30k tokens before answering — when the same answer lives in the
README and the repo map.

Routes
------
- ``OVERVIEW``      → repo_map + READMEs only, skip chunk retrieval
- ``SYMBOL_LOOKUP`` → symbol_graph first, BM25 second, no rerank needed
- ``EXACT_STRING``  → grep first, no embedding
- ``FILE_LOOKUP``   → repo_map filter for the named file, then read it
- ``HOW_X_WORKS``   → HyDE + hybrid + rerank + neighbor expand (full)
- ``DEBUG``         → grep error string + retrieve + read full files
- ``DEFAULT``       → existing pipeline (fallback when no rule fires)

Each route returns a ``RouteDecision`` telling the rest of the pipeline:
how much to seed, whether to use HyDE, the recommended top_k, and the
max tool turns budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.query_analyzer import QueryAnalysis, analyze as analyze_query


# ── Phrase tables (lowercase, substring match) ────────────────────────────────
OVERVIEW_PHRASES: tuple[str, ...] = (
    "what is this", "what's this", "whats this",
    "what does this repo", "what does this project",
    "summarize", "summarise", "summary of",
    "overview of", "describe the project", "describe this repo",
    "describe the codebase", "describe the code",
    "content of this", "contents of this",
    "what's in this repo", "whats in this repo", "what is in this repo",
    "explain the project", "explain this repo", "explain this codebase",
    "tell me about this", "tldr", "high level",
)

DEBUG_PHRASES: tuple[str, ...] = (
    "error", "failing", "doesn't work", "doesnt work", "not working",
    "bug", "broken", "exception", "crash", "stack trace", "stacktrace",
    "regression", "fix the", "why is", "why does", "why doesn't",
    "throws", "throwing", "raise", "raises",
)

HOW_PHRASES: tuple[str, ...] = (
    "how does", "how is", "how do", "how to", "how can", "how would",
    "where does", "what happens when", "walkthrough",
    "explain how", "walk me through",
)


@dataclass
class RouteDecision:
    """Plan a query before retrieval."""

    route: str
    analysis: QueryAnalysis
    use_hyde: bool = False
    use_multi_query: bool = False
    # "both"           → repo_map + retrieved chunks (legacy default)
    # "map_only"       → repo_map only (overview / file lookup)
    # "retrieval_only" → chunks only (symbol lookup)
    # "minimal"        → tiny stub, agent drives the rest (exact string)
    seed_strategy: str = "both"
    max_tool_turns: int = 16
    suggested_top_k: int = 30
    # ~tokens to budget for seed retrieval — lower for narrow routes.
    seed_token_budget: int = 30000
    note: str = ""
    # Files explicitly named in the query (extracted from path mentions).
    target_files: list[str] = field(default_factory=list)


def _word_count(q: str) -> int:
    return len([w for w in q.split() if w.strip()])


def classify(query: str) -> RouteDecision:
    """Pick a retrieval strategy for ``query``. Pure function; no I/O."""
    q_raw = (query or "").strip()
    q = q_raw.lower()
    analysis = analyze_query(q_raw)
    sym_count = len(analysis.symbols)
    path_count = len(analysis.paths)
    word_count = _word_count(q)

    # ── OVERVIEW: short, prose-only, with overview vocabulary. ────────────────
    # The biggest waste in the log: "content of this repo?" got the full
    # 14-turn agentic treatment when a 2k-token repo_map + READMEs would
    # answer it directly.
    if word_count <= 12 and any(p in q for p in OVERVIEW_PHRASES):
        return RouteDecision(
            route="OVERVIEW",
            analysis=analysis,
            use_hyde=False,
            use_multi_query=False,
            seed_strategy="map_only",
            max_tool_turns=4,
            suggested_top_k=8,
            seed_token_budget=6000,
            note="overview keyword + short query",
        )

    # ── EXACT_STRING: query contains quoted text. ─────────────────────────────
    # Grep is the right tool. Don't waste embedding budget.
    if '"' in q_raw or "'" in q_raw:
        return RouteDecision(
            route="EXACT_STRING",
            analysis=analysis,
            use_hyde=False,
            use_multi_query=False,
            seed_strategy="minimal",
            max_tool_turns=6,
            suggested_top_k=12,
            seed_token_budget=6000,
            note="quoted string",
        )

    # ── FILE_LOOKUP: file/path mentioned, no question semantics. ──────────────
    if (
        path_count >= 1
        and word_count <= 10
        and "?" not in q
        and not any(p in q for p in HOW_PHRASES)
        and not any(p in q for p in DEBUG_PHRASES)
    ):
        return RouteDecision(
            route="FILE_LOOKUP",
            analysis=analysis,
            use_hyde=False,
            use_multi_query=False,
            seed_strategy="map_only",
            max_tool_turns=6,
            suggested_top_k=10,
            seed_token_budget=8000,
            note="file mentioned, no how/why/error",
            target_files=list(analysis.paths),
        )

    # ── SYMBOL_LOOKUP: 1-3 idents, very short, not how/why. ───────────────────
    if (
        sym_count >= 1
        and word_count <= 6
        and not any(p in q for p in HOW_PHRASES)
        and not any(p in q for p in DEBUG_PHRASES)
    ):
        return RouteDecision(
            route="SYMBOL_LOOKUP",
            analysis=analysis,
            use_hyde=False,
            use_multi_query=False,
            seed_strategy="retrieval_only",
            max_tool_turns=8,
            suggested_top_k=15,
            seed_token_budget=12000,
            note="symbol-dominant short query",
        )

    # ── DEBUG: error / bug / failure terminology. ─────────────────────────────
    if any(p in q for p in DEBUG_PHRASES):
        return RouteDecision(
            route="DEBUG",
            analysis=analysis,
            use_hyde=True,
            use_multi_query=True,
            seed_strategy="both",
            max_tool_turns=20,
            suggested_top_k=30,
            seed_token_budget=35000,
            note="debug query",
        )

    # ── HOW_X_WORKS: explanatory / paraphrase-friendly prose. ─────────────────
    if any(p in q for p in HOW_PHRASES) or sym_count == 0:
        return RouteDecision(
            route="HOW_X_WORKS",
            analysis=analysis,
            use_hyde=True,
            use_multi_query=True,
            seed_strategy="both",
            max_tool_turns=16,
            suggested_top_k=30,
            seed_token_budget=30000,
            note="explanatory prose query",
        )

    # ── DEFAULT: fall back to current behavior. ───────────────────────────────
    return RouteDecision(
        route="DEFAULT",
        analysis=analysis,
        use_hyde=False,
        use_multi_query=False,
        seed_strategy="both",
        max_tool_turns=16,
        suggested_top_k=30,
        seed_token_budget=30000,
        note="default",
    )
