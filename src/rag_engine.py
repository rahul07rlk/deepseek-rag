"""Hybrid retrieval with adaptive alpha, boosts, reranking, and file expansion.

Pipeline (all stages are independently toggleable via env):

1. Query analysis: detect symbols, file/path mentions, pick HYBRID_ALPHA
   (symbol-heavy queries lean on BM25; prose-heavy queries lean on vector).

2. Hybrid recall: FAISS (vector, query-embedding LRU cached) + BM25
   (keyword), fused with Reciprocal Rank Fusion using the chosen alpha.

3. Symbol & path boosts: candidates whose stored symbol matches an
   identifier in the query, or whose file path matches a path mention,
   get an additive RRF score boost — guarantees exact-symbol queries
   surface the right code even if the embedder didn't rank it.

4. Cross-encoder rerank (optional, RERANKER_ENABLED): top CANDIDATE_POOL
   pairs are rescored jointly. ~5-10× more accurate than bi-encoder.

5. File grouping: surviving chunks are grouped by source file.

6. Per-file tier selection:
     - whole-file (≥ WHOLE_FILE_THRESHOLD hits, ≤ WHOLE_FILE_MAX_CHARS)
     - neighbor-expanded merged ranges (read from disk with ±PAD lines)
     - raw chunks (fallback)

7. Token-budget enforcement and grounding-instructed system prompt.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import (
    BM25_CACHE,
    CANDIDATE_POOL,
    HYBRID_ALPHA,
    MIN_RELEVANCE_SCORE,
    MIN_RERANK_SCORE,
    NEIGHBOR_EXPANSION,
    NEIGHBOR_PAD_LINES,
    PATH_BOOST,
    QUERY_ANALYSIS_ENABLED,
    QUERY_EMBED_CACHE_SIZE,
    RERANKER_BATCH_SIZE,
    RERANKER_ENABLED,
    RERANKER_TOP_N,
    SYMBOL_BOOST,
    TOKEN_BUDGET,
    TOP_K_CHUNKS,
    WHOLE_FILE_MAX_CHARS,
    WHOLE_FILE_THRESHOLD,
)
from src.indexer import _tokenize_for_bm25, get_model, get_reranker, get_store
from src.query_analyzer import QueryAnalysis, analyze as analyze_query
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger("rag_engine", "proxy.log")

_bm25_state: dict | None = None
_bm25_mtime: float | None = None

# (file_path, mtime) -> list[str] of file lines (with line endings)
_file_cache: dict[tuple[str, float], list[str]] = {}
_FILE_CACHE_MAX = 256


def _load_bm25():
    """Hot-reload BM25 cache if the file on disk has been updated."""
    global _bm25_state, _bm25_mtime
    if not BM25_CACHE.exists():
        return None
    mtime = BM25_CACHE.stat().st_mtime
    if _bm25_state is None or _bm25_mtime != mtime:
        with open(BM25_CACHE, "rb") as f:
            _bm25_state = pickle.load(f)
        _bm25_mtime = mtime
        logger.debug(f"BM25 cache loaded ({len(_bm25_state['ids'])} chunks)")
    return _bm25_state


def _read_file_lines(file_path: str) -> list[str] | None:
    """Read a file as a list of lines, cached by (path, mtime). None on failure."""
    p = Path(file_path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    key = (file_path, mtime)
    cached = _file_cache.get(key)
    if cached is not None:
        return cached
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    lines = text.splitlines(keepends=True)
    if len(_file_cache) >= _FILE_CACHE_MAX:
        # Cheap LRU-ish eviction: drop the oldest insertion.
        _file_cache.pop(next(iter(_file_cache)))
    _file_cache[key] = lines
    return lines


# ── Vector search (with query-embedding LRU cache) ────────────────────────────
@lru_cache(maxsize=QUERY_EMBED_CACHE_SIZE)
def _embed_query(query: str) -> bytes:
    """LRU-cached query embedding. Returns raw bytes for hashability;
    caller wraps back into ndarray. Saves ~100-200ms on repeat queries
    (e.g. follow-up turns in the same conversation).

    VoyageEmbedder exposes embed_query() which uses input_type='query' —
    asymmetric encoding that measurably improves retrieval vs 'document'.
    Local SentenceTransformer falls back to the standard encode() path.
    """
    model = get_model()
    if hasattr(model, "embed_query"):
        emb = model.embed_query(query)
    else:
        emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(emb, dtype=np.float32).tobytes()


def _vector_search(query: str, n: int) -> list[tuple[str, float]]:
    store = get_store()
    emb = np.frombuffer(_embed_query(query), dtype=np.float32)
    return store.search(emb, n)


# ── BM25 search ───────────────────────────────────────────────────────────────
def _bm25_search(query: str, n: int) -> list[tuple[str, float]]:
    state = _load_bm25()
    if state is None:
        return []
    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return []
    scores = state["bm25"].get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    max_score = scores[ranked[0]] if ranked and scores[ranked[0]] > 0 else 1.0
    return [
        (state["ids"][i], float(scores[i]) / max_score)
        for i in ranked
        if scores[i] > 0
    ]


# ── Fusion ────────────────────────────────────────────────────────────────────
def _rrf_fuse(
    vector: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    alpha: float,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(vector):
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha * (1.0 / (k + rank))
    for rank, (doc_id, _) in enumerate(bm25):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 - alpha) * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Block builders ────────────────────────────────────────────────────────────
def _format_block(meta: dict, body: str, header_extra: str = "") -> str:
    symbol = meta.get("symbol", "")
    repo = meta.get("repo", "")
    parts = [
        f"### [{repo}] `{meta['file']}`",
        f"**Language:** {meta['language']} | "
        f"**Lines:** {meta['start_line']}-{meta['end_line']}"
        + (f" | **Symbol:** `{symbol}`" if symbol else "")
        + header_extra,
    ]
    header = "\n".join(parts) + "\n\n"
    return f"{header}```{meta['language']}\n{body}\n```"


def _apply_query_boosts(
    fused: list[tuple[str, float]],
    by_id: dict[str, tuple[str, dict]],
    analysis: QueryAnalysis,
) -> list[tuple[str, float]]:
    """Add additive boosts to fused scores for chunks whose stored symbol
    or file path matches identifiers/paths extracted from the query.

    A clean exact-symbol query like ``resolveHandle`` should never miss its
    target chunk just because the embedder ranked something else higher.
    """
    if not analysis.symbols and not analysis.paths:
        return fused
    sym_lower = {s.lower() for s in analysis.symbols}
    path_lower = [p.lower().replace("\\", "/") for p in analysis.paths]
    boosted: dict[str, float] = dict(fused)
    for doc_id, _score in fused:
        if doc_id not in by_id:
            continue
        _, meta = by_id[doc_id]
        # Symbol boost: exact case-insensitive match against any query symbol.
        # Stored symbols look like "ClassName.methodName" or "func_name".
        meta_symbol = (meta.get("symbol") or "").lower()
        if meta_symbol and sym_lower:
            symbol_parts = {meta_symbol, *meta_symbol.split("."), *meta_symbol.split()}
            if symbol_parts & sym_lower:
                boosted[doc_id] = boosted.get(doc_id, 0.0) + SYMBOL_BOOST
        # Path boost: query mentioned a filename or partial path that occurs
        # in the chunk's file path.
        meta_file = (meta.get("file") or "").lower().replace("\\", "/")
        if meta_file and path_lower:
            for p in path_lower:
                if p in meta_file:
                    boosted[doc_id] = boosted.get(doc_id, 0.0) + PATH_BOOST
                    break
    return sorted(boosted.items(), key=lambda x: x[1], reverse=True)


def _rerank_candidates(
    query: str,
    materialized: list[tuple[str, float, str, dict]],
) -> list[tuple[str, float, float, str, dict]] | None:
    """Score (query, doc) pairs jointly with the cross-encoder.

    Returns [(str_id, rerank_score, fused_score, doc, meta), ...] sorted by
    rerank_score desc, truncated to RERANKER_TOP_N and floored by
    MIN_RERANK_SCORE. Returns None if reranker unavailable so the caller
    can fall back to the fusion-only path.
    """
    reranker = get_reranker()
    if reranker is None or not materialized:
        return None
    pairs = [(query, doc) for _, _, doc, _ in materialized]
    raw = reranker.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
    )
    # Both backends return raw logits / log-odds; sigmoid normalizes to [0, 1]
    # so MIN_RERANK_SCORE and the per-block "Relevance:" header are calibrated.
    norm = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
    scored: list[tuple[str, float, float, str, dict]] = []
    for i, (doc_id, fused_score, doc, meta) in enumerate(materialized):
        s = float(norm[i])
        if s < MIN_RERANK_SCORE:
            continue
        scored.append((doc_id, s, fused_score, doc, meta))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:RERANKER_TOP_N]


def _merge_ranges(
    ranges: list[tuple[int, int]], pad: int, max_line: int
) -> list[tuple[int, int]]:
    """Pad each range, then merge overlapping/adjacent ones. Returns 1-indexed
    inclusive ranges clamped to [1, max_line]."""
    if not ranges:
        return []
    padded = sorted(
        (max(1, s - pad), min(max_line, e + pad)) for s, e in ranges
    )
    merged: list[tuple[int, int]] = [padded[0]]
    for s, e in padded[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# ── Main retrieval ────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = TOP_K_CHUNKS,
    alpha: float = HYBRID_ALPHA,
) -> tuple[str, int, list[dict]]:
    """Return (context_string, token_count, list_of_used_block_metas).

    A "block" is one of:
      - a whole file (when ≥ WHOLE_FILE_THRESHOLD chunks land in it)
      - a neighbor-expanded line range read from disk
      - a raw chunk (fallback when expansion is disabled or disk read fails)

    `top_k` caps the number of blocks, not raw chunks.
    """
    # ── Stage 1: query analysis (adaptive alpha + boost targets). ─────────────
    if QUERY_ANALYSIS_ENABLED:
        analysis = analyze_query(query)
        effective_alpha = analysis.alpha
    else:
        analysis = QueryAnalysis(raw_query=query, alpha=alpha)
        effective_alpha = alpha

    vec_hits = _vector_search(query, CANDIDATE_POOL)
    bm_hits = _bm25_search(query, CANDIDATE_POOL)
    vec_sim: dict[str, float] = dict(vec_hits)

    fused = _rrf_fuse(vec_hits, bm_hits, alpha=effective_alpha)
    if not fused:
        logger.warning("No hits from either retriever.")
        return "", 0, []

    # Pull docs+metas for the FULL fused list (typically ≤ 2×CANDIDATE_POOL ≈
    # 120 entries) so boosting can promote chunks that weren't in the initial
    # top CANDIDATE_POOL — e.g. a chunk at fused rank 80 with a perfect
    # symbol match can rise to top-1. Each lookup is O(1) dict access; cost
    # is negligible. Without this, boosts could only re-order existing top-60.
    all_ids = [doc_id for doc_id, _ in fused]
    by_id = get_store().get_by_str_ids(all_ids)

    # ── Stage 1.5: symbol & path boosts on fused scores. ──────────────────────
    if QUERY_ANALYSIS_ENABLED and (analysis.symbols or analysis.paths):
        fused = _apply_query_boosts(fused, by_id, analysis)

    # Now take the top CANDIDATE_POOL of the (possibly boost-reordered) list
    # and materialize for downstream stages.
    materialized: list[tuple[str, float, str, dict]] = []
    for doc_id, fused_score in fused[:CANDIDATE_POOL]:
        if doc_id not in by_id:
            continue
        doc, meta = by_id[doc_id]
        materialized.append((doc_id, fused_score, doc, meta))

    # ── Stage 2: cross-encoder rerank (or fall back to fusion-only). ──────────
    # ranked entries: (str_id, primary_score, secondary_score, doc, meta)
    #   primary   = used as the per-block "Relevance" floor + display
    #   secondary = used as "Fused" display value
    # With reranker on:  primary = rerank score (sigmoid),  secondary = fused
    # With reranker off: primary = vec_sim,                 secondary = fused
    rerank_mode = "off"
    ranked: list[tuple[str, float, float, str, dict]] = []
    if RERANKER_ENABLED:
        reranked = _rerank_candidates(query, materialized)
        if reranked is not None:
            ranked = reranked
            rerank_mode = "on"

    if rerank_mode == "off":
        for doc_id, fused_score, doc, meta in materialized:
            relevance = vec_sim.get(doc_id, 0.5)
            if relevance < MIN_RELEVANCE_SCORE:
                continue
            ranked.append((doc_id, relevance, fused_score, doc, meta))

    if not ranked:
        logger.warning("All candidates filtered out (reranker=%s).", rerank_mode)
        return "", 0, []

    # ── Group by file, preserving the chosen rank order. ──────────────────────
    # file_hits[file] = list of (secondary, primary, doc, meta), best first.
    file_hits: dict[str, list[tuple[float, float, str, dict]]] = {}
    file_order: list[str] = []
    for doc_id, primary, secondary, doc, meta in ranked:
        f = meta["file"]
        if f not in file_hits:
            file_hits[f] = []
            file_order.append(f)
        file_hits[f].append((secondary, primary, doc, meta))

    # ── Emit blocks per file in order. ────────────────────────────────────────
    context_parts: list[str] = []
    used_metas: list[dict] = []
    total_tokens = 0
    blocks_emitted = 0

    for f in file_order:
        if blocks_emitted >= top_k or total_tokens >= TOKEN_BUDGET:
            break

        hits = file_hits[f]
        top_score = max(h[0] for h in hits)
        best_relevance = max(h[1] for h in hits)
        sample_meta = hits[0][3]
        lines = _read_file_lines(f)

        # ── Tier 1: whole-file emit ───────────────────────────────────────────
        whole_file_ok = (
            WHOLE_FILE_THRESHOLD > 0
            and len(hits) >= WHOLE_FILE_THRESHOLD
            and lines is not None
        )
        if whole_file_ok:
            full_text = "".join(lines).rstrip()
            if len(full_text) <= WHOLE_FILE_MAX_CHARS:
                meta = {
                    **sample_meta,
                    "start_line": 1,
                    "end_line": len(lines),
                    "symbol": "<whole file>",
                }
                header_extra = (
                    f" | **Relevance:** {best_relevance:.2f}"
                    f" | **Fused:** {top_score:.3f}"
                    f" | **Mode:** whole-file ({len(hits)} hits)"
                )
                # Count tokens on the FORMATTED block (header + body) so the
                # budget tracker matches what's actually injected into the prompt.
                formatted = _format_block(meta, full_text, header_extra)
                tokens = count_tokens(formatted)
                if total_tokens + tokens > TOKEN_BUDGET:
                    break
                context_parts.append(formatted)
                used_metas.append({
                    **meta,
                    "relevance": best_relevance,
                    "fused": top_score,
                    "tokens": tokens,
                    "mode": "whole_file",
                    "hit_count": len(hits),
                })
                total_tokens += tokens
                blocks_emitted += 1
                continue
            # Too big for whole-file → fall through to chunk/neighbor mode.

        # ── Tier 2: neighbor-expanded ranges ──────────────────────────────────
        if NEIGHBOR_EXPANSION and lines is not None:
            # Map each retrieved chunk's line range, keeping its best fused score.
            ranges_with_score: dict[tuple[int, int], float] = {}
            for fused_score, _rel, _doc, meta in hits:
                key = (meta["start_line"], meta["end_line"])
                if fused_score > ranges_with_score.get(key, -1.0):
                    ranges_with_score[key] = fused_score
            merged = _merge_ranges(
                list(ranges_with_score.keys()),
                pad=NEIGHBOR_PAD_LINES,
                max_line=len(lines),
            )
            for s, e in merged:
                if blocks_emitted >= top_k or total_tokens >= TOKEN_BUDGET:
                    break
                body = "".join(lines[s - 1 : e]).rstrip()
                if not body:
                    continue
                meta = {
                    **sample_meta,
                    "start_line": s,
                    "end_line": e,
                    "symbol": sample_meta.get("symbol") or "<expanded>",
                }
                header_extra = (
                    f" | **Relevance:** {best_relevance:.2f}"
                    f" | **Fused:** {top_score:.3f}"
                    f" | **Mode:** expanded (+{NEIGHBOR_PAD_LINES} lines)"
                )
                formatted = _format_block(meta, body, header_extra)
                tokens = count_tokens(formatted)
                if total_tokens + tokens > TOKEN_BUDGET:
                    break
                context_parts.append(formatted)
                used_metas.append({
                    **meta,
                    "relevance": best_relevance,
                    "fused": top_score,
                    "tokens": tokens,
                    "mode": "expanded",
                })
                total_tokens += tokens
                blocks_emitted += 1
            continue

        # ── Tier 3: raw chunks (fallback) ─────────────────────────────────────
        for fused_score, relevance, doc, meta in hits:
            if blocks_emitted >= top_k or total_tokens >= TOKEN_BUDGET:
                break
            header_extra = (
                f" | **Relevance:** {relevance:.2f}"
                f" | **Fused:** {fused_score:.3f}"
            )
            formatted = _format_block(meta, doc, header_extra)
            tokens = count_tokens(formatted)
            if total_tokens + tokens > TOKEN_BUDGET:
                break
            context_parts.append(formatted)
            used_metas.append({
                **meta,
                "relevance": relevance,
                "fused": fused_score,
                "tokens": tokens,
                "mode": "chunk",
            })
            total_tokens += tokens
            blocks_emitted += 1

    context_str = "\n\n---\n\n".join(context_parts)
    actual_tokens = count_tokens(context_str)

    files_cited = sorted({m["filename"] for m in used_metas})
    mode_counts: dict[str, int] = {}
    for m in used_metas:
        mode_counts[m["mode"]] = mode_counts.get(m["mode"], 0) + 1
    note_suffix = f" ({analysis.note})" if analysis.note else ""
    logger.info(
        f"Retrieved {len(used_metas)} blocks ({mode_counts}) | "
        f"reranker={rerank_mode} | alpha={effective_alpha:.2f}{note_suffix} | "
        f"~{actual_tokens} tokens | Files: {files_cited}"
    )
    return context_str, actual_tokens, used_metas


def build_messages(
    user_query: str,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    context_str, token_count, metas = retrieve(user_query)

    files_cited = sorted({m["file"] for m in metas})
    files_summary = "\n".join(f"  - {f}" for f in files_cited) or "  (none)"

    system_message = f"""You are an expert software engineer with full semantic access to the codebase.

## Retrieved Codebase Context
The following code was retrieved as the most relevant to the current query
via hybrid BM25 + vector search, then expanded with neighboring lines and
whole-file pulls where multiple chunks landed in the same file. These are
REAL excerpts from the actual repository - treat them as ground truth.

{context_str}

## Files Referenced Above
{files_summary}

## Your Instructions
- Always cite exact file paths and line numbers when referencing code.
- If suggesting edits, show precise diffs or full replacement blocks.
- If the retrieved context does not contain enough information to answer,
  say so explicitly - do NOT hallucinate code that isn't shown above.
- Think step by step before producing any code changes.
- Injected context: {token_count} tokens across {len(metas)} blocks."""

    messages: list[dict] = [{"role": "system", "content": system_message}]
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") in ("user", "assistant"):
                messages.append(msg)
    messages.append({"role": "user", "content": user_query})
    return messages
