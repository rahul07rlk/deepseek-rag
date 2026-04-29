"""Repo-map pre-pass (Aider-style structural summary).

Builds a compact "table of contents" of every indexed file: relative path,
language, and the names of its top-level definitions (classes / functions /
interfaces / structs). The map is persisted to ``codebase_index/repo_map.json``
and rebuilt whenever the chunk store changes.

At query time, ``relevant_repo_map(query, ...)`` ranks files using BM25 over
their {path tokens, symbol names} and returns the top-N as a markdown block
that ``rag_engine.build_messages`` injects *before* the chunk-level context.

Why this exists
---------------
Pure chunk retrieval struggles on broad, "where do I look" queries —
symbol/path matches are diluted by many tiny chunks. A repo map gives the
LLM a global view of the codebase first, so it can reason about *which*
files to focus on before reading the (still chunk-level) context.

The map is built from the existing chunk index — no new file walking, no new
parsing pass — so it costs ~milliseconds even for thousands of chunks.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import (
    REPO_MAP_PATH,
    REPO_MAP_TOKEN_BUDGET,
    REPO_MAP_TOP_FILES,
    REPO_PATHS,
)
from src.indexer import _tokenize_for_bm25
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger("repo_map", "indexer.log")

# ── In-memory cache (reloaded when the JSON mtime changes) ────────────────────
_cache: dict | None = None
_cache_mtime: float | None = None


def _relpath(file_str: str) -> str:
    """Convert an absolute indexed file path to a repo-relative form for display."""
    p = Path(file_str)
    for rp in REPO_PATHS:
        try:
            return f"{rp.name}/" + str(p.relative_to(rp)).replace("\\", "/")
        except ValueError:
            continue
    return str(p).replace("\\", "/")


def build_repo_map() -> dict:
    """Walk the chunk store and emit one entry per file with its symbols."""
    from src.indexer import get_store  # avoid circular import at module load

    store = get_store()
    if store.count() == 0:
        logger.info("Repo-map: nothing to summarize (empty store).")
        REPO_MAP_PATH.write_text(json.dumps({"files": [], "version": 2}), encoding="utf-8")
        return {"files": [], "version": 2}

    _ids, _docs, metas = store.get_all()
    by_file: dict[str, dict] = {}

    for meta in metas:
        f = meta["file"]
        entry = by_file.setdefault(f, {
            "file": f,
            "rel": _relpath(f),
            "filename": meta.get("filename", ""),
            "language": meta.get("language", ""),
            "repo": meta.get("repo", ""),
            "symbols": [],
            "_seen": set(),
            "max_end_line": 0,
        })
        sym = (meta.get("symbol") or "").strip()
        if sym and sym not in entry["_seen"] and not sym.startswith("<"):
            entry["_seen"].add(sym)
            entry["symbols"].append(sym)
        end_line = int(meta.get("end_line", 0) or 0)
        if end_line > entry["max_end_line"]:
            entry["max_end_line"] = end_line

    files: list[dict] = []
    for entry in by_file.values():
        del entry["_seen"]
        # Cap to the most informative symbols — anything past 30 is noise for
        # the LLM's spatial reasoning. Order is insertion order which
        # approximates source order via chunk emission order.
        entry["symbols"] = entry["symbols"][:30]
        entry["lines"] = entry["max_end_line"]
        del entry["max_end_line"]
        files.append(entry)

    files.sort(key=lambda x: x["rel"])
    payload = {"files": files, "version": 2}
    REPO_MAP_PATH.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Repo-map built ({len(files)} files) -> {REPO_MAP_PATH.name}")
    return payload


def _load() -> dict | None:
    """Hot-reload the persisted repo map on mtime change."""
    global _cache, _cache_mtime
    if not REPO_MAP_PATH.exists():
        return None
    mtime = REPO_MAP_PATH.stat().st_mtime
    if _cache is None or _cache_mtime != mtime:
        try:
            _cache = json.loads(REPO_MAP_PATH.read_text(encoding="utf-8"))
            _cache_mtime = mtime
        except Exception as e:
            logger.warning(f"Failed to load repo-map: {e}")
            return None
    return _cache


def _file_tokens(entry: dict) -> list[str]:
    """Tokens used for BM25 over (path + symbols + filename + language)."""
    path_tokens = _tokenize_for_bm25(entry["rel"])
    sym_tokens: list[str] = []
    for s in entry["symbols"]:
        sym_tokens.extend(_tokenize_for_bm25(s))
    name_tokens = _tokenize_for_bm25(entry["filename"])
    lang_tokens = _tokenize_for_bm25(entry.get("language", ""))
    return path_tokens + sym_tokens + name_tokens + lang_tokens


def relevant_repo_map(
    query: str,
    top_files: int = REPO_MAP_TOP_FILES,
    token_budget: int = REPO_MAP_TOKEN_BUDGET,
) -> tuple[str, list[dict]]:
    """Return (markdown_block, ranked_entries) for injection into the prompt.

    Ranks files by BM25 over their path + symbol vocabulary. If the query has
    no usable tokens (rare — e.g. all stopwords) we fall back to a stable
    alphabetical slice so the model still sees *some* structural overview.
    """
    payload = _load()
    if not payload or not payload.get("files"):
        return "", []

    files: list[dict] = payload["files"]
    tokens = _tokenize_for_bm25(query or "")
    if not tokens:
        ranked = files[:top_files]
    else:
        corpus = [_file_tokens(e) for e in files]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(files)), key=lambda i: scores[i], reverse=True)
        ranked = [files[i] for i in order[:top_files] if scores[i] > 0]
        # Always include at least 5 entries — the spatial map is useful even
        # when BM25 finds nothing. Falls back to the top-of-alphabet slice.
        if len(ranked) < 5:
            picked = {e["file"] for e in ranked}
            for e in files:
                if e["file"] not in picked:
                    ranked.append(e)
                    if len(ranked) >= 5:
                        break

    # ── Render: group by repo, list files with their symbols. ────────────────
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in ranked:
        by_repo[e.get("repo", "?")].append(e)

    lines: list[str] = []
    for repo in sorted(by_repo.keys()):
        lines.append(f"### {repo}")
        for e in by_repo[repo]:
            sym_str = ", ".join(e["symbols"][:12]) if e["symbols"] else "(no top-level symbols)"
            lines.append(f"- `{e['rel']}` ({e['language']}, {e['lines']} lines) — {sym_str}")
        lines.append("")

    # Trim to token budget greedily from the bottom.
    rendered = "\n".join(lines).rstrip()
    while count_tokens(rendered) > token_budget and lines:
        lines.pop()
        rendered = "\n".join(lines).rstrip()

    return rendered, ranked


if __name__ == "__main__":
    build_repo_map()
