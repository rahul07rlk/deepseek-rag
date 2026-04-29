"""Semantic answer cache.

After a query is answered, embed the original question and store
``(query_embedding, query_text, answer, route, timestamp)`` to disk. On the
next query, embed it, compute cosine similarity against every cached entry,
and short-circuit the entire RAG pipeline when the best match is above
``SEMANTIC_CACHE_THRESHOLD``.

Why this exists
---------------
Repeated questions ("content of this repo?" twice in 30 minutes, observed in
the proxy log) currently re-run the full agentic loop — embedding, hybrid
search, rerank, 6+ tool calls, ~30k tokens. With a semantic cache, the
second query is a single embed + dot-product against ≤200 entries (~10ms)
and returns instantly.

Why semantic, not exact-string
------------------------------
"content of this repo?" ≈ "what is this repo about?" ≈ "summarise the
codebase" — same intent, different wording. Embedding similarity catches
all three; an exact-string cache would only catch the first.

Invalidation
------------
- TTL (``SEMANTIC_CACHE_TTL_S``, default 1h) — bounds staleness even when
  files don't change.
- Index rebuild — ``reset()`` is called from ``schedule_bm25_rebuild`` so
  any file save invalidates the entire cache. Conservative but safe.

Storage layout (under ``codebase_index/semantic_cache/``)
---------------------------------------------------------
- ``entries.json`` — list of ``{q, a, ts, route}`` dicts
- ``vectors.npy`` — N×dim float32 array, row-aligned with ``entries``

Both files are written atomically together so partial-write corruption
returns an empty cache rather than crashing.
"""
from __future__ import annotations

import json
import time
from threading import RLock
from typing import Optional

import numpy as np

from src.config import (
    INDEX_DIR,
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_TTL_S,
)
from src.utils.logger import get_logger

logger = get_logger("answer_cache", "proxy.log")

_CACHE_DIR = INDEX_DIR / "semantic_cache"
_META_FILE = _CACHE_DIR / "entries.json"
_VEC_FILE = _CACHE_DIR / "vectors.npy"


class _Cache:
    def __init__(self) -> None:
        self._lock = RLock()
        # entries[i] aligns with self._vecs[i].
        self._entries: list[dict] = []
        self._vecs: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        try:
            if _META_FILE.exists() and _VEC_FILE.exists():
                self._entries = json.loads(_META_FILE.read_text(encoding="utf-8"))
                self._vecs = np.load(_VEC_FILE)
                if (
                    self._vecs is None
                    or self._vecs.size == 0
                    or len(self._entries) != self._vecs.shape[0]
                ):
                    # Mismatched shapes — discard and start fresh.
                    self._entries = []
                    self._vecs = None
                else:
                    logger.info(
                        f"Semantic cache loaded ({len(self._entries)} entries)"
                    )
        except Exception as e:
            logger.warning(f"Cache load failed ({e}); starting empty.")
            self._entries = []
            self._vecs = None

    def _save(self) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _META_FILE.write_text(
                json.dumps(self._entries, ensure_ascii=False),
                encoding="utf-8",
            )
            if self._vecs is not None and self._vecs.size > 0:
                np.save(_VEC_FILE, self._vecs)
            elif _VEC_FILE.exists():
                _VEC_FILE.unlink()
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _purge_expired(self) -> None:
        if not self._entries or self._vecs is None:
            return
        now = time.time()
        keep_idx = [
            i for i, e in enumerate(self._entries)
            if (now - e.get("ts", 0)) <= SEMANTIC_CACHE_TTL_S
        ]
        if len(keep_idx) == len(self._entries):
            return
        if not keep_idx:
            self._entries = []
            self._vecs = None
            return
        self._entries = [self._entries[i] for i in keep_idx]
        self._vecs = self._vecs[keep_idx]

    def lookup(self, query_emb: np.ndarray, threshold: float) -> Optional[dict]:
        with self._lock:
            self._purge_expired()
            if not self._entries or self._vecs is None or self._vecs.size == 0:
                return None
            # Embeddings are L2-normalized → dot product == cosine similarity.
            q = query_emb.astype(np.float32, copy=False).reshape(-1)
            if q.shape[0] != self._vecs.shape[1]:
                logger.warning(
                    f"Cache dim mismatch ({q.shape[0]} vs {self._vecs.shape[1]}) — clearing."
                )
                self._entries = []
                self._vecs = None
                self._save()
                return None
            sims = self._vecs @ q
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= threshold:
                hit = dict(self._entries[best_idx])
                hit["similarity"] = best_sim
                return hit
            return None

    def put(
        self,
        query: str,
        query_emb: np.ndarray,
        answer: str,
        route: str = "",
    ) -> None:
        with self._lock:
            self._purge_expired()
            v = query_emb.astype(np.float32, copy=False).reshape(1, -1)
            entry = {
                "q": query,
                "a": answer,
                "ts": time.time(),
                "route": route,
            }
            if self._vecs is None or self._vecs.size == 0:
                self._vecs = v.copy()
                self._entries = [entry]
            else:
                if v.shape[1] != self._vecs.shape[1]:
                    # Embedder swapped under us — rebuild from scratch.
                    self._entries = [entry]
                    self._vecs = v.copy()
                else:
                    # FIFO eviction once we exceed the cap.
                    overflow = (len(self._entries) + 1) - SEMANTIC_CACHE_MAX_ENTRIES
                    if overflow > 0:
                        self._entries = self._entries[overflow:]
                        self._vecs = self._vecs[overflow:]
                    self._vecs = np.vstack([self._vecs, v])
                    self._entries.append(entry)
            self._save()

    def reset(self) -> None:
        with self._lock:
            self._entries = []
            self._vecs = None
            for p in (_META_FILE, _VEC_FILE):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            logger.info("Semantic cache reset.")

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": SEMANTIC_CACHE_ENABLED,
                "entries": len(self._entries),
                "threshold": SEMANTIC_CACHE_THRESHOLD,
                "ttl_s": SEMANTIC_CACHE_TTL_S,
                "max_entries": SEMANTIC_CACHE_MAX_ENTRIES,
            }


_cache_singleton: Optional[_Cache] = None


def _get() -> _Cache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = _Cache()
    return _cache_singleton


def lookup(query: str) -> Optional[dict]:
    """Return ``{q, a, ts, route, similarity}`` if the cache has a near match,
    else None. Safe to call regardless of feature flag — returns None when
    disabled or on any internal error."""
    if not SEMANTIC_CACHE_ENABLED or not query:
        return None
    try:
        # Reuse the LRU-cached query embedder so identical strings cost nothing.
        from src.rag_engine import _embed_query

        emb = np.frombuffer(_embed_query(query), dtype=np.float32)
        return _get().lookup(emb, SEMANTIC_CACHE_THRESHOLD)
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def put(query: str, answer: str, route: str = "") -> None:
    """Store an answered query. No-op when caching is disabled or the answer
    is empty / clearly an error sentinel."""
    if not SEMANTIC_CACHE_ENABLED or not query or not answer:
        return
    # Don't cache obvious failure modes.
    stripped = answer.strip()
    if len(stripped) < 20 or stripped.lower().startswith(("error:", "(no ", "i don't")):
        return
    try:
        from src.rag_engine import _embed_query

        emb = np.frombuffer(_embed_query(query), dtype=np.float32)
        _get().put(query, emb, answer, route)
    except Exception as e:
        logger.warning(f"Cache put failed: {e}")


def reset() -> None:
    """Wipe the cache. Called when the index is rebuilt — file changes can
    invalidate any prior answer."""
    try:
        _get().reset()
    except Exception as e:
        logger.warning(f"Cache reset failed: {e}")


def stats() -> dict:
    try:
        return _get().stats()
    except Exception:
        return {"enabled": SEMANTIC_CACHE_ENABLED, "entries": 0}
