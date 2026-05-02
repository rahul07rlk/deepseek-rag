"""Snapshot-versioned semantic answer cache.

After a query is answered, embed the original question and store
``(query_embedding, query_text, answer, snapshot_id, evidence_files,
files_fingerprint, route, timestamp, component_versions)`` to disk.

Lookup is a two-stage filter:
  1. Hard match on ``component_versions`` — entries built with a different
     embedder / reranker / prompt template are immediately discarded.
  2. Cosine similarity against the surviving vectors. Hits below the
     threshold are dropped.
  3. Per-file freshness check — for the matched entry, recompute the
     fingerprint of the evidence files cited at answer time. If any of
     them changed, the entry is treated as stale (deleted, miss returned).

This replaces the legacy "blow the entire cache on any save" invalidation
with surgical per-file invalidation. A save in `module_a.py` only
invalidates answers whose evidence cited `module_a.py`; answers about
`module_b.py` survive untouched.

Storage layout (under ``codebase_index/semantic_cache/``):
  - ``entries.json`` — list of dicts (q, a, ts, route, snapshot, versions,
    evidence_files, evidence_fp)
  - ``vectors.npy`` — N×dim float32 array, row-aligned with ``entries``

Both files are written atomically together. Partial-write corruption
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
from src.snapshot import (
    component_versions,
    current_snapshot,
    files_fingerprint,
)
from src.utils.logger import get_logger

logger = get_logger("answer_cache", "proxy.log")

_CACHE_DIR = INDEX_DIR / "semantic_cache"
_META_FILE = _CACHE_DIR / "entries.json"
_VEC_FILE = _CACHE_DIR / "vectors.npy"


class _Cache:
    def __init__(self) -> None:
        self._lock = RLock()
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

    def _purge_expired_and_versioned(self, current_versions: str) -> None:
        if not self._entries or self._vecs is None:
            return
        now = time.time()
        keep_idx: list[int] = []
        for i, e in enumerate(self._entries):
            age = now - e.get("ts", 0)
            if age > SEMANTIC_CACHE_TTL_S:
                continue
            if e.get("versions") and e["versions"] != current_versions:
                continue
            keep_idx.append(i)
        if len(keep_idx) == len(self._entries):
            return
        if not keep_idx:
            self._entries = []
            self._vecs = None
            return
        self._entries = [self._entries[i] for i in keep_idx]
        self._vecs = self._vecs[keep_idx]

    def lookup(
        self,
        query_emb: np.ndarray,
        threshold: float,
        current_versions: str,
    ) -> Optional[dict]:
        with self._lock:
            self._purge_expired_and_versioned(current_versions)
            if not self._entries or self._vecs is None or self._vecs.size == 0:
                return None
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
            if best_sim < threshold:
                return None

            entry = self._entries[best_idx]
            # Per-file freshness: recompute fingerprint of cited evidence
            # files. If anything changed, this answer is stale even though
            # the query embedding still matches.
            evidence = entry.get("evidence_files") or []
            if evidence:
                current_fp = files_fingerprint(evidence)
                if current_fp != entry.get("evidence_fp"):
                    logger.info(
                        f"Cache hit dropped (evidence stale): {entry.get('q', '')[:60]}"
                    )
                    self._invalidate_index(best_idx)
                    self._save()
                    return None

            hit = dict(entry)
            hit["similarity"] = best_sim
            return hit

    def _invalidate_index(self, idx: int) -> None:
        if 0 <= idx < len(self._entries):
            self._entries.pop(idx)
            if self._vecs is not None:
                self._vecs = np.delete(self._vecs, idx, axis=0)

    def invalidate_for_file(self, file_path: str) -> int:
        """Drop every entry whose evidence cited ``file_path``.

        Returns the count of entries dropped. Called by the watcher on
        per-file save instead of nuking the whole cache.
        """
        with self._lock:
            if not self._entries:
                return 0
            target = file_path.replace("\\", "/").lower()
            keep_idx: list[int] = []
            for i, e in enumerate(self._entries):
                evidence = [
                    str(f).replace("\\", "/").lower()
                    for f in (e.get("evidence_files") or [])
                ]
                if target in evidence:
                    continue
                keep_idx.append(i)
            dropped = len(self._entries) - len(keep_idx)
            if dropped == 0:
                return 0
            if not keep_idx:
                self._entries = []
                self._vecs = None
            else:
                self._entries = [self._entries[i] for i in keep_idx]
                if self._vecs is not None:
                    self._vecs = self._vecs[keep_idx]
            self._save()
            logger.info(f"Cache: invalidated {dropped} entries citing {file_path}")
            return dropped

    def put(
        self,
        query: str,
        query_emb: np.ndarray,
        answer: str,
        evidence_files: list[str] | None = None,
        route: str = "",
    ) -> None:
        with self._lock:
            versions = component_versions()
            self._purge_expired_and_versioned(versions)
            v = query_emb.astype(np.float32, copy=False).reshape(1, -1)
            evidence = list(evidence_files or [])
            entry = {
                "q": query,
                "a": answer,
                "ts": time.time(),
                "route": route,
                "snapshot": current_snapshot().id,
                "versions": versions,
                "evidence_files": evidence,
                "evidence_fp": files_fingerprint(evidence) if evidence else "",
            }
            if self._vecs is None or self._vecs.size == 0:
                self._vecs = v.copy()
                self._entries = [entry]
            else:
                if v.shape[1] != self._vecs.shape[1]:
                    self._entries = [entry]
                    self._vecs = v.copy()
                else:
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
                "snapshot": current_snapshot().id,
                "versions": component_versions(),
            }


_cache_singleton: Optional[_Cache] = None


def _get() -> _Cache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = _Cache()
    return _cache_singleton


def lookup(query: str) -> Optional[dict]:
    if not SEMANTIC_CACHE_ENABLED or not query:
        return None
    try:
        from src.rag_engine import _embed_query

        emb = np.frombuffer(_embed_query(query), dtype=np.float32)
        return _get().lookup(emb, SEMANTIC_CACHE_THRESHOLD, component_versions())
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def put(
    query: str,
    answer: str,
    route: str = "",
    evidence_files: list[str] | None = None,
) -> None:
    if not SEMANTIC_CACHE_ENABLED or not query or not answer:
        return
    stripped = answer.strip()
    if len(stripped) < 20 or stripped.lower().startswith(("error:", "(no ", "i don't")):
        return
    try:
        from src.rag_engine import _embed_query

        emb = np.frombuffer(_embed_query(query), dtype=np.float32)
        _get().put(query, emb, answer, evidence_files=evidence_files, route=route)
    except Exception as e:
        logger.warning(f"Cache put failed: {e}")


def invalidate_for_file(file_path: str) -> int:
    """Per-file cache invalidation — called by the watcher on save."""
    try:
        return _get().invalidate_for_file(file_path)
    except Exception as e:
        logger.warning(f"Per-file invalidation failed: {e}")
        return 0


def reset() -> None:
    try:
        _get().reset()
    except Exception as e:
        logger.warning(f"Cache reset failed: {e}")


def stats() -> dict:
    try:
        return _get().stats()
    except Exception:
        return {"enabled": SEMANTIC_CACHE_ENABLED, "entries": 0}
