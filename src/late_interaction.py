"""ColBERT-style late-interaction retrieval.

A bi-encoder (FAISS + Qwen3) compresses each chunk into one vector.
That throws away token-level structure: ``processData`` and
``processUserData`` collapse onto similar points in space, and a query
mentioning the exact identifier doesn't get extra weight for it being
the exact identifier.

Late interaction keeps a small set of token-level vectors per chunk
(``MaxSim`` aggregation: query token → max similarity over chunk tokens,
summed). For identifier-heavy code queries this gives meaningful recall
gains and works on a GTX 1650 because:

  - the model is small (PyLate-style ColBERT distillations: ~110M params)
  - per-chunk storage is bounded (≤32 token vectors × 128d ≈ 16 KB)
  - retrieval is a sparse dot-product, not a full transformer pass

Implementation strategy:

  - Use ``pylate`` (preferred) if installed: it ships pre-trained
    ColBERT models for code and handles the indexing format. Fall back
    to a no-op shim when pylate isn't available so the rest of the
    system runs unaffected.
  - Index file is independent of FAISS / BM25 — lives at
    ``codebase_index/late_interaction/`` and is rebuilt by the indexer.
  - At query time, retrieval fuses the late-interaction top-K with the
    existing FAISS+BM25 results via RRF before reranking.

API (mirrors ``rag_engine`` callsites):

  - ``LateInteractionStore.index(items)``: bulk add (id, text)
  - ``LateInteractionStore.search(query, k)`` -> [(id, score)]
  - ``available()`` -> bool
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from src.config import INDEX_DIR
from src.utils.logger import get_logger

logger = get_logger("late_interaction", "indexer.log")

_LI_DIR = INDEX_DIR / "late_interaction"


def _enabled_flag() -> bool:
    return os.getenv("LATE_INTERACTION_ENABLED", "true").lower() == "true"


def _model_name() -> str:
    return os.getenv(
        "LATE_INTERACTION_MODEL",
        "lightonai/Reason-ModernColBERT",  # ~110M params, code-friendly
    )


class LateInteractionStore:
    """Wraps a pylate index with the same surface shape as VectorStore.

    Methods are no-ops when pylate isn't installed — the callsite checks
    ``self.available`` and skips this branch.
    """

    def __init__(self):
        self._available = False
        self._encoder = None
        self._index = None
        self._retriever = None
        self._init()

    def _init(self) -> None:
        if not _enabled_flag():
            return
        try:
            from pylate import indexes, models, retrieve  # type: ignore
        except ImportError:
            logger.info(
                "pylate not installed — late-interaction disabled. "
                "`pip install pylate` to enable."
            )
            return
        try:
            _LI_DIR.mkdir(parents=True, exist_ok=True)
            self._encoder = models.ColBERT(
                model_name_or_path=_model_name(),
                device=os.getenv("LATE_INTERACTION_DEVICE", "cpu"),
            )
            self._index = indexes.PLAID(
                index_folder=str(_LI_DIR),
                index_name="code",
                override=False,
            )
            self._retriever = retrieve.ColBERT(index=self._index)
            self._available = True
            logger.info(f"Late-interaction online: {_model_name()}")
        except Exception as e:
            logger.warning(f"Late-interaction init failed: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ── Indexing ─────────────────────────────────────────────────────────────
    def index(self, items: Iterable[tuple[str, str]],
               batch_size: int = 32) -> int:
        """Add (id, text) pairs to the index. Returns count added.

        Re-indexing is idempotent at the id level — pylate's PLAID index
        replaces existing ids on add.
        """
        if not self._available:
            return 0
        ids: list[str] = []
        texts: list[str] = []
        added = 0
        for cid, txt in items:
            ids.append(cid)
            texts.append(txt)
            if len(ids) >= batch_size:
                added += self._flush(ids, texts)
                ids, texts = [], []
        if ids:
            added += self._flush(ids, texts)
        return added

    def _flush(self, ids: list[str], texts: list[str]) -> int:
        try:
            embeddings = self._encoder.encode(
                texts, is_query=False, show_progress_bar=False,
            )
            self._index.add_documents(documents_ids=ids, documents_embeddings=embeddings)
            return len(ids)
        except Exception as e:
            logger.warning(f"Late-interaction index batch failed: {e}")
            return 0

    # ── Retrieval ────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        """Return [(doc_id, score)] sorted by descending score."""
        if not self._available or not query:
            return []
        try:
            q_emb = self._encoder.encode(
                [query], is_query=True, show_progress_bar=False,
            )
            results = self._retriever.retrieve(queries_embeddings=q_emb, k=k)
            # results is list[list[dict]]; we have a single query.
            row = results[0] if results else []
            return [(r["id"], float(r["score"])) for r in row]
        except Exception as e:
            logger.debug(f"Late-interaction search failed: {e}")
            return []

    def reset(self) -> None:
        if not self._available:
            return
        try:
            # Best-effort: drop the index folder so next call reinitializes.
            import shutil
            shutil.rmtree(_LI_DIR, ignore_errors=True)
            _LI_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._available = False
        self._init()


_singleton: LateInteractionStore | None = None


def get_store() -> LateInteractionStore:
    global _singleton
    if _singleton is None:
        _singleton = LateInteractionStore()
    return _singleton


def available() -> bool:
    return get_store().available


def search(query: str, k: int = 30) -> list[tuple[str, float]]:
    return get_store().search(query, k)
