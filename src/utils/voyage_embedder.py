"""Voyage AI cloud embedder — drop-in for SentenceTransformer.

Implements the same `.encode()` and `.get_sentence_embedding_dimension()`
interface so the indexer and rag_engine need no structural changes.

Key difference from local models: uses `input_type="document"` for bulk
encoding (indexing) and `input_type="query"` via the dedicated `embed_query()`
method called from rag_engine._embed_query — this asymmetric encoding is what
makes voyage-code-3 outperform bi-encoder baselines on code retrieval.
"""
from __future__ import annotations

import numpy as np

_DIMS: dict[str, int] = {
    "voyage-code-3": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-large-2": 1536,
    "voyage-2": 1024,
}

_API_BATCH_LIMIT = 128  # Voyage API max texts per call


class VoyageEmbedder:
    """Wraps voyageai.Client to look like SentenceTransformer."""

    def __init__(self, model: str, api_key: str) -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai package not installed. Run: pip install voyageai"
            ) from exc
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dim = _DIMS.get(model, 1024)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string with query-optimised input_type."""
        result = self._client.embed([query], model=self._model, input_type="query")
        arr = np.array(result.embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode document texts for indexing (input_type='document').

        Batches automatically at min(batch_size, _API_BATCH_LIMIT) per call
        to stay within Voyage API limits while respecting caller preference.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        effective_batch = min(batch_size, _API_BATCH_LIMIT)
        all_embs: list[list[float]] = []
        for i in range(0, len(sentences), effective_batch):
            batch = sentences[i : i + effective_batch]
            result = self._client.embed(
                batch, model=self._model, input_type="document"
            )
            all_embs.extend(result.embeddings)

        arr = np.array(all_embs, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr
