"""Voyage AI cloud reranker — drop-in for CrossEncoder / Qwen3Reranker.

Exposes `.predict(pairs, ...)` so `_rerank_candidates` in rag_engine.py
needs no changes. Voyage returns calibrated [0, 1] relevance scores;
we convert to log-odds so the existing sigmoid normalisation in rag_engine
round-trips back to the same values.
"""
from __future__ import annotations

import numpy as np


class VoyageReranker:
    """Wraps voyageai.Client.rerank() to look like CrossEncoder."""

    def __init__(self, model: str, api_key: str) -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai package not installed. Run: pip install voyageai"
            ) from exc
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Score (query, doc) pairs. All pairs must share the same query.

        Returns log-odds so that rag_engine's sigmoid normalisation recovers
        the original [0, 1] Voyage relevance scores.
        """
        if not pairs:
            return np.array([], dtype=np.float64)

        query = pairs[0][0]
        documents = [doc for _, doc in pairs]

        result = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=len(documents),
            truncation=True,
        )

        # result.results: list of RerankingObject with .index and .relevance_score
        # Initialise to near-zero probability for any docs not returned.
        scores = np.full(len(documents), 1e-7, dtype=np.float64)
        for item in result.results:
            scores[item.index] = float(item.relevance_score)

        # Convert [0,1] → log-odds so sigmoid(logit(p)) == p
        eps = 1e-7
        scores = np.clip(scores, eps, 1.0 - eps)
        return np.log(scores / (1.0 - scores))
