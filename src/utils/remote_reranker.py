"""HTTP client for the standalone Qwen3 rerank microservice.

Drop-in replacement for sentence_transformers.CrossEncoder / Qwen3Reranker /
VoyageReranker — same .predict(pairs, batch_size, show_progress_bar) shape
the rest of the pipeline expects. Activated by RERANKER_PROVIDER=remote.

The companion server lives at deepseek-rag/rerank_server/ — copy that
folder to the second laptop and run start.bat there.
"""
from __future__ import annotations

import httpx


class RemoteReranker:
    """POST (query, docs) over LAN to the Qwen3 rerank server."""

    def __init__(
        self,
        url: str,
        timeout: float = 60.0,
    ) -> None:
        if not url:
            raise ValueError("RemoteReranker requires a URL (REMOTE_RERANKER_URL).")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        # Fail fast at load time if the server is unreachable — the indexer
        # would otherwise log a generic loader error and silently fall back to
        # fusion-only ranking on the first query.
        try:
            r = self._client.get(f"{self.url}/health")
            r.raise_for_status()
            info = r.json()
            print(
                f"[RemoteReranker] Connected to {self.url} "
                f"(model={info.get('model')}, device={info.get('device')})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Rerank server at {self.url} is not reachable: {e}. "
                f"Start it on the second laptop with rerank_server\\start.bat "
                f"and verify the LAN IP / Windows Firewall rule."
            ) from e

    def predict(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 8,
        show_progress_bar: bool = False,
    ) -> list[float]:
        if not sentences:
            return []
        # All pairs in a rerank call share the same query (one user query
        # against many candidate docs); the server contract reflects that
        # to halve payload size vs sending the query per pair.
        query = sentences[0][0]
        docs = [d for _, d in sentences]
        try:
            r = self._client.post(
                f"{self.url}/rerank",
                json={"query": query, "docs": docs, "batch_size": int(batch_size)},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            # Bubble up so the engine can fall back to the fusion-only path
            # rather than silently returning bad scores.
            raise RuntimeError(f"Remote rerank failed: {e}") from e
        data = r.json()
        return [float(s) for s in data.get("scores", [])]

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
