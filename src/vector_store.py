"""FAISS-backed persistent vector store.

Replaces ChromaDB. Uses IndexIDMap2(IndexFlatIP) so that:
  - Embeddings normalized to unit length => inner product == cosine similarity
  - Individual vectors can be removed by ID (needed for incremental re-index)
  - No C++ compilation required on the host machine (faiss-cpu has pre-built wheels)

Persistence layout inside INDEX_DIR:
  faiss.index      - FAISS binary index
  store_meta.json  - all chunk text + metadata, keyed by integer FAISS ID
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    _INDEX_FILE = "faiss.index"
    _META_FILE = "store_meta.json"

    def __init__(self, index_dir: Path, dim: int, embed_model: str = "") -> None:
        self._dir = index_dir
        self._dim = dim
        self._embed_model = embed_model
        self._index_path = index_dir / self._INDEX_FILE
        self._meta_path = index_dir / self._META_FILE

        # int_id -> {"str_id": str, "doc": str, "meta": dict}
        self._data: dict[int, dict] = {}
        # str_id -> int_id  (reverse lookup)
        self._str_to_int: dict[str, int] = {}
        self._next_id: int = 0

        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _make_index(self) -> faiss.Index:
        inner = faiss.IndexFlatIP(self._dim)
        return faiss.IndexIDMap2(inner)

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            with open(self._meta_path, encoding="utf-8") as f:
                saved = json.load(f)
            stored_model = saved.get("embed_model", "")
            if stored_model and stored_model != self._embed_model and saved.get("count", 0) > 0:
                # Dimension mismatch — loading the stale FAISS index would crash at search time.
                # Start fresh; the indexer will rebuild with the correct model.
                # ASCII-only message: cp1252 consoles (Windows default) choke on
                # arrows / em-dashes when stdout isn't reconfigured to UTF-8.
                print(
                    f"[VectorStore] Embedding model changed "
                    f"'{stored_model}' -> '{self._embed_model}'. "
                    f"Stale index discarded - re-run the indexer to rebuild."
                )
                self._index = self._make_index()
                # Reset the in-memory state too, otherwise repo-map / stats
                # see a non-empty self._data that's now misaligned with FAISS.
                self._data.clear()
                self._str_to_int.clear()
                self._next_id = 0
                return
            self._index = faiss.read_index(str(self._index_path))
            self._next_id = saved["next_id"]
            self._embed_model = stored_model or self._embed_model
            for k, v in saved["data"].items():
                iid = int(k)
                self._data[iid] = v
                self._str_to_int[v["str_id"]] = iid
        else:
            self._index = self._make_index()

    def save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "next_id": self._next_id,
                    "embed_model": self._embed_model,
                    "count": len(self._data),
                    "data": {str(k): v for k, v in self._data.items()},
                },
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    # ── Write operations ──────────────────────────────────────────────────────

    def add(
        self,
        str_ids: list[str],
        docs: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
    ) -> None:
        if not str_ids:
            return
        int_ids: list[int] = []
        for str_id in str_ids:
            iid = self._next_id
            self._next_id += 1
            self._str_to_int[str_id] = iid
            int_ids.append(iid)

        for iid, str_id, doc, meta in zip(int_ids, str_ids, docs, metadatas):
            self._data[iid] = {"str_id": str_id, "doc": doc, "meta": meta}

        vecs = np.asarray(embeddings, dtype=np.float32)
        self._index.add_with_ids(vecs, np.array(int_ids, dtype=np.int64))

    def delete_by_file(self, file_str: str) -> int:
        """Remove all chunks for file_str. Returns number of chunks removed."""
        to_remove = [
            iid for iid, v in self._data.items()
            if v["meta"].get("file") == file_str
        ]
        if not to_remove:
            return 0
        self._index.remove_ids(np.array(to_remove, dtype=np.int64))
        for iid in to_remove:
            str_id = self._data[iid]["str_id"]
            self._str_to_int.pop(str_id, None)
            del self._data[iid]
        return len(to_remove)

    def delete_by_file_prefix(self, file_str: str) -> int:
        """Alias kept for clarity; same as delete_by_file."""
        return self.delete_by_file(file_str)

    # ── Read operations ───────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self._data)

    def search(self, query: np.ndarray, n: int) -> list[tuple[str, float]]:
        """Return list of (str_id, cosine_similarity) sorted descending."""
        actual_n = min(n, self.count())
        if actual_n == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, int_ids = self._index.search(q, actual_n)
        results: list[tuple[str, float]] = []
        for score, iid in zip(scores[0], int_ids[0]):
            if iid == -1:
                continue
            v = self._data.get(int(iid))
            if v:
                results.append((v["str_id"], float(score)))
        return results

    def get_by_str_ids(self, str_ids: list[str]) -> dict[str, tuple[str, dict]]:
        """Return {str_id: (doc, meta)} for each found ID."""
        out: dict[str, tuple[str, dict]] = {}
        for str_id in str_ids:
            iid = self._str_to_int.get(str_id)
            if iid is not None:
                v = self._data.get(iid)
                if v:
                    out[str_id] = (v["doc"], v["meta"])
        return out

    def get_by_file(self, file_str: str) -> list[str]:
        """Return str_ids of all chunks for file_str."""
        return [
            v["str_id"] for v in self._data.values()
            if v["meta"].get("file") == file_str
        ]

    def get_all(self) -> tuple[list[str], list[str], list[dict]]:
        """Return (str_ids, docs, metas) for every stored chunk."""
        str_ids, docs, metas = [], [], []
        for v in self._data.values():
            str_ids.append(v["str_id"])
            docs.append(v["doc"])
            metas.append(v["meta"])
        return str_ids, docs, metas
