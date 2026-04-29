import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from src import indexer
from src.vector_store import VectorStore


@pytest.fixture
def workspace_tmp():
    root = Path(__file__).parent / "_tmp_incremental_indexer"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


def _store_with_file(index_dir: Path, fpath: Path) -> VectorStore:
    store = VectorStore(index_dir, dim=2, embed_model="test-model")
    store.add(
        [f"{fpath}::chunk_0"],
        ["stale code"],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [{"file": str(fpath), "filename": fpath.name, "hash": "old"}],
    )
    store.save()
    return store


def test_single_file_empty_chunk_result_removes_stale_chunks(
    workspace_tmp: Path, monkeypatch
):
    fpath = workspace_tmp / "sample.py"
    fpath.write_text("x = 1\n", encoding="utf-8")
    store = _store_with_file(workspace_tmp / "index", fpath)
    events: list[str] = []

    monkeypatch.setattr(indexer, "get_store", lambda: store)
    monkeypatch.setattr(indexer, "chunk_file", lambda _: [])
    monkeypatch.setattr(indexer, "schedule_bm25_rebuild", lambda: events.append("bm25"))
    monkeypatch.setattr(
        indexer, "_update_symbol_graph_for_file", lambda _: events.append("graph")
    )
    monkeypatch.setattr(
        indexer,
        "get_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    indexer.index_single_file(fpath)

    assert store.count() == 0
    assert events == ["graph", "bm25"]
    reloaded = VectorStore(workspace_tmp / "index", dim=2, embed_model="test-model")
    assert reloaded.count() == 0
    assert reloaded.get_by_file(str(fpath)) == []


def test_single_file_noise_filter_removes_stale_chunks(workspace_tmp: Path, monkeypatch):
    fpath = workspace_tmp / "sample.py"
    fpath.write_text("x = 1\n", encoding="utf-8")
    store = _store_with_file(workspace_tmp / "index", fpath)
    events: list[str] = []

    monkeypatch.setattr(indexer, "get_store", lambda: store)
    monkeypatch.setattr(indexer, "_is_noise", lambda _: (True, "too large"))
    monkeypatch.setattr(indexer, "schedule_bm25_rebuild", lambda: events.append("bm25"))
    monkeypatch.setattr(
        indexer, "_update_symbol_graph_for_file", lambda _: events.append("graph")
    )
    monkeypatch.setattr(
        indexer,
        "get_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    indexer.index_single_file(fpath)

    assert store.count() == 0
    assert events == ["graph", "bm25"]
    reloaded = VectorStore(workspace_tmp / "index", dim=2, embed_model="test-model")
    assert reloaded.count() == 0


def test_single_file_contextualizes_embedding_but_stores_raw(
    workspace_tmp: Path, monkeypatch
):
    fpath = workspace_tmp / "sample.py"
    fpath.write_text(
        "def answer():\n    return 42\n",
        encoding="utf-8",
    )
    store = VectorStore(workspace_tmp / "index", dim=2, embed_model="test-model")
    events: list[str] = []

    class FakeModel:
        seen_docs: list[str] = []

        def encode(self, docs, **_kwargs):
            self.seen_docs = list(docs)
            return np.asarray([[1.0, 0.0] for _ in docs], dtype=np.float32)

    fake_model = FakeModel()

    monkeypatch.setattr(indexer, "get_store", lambda: store)
    monkeypatch.setattr(indexer, "get_model", lambda: fake_model)
    monkeypatch.setattr(indexer, "schedule_bm25_rebuild", lambda: events.append("bm25"))
    monkeypatch.setattr(
        indexer, "_update_symbol_graph_for_file", lambda _: events.append("graph")
    )
    monkeypatch.setattr(indexer, "CONTEXTUAL_RETRIEVAL_MODE", "rules")
    monkeypatch.setattr(indexer, "CONTEXTUAL_PREFIX_MAX_CHARS", 300)

    indexer.index_single_file(fpath)

    assert fake_model.seen_docs
    assert fake_model.seen_docs[0].startswith("This chunk is from")
    assert "def answer()" in fake_model.seen_docs[0]

    stored = store.get_by_str_ids([f"{fpath}::chunk_0"])[f"{fpath}::chunk_0"]
    raw_doc, meta = stored
    assert raw_doc.startswith("def answer()")
    assert "This chunk is from" not in raw_doc
    assert meta["contextual_version"].startswith("rules:")
    assert meta["contextual_prefix"].startswith("This chunk is from")
    assert events == ["graph", "bm25"]
