import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def workspace_tmp():
    root = Path(__file__).parent / "_tmp_vector_store"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


def _emb(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_sidecar_tracks_file_chunk_ids(workspace_tmp: Path):
    store = VectorStore(workspace_tmp, dim=2, embed_model="test-model")
    store.add(
        ["a.py::chunk_0", "a.py::chunk_1", "b.py::chunk_0"],
        ["alpha", "beta", "gamma"],
        _emb([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
        [
            {"file": "a.py", "language": "python"},
            {"file": "a.py", "language": "python"},
            {"file": "b.py", "language": "python"},
        ],
    )
    store.save()

    state = json.loads((workspace_tmp / "index_state.json").read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert state["files"] == {"a.py": [0, 1], "b.py": [2]}

    reloaded = VectorStore(workspace_tmp, dim=2, embed_model="test-model")
    assert reloaded.get_by_file("a.py") == ["a.py::chunk_0", "a.py::chunk_1"]

    assert reloaded.delete_by_file("a.py") == 2
    assert reloaded.get_by_file("a.py") == []
    assert reloaded.count() == 1
    reloaded.save()

    state = json.loads((workspace_tmp / "index_state.json").read_text(encoding="utf-8"))
    assert state["files"] == {"b.py": [2]}


def test_existing_str_id_replaces_old_vector_and_metadata(workspace_tmp: Path):
    store = VectorStore(workspace_tmp, dim=2, embed_model="test-model")
    store.add(
        ["a.py::chunk_0"],
        ["old"],
        _emb([[1.0, 0.0]]),
        [{"file": "a.py", "hash": "old"}],
    )

    store.add(
        ["a.py::chunk_0"],
        ["new"],
        _emb([[0.0, 1.0]]),
        [{"file": "a.py", "hash": "new"}],
    )

    assert store.count() == 1
    assert store.get_by_file("a.py") == ["a.py::chunk_0"]
    doc, meta = store.get_by_str_ids(["a.py::chunk_0"])["a.py::chunk_0"]
    assert doc == "new"
    assert meta["hash"] == "new"


def test_missing_sidecar_is_repaired_from_metadata(workspace_tmp: Path):
    store = VectorStore(workspace_tmp, dim=2, embed_model="test-model")
    store.add(
        ["a.py::chunk_0"],
        ["alpha"],
        _emb([[1.0, 0.0]]),
        [{"file": "a.py"}],
    )
    store.save()
    (workspace_tmp / "index_state.json").unlink()

    reloaded = VectorStore(workspace_tmp, dim=2, embed_model="test-model")
    assert reloaded.get_by_file("a.py") == ["a.py::chunk_0"]
    reloaded.save()

    state = json.loads((workspace_tmp / "index_state.json").read_text(encoding="utf-8"))
    assert state["files"] == {"a.py": [0]}
