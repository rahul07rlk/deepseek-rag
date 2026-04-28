"""Chunker unit tests. Runs in isolation - no GPU / no network required."""
from pathlib import Path

from src.chunker import chunk_file


def test_python_ast_chunking(tmp_path: Path):
    f = tmp_path / "example.py"
    f.write_text(
        '''"""Module docstring."""
import os

CONST = 1

def alpha(x):
    """One."""
    return x + 1

class Service:
    def handle(self, req):
        return req

    async def fetch(self, url):
        return url

def beta():
    pass
''',
        encoding="utf-8",
    )
    chunks = chunk_file(f)
    symbols = [c.get("symbol") for c in chunks]
    assert "alpha" in symbols
    assert "beta" in symbols
    assert "class Service" in symbols
    # Methods are indexed separately from the class chunk.
    assert "Service.handle" in symbols
    assert "Service.fetch" in symbols


def test_non_python_falls_back_to_windows(tmp_path: Path):
    f = tmp_path / "example.md"
    f.write_text("# Title\n" + ("some line\n" * 120), encoding="utf-8")
    chunks = chunk_file(f)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["start_line"] >= 1
        assert c["end_line"] >= c["start_line"]


def test_empty_file(tmp_path: Path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    assert chunk_file(f) == []


def test_unparseable_python_falls_back(tmp_path: Path):
    f = tmp_path / "broken.py"
    f.write_text("def oops(:\n    pass\n" * 20, encoding="utf-8")
    chunks = chunk_file(f)
    # Should still produce chunks via the window fallback.
    assert len(chunks) > 0
