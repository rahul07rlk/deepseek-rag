"""Backend interface for the multi-graph code intelligence layer.

A backend stores nodes and edges and answers traversal queries. Two
implementations live alongside this:

  - ``SqliteBackend`` (always available, default fallback)
  - ``KuzuBackend``   (used when ``kuzu`` is installed; faster multi-hop)

Schema is normalized: every node has a stable string id, every edge
has a (src_id, dst_id, kind, attrs) shape. Backends translate this
shape to whatever native primitives they use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


# ── Node and edge value types ────────────────────────────────────────────────
@dataclass
class GraphNode:
    id: str            # globally unique (e.g. "file:src/foo.py", "sym:src/foo.py:Foo.bar")
    kind: str          # File | Module | Symbol | Test
    name: str          # display name
    file: str = ""     # owning file path (empty for File nodes themselves)
    line: int = 0      # definition line (0 if N/A)
    repo: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    src_id: str
    dst_id: str
    kind: str          # CALLS | IMPORTS | DEFINES | INHERITS | IMPLEMENTS | TESTS | TOUCHED_IN
    line: int = 0      # source line of the edge (0 if N/A)
    repo: str = ""
    weight: float = 1.0


# ── Backend protocol ─────────────────────────────────────────────────────────
class GraphBackend(Protocol):
    """All concrete backends implement this surface."""

    backend_name: str

    def initialize(self) -> None: ...
    def close(self) -> None: ...
    def clear(self) -> None: ...

    # Bulk ingestion (used at full-rebuild time).
    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None: ...
    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None: ...

    # Per-file incremental updates (used by the watcher).
    def delete_file(self, file_path: str) -> int: ...

    # Lookup primitives.
    def find_definitions(self, name: str, limit: int = 50) -> list[dict]: ...
    def edges_into(self, dst_id: str, kind: str | None = None,
                   limit: int = 100) -> list[dict]: ...
    def edges_out(self, src_id: str, kind: str | None = None,
                  limit: int = 100) -> list[dict]: ...

    # Multi-hop traversal — backend-native when available, generic walk
    # otherwise. Edge weights are returned so callers can rank.
    def multi_hop(
        self,
        seed_id: str,
        kinds: list[str] | None,
        max_hops: int,
        max_results: int,
    ) -> list[dict]: ...

    def stats(self) -> dict: ...
