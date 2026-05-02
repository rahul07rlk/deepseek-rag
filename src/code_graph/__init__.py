"""Multi-graph code intelligence.

Replaces the flat ``symbol_graph`` (definitions + refs + imports in SQLite)
with a richer property graph:

  Node types     : File, Module, Symbol, Test
  Edge types     : DEFINES, CALLS, IMPORTS, INHERITS, IMPLEMENTS,
                   TESTS, TOUCHED_IN (git lineage)

Backend is swappable:
  - ``kuzu``    : embedded property graph (preferred when ``kuzu`` is
                  installed; native Cypher, multi-hop traversal in C++)
  - ``sqlite``  : same shape over SQLite tables (always available, used
                  as fallback and for backward compat with the old graph)

The public API in ``graph.py`` is identical regardless of backend, so
callers (rag_engine, agentic tools, eval) don't care which is active.
"""
from src.code_graph.graph import (  # re-export
    GraphEngine,
    get_engine,
    find_definitions,
    find_callers,
    find_callees,
    find_importers,
    find_implementations,
    tests_for_file,
    neighbors_for_chunk,
    multi_hop_neighbors,
    graph_stats,
    rebuild,
    update_for_file,
    delete_for_file,
)

__all__ = [
    "GraphEngine",
    "get_engine",
    "find_definitions",
    "find_callers",
    "find_callees",
    "find_importers",
    "find_implementations",
    "tests_for_file",
    "neighbors_for_chunk",
    "multi_hop_neighbors",
    "graph_stats",
    "rebuild",
    "update_for_file",
    "delete_for_file",
]
