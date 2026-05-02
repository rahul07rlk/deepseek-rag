"""Public API for the multi-graph code intelligence layer.

This module is the only thing other code (rag_engine, agentic, eval)
should talk to. It picks a backend at startup based on
``GRAPH_BACKEND`` and exposes the high-level operations RAG needs:

  - find_definitions, find_callers, find_callees, find_importers,
    find_implementations, tests_for_file
  - neighbors_for_chunk: 1-hop expansion (compat with the legacy API)
  - multi_hop_neighbors: budgeted N-hop traversal
  - rebuild / update_for_file / delete_for_file: ingestion entry points
"""
from __future__ import annotations

import os
from pathlib import Path

from src.code_graph.backend import GraphBackend
from src.code_graph.extract import (
    collect_recent_changes,
    extract,
    file_id,
    symbol_id,
)
from src.config import (
    INDEX_DIR,
    INDEXED_EXTENSIONS,
    IGNORED_DIRS,
    IGNORED_FILENAMES,
    IGNORED_SUFFIXES,
    MAX_FILE_BYTES,
    REPO_PATHS,
)
from src.utils.logger import get_logger

logger = get_logger("code_graph", "indexer.log")


_GRAPH_DB_DIR = INDEX_DIR / "code_graph"


class GraphEngine:
    def __init__(self, backend: GraphBackend):
        self.backend = backend
        self.backend.initialize()

    # ── High-level lookups ───────────────────────────────────────────────────
    def find_definitions(self, name: str, limit: int = 50) -> list[dict]:
        return self.backend.find_definitions(name, limit)

    def find_callers(self, name: str, limit: int = 50) -> list[dict]:
        """Files/symbols whose CALLS edges point at ``name``.

        We don't always know the exact symbol id of the target, so we
        match by name on placeholder ids first, then resolve to real
        nodes when possible.
        """
        # Definitions of the symbol — these are concrete dst nodes.
        defs = self.find_definitions(name, limit=20)
        callers: list[dict] = []
        seen: set[str] = set()
        # Collect edges into the by-name placeholder.
        placeholder = f"sym:?::{name}"
        for e in self.backend.edges_into(placeholder, kind="CALLS", limit=limit * 3):
            if e["src_id"] in seen:
                continue
            seen.add(e["src_id"])
            callers.append(e)
            if len(callers) >= limit:
                return callers
        for d in defs:
            for e in self.backend.edges_into(d["id"], kind="CALLS", limit=limit):
                if e["src_id"] in seen:
                    continue
                seen.add(e["src_id"])
                callers.append(e)
                if len(callers) >= limit:
                    return callers
        return callers

    def find_callees(self, src_id: str, limit: int = 50) -> list[dict]:
        return self.backend.edges_out(src_id, kind="CALLS", limit=limit)

    def find_importers(self, target_substring: str, limit: int = 50) -> list[dict]:
        # Reach into the backend with a substring match — both backends
        # implement edges_into(target_id), so we match the placeholder
        # `module:<target>` first and substring-search modules.
        # SQLite implements substring; Kuzu requires a different query.
        if hasattr(self.backend, "find_importers_substring"):
            return self.backend.find_importers_substring(target_substring, limit)  # type: ignore
        # Generic path: scan modules whose name contains the substring,
        # then collect IMPORTS edges into them.
        return _generic_substring_importers(self.backend, target_substring, limit)

    def find_implementations(self, base_name: str, limit: int = 50) -> list[dict]:
        """Symbols inheriting from a given base class/interface."""
        return self.backend.edges_into(f"sym:?::{base_name}", kind="INHERITS", limit=limit)

    def tests_for_file(self, file_path: str, limit: int = 20) -> list[dict]:
        """Files whose TESTS edges point at this file's module."""
        # Tests link to the imported module, not the file directly. We
        # accept either a file path or a module name; for a file path we
        # resolve to the most likely module name (file stem).
        stem = Path(file_path).stem
        return self.backend.edges_into(f"module:{stem}", kind="TESTS", limit=limit)

    def neighbors_for_chunk(
        self,
        file: str,
        symbol: str = "",
        start_line: int = 1,
        end_line: int = 1,
        limit: int = 12,
    ) -> list[dict]:
        """1-hop graph neighbors. Compat shim for the legacy API used by
        ``rag_engine._expand_ranked_with_call_graph``."""
        out: list[dict] = []
        seen: set[tuple[str, int, str, str]] = set()

        def add(relation: str, file_: str, line: int, sym: str, repo: str) -> None:
            key = (relation, file_, line, sym)
            if key in seen or len(out) >= limit:
                return
            seen.add(key)
            out.append({"relation": relation, "file": file_, "line": line,
                        "symbol": sym, "repo": repo})

        sym = (symbol or "").strip().split()[-1] if symbol else ""
        if "." in sym:
            sym = sym.rsplit(".", 1)[-1]

        if sym:
            # Callers
            for e in self.find_callers(sym, limit=limit * 2):
                # src_id is "sym:<file>::<name>" or "file:<file>"
                src = e.get("src_id", "")
                file_ = _id_to_file(src)
                if file_ and file_ != file:
                    add("caller", file_, int(e.get("line") or 0), sym,
                        e.get("repo") or "")

        # Callees: walk CALLS edges out of the chunk's file.
        fid = file_id(file)
        for e in self.backend.edges_out(fid, kind="CALLS", limit=limit * 2):
            dst = e.get("dst_id", "")
            file_ = _id_to_file(dst) or ""
            target_name = dst.rsplit("::", 1)[-1] if "::" in dst else ""
            if file_ and file_ != file:
                add("callee", file_, int(e.get("line") or 0), target_name,
                    e.get("repo") or "")

        # Importers: files whose IMPORTS hit a module matching this file's stem.
        stem = Path(file).stem
        if stem:
            for e in self.backend.edges_into(f"module:{stem}", kind="IMPORTS",
                                              limit=limit * 2):
                src = e.get("src_id", "")
                file_ = _id_to_file(src)
                if file_ and file_ != file:
                    add("importer", file_, int(e.get("line") or 0), stem,
                        e.get("repo") or "")
        return out

    def multi_hop_neighbors(
        self,
        seed_file: str,
        seed_symbol: str = "",
        kinds: list[str] | None = None,
        max_hops: int = 2,
        max_results: int = 25,
    ) -> list[dict]:
        """Budgeted N-hop traversal starting from a chunk's file/symbol.

        ``kinds`` filters the edges followed (e.g. ['CALLS', 'IMPORTS']
        for "what does this code reach?", ['INHERITS'] for "type
        hierarchy"). ``max_hops`` caps depth, ``max_results`` caps width.
        """
        if seed_symbol:
            seed_id = symbol_id(seed_file, seed_symbol)
        else:
            seed_id = file_id(seed_file)
        return self.backend.multi_hop(seed_id, kinds, max_hops, max_results)

    # ── Ingestion ────────────────────────────────────────────────────────────
    def rebuild(self) -> dict:
        """Walk every configured repo and rebuild the graph from scratch."""
        self.backend.clear()
        recent = collect_recent_changes(REPO_PATHS, lookback=30)
        all_nodes: list = []
        all_edges: list = []
        files_processed = 0
        for repo in REPO_PATHS:
            for p in repo.rglob("*"):
                if not p.is_file() or not _is_indexable(p, repo):
                    continue
                nodes, edges = extract(p, repo.name, recent)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                files_processed += 1
                # Flush periodically to keep memory bounded on big repos.
                if len(all_nodes) > 10_000 or len(all_edges) > 20_000:
                    self.backend.upsert_nodes(all_nodes)
                    self.backend.upsert_edges(all_edges)
                    all_nodes = []
                    all_edges = []
        if all_nodes:
            self.backend.upsert_nodes(all_nodes)
        if all_edges:
            self.backend.upsert_edges(all_edges)
        stats = self.backend.stats()
        stats["files_processed"] = files_processed
        logger.info(f"Code graph rebuilt: {stats}")
        return stats

    def update_for_file(self, file_path: str | Path) -> dict:
        p = Path(file_path)
        s = str(p).replace("\\", "/")
        if not p.exists():
            self.backend.delete_file(s)
            return {"deleted": True, "file": s}
        repo = _resolve_repo(p)
        nodes, edges = extract(p, repo, recent_change_set=None)
        self.backend.delete_file(s)
        self.backend.upsert_nodes(nodes)
        self.backend.upsert_edges(edges)
        return {"file": s, "nodes": len(nodes), "edges": len(edges)}

    def delete_for_file(self, file_path: str | Path) -> int:
        s = str(Path(file_path)).replace("\\", "/")
        return self.backend.delete_file(s)

    def stats(self) -> dict:
        return self.backend.stats()


# ── Backend selection + singleton ────────────────────────────────────────────
_engine: GraphEngine | None = None


def _make_backend() -> GraphBackend:
    requested = os.getenv("GRAPH_BACKEND", "auto").lower()
    if requested in ("auto", "kuzu"):
        try:
            from src.code_graph.kuzu_backend import KuzuBackend
            backend = KuzuBackend(_GRAPH_DB_DIR / "kuzu_db")
            backend.initialize()
            logger.info("Code graph backend: Kuzu")
            return backend
        except Exception as e:
            if requested == "kuzu":
                # Explicit request — surface the failure.
                raise
            logger.info(f"Kuzu unavailable ({type(e).__name__}); using SQLite backend.")
    from src.code_graph.sqlite_backend import SqliteBackend
    backend = SqliteBackend(_GRAPH_DB_DIR / "graph.sqlite")
    backend.initialize()
    logger.info("Code graph backend: SQLite")
    return backend


def get_engine() -> GraphEngine:
    global _engine
    if _engine is None:
        _engine = GraphEngine(_make_backend())
    return _engine


# ── Module-level convenience wrappers (mirror legacy symbol_graph API) ───────
def find_definitions(name: str, limit: int = 50) -> list[dict]:
    return get_engine().find_definitions(name, limit)


def find_callers(name: str, limit: int = 50) -> list[dict]:
    return get_engine().find_callers(name, limit)


def find_callees(src_id: str, limit: int = 50) -> list[dict]:
    return get_engine().find_callees(src_id, limit)


def find_importers(target: str, limit: int = 50) -> list[dict]:
    return get_engine().find_importers(target, limit)


def find_implementations(base_name: str, limit: int = 50) -> list[dict]:
    return get_engine().find_implementations(base_name, limit)


def tests_for_file(file_path: str, limit: int = 20) -> list[dict]:
    return get_engine().tests_for_file(file_path, limit)


def neighbors_for_chunk(file: str, symbol: str = "", start_line: int = 1,
                         end_line: int = 1, limit: int = 12) -> list[dict]:
    return get_engine().neighbors_for_chunk(file, symbol, start_line, end_line, limit)


def multi_hop_neighbors(seed_file: str, seed_symbol: str = "",
                         kinds: list[str] | None = None, max_hops: int = 2,
                         max_results: int = 25) -> list[dict]:
    return get_engine().multi_hop_neighbors(
        seed_file, seed_symbol, kinds, max_hops, max_results,
    )


def graph_stats() -> dict:
    return get_engine().stats()


def rebuild() -> dict:
    return get_engine().rebuild()


def update_for_file(file_path: str | Path) -> dict:
    return get_engine().update_for_file(file_path)


def delete_for_file(file_path: str | Path) -> int:
    return get_engine().delete_for_file(file_path)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _id_to_file(node_id: str) -> str:
    if node_id.startswith("file:"):
        return node_id[len("file:"):]
    if node_id.startswith("sym:"):
        rest = node_id[len("sym:"):]
        if "::" in rest:
            return rest.split("::", 1)[0]
    return ""


def _is_indexable(p: Path, repo: Path) -> bool:
    if p.suffix not in INDEXED_EXTENSIONS:
        return False
    if p.name in IGNORED_FILENAMES:
        return False
    lower = p.name.lower()
    if any(lower.endswith(s) for s in IGNORED_SUFFIXES):
        return False
    try:
        size = p.stat().st_size
    except OSError:
        return False
    if size == 0 or size > MAX_FILE_BYTES:
        return False
    try:
        if any(part in IGNORED_DIRS for part in p.relative_to(repo).parts):
            return False
    except ValueError:
        return False
    return True


def _resolve_repo(file_path: Path) -> str:
    for rp in REPO_PATHS:
        try:
            file_path.relative_to(rp)
            return rp.name
        except ValueError:
            continue
    return file_path.parts[0] if file_path.parts else "unknown"


def _generic_substring_importers(backend: GraphBackend, sub: str,
                                   limit: int) -> list[dict]:
    """Fallback scan for backends that don't have a native substring helper."""
    # We don't have a generic name-search on edges in the protocol, so
    # we approximate with a Kuzu/SQLite-friendly best effort: query
    # edges_into(module:<sub>) directly. Substring users often pass
    # an exact stem.
    return backend.edges_into(f"module:{sub}", kind="IMPORTS", limit=limit)


if __name__ == "__main__":
    print(rebuild())
