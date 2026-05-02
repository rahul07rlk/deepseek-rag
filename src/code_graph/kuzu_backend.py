"""Kuzu implementation of the multi-graph backend.

Kuzu is an embedded property-graph database with native Cypher and
column-store storage. For the RAG graph workload (call hierarchies,
import chains, test→code links) it is roughly 10-50× faster than the
SQLite backend on multi-hop traversal — the difference matters once
the graph has >100k edges.

The backend is loaded lazily; ``import kuzu`` happens inside
``initialize`` so the rest of the system runs unaffected when Kuzu
isn't installed.

Schema (Cypher):

    CREATE NODE TABLE node(
        id STRING, kind STRING, name STRING, file STRING,
        line INT64, repo STRING, extra STRING, PRIMARY KEY(id)
    );
    CREATE REL TABLE edge(
        FROM node TO node, kind STRING, line INT64, repo STRING, weight DOUBLE
    );
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.code_graph.backend import GraphBackend, GraphEdge, GraphNode


class KuzuBackend(GraphBackend):
    backend_name = "kuzu"

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db = None
        self._conn = None
        self._kuzu = None

    def initialize(self) -> None:
        try:
            import kuzu  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Kuzu backend requested but `kuzu` is not installed. "
                "Run `pip install kuzu` or set GRAPH_BACKEND=sqlite."
            ) from e
        self._kuzu = kuzu
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self.db_path))
        self._conn = kuzu.Connection(self._db)
        # CREATE TABLE statements are idempotent in Kuzu (IF NOT EXISTS-style).
        try:
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS node(
                    id STRING, kind STRING, name STRING, file STRING,
                    line INT64, repo STRING, extra STRING, PRIMARY KEY(id)
                )
            """)
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS edge(
                    FROM node TO node,
                    kind STRING, line INT64, repo STRING, weight DOUBLE
                )
            """)
        except Exception:
            # Older Kuzu versions don't accept IF NOT EXISTS — try plain CREATE.
            try:
                self._conn.execute("""
                    CREATE NODE TABLE node(
                        id STRING, kind STRING, name STRING, file STRING,
                        line INT64, repo STRING, extra STRING, PRIMARY KEY(id)
                    )
                """)
            except Exception:
                pass
            try:
                self._conn.execute("""
                    CREATE REL TABLE edge(
                        FROM node TO node,
                        kind STRING, line INT64, repo STRING, weight DOUBLE
                    )
                """)
            except Exception:
                pass

    def close(self) -> None:
        # Kuzu auto-flushes; nothing extra needed.
        self._conn = None
        self._db = None

    def clear(self) -> None:
        if self._conn is None:
            self.initialize()
        self._conn.execute("MATCH ()-[e:edge]->() DELETE e")
        self._conn.execute("MATCH (n:node) DELETE n")

    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        if self._conn is None:
            self.initialize()
        for n in nodes:
            extra = json.dumps(n.extra)
            # MERGE keeps node identity unique on id.
            self._conn.execute(
                "MERGE (x:node {id: $id}) "
                "SET x.kind=$kind, x.name=$name, x.file=$file, "
                "x.line=$line, x.repo=$repo, x.extra=$extra",
                {"id": n.id, "kind": n.kind, "name": n.name, "file": n.file,
                 "line": int(n.line), "repo": n.repo, "extra": extra},
            )

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        if self._conn is None:
            self.initialize()
        for e in edges:
            self._conn.execute(
                "MATCH (a:node {id: $sid}), (b:node {id: $did}) "
                "CREATE (a)-[:edge {kind: $kind, line: $line, "
                "repo: $repo, weight: $weight}]->(b)",
                {"sid": e.src_id, "did": e.dst_id, "kind": e.kind,
                 "line": int(e.line), "repo": e.repo, "weight": float(e.weight)},
            )

    def delete_file(self, file_path: str) -> int:
        if self._conn is None:
            self.initialize()
        # Drop edges touching nodes from this file, then the nodes.
        self._conn.execute(
            "MATCH (n:node {file: $f})-[r:edge]-() DELETE r",
            {"f": file_path},
        )
        result = self._conn.execute(
            "MATCH (n:node {file: $f}) DELETE n RETURN COUNT(*) AS c",
            {"f": file_path},
        )
        # Kuzu Result has get_next; we just sum.
        n = 0
        try:
            while result.has_next():
                row = result.get_next()
                n += int(row[0])
        except Exception:
            pass
        return n

    def find_definitions(self, name: str, limit: int = 50) -> list[dict]:
        if self._conn is None:
            self.initialize()
        result = self._conn.execute(
            "MATCH (n:node) WHERE n.kind = 'Symbol' AND n.name = $n "
            "RETURN n.id, n.kind, n.name, n.file, n.line, n.repo, n.extra LIMIT $lim",
            {"n": name, "lim": int(limit)},
        )
        return _result_to_dicts(result)

    def edges_into(self, dst_id: str, kind: str | None = None,
                   limit: int = 100) -> list[dict]:
        if self._conn is None:
            self.initialize()
        if kind:
            q = ("MATCH (a:node)-[r:edge]->(b:node {id: $d}) "
                 "WHERE r.kind = $k "
                 "RETURN a.id, b.id, r.kind, r.line, r.repo, r.weight LIMIT $lim")
            params = {"d": dst_id, "k": kind, "lim": int(limit)}
        else:
            q = ("MATCH (a:node)-[r:edge]->(b:node {id: $d}) "
                 "RETURN a.id, b.id, r.kind, r.line, r.repo, r.weight LIMIT $lim")
            params = {"d": dst_id, "lim": int(limit)}
        return _edges_to_dicts(self._conn.execute(q, params))

    def edges_out(self, src_id: str, kind: str | None = None,
                  limit: int = 100) -> list[dict]:
        if self._conn is None:
            self.initialize()
        if kind:
            q = ("MATCH (a:node {id: $s})-[r:edge]->(b:node) "
                 "WHERE r.kind = $k "
                 "RETURN a.id, b.id, r.kind, r.line, r.repo, r.weight LIMIT $lim")
            params = {"s": src_id, "k": kind, "lim": int(limit)}
        else:
            q = ("MATCH (a:node {id: $s})-[r:edge]->(b:node) "
                 "RETURN a.id, b.id, r.kind, r.line, r.repo, r.weight LIMIT $lim")
            params = {"s": src_id, "lim": int(limit)}
        return _edges_to_dicts(self._conn.execute(q, params))

    def multi_hop(
        self,
        seed_id: str,
        kinds: list[str] | None,
        max_hops: int,
        max_results: int,
    ) -> list[dict]:
        if self._conn is None:
            self.initialize()
        if max_hops <= 0:
            return []
        # Variable-length match. Kuzu uses [r:edge*1..N] syntax.
        kind_filter = ""
        params: dict = {"sid": seed_id, "lim": int(max_results)}
        if kinds:
            kind_filter = " WHERE ALL(rel IN r WHERE rel.kind IN $kinds)"
            params["kinds"] = kinds
        q = (
            f"MATCH (a:node {{id: $sid}})-[r:edge*1..{int(max_hops)}]->(b:node)"
            f"{kind_filter} "
            "RETURN DISTINCT b.id, b.kind, b.name, b.file, b.line, b.repo, b.extra, "
            "size(r) AS hop "
            "LIMIT $lim"
        )
        try:
            result = self._conn.execute(q, params)
        except Exception:
            # Older Kuzu versions: fall back to fixed 1-hop and walk in Python.
            return self._fallback_multi_hop(seed_id, kinds, max_hops, max_results)
        rows = []
        try:
            while result.has_next():
                r = result.get_next()
                rows.append({
                    "id": r[0], "kind": r[1], "name": r[2],
                    "file": r[3], "line": r[4], "repo": r[5],
                    "extra": _safe_json(r[6]),
                    "hop": int(r[7]) if len(r) > 7 else 1,
                })
        except Exception:
            pass
        rows.sort(key=lambda x: (x.get("hop", 1)))
        return rows[:max_results]

    def _fallback_multi_hop(
        self,
        seed_id: str,
        kinds: list[str] | None,
        max_hops: int,
        max_results: int,
    ) -> list[dict]:
        """Python BFS for older Kuzu versions that lack variable-length match."""
        seen: set[str] = {seed_id}
        out: list[dict] = []
        frontier: list[tuple[str, int]] = [(seed_id, 0)]
        while frontier and len(out) < max_results:
            node_id, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            for e in self.edges_out(node_id, limit=200):
                if kinds and e["kind"] not in kinds:
                    continue
                if e["dst_id"] in seen:
                    continue
                seen.add(e["dst_id"])
                # Pull node info.
                result = self._conn.execute(
                    "MATCH (n:node {id: $i}) RETURN "
                    "n.id, n.kind, n.name, n.file, n.line, n.repo, n.extra",
                    {"i": e["dst_id"]},
                )
                if result.has_next():
                    r = result.get_next()
                    out.append({
                        "id": r[0], "kind": r[1], "name": r[2],
                        "file": r[3], "line": r[4], "repo": r[5],
                        "extra": _safe_json(r[6]), "hop": hop + 1,
                        "edge_kind": e["kind"],
                    })
                frontier.append((e["dst_id"], hop + 1))
        return out

    def stats(self) -> dict:
        if self._conn is None:
            try:
                self.initialize()
            except Exception:
                return {"backend": self.backend_name, "nodes": 0, "edges": 0}
        try:
            n_res = self._conn.execute("MATCH (n:node) RETURN COUNT(*) AS c")
            e_res = self._conn.execute("MATCH ()-[e:edge]->() RETURN COUNT(*) AS c")
            n = e = 0
            if n_res.has_next():
                n = int(n_res.get_next()[0])
            if e_res.has_next():
                e = int(e_res.get_next()[0])
            return {"backend": self.backend_name, "nodes": n, "edges": e}
        except Exception:
            return {"backend": self.backend_name, "nodes": 0, "edges": 0}


def _result_to_dicts(result) -> list[dict]:
    out: list[dict] = []
    try:
        while result.has_next():
            r = result.get_next()
            out.append({
                "id": r[0], "kind": r[1], "name": r[2], "file": r[3],
                "line": r[4], "repo": r[5], "extra": _safe_json(r[6]),
            })
    except Exception:
        pass
    return out


def _edges_to_dicts(result) -> list[dict]:
    out: list[dict] = []
    try:
        while result.has_next():
            r = result.get_next()
            out.append({
                "src_id": r[0], "dst_id": r[1], "kind": r[2],
                "line": r[3], "repo": r[4], "weight": r[5],
            })
    except Exception:
        pass
    return out


def _safe_json(s: str) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}
