"""SQLite implementation of the multi-graph backend.

Schema:

    nodes(id PK, kind, name, file, line, repo, extra_json)
    edges(src_id, dst_id, kind, line, repo, weight)

Indexed for the access patterns we actually use:
  - find_definitions      : nodes by (kind='Symbol', name)
  - edges_into / edges_out: edges by (dst_id|src_id, kind)
  - delete_file           : nodes & edges by file path

Multi-hop traversal is a Python BFS — fine for k≤3 with hop budgets.
For deeper traversal at scale, use the Kuzu backend.
"""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from src.code_graph.backend import GraphBackend, GraphEdge, GraphNode


class SqliteBackend(GraphBackend):
    backend_name = "sqlite"

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    file TEXT NOT NULL DEFAULT '',
                    line INTEGER NOT NULL DEFAULT 0,
                    repo TEXT NOT NULL DEFAULT '',
                    extra_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_kind_name ON nodes(kind, name);
                CREATE INDEX IF NOT EXISTS idx_nodes_file      ON nodes(file);

                CREATE TABLE IF NOT EXISTS edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    kind   TEXT NOT NULL,
                    line   INTEGER NOT NULL DEFAULT 0,
                    repo   TEXT NOT NULL DEFAULT '',
                    weight REAL NOT NULL DEFAULT 1.0
                );
                CREATE INDEX IF NOT EXISTS idx_edges_dst  ON edges(dst_id, kind);
                CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges(src_id, kind);
                CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
            """)

    def close(self) -> None:
        # Connections are per-call; nothing global to release.
        return

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM nodes")
            c.execute("DELETE FROM edges")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [
            (n.id, n.kind, n.name, n.file, n.line, n.repo, json.dumps(n.extra))
            for n in nodes
        ]
        if not rows:
            return
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO nodes(id,kind,name,file,line,repo,extra_json) "
                "VALUES (?,?,?,?,?,?,?)",
                rows,
            )

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        rows = [
            (e.src_id, e.dst_id, e.kind, e.line, e.repo, e.weight)
            for e in edges
        ]
        if not rows:
            return
        with self._conn() as c:
            c.executemany(
                "INSERT INTO edges(src_id,dst_id,kind,line,repo,weight) VALUES (?,?,?,?,?,?)",
                rows,
            )

    def delete_file(self, file_path: str) -> int:
        with self._conn() as c:
            # Drop nodes owned by this file plus their edges.
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM nodes WHERE file = ?", (file_path,)
            ).fetchall()]
            n = c.execute("DELETE FROM nodes WHERE file = ?", (file_path,)).rowcount or 0
            if ids:
                placeholders = ",".join("?" * len(ids))
                c.execute(
                    f"DELETE FROM edges WHERE src_id IN ({placeholders}) "
                    f"OR dst_id IN ({placeholders})",
                    ids + ids,
                )
            return n

    def find_definitions(self, name: str, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,kind,name,file,line,repo,extra_json FROM nodes "
                "WHERE kind='Symbol' AND name = ? LIMIT ?",
                (name, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def edges_into(self, dst_id: str, kind: str | None = None,
                   limit: int = 100) -> list[dict]:
        sql = ("SELECT src_id, dst_id, kind, line, repo, weight "
               "FROM edges WHERE dst_id = ?")
        params: list = [dst_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def edges_out(self, src_id: str, kind: str | None = None,
                  limit: int = 100) -> list[dict]:
        sql = ("SELECT src_id, dst_id, kind, line, repo, weight "
               "FROM edges WHERE src_id = ?")
        params: list = [src_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def multi_hop(
        self,
        seed_id: str,
        kinds: list[str] | None,
        max_hops: int,
        max_results: int,
    ) -> list[dict]:
        """BFS up to ``max_hops`` from ``seed_id``. Edges of any kind in
        ``kinds`` are followed; ``None`` follows everything. Returns
        deduplicated node dicts ordered by hop distance, then weight.
        """
        if max_hops <= 0:
            return []
        seen: set[str] = {seed_id}
        out: list[tuple[int, float, dict]] = []  # (hop, neg_weight_sum, node_dict)
        frontier: deque[tuple[str, int, float]] = deque([(seed_id, 0, 0.0)])
        with self._conn() as c:
            while frontier and len(out) < max_results * 4:
                node_id, hop, weight_sum = frontier.popleft()
                if hop >= max_hops:
                    continue
                # Outgoing edges only — keeps traversal directional and
                # makes "callers of X" different from "callees of X".
                params: list = [node_id]
                kind_clause = ""
                if kinds:
                    placeholders = ",".join("?" * len(kinds))
                    kind_clause = f" AND kind IN ({placeholders})"
                    params.extend(kinds)
                rows = c.execute(
                    f"SELECT src_id,dst_id,kind,line,repo,weight FROM edges "
                    f"WHERE src_id = ?{kind_clause} LIMIT 200",
                    params,
                ).fetchall()
                for r in rows:
                    nxt = r["dst_id"]
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    nrow = c.execute(
                        "SELECT id,kind,name,file,line,repo,extra_json "
                        "FROM nodes WHERE id = ?", (nxt,),
                    ).fetchone()
                    if nrow is None:
                        continue
                    nd = _row_to_dict(nrow)
                    nd["hop"] = hop + 1
                    nd["edge_kind"] = r["kind"]
                    out.append((hop + 1, -(weight_sum + r["weight"]), nd))
                    frontier.append((nxt, hop + 1, weight_sum + r["weight"]))
        out.sort(key=lambda t: (t[0], t[1]))
        return [nd for _, _, nd in out[:max_results]]

    def stats(self) -> dict:
        if not self.db_path.exists():
            return {"backend": self.backend_name, "nodes": 0, "edges": 0}
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
            e = c.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
            by_kind = {}
            for r in c.execute("SELECT kind, COUNT(*) c FROM edges GROUP BY kind").fetchall():
                by_kind[r["kind"]] = r["c"]
        return {
            "backend": self.backend_name,
            "nodes": n,
            "edges": e,
            "edges_by_kind": by_kind,
        }


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if "extra_json" in d:
        try:
            d["extra"] = json.loads(d.pop("extra_json"))
        except Exception:
            d["extra"] = {}
    return d
