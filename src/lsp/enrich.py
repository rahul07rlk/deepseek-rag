"""LSP-driven graph enrichment.

After the heuristic graph is built (regex defs + tree-sitter refs), this
pass walks the document symbols emitted by the LSP server for each
indexable file and asks ``textDocument/references`` for every defined
symbol. The returned locations are ground-truth callers — typed,
inheritance-aware, scope-aware — and are inserted as high-weight CALLS
edges that override the heuristic ones.

Cost: roughly one ``references`` call per symbol per file. Pyright +
tsserver handle thousands of symbols in seconds; gopls and rust-analyzer
need warm-up time on first invocation. This pass is therefore optional
and runs asynchronously after the cheap regex/tree-sitter graph is
already serving queries.

Toggle with ``LSP_ENRICH_ENABLED``. Defaults to ``true`` only when at
least one LSP server is detected on PATH.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from src.code_graph import get_engine
from src.code_graph.backend import GraphEdge
from src.code_graph.extract import file_id, symbol_id
from src.config import (
    INDEX_DIR,
    INDEXED_EXTENSIONS,
    IGNORED_DIRS,
    IGNORED_FILENAMES,
    IGNORED_SUFFIXES,
    MAX_FILE_BYTES,
    REPO_PATHS,
)
from src.lsp.manager import get_manager
from src.lsp.servers import installed_servers
from src.utils.logger import get_logger

logger = get_logger("lsp_enrich", "indexer.log")

_LSP_EDGE_WEIGHT = 2.0  # higher than heuristic CALLS (1.0) so rerank prefers


def lsp_enabled() -> bool:
    if os.getenv("LSP_ENRICH_ENABLED", "").lower() == "false":
        return False
    return bool(installed_servers())


def enrich(max_files: int | None = None,
            per_file_timeout: float = 4.0) -> dict:
    """Walk every indexable file across REPO_PATHS, ask its LSP server
    for documentSymbol → references, write ground-truth CALLS edges.

    Returns a stats dict.
    """
    if not lsp_enabled():
        return {"enabled": False, "reason": "no LSP servers installed"}

    mgr = get_manager()
    engine = get_engine()
    backend = engine.backend

    files = _walk_indexable()
    if max_files is not None:
        files = files[:max_files]

    edges_added = 0
    files_processed = 0
    skipped = 0
    t0 = time.time()
    pending: list[GraphEdge] = []

    for fp, repo_name in files:
        files_processed += 1
        try:
            symbols = mgr.symbols(fp)
        except Exception:
            symbols = []
        if not symbols:
            skipped += 1
            continue
        fid = file_id(str(fp).replace("\\", "/"))
        for sym in symbols:
            name = sym.get("name", "")
            line = int(sym.get("line", 0))
            ch = int(sym.get("character", 0))
            if not name:
                continue
            top_name = name.split(".")[-1]
            try:
                refs = mgr.refs(fp, line, ch, include_decl=False)
            except Exception:
                continue
            if not refs:
                continue
            target_id = symbol_id(str(fp).replace("\\", "/"), top_name)
            for r in refs:
                ref_path = (r.get("path") or "").replace("\\", "/")
                if not ref_path:
                    continue
                pending.append(GraphEdge(
                    src_id=file_id(ref_path),
                    dst_id=target_id,
                    kind="CALLS",
                    line=int(r.get("line") or 0) + 1,
                    repo=repo_name,
                    weight=_LSP_EDGE_WEIGHT,
                ))
            if len(pending) >= 5_000:
                backend.upsert_edges(pending)
                edges_added += len(pending)
                pending = []
        # Bound per-file: if we've spent too long, drop close & move on.
        if (time.time() - t0) / max(1, files_processed) > per_file_timeout:
            logger.info("LSP enrichment running slow; will continue but log it.")

    if pending:
        backend.upsert_edges(pending)
        edges_added += len(pending)

    stats = {
        "enabled": True,
        "files_processed": files_processed,
        "files_skipped": skipped,
        "edges_added": edges_added,
        "elapsed_s": round(time.time() - t0, 2),
        "servers": mgr.installed(),
    }
    logger.info(f"LSP enrichment: {stats}")

    # Persist a marker so we can show this in /stats.
    try:
        marker = INDEX_DIR / "code_graph" / "lsp_enrich.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        import json
        marker.write_text(json.dumps(stats), encoding="utf-8")
    except Exception:
        pass

    return stats


def _walk_indexable() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for repo in REPO_PATHS:
        for p in repo.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel_parts = p.relative_to(repo).parts
            except ValueError:
                continue
            if any(part in IGNORED_DIRS for part in rel_parts):
                continue
            if p.suffix not in INDEXED_EXTENSIONS:
                continue
            if p.name in IGNORED_FILENAMES:
                continue
            if any(p.name.lower().endswith(s) for s in IGNORED_SUFFIXES):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size == 0 or size > MAX_FILE_BYTES:
                continue
            out.append((p, repo.name))
    return out


if __name__ == "__main__":
    print(enrich())
