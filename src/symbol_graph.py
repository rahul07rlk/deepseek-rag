"""Symbol graph: per-file definitions + cross-file references via SQLite.

Pure-text retrieval can't answer the questions developers actually ask:

  - "who calls ``resolveHandle``?"
  - "what implements ``IUserRepo``?"
  - "what files import from ``services/auth.ts``?"

Tree-sitter already gives us the AST at chunk time; this module stores
(symbol, file, kind, line) tuples in a tiny SQLite database alongside the
FAISS index, then exposes targeted lookups for the agentic tool layer.

Schema (kept deliberately minimal — 3 tables, no joins on the hot path):

  definitions(symbol, file, kind, start_line, end_line, repo)
  references(symbol, file, line, repo)
  imports(file, target, alias, line, repo)

Population is best-effort:
  - **Definitions / imports** use language-specific regex (cheap, reliable
    for the small surface of `def`/`function`/`class`/`interface`/`import`
    keywords).
  - **References** use the tree-sitter AST extractor in
    ``tree_sitter_chunker``. Only meaningful identifier nodes are recorded;
    comments and string literals are naturally excluded because tree-sitter
    doesn't emit identifier child nodes inside them. For languages
    tree-sitter doesn't cover (or on parse failure), refs are skipped
    entirely — we'd rather have no refs than the noise the legacy regex
    produced (every \\w+ token across the whole file, including JSDoc,
    log strings, error messages, …).

The agentic tools are robust to missing data — graph lookups that return
nothing trigger a fallback to grep/retrieve at the tool layer.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import (
    IGNORED_DIRS,
    IGNORED_FILENAMES,
    IGNORED_SUFFIXES,
    INDEX_DIR,
    INDEXED_EXTENSIONS,
    MAX_FILE_BYTES,
    REPO_PATHS,
)
from src.utils.logger import get_logger

logger = get_logger("symbol_graph", "indexer.log")

DB_PATH = INDEX_DIR / "symbol_graph.sqlite"

# ── Definition + import patterns (regex-based) ───────────────────────────────
# References are NOT regex-based any more — see ``_extract_file``. These
# patterns only cover declaration keywords (def/class/function/interface/
# import) which are small, syntactically narrow, and stable enough that
# tree-sitter would be overkill at the precision we need.
_DEF_RE_BY_LANG: dict[str, list[tuple[str, re.Pattern]]] = {
    "python": [
        ("function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)),
        ("class",    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.M)),
    ],
    "typescript": [
        ("function",  re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M)),
        ("class",     re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M)),
        ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M)),
        ("type",      re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M)),
        ("enum",      re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M)),
        ("const",     re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?(?:\(|function)", re.M)),
    ],
    "go": [
        ("function", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", re.M)),
        ("type",     re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)", re.M)),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
        ("struct",   re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
        ("enum",     re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
        ("trait",    re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
    ],
    "java": [
        ("class",     re.compile(r"^\s*(?:public|private|protected)?\s*(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
        ("interface", re.compile(r"^\s*(?:public|private|protected)?\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
    ],
}
_DEF_RE_BY_LANG["javascript"] = _DEF_RE_BY_LANG["typescript"]
_DEF_RE_BY_LANG["tsx"]        = _DEF_RE_BY_LANG["typescript"]
_DEF_RE_BY_LANG["jsx"]        = _DEF_RE_BY_LANG["typescript"]
_DEF_RE_BY_LANG["csharp"]     = _DEF_RE_BY_LANG["java"]
_DEF_RE_BY_LANG["kotlin"]     = _DEF_RE_BY_LANG["java"]

_IMPORT_RE_BY_LANG: dict[str, re.Pattern] = {
    # Captures the module path. Aliases are not extracted here — the path is
    # what matters for the call graph; aliases live with the references row.
    "python":     re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))", re.M),
    "typescript": re.compile(r"""^\s*(?:import|export)\s+(?:.*?from\s+)?["']([^"']+)["']""", re.M),
    "go":         re.compile(r"""^\s*import\s+(?:\([^)]*\)|"([^"]+)")""", re.M | re.S),
    "rust":       re.compile(r"^\s*use\s+([\w:]+)", re.M),
    "java":       re.compile(r"^\s*import\s+([\w.]+);", re.M),
}
_IMPORT_RE_BY_LANG["javascript"] = _IMPORT_RE_BY_LANG["typescript"]
_IMPORT_RE_BY_LANG["tsx"]        = _IMPORT_RE_BY_LANG["typescript"]
_IMPORT_RE_BY_LANG["jsx"]        = _IMPORT_RE_BY_LANG["typescript"]
_IMPORT_RE_BY_LANG["csharp"]     = re.compile(r"^\s*using\s+([\w.]+);", re.M)
_IMPORT_RE_BY_LANG["kotlin"]     = re.compile(r"^\s*import\s+([\w.]+)", re.M)

_LANG_BY_EXT = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx",
    ".go": "go", ".rs": "rust",
    ".java": "java", ".cs": "csharp", ".kt": "kotlin",
}


# ── DB lifecycle ──────────────────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS definitions (
            symbol     TEXT NOT NULL,
            file       TEXT NOT NULL,
            kind       TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line   INTEGER,
            repo       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_def_symbol ON definitions(symbol);
        CREATE INDEX IF NOT EXISTS idx_def_file   ON definitions(file);

        CREATE TABLE IF NOT EXISTS refs (
            symbol TEXT NOT NULL,
            file   TEXT NOT NULL,
            line   INTEGER NOT NULL,
            repo   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ref_symbol ON refs(symbol);
        CREATE INDEX IF NOT EXISTS idx_ref_file   ON refs(file);

        CREATE TABLE IF NOT EXISTS imports (
            file   TEXT NOT NULL,
            target TEXT NOT NULL,
            line   INTEGER,
            repo   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_imp_file   ON imports(file);
        CREATE INDEX IF NOT EXISTS idx_imp_target ON imports(target);
    """)
    return conn


@contextmanager
def _txn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Extraction ────────────────────────────────────────────────────────────────
def _is_indexable(fpath: Path) -> bool:
    if fpath.suffix not in INDEXED_EXTENSIONS:
        return False
    if fpath.name in IGNORED_FILENAMES:
        return False
    lower = fpath.name.lower()
    if any(lower.endswith(s) for s in IGNORED_SUFFIXES):
        return False
    try:
        size = fpath.stat().st_size
    except OSError:
        return False
    if size == 0 or size > MAX_FILE_BYTES:
        return False
    if any(part in IGNORED_DIRS for part in fpath.parts):
        return False
    return True


def _walk_indexable() -> list[tuple[Path, str]]:
    """Yield (path, repo_name) for every indexable source file."""
    out: list[tuple[Path, str]] = []
    for repo in REPO_PATHS:
        repo_name = repo.name
        for p in repo.rglob("*"):
            if not p.is_file():
                continue
            if any(part in IGNORED_DIRS for part in p.relative_to(repo).parts):
                continue
            if _is_indexable(p):
                out.append((p, repo_name))
    return out


def _extract_file(
    fpath: Path,
    repo: str,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Return (definitions, refs, imports) tuples ready for executemany().

    Definitions and imports use language-specific regex (cheap and reliable
    for those constructs).

    References use the **tree-sitter AST extractor** when the language is
    supported — only meaningful identifier nodes (call/member/type/field
    names) are captured, with comments and string literals naturally
    excluded because tree-sitter doesn't emit identifier child nodes inside
    them. For languages tree-sitter doesn't cover (or when parsing fails)
    refs are skipped entirely — we'd rather have no refs than the noise the
    legacy regex produced (every \\w+ token across the whole file, including
    inside JSDoc, log strings, error messages, …).
    """
    lang = _LANG_BY_EXT.get(fpath.suffix)
    if lang is None:
        return [], [], []
    try:
        text = fpath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], [], []

    file_str = str(fpath)
    defs: list[tuple] = []
    imps: list[tuple] = []
    refs: list[tuple] = []

    # Definitions.
    for kind, pat in _DEF_RE_BY_LANG.get(lang, []):
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            defs.append((m.group(1), file_str, kind, line, line, repo))

    # Imports.
    imp_pat = _IMPORT_RE_BY_LANG.get(lang)
    if imp_pat is not None:
        for m in imp_pat.finditer(text):
            target = next((g for g in m.groups() if g), None)
            if not target:
                continue
            line = text.count("\n", 0, m.start()) + 1
            imps.append((file_str, target.strip(), line, repo))

    # References — AST-driven, comment/string-aware.
    try:
        from src.tree_sitter_chunker import extract_references_with_tree_sitter
        ast_refs = extract_references_with_tree_sitter(text, fpath.suffix)
    except Exception as e:
        logger.debug(f"AST refs unavailable for {fpath.name}: {e}")
        ast_refs = None

    if ast_refs is not None:
        # Filter out the binding-site identifiers — for each definition row
        # extracted above, the def's own name on its def-line is not a
        # "reference" (it's the declaration). Without this exclusion,
        # find_callers would always include the file where the symbol is
        # defined, which we then re-filter at query time anyway — but
        # filtering here saves rows in the DB and makes graph_stats honest.
        def_keys = {(d[0], d[3]) for d in defs}
        for sym, line in ast_refs:
            if (sym, line) in def_keys:
                continue
            refs.append((sym, file_str, line, repo))

    return defs, refs, imps


def build_symbol_graph() -> dict:
    """Rebuild the entire graph from the configured REPO_PATHS."""
    files = _walk_indexable()
    if not files:
        logger.info("Symbol graph: no indexable files found.")
        return {"definitions": 0, "refs": 0, "imports": 0, "files": 0}

    all_defs: list[tuple] = []
    all_refs: list[tuple] = []
    all_imps: list[tuple] = []
    for fpath, repo in files:
        d, r, i = _extract_file(fpath, repo)
        all_defs.extend(d)
        all_refs.extend(r)
        all_imps.extend(i)

    with _txn() as conn:
        conn.execute("DELETE FROM definitions")
        conn.execute("DELETE FROM refs")
        conn.execute("DELETE FROM imports")
        conn.executemany(
            "INSERT INTO definitions(symbol,file,kind,start_line,end_line,repo) VALUES (?,?,?,?,?,?)",
            all_defs,
        )
        conn.executemany(
            "INSERT INTO refs(symbol,file,line,repo) VALUES (?,?,?,?)",
            all_refs,
        )
        conn.executemany(
            "INSERT INTO imports(file,target,line,repo) VALUES (?,?,?,?)",
            all_imps,
        )

    summary = {
        "definitions": len(all_defs),
        "refs": len(all_refs),
        "imports": len(all_imps),
        "files": len(files),
    }
    logger.info(f"Symbol graph built: {summary}")
    return summary


# ── Incremental updates (used by the watcher on per-file save) ───────────────
def _resolve_repo(fpath: Path) -> str:
    """Best-effort repo name lookup so a file save can map to its repo without
    a full walk."""
    for rp in REPO_PATHS:
        try:
            fpath.relative_to(rp)
            return rp.name
        except ValueError:
            continue
    return fpath.parts[0] if fpath.parts else "unknown"


def delete_for_file(fpath: Path | str) -> int:
    """Drop every definition/ref/import row for ``fpath``. Returns rows deleted.

    Cheap (indexed by file) — runs in microseconds even for large graphs.
    """
    file_str = str(fpath)
    if not DB_PATH.exists():
        return 0
    with _txn() as conn:
        d = conn.execute("DELETE FROM definitions WHERE file = ?", (file_str,)).rowcount
        r = conn.execute("DELETE FROM refs WHERE file = ?", (file_str,)).rowcount
        i = conn.execute("DELETE FROM imports WHERE file = ?", (file_str,)).rowcount
    return (d or 0) + (r or 0) + (i or 0)


def update_for_file(fpath: Path | str) -> dict:
    """Idempotent per-file refresh: delete old rows, re-extract, re-insert.

    Mirrors what ``index_single_file`` does for the FAISS index. Together they
    let the watcher keep the entire system (vectors + BM25 + symbol graph) in
    sync with disk without ever requiring a full reindex.
    """
    fpath = Path(fpath)
    file_str = str(fpath)
    if not _is_indexable(fpath) or not fpath.exists():
        # File deleted or no longer indexable — wipe its rows and stop.
        delete_for_file(file_str)
        return {"definitions": 0, "refs": 0, "imports": 0}

    repo = _resolve_repo(fpath)
    defs, refs, imps = _extract_file(fpath, repo)

    with _txn() as conn:
        conn.execute("DELETE FROM definitions WHERE file = ?", (file_str,))
        conn.execute("DELETE FROM refs WHERE file = ?", (file_str,))
        conn.execute("DELETE FROM imports WHERE file = ?", (file_str,))
        if defs:
            conn.executemany(
                "INSERT INTO definitions(symbol,file,kind,start_line,end_line,repo) VALUES (?,?,?,?,?,?)",
                defs,
            )
        if refs:
            conn.executemany(
                "INSERT INTO refs(symbol,file,line,repo) VALUES (?,?,?,?)",
                refs,
            )
        if imps:
            conn.executemany(
                "INSERT INTO imports(file,target,line,repo) VALUES (?,?,?,?)",
                imps,
            )
    return {"definitions": len(defs), "refs": len(refs), "imports": len(imps)}


# ── Query API used by the agentic tool layer ─────────────────────────────────
def find_definitions(symbol: str, limit: int = 50) -> list[dict]:
    if not symbol:
        return []
    with _txn() as conn:
        rows = conn.execute(
            "SELECT symbol, file, kind, start_line, end_line, repo "
            "FROM definitions WHERE symbol = ? LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def find_callers(symbol: str, limit: int = 50) -> list[dict]:
    """Files referencing `symbol`. Excludes the file(s) where `symbol` is
    defined so the result is genuinely about callers, not the def itself.
    """
    if not symbol:
        return []
    with _txn() as conn:
        def_files = {r["file"] for r in conn.execute(
            "SELECT file FROM definitions WHERE symbol = ?", (symbol,)
        ).fetchall()}
        rows = conn.execute(
            "SELECT symbol, file, line, repo FROM refs WHERE symbol = ? LIMIT ?",
            (symbol, max(limit * 4, 200)),
        ).fetchall()
    out: list[dict] = []
    seen_files: set[str] = set()
    for r in rows:
        if r["file"] in def_files:
            continue
        out.append(dict(r))
        seen_files.add(r["file"])
        if len(out) >= limit:
            break
    return out


def find_importers(target_substring: str, limit: int = 50) -> list[dict]:
    """Files whose import statements contain `target_substring` as a
    substring. Substring match handles both relative imports (./Foo) and
    package imports (services/auth)."""
    if not target_substring:
        return []
    with _txn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT file, target, line, repo FROM imports "
            "WHERE target LIKE ? LIMIT ?",
            (f"%{target_substring}%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


def neighbors_for_chunk(
    file: str,
    symbol: str = "",
    start_line: int = 1,
    end_line: int = 1,
    limit: int = 12,
) -> list[dict]:
    """Best-effort 1-hop graph neighbors for a retrieved chunk.

    Returns rows shaped as:
      {"relation": "caller"|"callee"|"importer", "file": str,
       "line": int, "symbol": str, "repo": str}

    The graph is intentionally approximate: refs are token-based, so this
    favors recall and leaves final relevance to token budget + reranking.
    """
    if not DB_PATH.exists():
        return []

    out: list[dict] = []
    seen: set[tuple[str, int, str, str]] = set()

    def add(relation: str, row, sym: str) -> None:
        if len(out) >= limit:
            return
        line = int(row["line"] or 1)
        key = (relation, row["file"], line, sym)
        if key in seen:
            return
        seen.add(key)
        out.append({
            "relation": relation,
            "file": row["file"],
            "line": line,
            "symbol": sym,
            "repo": row["repo"],
        })

    simple_symbol = (symbol or "").strip()
    if simple_symbol.startswith("class "):
        simple_symbol = simple_symbol.split(" ", 1)[1]
    if "." in simple_symbol:
        simple_symbol = simple_symbol.rsplit(".", 1)[-1]
    if simple_symbol.startswith("<"):
        simple_symbol = ""

    with _txn() as conn:
        if simple_symbol:
            caller_rows = conn.execute(
                "SELECT symbol,file,line,repo FROM refs "
                "WHERE symbol = ? AND file != ? LIMIT ?",
                (simple_symbol, file, max(limit * 3, 50)),
            ).fetchall()
            for row in caller_rows:
                add("caller", row, simple_symbol)
                if len(out) >= limit:
                    return out

        callee_rows = conn.execute(
            """
            SELECT DISTINCT r.symbol AS symbol,
                            d.file AS file,
                            d.start_line AS line,
                            d.repo AS repo
            FROM refs r
            JOIN definitions d ON d.symbol = r.symbol
            WHERE r.file = ?
              AND r.line BETWEEN ? AND ?
              AND d.file != ?
            LIMIT ?
            """,
            (file, start_line, end_line, file, max(limit * 3, 50)),
        ).fetchall()
        for row in callee_rows:
            add("callee", row, row["symbol"])
            if len(out) >= limit:
                return out

        stem = Path(file).stem
        if stem:
            importer_rows = conn.execute(
                "SELECT DISTINCT file,target,line,repo FROM imports "
                "WHERE file != ? AND target LIKE ? LIMIT ?",
                (file, f"%{stem}%", max(limit * 2, 30)),
            ).fetchall()
            for row in importer_rows:
                add("importer", row, row["target"])
                if len(out) >= limit:
                    return out

    return out


def graph_stats() -> dict:
    if not DB_PATH.exists():
        return {"definitions": 0, "refs": 0, "imports": 0}
    with _txn() as conn:
        d = conn.execute("SELECT COUNT(*) c FROM definitions").fetchone()["c"]
        r = conn.execute("SELECT COUNT(*) c FROM refs").fetchone()["c"]
        i = conn.execute("SELECT COUNT(*) c FROM imports").fetchone()["c"]
    return {"definitions": d, "refs": r, "imports": i}


# ── Multi-graph compat layer ─────────────────────────────────────────────────
# When GRAPH_BACKEND is set (kuzu / sqlite via code_graph), redirect the
# legacy callers (rag_engine, agentic, watcher) to the new engine. Functions
# above are kept verbatim for backward compatibility — they remain the
# fallback when the new graph isn't built yet.
def _use_new_graph() -> bool:
    import os
    return os.getenv("USE_CODE_GRAPH", "true").lower() == "true"


_legacy_neighbors_for_chunk = neighbors_for_chunk
_legacy_find_definitions = find_definitions
_legacy_find_callers = find_callers
_legacy_find_importers = find_importers


def neighbors_for_chunk(file: str, symbol: str = "", start_line: int = 1,
                         end_line: int = 1, limit: int = 12) -> list[dict]:  # noqa: F811
    if _use_new_graph():
        try:
            from src.code_graph import neighbors_for_chunk as _new
            return _new(file, symbol, start_line, end_line, limit)
        except Exception as e:
            logger.debug(f"code_graph fallback to legacy: {e}")
    return _legacy_neighbors_for_chunk(file, symbol, start_line, end_line, limit)


def find_definitions(symbol: str, limit: int = 50) -> list[dict]:  # noqa: F811
    if _use_new_graph():
        try:
            from src.code_graph import find_definitions as _new
            return _new(symbol, limit)
        except Exception:
            pass
    return _legacy_find_definitions(symbol, limit)


def find_callers(symbol: str, limit: int = 50) -> list[dict]:  # noqa: F811
    if _use_new_graph():
        try:
            from src.code_graph import find_callers as _new
            return _new(symbol, limit)
        except Exception:
            pass
    return _legacy_find_callers(symbol, limit)


def find_importers(target_substring: str, limit: int = 50) -> list[dict]:  # noqa: F811
    if _use_new_graph():
        try:
            from src.code_graph import find_importers as _new
            return _new(target_substring, limit)
        except Exception:
            pass
    return _legacy_find_importers(target_substring, limit)


if __name__ == "__main__":
    print(build_symbol_graph())
