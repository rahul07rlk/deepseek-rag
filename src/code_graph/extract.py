"""Graph extraction — turn source files into (nodes, edges).

Definitions and imports come from regex patterns (cheap and stable for
declaration keywords across most languages). References, inheritance,
and call relationships come from tree-sitter when available.

Test→code edges are heuristic for now (test file name matches a code
file name, or test imports a code module). LSP integration will replace
the heuristic with ground truth where supported.

Git lineage edges (TOUCHED_IN) are added by the indexer using the
``recent_change_files`` set so the graph knows which files churned
recently — used for "what's the latest change to this area?" queries.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.code_graph.backend import GraphEdge, GraphNode

# ── Reuse the regex set from the legacy symbol_graph but extend it. ──────────
_LANG_BY_EXT = {
    ".py": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx",
    ".go": "go", ".rs": "rust",
    ".java": "java", ".cs": "csharp", ".kt": "kotlin",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".php": "php", ".rb": "ruby", ".swift": "swift",
    ".scala": "scala",
}

_DEF_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
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
    "cpp": [
        ("function", re.compile(r"^\s*(?:[\w:<>*&]+\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{", re.M)),
        ("class",    re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)),
    ],
}
_DEF_PATTERNS["javascript"] = _DEF_PATTERNS["typescript"]
_DEF_PATTERNS["tsx"] = _DEF_PATTERNS["typescript"]
_DEF_PATTERNS["jsx"] = _DEF_PATTERNS["typescript"]
_DEF_PATTERNS["csharp"] = _DEF_PATTERNS["java"]
_DEF_PATTERNS["kotlin"] = _DEF_PATTERNS["java"]
_DEF_PATTERNS["c"] = _DEF_PATTERNS["cpp"]

_IMPORT_PATTERNS: dict[str, re.Pattern] = {
    "python":     re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))", re.M),
    "typescript": re.compile(r"""^\s*(?:import|export)\s+(?:.*?from\s+)?["']([^"']+)["']""", re.M),
    "go":         re.compile(r'\bimport\s+(?:\(([^)]*)\)|"([^"]+)")', re.M | re.S),
    "rust":       re.compile(r"^\s*use\s+([\w:]+)", re.M),
    "java":       re.compile(r"^\s*import\s+([\w.]+);", re.M),
    "csharp":     re.compile(r"^\s*using\s+([\w.]+);", re.M),
    "kotlin":     re.compile(r"^\s*import\s+([\w.]+)", re.M),
    "cpp":        re.compile(r"""^\s*#\s*include\s+[<"]([^>"]+)[>"]""", re.M),
}
_IMPORT_PATTERNS["javascript"] = _IMPORT_PATTERNS["typescript"]
_IMPORT_PATTERNS["tsx"] = _IMPORT_PATTERNS["typescript"]
_IMPORT_PATTERNS["jsx"] = _IMPORT_PATTERNS["typescript"]
_IMPORT_PATTERNS["c"] = _IMPORT_PATTERNS["cpp"]

# Inheritance / implementation patterns — coarse but useful for boost signals.
_INHERIT_PATTERNS: dict[str, re.Pattern] = {
    "python":     re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]+)\)", re.M),
    "typescript": re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+extends\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M),
    "java":       re.compile(r"^\s*(?:public|private|protected)?\s*(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\s+extends\s+([A-Za-z_][A-Za-z0-9_]*)", re.M),
}
_INHERIT_PATTERNS["javascript"] = _INHERIT_PATTERNS["typescript"]
_INHERIT_PATTERNS["tsx"] = _INHERIT_PATTERNS["typescript"]


def file_id(path: str) -> str:
    return f"file:{str(path).replace(chr(92), '/')}"


def symbol_id(file: str, name: str) -> str:
    return f"sym:{str(file).replace(chr(92), '/')}::{name}"


def detect_language(path: Path) -> str | None:
    return _LANG_BY_EXT.get(path.suffix.lower())


def extract(
    file_path: Path,
    repo: str,
    recent_change_set: set[str] | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Return (nodes, edges) extracted from a single source file.

    Always emits a ``File`` node. Emits ``Symbol`` nodes for every
    declaration found, and edges for IMPORTS / DEFINES / INHERITS /
    CALLS / TESTS where extractable.
    """
    lang = detect_language(file_path)
    if lang is None:
        return [], []
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], []

    file_str = str(file_path).replace("\\", "/")
    fid = file_id(file_str)
    nodes: list[GraphNode] = [
        GraphNode(id=fid, kind="File", name=file_path.name,
                  file=file_str, line=0, repo=repo)
    ]
    edges: list[GraphEdge] = []

    # ── Definitions → Symbol nodes + DEFINES edges. ──────────────────────────
    sym_id_by_name: dict[str, str] = {}
    for kind, pat in _DEF_PATTERNS.get(lang, []):
        for m in pat.finditer(text):
            name = m.group(1)
            if not name:
                continue
            line = text.count("\n", 0, m.start()) + 1
            sid = symbol_id(file_str, name)
            sym_id_by_name[name] = sid
            nodes.append(GraphNode(
                id=sid, kind="Symbol", name=name,
                file=file_str, line=line, repo=repo,
                extra={"decl_kind": kind, "language": lang},
            ))
            edges.append(GraphEdge(
                src_id=fid, dst_id=sid, kind="DEFINES",
                line=line, repo=repo,
            ))

    # ── Imports → IMPORTS edges (target node may not exist yet; we still
    # record the edge so callers can resolve later via name).
    imp_pat = _IMPORT_PATTERNS.get(lang)
    if imp_pat is not None:
        for m in imp_pat.finditer(text):
            target = next((g for g in m.groups() if g), None)
            if not target:
                continue
            target = target.strip()
            line = text.count("\n", 0, m.start()) + 1
            target_node_id = f"module:{target}"
            # Create a placeholder Module node so the edge resolves.
            nodes.append(GraphNode(
                id=target_node_id, kind="Module", name=target,
                file="", line=0, repo=repo,
            ))
            edges.append(GraphEdge(
                src_id=fid, dst_id=target_node_id, kind="IMPORTS",
                line=line, repo=repo,
            ))

    # ── Inheritance → INHERITS edges.
    inh_pat = _INHERIT_PATTERNS.get(lang)
    if inh_pat is not None:
        for m in inh_pat.finditer(text):
            child = m.group(1)
            parents_str = m.group(2) or ""
            child_id = sym_id_by_name.get(child)
            if not child_id:
                continue
            for parent in re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", parents_str):
                # Parent may not be in this file — record by-name placeholder.
                parent_id = f"sym:?::{parent}"
                edges.append(GraphEdge(
                    src_id=child_id, dst_id=parent_id, kind="INHERITS",
                    repo=repo,
                ))

    # ── References → CALLS edges (best-effort via tree-sitter). ──────────────
    try:
        from src.tree_sitter_chunker import extract_references_with_tree_sitter
        ts_refs = extract_references_with_tree_sitter(text, file_path.suffix) or []
    except Exception:
        ts_refs = []

    # We only emit CALLS edges when the referenced name has a definition
    # somewhere we can hint at. Use a "by-name" placeholder so the resolver
    # can connect them after all files are processed.
    seen_calls: set[tuple[str, str]] = set()
    for sym, line in ts_refs:
        if not sym:
            continue
        # Find the enclosing symbol (the most recent definition before `line`).
        enclosing = _enclosing_symbol(sym_id_by_name, text, line)
        src = enclosing or fid
        dst = f"sym:?::{sym}"
        if (src, dst) in seen_calls:
            continue
        seen_calls.add((src, dst))
        edges.append(GraphEdge(
            src_id=src, dst_id=dst, kind="CALLS",
            line=line, repo=repo,
        ))

    # ── Test → code linkage (heuristic). ────────────────────────────────────
    if _is_test_file(file_path):
        # The file under test is usually named without "test_" prefix /
        # "_test" / ".test." suffix; the test imports often cite the
        # production module. We add TESTS edges for every IMPORTS edge
        # this file emits — kept loose intentionally; LSP can replace.
        for e in edges:
            if e.kind == "IMPORTS" and e.src_id == fid:
                edges.append(GraphEdge(
                    src_id=fid, dst_id=e.dst_id, kind="TESTS",
                    line=e.line, repo=repo, weight=0.5,
                ))

    # ── TOUCHED_IN edges from git churn (added later by the indexer). ────────
    if recent_change_set and str(file_path.resolve()) in recent_change_set:
        # Mark the File node as recently touched. We don't have a commit
        # node yet — the simplest representation is an extra attribute on
        # the File node; ``CALL_GRAPH`` checks ``extra.touched_in_recent``.
        nodes[0].extra["touched_in_recent"] = True

    return nodes, edges


def _enclosing_symbol(
    sym_id_by_name: dict[str, str],
    source: str,
    line: int,
) -> str | None:
    """Best-effort: return the symbol id whose definition is closest to and
    above ``line``. Naive but works without a full AST walk."""
    # Walk symbols in order of declaration line and pick the last one
    # whose start_line <= line.
    best: tuple[int, str] | None = None
    line_offset = 0
    pat = re.compile(r"^[ \t]*(?:def|class|function|fn|func|interface|type)\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.M)
    for m in pat.finditer(source):
        decl_line = source.count("\n", 0, m.start()) + 1
        if decl_line > line:
            break
        sid = sym_id_by_name.get(m.group(1))
        if sid is not None:
            best = (decl_line, sid)
    return best[1] if best else None


def _is_test_file(path: Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(token in s for token in (
        "/tests/", "/test/", "/__tests__/", "/spec/", "/specs/", "/e2e/",
        ".test.", ".spec.", "_test.go", "test_",
    ))


def collect_recent_changes(repos: list[Path], lookback: int = 30) -> set[str]:
    """Files touched in the last N commits across all repos. Used to
    stamp ``TOUCHED_IN`` lineage on File nodes."""
    out: set[str] = set()
    for repo in repos:
        try:
            r = subprocess.run(
                ["git", "-C", str(repo), "log",
                 f"-{lookback}", "--name-only", "--pretty=format:"],
                capture_output=True, text=True, encoding='utf-8', timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line:
                        out.add(str((repo / line).resolve()))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return out
