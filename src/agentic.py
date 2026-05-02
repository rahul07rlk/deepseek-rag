"""Agentic retrieval tools — DeepSeek tool-use loop.

Exposes retrieval primitives as OpenAI/DeepSeek-format tools the LLM can call
iteratively. Modeled on Cursor / Claude Code / Aider: the model decides what
to look at next instead of getting a one-shot stuffed prompt.

Tools provided
--------------
- ``retrieve``       hybrid retrieval over the existing FAISS+BM25 index
- ``read_file``      bounded slice of any indexed file (1-indexed, inclusive)
- ``grep``           regex over the corpus, returns path:line:match snippets
- ``find_symbol``    definitions of a symbol (file, kind, line)
- ``find_callers``   files referencing a symbol (excluding its own def site)
- ``find_importers`` files whose import statements match a substring
- ``repo_map``       structural overview filtered by an optional query

The tool layer is read-only by design — none of these mutate the workspace.
The proxy server is responsible for the round-trip loop with DeepSeek; this
module owns the schema + the dispatcher.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import (
    AGENTIC_MAX_TOOL_TURNS,
    AGENTIC_SEED_TOKEN_BUDGET,
    AGENTIC_TOOL_GREP_MAX_HITS,
    AGENTIC_TOOL_MEMOIZATION,
    AGENTIC_TOOL_READ_MAX_LINES,
    IGNORED_DIRS,
    IGNORED_FILENAMES,
    IGNORED_SUFFIXES,
    INDEXED_EXTENSIONS,
    REPO_PATHS,
)
from src.utils.logger import get_logger

logger = get_logger("agentic", "proxy.log")


# ── Tool schemas (OpenAI / DeepSeek tool-use format) ─────────────────────────
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Hybrid (BM25 + vector) retrieval over the indexed codebase. "
                "Returns code blocks with file paths and line numbers. Use "
                "this for natural-language questions ('how does auth work') "
                "or when you need broader semantic context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "top_k": {"type": "integer", "description": "Max blocks to return.", "default": 12},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a slice of a file from disk by absolute or repo-relative path. "
                "Use after find_symbol/grep to inspect specific code at exact lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path or repo-relative path."},
                    "start_line": {"type": "integer", "description": "1-indexed start line.", "default": 1},
                    "end_line": {"type": "integer", "description": "1-indexed end line (inclusive). 0 = end of file.", "default": 0},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": (
                "Read MANY file slices in ONE tool call. Strongly preferred over "
                "multiple read_file calls — each round-trip costs an LLM turn, "
                "this batches them. Each item has the same fields as read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of {path, start_line?, end_line?} objects.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "start_line": {"type": "integer", "default": 1},
                                "end_line": {"type": "integer", "default": 0},
                            },
                            "required": ["path"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Regex search across the indexed source tree. Returns "
                "path:line:match. Use this for exact strings (error messages, "
                "API endpoints, config keys) when you don't know which file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex (case-insensitive)."},
                    "path_glob": {"type": "string", "description": "Optional substring to filter file paths."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": (
                "Locate definitions of a symbol (function, class, interface, "
                "type, etc.) across the codebase. Use when the query mentions "
                "a specific identifier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Exact identifier name."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_callers",
            "description": (
                "Files that reference a symbol but don't define it. Use to "
                "trace impact of a change ('who calls foo?')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Exact identifier name."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_importers",
            "description": (
                "Files whose import statements contain the given substring. "
                "Use to find every place that imports a module or path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Substring of the import target (path or package)."},
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_map",
            "description": (
                "Compressed structural overview of the codebase: file paths, "
                "languages, sizes, top-level symbols. Pass an optional query "
                "to bias the ranking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional ranking query.", "default": ""},
                    "top_files": {"type": "integer", "description": "Max files to list.", "default": 30},
                },
            },
        },
    },
    # ── Multi-graph: implementations + multi-hop traversal ──────────────────
    {
        "type": "function",
        "function": {
            "name": "find_implementations",
            "description": (
                "Symbols inheriting from / implementing a given base class or "
                "interface. Use for type-hierarchy questions: 'who implements "
                "IUserRepo?', 'subclasses of BaseHandler?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Base class / interface name."},
                },
                "required": ["base"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_neighbors",
            "description": (
                "Multi-hop graph traversal from a seed file (and optional "
                "symbol). Follows CALLS / IMPORTS / INHERITS / TESTS edges up "
                "to max_hops. Use this when 1-hop callers/callees aren't "
                "enough — e.g. 'trace this request from handler to DB'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Seed file path."},
                    "symbol": {"type": "string", "description": "Optional starting symbol.", "default": ""},
                    "kinds": {
                        "type": "array",
                        "description": "Edge kinds to follow.",
                        "items": {"type": "string"},
                        "default": ["CALLS", "IMPORTS"],
                    },
                    "max_hops": {"type": "integer", "default": 2},
                    "max_results": {"type": "integer", "default": 25},
                },
                "required": ["file"],
            },
        },
    },
    # ── LSP: ground-truth defs/refs ─────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "lsp_definition",
            "description": (
                "Use a real language server (pyright/tsserver/gopls/rust-"
                "analyzer/clangd/etc) to find the EXACT definition of a "
                "symbol at a position. Use this when find_symbol returns "
                "ambiguous matches or when types matter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer", "description": "1-indexed line."},
                    "character": {"type": "integer", "description": "0-indexed column.", "default": 0},
                },
                "required": ["file", "line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_references",
            "description": (
                "Use a real language server to find ALL references of a "
                "symbol (typed, scope-aware). Better than find_callers when "
                "the same name is reused in different contexts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "character": {"type": "integer", "default": 0},
                },
                "required": ["file", "line"],
            },
        },
    },
    # ── Sandbox: verify a code snippet before returning ─────────────────────
    {
        "type": "function",
        "function": {
            "name": "verify_code",
            "description": (
                "Run static checks (syntax, types, lint, compile) on a code "
                "snippet using the appropriate toolchain. Use this BEFORE "
                "returning generated code so you catch hallucinated imports "
                "or type errors. Returns pass/fail per check."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "python|typescript|go|rust|java|cpp|c"},
                    "code": {"type": "string", "description": "Source to check."},
                },
                "required": ["language", "code"],
            },
        },
    },
    # ── Git: recent changes / history ───────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "recent_changes",
            "description": (
                "Files modified in the last N git commits across all repos. "
                "Use for debugging questions where the answer likely lies in "
                "a recent change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback": {"type": "integer", "default": 20},
                },
            },
        },
    },
]


# ── Path helpers ──────────────────────────────────────────────────────────────
def _resolve_path(p: str) -> Path | None:
    """Accept absolute paths, repo-relative paths, and ``RepoName/sub/file.ts``."""
    if not p:
        return None
    cand = Path(p)
    if cand.is_absolute() and cand.exists():
        return cand
    norm = p.replace("\\", "/").lstrip("./")
    for repo in REPO_PATHS:
        # Strip the leading repo name if present.
        if norm.startswith(repo.name + "/"):
            tail = norm[len(repo.name) + 1 :]
            joined = repo / tail
            if joined.exists():
                return joined
        joined = repo / norm
        if joined.exists():
            return joined
    return None


def _is_skippable_file(p: Path) -> bool:
    if p.name in IGNORED_FILENAMES:
        return True
    lower = p.name.lower()
    if any(lower.endswith(s) for s in IGNORED_SUFFIXES):
        return True
    if any(part in IGNORED_DIRS for part in p.parts):
        return True
    return False


# ── Tool implementations ──────────────────────────────────────────────────────
def tool_retrieve(query: str, top_k: int = 12) -> str:
    """Call the existing one-shot retriever with a tighter cap so the agent
    can call it multiple times within the seed budget."""
    from src.rag_engine import retrieve  # avoid circular import at module load
    ctx, ntok, metas = retrieve(query, top_k=int(top_k), token_budget=20000)
    if not ctx:
        return "(no results)"
    return f"Retrieved {len(metas)} blocks (~{ntok} tokens):\n\n{ctx}"


def tool_read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    fpath = _resolve_path(path)
    if fpath is None:
        return f"Error: file not found: {path}"
    if _is_skippable_file(fpath):
        return f"Error: refusing to read filtered file: {fpath.name}"
    try:
        lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as e:
        return f"Error: {e}"
    total = len(lines)
    s = max(1, int(start_line))
    e = total if int(end_line) <= 0 else min(total, int(end_line))
    if s > total:
        return f"Error: start_line {s} > file length {total}"
    if e - s + 1 > AGENTIC_TOOL_READ_MAX_LINES:
        e = s + AGENTIC_TOOL_READ_MAX_LINES - 1
    body = "\n".join(lines[s - 1 : e])
    header = f"{fpath} (lines {s}-{e} of {total})"
    return f"{header}\n```\n{body}\n```"


def _grep_corpus(pattern: str, path_glob: str = "") -> list[tuple[str, int, str]]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Bad regex: {e}") from e
    glob_lower = path_glob.lower().replace("\\", "/") if path_glob else ""
    hits: list[tuple[str, int, str]] = []
    for repo in REPO_PATHS:
        for fpath in repo.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix not in INDEXED_EXTENSIONS:
                continue
            if _is_skippable_file(fpath):
                continue
            rel = str(fpath).replace("\\", "/").lower()
            if glob_lower and glob_lower not in rel:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    snippet = line.strip()
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "…"
                    hits.append((str(fpath), i, snippet))
                    if len(hits) >= AGENTIC_TOOL_GREP_MAX_HITS:
                        return hits
    return hits


def tool_grep(pattern: str, path_glob: str = "") -> str:
    try:
        hits = _grep_corpus(pattern, path_glob or "")
    except ValueError as e:
        return f"Error: {e}"
    if not hits:
        return "(no matches)"
    lines = [f"{f}:{ln}: {snip}" for f, ln, snip in hits]
    suffix = ""
    if len(hits) >= AGENTIC_TOOL_GREP_MAX_HITS:
        suffix = f"\n(truncated at {AGENTIC_TOOL_GREP_MAX_HITS} hits)"
    return "\n".join(lines) + suffix


def tool_find_symbol(symbol: str) -> str:
    from src.symbol_graph import find_definitions
    rows = find_definitions(symbol)
    if not rows:
        return f"No definitions found for `{symbol}`."
    out = [f"{len(rows)} definition(s) of `{symbol}`:"]
    for r in rows:
        out.append(f"- [{r['kind']}] {r['file']}:{r['start_line']} (repo={r['repo']})")
    return "\n".join(out)


def tool_find_callers(symbol: str) -> str:
    from src.symbol_graph import find_callers
    rows = find_callers(symbol)
    if not rows:
        return f"No callers found for `{symbol}`."
    out = [f"{len(rows)} call site(s) of `{symbol}`:"]
    for r in rows:
        out.append(f"- {r['file']}:{r['line']} (repo={r['repo']})")
    return "\n".join(out)


def tool_find_importers(target: str) -> str:
    from src.symbol_graph import find_importers
    rows = find_importers(target)
    if not rows:
        return f"No files import `{target}`."
    out = [f"{len(rows)} importer(s) of `{target}`:"]
    for r in rows:
        out.append(f"- {r['file']}:{r['line']}  imports `{r['target']}`")
    return "\n".join(out)


def tool_repo_map(query: str = "", top_files: int = 30) -> str:
    from src.repo_map import relevant_repo_map
    rendered, _ = relevant_repo_map(query or "", top_files=int(top_files))
    return rendered or "(repo map empty — run reindex)"


def tool_find_implementations(base: str) -> str:
    try:
        from src.code_graph import find_implementations
    except Exception:
        return "(graph unavailable)"
    rows = find_implementations(base)
    if not rows:
        return f"No implementations found for `{base}`."
    out = [f"{len(rows)} implementer(s) of `{base}`:"]
    for r in rows:
        sid = r.get("src_id", "")
        out.append(f"- {sid} (line {r.get('line', 0)})")
    return "\n".join(out)


def tool_graph_neighbors(
    file: str, symbol: str = "",
    kinds: list | None = None,
    max_hops: int = 2, max_results: int = 25,
) -> str:
    try:
        from src.code_graph import multi_hop_neighbors
    except Exception:
        return "(graph unavailable)"
    rows = multi_hop_neighbors(
        seed_file=file,
        seed_symbol=symbol or "",
        kinds=list(kinds) if kinds else None,
        max_hops=int(max_hops),
        max_results=int(max_results),
    )
    if not rows:
        return f"No graph neighbors within {max_hops} hops of {file}{':' + symbol if symbol else ''}."
    out = [f"{len(rows)} neighbor(s):"]
    for r in rows:
        out.append(
            f"- [hop={r.get('hop', '?')} via {r.get('edge_kind', '?')}] "
            f"{r.get('kind', '?')} {r.get('name', '?')} @ {r.get('file', '')}:{r.get('line', 0)}"
        )
    return "\n".join(out)


def tool_lsp_definition(file: str, line: int, character: int = 0) -> str:
    fpath = _resolve_path(file)
    if fpath is None:
        return f"Error: file not found: {file}"
    try:
        from src.lsp import get_manager
        mgr = get_manager()
        # LSP positions are 0-indexed; tool surface uses 1-indexed lines.
        defs = mgr.defs(fpath, max(0, int(line) - 1), int(character))
    except Exception as e:
        return f"LSP unavailable: {type(e).__name__}: {e}"
    if not defs:
        return "(no definition found via LSP — try find_symbol fallback)"
    out = ["LSP definitions:"]
    for d in defs:
        out.append(f"- {d.get('path', '?')}:{int(d.get('line', 0)) + 1}")
    return "\n".join(out)


def tool_lsp_references(file: str, line: int, character: int = 0) -> str:
    fpath = _resolve_path(file)
    if fpath is None:
        return f"Error: file not found: {file}"
    try:
        from src.lsp import get_manager
        mgr = get_manager()
        refs = mgr.refs(fpath, max(0, int(line) - 1), int(character))
    except Exception as e:
        return f"LSP unavailable: {type(e).__name__}: {e}"
    if not refs:
        return "(no references found via LSP — try find_callers fallback)"
    out = [f"LSP references ({len(refs)}):"]
    for r in refs:
        out.append(f"- {r.get('path', '?')}:{int(r.get('line', 0)) + 1}")
    return "\n".join(out)


def tool_verify_code(language: str, code: str) -> str:
    try:
        from src.sandbox import verify_code
    except Exception as e:
        return f"Sandbox unavailable: {type(e).__name__}: {e}"
    reports = verify_code(code, language)
    if not reports:
        return "(no checks ran)"
    out = []
    all_passed = True
    for r in reports:
        status = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        if not r.passed and not r.skipped:
            all_passed = False
        out.append(f"[{status}] {r.tool} ({r.duration_s:.2f}s)")
        if r.stderr.strip():
            out.append(r.stderr.strip()[:1500])
        if r.error:
            out.append(f"error: {r.error}")
    out.insert(0, f"Verification: {'OK' if all_passed else 'FAILED'}")
    return "\n".join(out)


def tool_recent_changes(lookback: int = 20) -> str:
    try:
        from src.code_graph.extract import collect_recent_changes
        from src.config import REPO_PATHS
    except Exception as e:
        return f"git unavailable: {type(e).__name__}: {e}"
    files = collect_recent_changes(REPO_PATHS, lookback=int(lookback))
    if not files:
        return f"No git-tracked changes in the last {lookback} commits."
    sample = sorted(files)[:80]
    out = [f"{len(files)} files changed in last {lookback} commits (showing first 80):"]
    out.extend(f"- {f}" for f in sample)
    return "\n".join(out)


def tool_read_files(items: list) -> str:
    """Batched read_file. Each item = {path, start_line?, end_line?}."""
    if not isinstance(items, list) or not items:
        return "Error: read_files requires a non-empty 'items' list"
    parts: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            parts.append(f"### Item {i+1}\nError: each item must be an object")
            continue
        path = item.get("path", "")
        start = item.get("start_line", 1)
        end = item.get("end_line", 0)
        body = tool_read_file(path, start, end)
        parts.append(f"### Item {i+1}: {path}\n{body}")
    return "\n\n---\n\n".join(parts)


# ── Dispatcher ────────────────────────────────────────────────────────────────
_DISPATCH = {
    "retrieve":             lambda a: tool_retrieve(a.get("query", ""), a.get("top_k", 12)),
    "read_file":            lambda a: tool_read_file(a.get("path", ""), a.get("start_line", 1), a.get("end_line", 0)),
    "read_files":           lambda a: tool_read_files(a.get("items", [])),
    "grep":                 lambda a: tool_grep(a.get("pattern", ""), a.get("path_glob", "")),
    "find_symbol":          lambda a: tool_find_symbol(a.get("symbol", "")),
    "find_callers":         lambda a: tool_find_callers(a.get("symbol", "")),
    "find_importers":       lambda a: tool_find_importers(a.get("target", "")),
    "repo_map":             lambda a: tool_repo_map(a.get("query", ""), a.get("top_files", 30)),
    "find_implementations": lambda a: tool_find_implementations(a.get("base", "")),
    "graph_neighbors":      lambda a: tool_graph_neighbors(
        a.get("file", ""), a.get("symbol", ""),
        a.get("kinds"), a.get("max_hops", 2), a.get("max_results", 25),
    ),
    "lsp_definition":       lambda a: tool_lsp_definition(
        a.get("file", ""), a.get("line", 1), a.get("character", 0),
    ),
    "lsp_references":       lambda a: tool_lsp_references(
        a.get("file", ""), a.get("line", 1), a.get("character", 0),
    ),
    "verify_code":          lambda a: tool_verify_code(
        a.get("language", ""), a.get("code", ""),
    ),
    "recent_changes":       lambda a: tool_recent_changes(a.get("lookback", 20)),
}


def _normalize_args_for_memo(name: str, args: dict) -> str:
    """Deterministic memo key. We re-serialize the dict so that
    ``{a:1,b:2}`` and ``{b:2,a:1}`` (semantically identical) hash the same."""
    try:
        # Sort keys so memo keys are order-independent.
        return f"{name}::{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    except (TypeError, ValueError):
        return f"{name}::{repr(args)}"


def dispatch(
    name: str,
    arguments_json: str,
    memo: dict | None = None,
) -> str:
    """Run a tool by name. Always returns a string (DeepSeek expects str
    content in tool messages). Errors are returned as plain text so the
    model can react instead of the loop crashing.

    Args:
        memo: when provided and AGENTIC_TOOL_MEMOIZATION is on, results are
            cached by ``(tool_name, normalized_args)`` for the duration of
            the agentic session. Q1's repeated read_file of the same path
            (observed in the proxy log) becomes O(1) on the second call.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool `{name}`"
    try:
        args: Any = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return f"Error: bad tool arguments JSON: {e}"
    if not isinstance(args, dict):
        return "Error: tool arguments must be a JSON object"

    memo_key: str | None = None
    if memo is not None and AGENTIC_TOOL_MEMOIZATION:
        memo_key = _normalize_args_for_memo(name, args)
        cached = memo.get(memo_key)
        if cached is not None:
            # Tag the cached return so the LLM sees this is a memo hit
            # (so it doesn't think the tool ran twice with the same result
            # by coincidence and re-trigger an investigation).
            return f"(cached from earlier in this session)\n{cached}"

    try:
        result = fn(args)
    except Exception as e:
        logger.exception(f"Tool {name} failed: {e}")
        return f"Error: tool `{name}` raised {type(e).__name__}: {e}"
    # Hard cap on tool-result size so a runaway grep doesn't blow up context.
    if len(result) > 24000:
        result = result[:24000] + f"\n…(truncated at 24000 chars)"
    if memo_key is not None:
        memo[memo_key] = result
    return result


# ── Repo scope detection ─────────────────────────────────────────────────────
def _detect_repo_scope(user_query: str) -> list[str]:
    """Return names of repos explicitly mentioned in the query.

    Used to inject a SCOPE constraint into the system prompt so the agent
    doesn't drift into unrelated repos when the user targets a specific one.
    E.g. "in deepseek-rag readme add content" → scoped to deepseek-rag only.

    Repos are checked longest-name-first so "aelvyris-backend" is consumed
    before "aelvyris" can false-positive as a substring match.
    """
    q_remaining = user_query.lower()
    scoped: list[str] = []
    # Longest name first prevents shorter names from matching inside longer ones.
    for rp in sorted(REPO_PATHS, key=lambda p: len(p.name), reverse=True):
        name = rp.name.lower()
        if name in q_remaining:
            scoped.append(rp.name)
            # Mask matched span so its prefix doesn't fire on a later iteration.
            q_remaining = q_remaining.replace(name, " " * len(name))
    return scoped


# ── Workspace layout hint (so the LLM emits Continue-resolvable paths) ───────
def _workspace_layout_section() -> str:
    """Tell the model where files live so its code suggestions carry paths
    Continue's "Apply" button can actually resolve.

    The IDE workspace root (the folder VS Code has open) is the COMMON parent
    of every indexed repo. Continue resolves a path quoted in a code block
    against that root: a bare ``src/utils/foo.ts`` tries to land at
    ``<workspace>/src/utils/foo.ts`` and fails when there is no such folder
    (multi-repo workspaces are exactly this case).

    To make Apply work for both edits AND new files, the model must:
      1. Use **workspace-relative** paths (include the repo folder name).
      2. Put the path in the **code-fence info string**, the format
         Continue's apply parser auto-detects:
            ```language path/relative/to/workspace.ext
      3. Use the same path on a "**File:**" header before the fence as a
         human-readable belt-and-suspenders cue.
    """
    if not REPO_PATHS:
        return ""

    # Heuristic: workspace root is the common parent of every indexed repo
    # (in the user's case, c:\My_Projects). When the repos don't share a
    # common parent, fall back to listing them individually — better to be
    # explicit than guess wrong.
    try:
        from os.path import commonpath
        common = commonpath([str(rp) for rp in REPO_PATHS])
    except (ValueError, OSError):
        common = ""

    workspace_root = common.replace("\\", "/") if common else ""
    repo_examples: list[str] = []
    for rp in REPO_PATHS[:6]:
        try:
            rel = rp.relative_to(common) if common else rp
            rel_str = str(rel).replace("\\", "/") if common else str(rp).replace("\\", "/")
        except ValueError:
            rel_str = str(rp).replace("\\", "/")
        repo_examples.append(f"  - `{rel_str}/`")
    repo_lines = "\n".join(repo_examples)

    # Build a concrete example using the first repo so the format is
    # unambiguous to the model.
    first_repo = repo_examples[0].strip("- `/ ") if repo_examples else "RepoName"
    example_path = f"{first_repo}/src/utils/example.ts"

    workspace_line = (
        f"The IDE workspace root is `{workspace_root}/`. Each indexed repo "
        "is a folder directly under it:\n"
    ) if workspace_root else (
        "Each indexed repo is a top-level folder in the IDE workspace:\n"
    )

    return (
        "\n## Workspace layout (CRITICAL — read before answering)\n"
        f"{workspace_line}"
        f"{repo_lines}\n\n"
        "The IDE's **Apply** button resolves the path quoted in your code "
        "block AGAINST THE WORKSPACE ROOT. A bare path like "
        "`src/utils/foo.ts` resolves to `<workspace-root>/src/utils/foo.ts` "
        "— which does NOT exist (no repo lives directly under the root) — "
        "so Apply fails with \"Could not resolve filepath to apply changes\" "
        "and the user CANNOT use your code. This is the most common failure "
        "mode and you must avoid it.\n\n"
        "### MANDATORY format for every code suggestion\n"
        "Every fenced code block MUST follow this **exact** structure — no "
        "exceptions. Both the header line AND the code fence info string are "
        "required; Continue reads the fence info string to locate the file, "
        "and the header gives the user a human-readable cue.\n\n"
        "**Editing an existing file:**\n"
        f"**File:** `{example_path}`\n"
        f"```typescript {example_path}\n"
        "// COMPLETE file content — every line, not a partial snippet\n"
        "```\n\n"
        "**Creating a brand-new file** (Continue 1.x will create it on disk "
        "automatically when Apply is clicked if the path does not yet exist):\n"
        f"**File (new):** `{example_path}`\n"
        f"```typescript {example_path}\n"
        "// COMPLETE new file content\n"
        "```\n\n"
        "### Rules — non-negotiable\n"
        "1. The path in the **header** and the path in the **fence info "
        "string** MUST be identical character-for-character. A mismatch "
        "causes Apply to silently fail or open the wrong file.\n"
        "2. **NEVER** emit a path starting with `src/`, `app/`, `lib/`, "
        "`components/`, `pages/`, `utils/`, etc. without the repo folder "
        "name in front. Those fail to resolve against the workspace root.\n"
        "3. Always output the **COMPLETE** file — every single line. "
        "Do NOT use `// ... rest unchanged`, `// existing code`, "
        "`// TODO: keep previous logic`, or any other abbreviation. "
        "Continue's Apply patch requires the full file to merge correctly. "
        "The output budget is 100 000 tokens — use it.\n"
        "4. For **edits**, copy the EXACT path returned by `read_file`, "
        "`grep`, or `find_symbol` — never reconstruct it from memory.\n"
        "5. For **new files**, infer the repo prefix from the user's "
        "request (\"in aelvyris-backend\" → `Aelvyris-Backend/`) or from "
        "the imports the file would contain. When uncertain, ASK the user "
        "before generating code.\n"
        "6. On Windows the indexed paths use the double-folder nesting "
        "pattern in some repos (e.g. `Aelvyris-Backend/Aelvyris-Backend/"
        "src/...`). Mirror EXACTLY whatever path your tools returned — "
        "never shorten or rewrite it.\n"
        "7. After generating an edit or new file, state clearly: "
        "\"Click **Apply** on the code block above to apply this change.\" "
        "so the user knows the workflow.\n"
    )


# ── Seed-message construction ─────────────────────────────────────────────────
def build_seed_messages(
    user_query: str,
    conversation_history: list[dict] | None = None,
    route=None,
    vector_query_override: str | None = None,
    session_context: dict | None = None,
) -> list[dict]:
    """Build the initial messages for the agentic loop.

    Args:
        route: optional ``RouteDecision`` from ``query_router``. When given,
            the seed shape is tailored to the route — e.g. OVERVIEW gets the
            repo map only (no chunk retrieval), SYMBOL_LOOKUP gets retrieval
            only (no map), DEBUG gets both with a much higher budget.
        vector_query_override: optional HyDE-expanded vector query. Pulls the
            initial retrieved context from a code-shaped paraphrase of the
            user's prose, dramatically improving recall on prose-y questions.
    """
    from src.rag_engine import retrieve
    from src.repo_map import relevant_repo_map

    # Defaults preserve current behavior when no route is supplied.
    seed_strategy = "both"
    seed_token_budget = AGENTIC_SEED_TOKEN_BUDGET
    max_turns = AGENTIC_MAX_TOOL_TURNS
    suggested_top_k = 8
    use_multi = None
    route_note = ""
    target_files: list[str] = []

    if route is not None:
        seed_strategy = route.seed_strategy
        seed_token_budget = min(seed_token_budget, route.seed_token_budget)
        max_turns = min(AGENTIC_MAX_TOOL_TURNS, route.max_tool_turns)
        suggested_top_k = max(4, route.suggested_top_k // 4)  # seed is a slice
        use_multi = route.use_multi_query
        route_note = f"{route.route} ({route.note})"
        target_files = list(route.target_files or [])

    # ── Section 1: repo map (skipped when retrieval-only / minimal). ──────────
    map_section = ""
    if seed_strategy in ("both", "map_only"):
        repo_map_block, _ = relevant_repo_map(user_query)
        if repo_map_block:
            map_section = (
                "\n## Repository Map\n"
                "Compressed view of the codebase. Use it to decide which tools "
                "to call next.\n\n"
                f"{repo_map_block}\n"
            )

    # ── Section 2: retrieved chunks (skipped when map-only / minimal). ────────
    ctx_section = ""
    metas: list[dict] = []
    confidence_hint = ""
    if seed_strategy in ("both", "retrieval_only"):
        seed_ctx, ntok, metas = retrieve(
            user_query,
            top_k=suggested_top_k,
            token_budget=seed_token_budget,
            vector_query_override=vector_query_override,
            use_multi_query=use_multi,
            conversation_history=conversation_history,
        )
        if seed_ctx:
            files_cited = sorted({m["file"] for m in metas})
            files_summary = "\n".join(f"  - {f}" for f in files_cited) or "  (none)"
            ctx_section = (
                "\n## Initial Retrieved Context\n"
                f"Hybrid retrieval pre-pulled {len(metas)} blocks (~{ntok} tokens) "
                "as a starting point. Use the tools below to dig deeper.\n\n"
                f"{seed_ctx}\n\n### Files Referenced Above\n{files_summary}\n"
            )
            # CRAG-style low-confidence hint — when reranker scores are weak,
            # the LLM should validate via tools rather than trust the seed.
            if metas and metas[0].get("low_confidence"):
                confidence_hint = (
                    "\n⚠ **Retrieval confidence is LOW** — the reranker did not "
                    "find a strong match. The above context may be tangential. "
                    "Call `grep` or `find_symbol` to validate before answering, "
                    "and explicitly say so if the codebase doesn't contain what "
                    "was asked.\n"
                )

    # ── Section 3: route-specific hint ────────────────────────────────────────
    route_hint = ""
    if route is not None:
        if route.route == "OVERVIEW":
            route_hint = (
                "\n## Route: OVERVIEW\n"
                "This is a broad question about the project. The repo map above "
                "should already give you most of the answer. If you need more "
                "context, call `read_file` on README files or top-level entry "
                "points (main.ts, index.ts, etc.) — do NOT spam `retrieve`.\n"
            )
        elif route.route == "FILE_LOOKUP":
            files_str = ", ".join(target_files) if target_files else "(see query)"
            route_hint = (
                "\n## Route: FILE_LOOKUP\n"
                f"User is asking about specific file(s): {files_str}. Use "
                "`read_file` on the named file(s) directly. Skip `retrieve`.\n"
            )
        elif route.route == "SYMBOL_LOOKUP":
            route_hint = (
                "\n## Route: SYMBOL_LOOKUP\n"
                "User is asking about a specific identifier. Start with "
                "`find_symbol`, then `read_file` at the returned location. "
                "Avoid `retrieve` — it dilutes the answer.\n"
            )
        elif route.route == "EXACT_STRING":
            route_hint = (
                "\n## Route: EXACT_STRING\n"
                "User quoted a literal string. Use `grep` first.\n"
            )
        elif route.route == "DEBUG":
            route_hint = (
                "\n## Route: DEBUG\n"
                "User reported an error / failure. Search by the error string "
                "via `grep`, then `read_file` the surrounding handler.\n"
            )
        elif route.route == "WRITE_FILE":
            files_str = ", ".join(target_files) if target_files else "(file named in the query)"
            route_hint = (
                "\n## Route: WRITE_FILE\n"
                f"The user wants to add or modify content in: {files_str}.\n"
                "Follow this exact procedure — no deviation:\n"
                "1. Call `read_file` on the target file to get its CURRENT content.\n"
                "2. Make ONLY the changes the user requested.\n"
                "3. Output the COMPLETE updated file — no diffs, no partial snippets,\n"
                "   no `// ... rest unchanged`. The user needs the full file to Apply.\n"
                "Do NOT retrieve, summarize, or explore other files/repos. "
                "Stay focused on the specific file the user named.\n"
            )
        elif route.route == "HOW_X_WORKS":
            route_hint = (
                "\n## Route: HOW_X_WORKS\n"
                "This is an explanatory question. Use `retrieve` to pull relevant "
                "implementations, then `read_file` at the returned locations for "
                "detail. Stay focused on the specific subject the user asked about. "
                "Do NOT produce unsolicited summaries of other components or repos.\n"
            )
        elif route.route == "IMPLEMENT_FEATURE":
            route_hint = (
                "\n## Route: IMPLEMENT_FEATURE\n"
                "The user wants new code written. Before writing:\n"
                "1. Use `retrieve` or `find_symbol` to find existing patterns, "
                "   interfaces, and conventions in the relevant repo.\n"
                "2. Use `read_file` on closely related files to match style.\n"
                "3. Output COMPLETE, production-ready code that integrates cleanly "
                "   with what already exists. Follow the workspace path format.\n"
            )
        elif route.route == "REFACTOR":
            route_hint = (
                "\n## Route: REFACTOR\n"
                "The user wants existing code improved. Procedure:\n"
                "1. Use `read_file` on the target file(s) to get the full current code.\n"
                "2. Understand the existing structure before proposing changes.\n"
                "3. Output the COMPLETE refactored file — not a diff, not partial. "
                "   The user needs the full file to Apply in the IDE.\n"
                "Do not change behavior without flagging it — refactors must be safe.\n"
            )

    # ── Session context (soft hint from conversation history) ────────────────
    # When the conversation has established a working context (recent turns
    # retrieved files from Aelvyris-Backend, or the user asked about specific
    # files there), carry that forward as a soft scope bias for the current
    # query — but ONLY when the current query has no explicit repo mention.
    # This fixes multi-context chats: "how does error handling work?" after
    # 5 turns about Backend should search Backend, not all 5 repos equally.
    session_ctx_section = ""
    current_explicit_scope = _detect_repo_scope(user_query)
    if session_context and not current_explicit_scope:
        history_repos = [r for r in (session_context.get("repos") or []) if r]
        history_files = [f for f in (session_context.get("files") or []) if f]
        if history_repos or history_files:
            parts: list[str] = []
            if history_repos:
                parts.append(
                    "Repos from recent turns: "
                    + ", ".join(f"`{r}`" for r in history_repos[:4])
                )
            if history_files:
                parts.append(
                    "Files referenced recently: "
                    + ", ".join(f"`{f}`" for f in history_files[:6])
                )
            session_ctx_section = (
                "\n## Conversation Context\n"
                + "\n".join(parts) + "\n"
                "The current query does not name a specific repo. Based on the "
                "conversation so far, prioritize the repos/files above in your "
                "tool calls — unless the query content clearly points elsewhere. "
                "This is a soft hint, not a hard constraint.\n"
            )

    # ── Repo scope constraint ─────────────────────────────────────────────────
    # When the user names a specific repo in their query, inject a hard scope
    # so the agent cannot drift into unrelated repos during tool calls.
    repo_scope = _detect_repo_scope(user_query)
    scope_section = ""
    if repo_scope:
        scope_names = " and ".join(f"`{r}`" for r in repo_scope)
        scope_section = (
            f"\n⚠ **SCOPE CONSTRAINT — MANDATORY**: The user's query explicitly "
            f"references {scope_names}. You MUST restrict ALL tool calls "
            f"(read_file, grep, retrieve, find_symbol, find_callers, find_importers) "
            f"to files within {scope_names}. Do NOT read, summarize, or explore "
            "files from any other repo unless the user explicitly asks. Violating "
            "this constraint is the #1 failure mode — a complete summary of the "
            "wrong repo is not a valid answer.\n"
        )

    workspace_section = _workspace_layout_section()
    system_message = f"""You are an expert software engineer with full agentic access to the codebase.
{map_section}{ctx_section}{confidence_hint}{route_hint}{session_ctx_section}{scope_section}{workspace_section}
## Your Tools
You can iteratively call these tools to investigate the code before answering:
  - `retrieve(query, top_k)`       — hybrid semantic search
  - `grep(pattern, path_glob?)`    — regex search by line
  - `read_file(path, start, end)`  — read a specific slice
  - `read_files(items)`            — BATCH read; pass [{{path, start_line?, end_line?}}, ...]
  - `find_symbol(symbol)`          — locate a definition
  - `find_callers(symbol)`         — find call sites
  - `find_importers(target)`       — find files that import a path/package
  - `repo_map(query?, top_files?)` — get a fresh structural map

## Your Instructions
- **Batch reads**: when you need 2+ files, call `read_files` with all of them in
  ONE call. Each read_file round-trip is an LLM turn — batching saves seconds.
- **Don't re-read**: tool results are memoized within this session. If you
  already saw a result in an earlier turn, use it; don't call again.
- For broad questions, start with `repo_map` or `retrieve` to orient yourself.
- For specific symbols, prefer `find_symbol` → `read_file` over `grep`.
- For exact strings (error messages, config keys), use `grep`.
- ALWAYS cite exact file paths and line numbers in your final answer.
- Do not hallucinate code. If a tool returns nothing useful, try a different
  tool or admit the codebase doesn't contain what was asked for.
- Only emit your final answer after you have enough evidence — keep calling
  tools until you do, up to {max_turns} tool turns.
- **Stay on task**: answer EXACTLY what the user asked. If the user asks to
  edit file X, read X and produce the modified file — do NOT wander into
  unrelated repos or produce unsolicited codebase summaries. Breadth is not
  a virtue here; precision is."""

    messages: list[dict] = [{"role": "system", "content": system_message}]
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") in ("user", "assistant"):
                messages.append(msg)
    messages.append({"role": "user", "content": user_query})
    if route_note:
        logger.info(f"Route: {route_note} | seed={seed_strategy} | turns<={max_turns}")
    return messages
