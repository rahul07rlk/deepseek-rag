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
    "retrieve":       lambda a: tool_retrieve(a.get("query", ""), a.get("top_k", 12)),
    "read_file":      lambda a: tool_read_file(a.get("path", ""), a.get("start_line", 1), a.get("end_line", 0)),
    "read_files":     lambda a: tool_read_files(a.get("items", [])),
    "grep":           lambda a: tool_grep(a.get("pattern", ""), a.get("path_glob", "")),
    "find_symbol":    lambda a: tool_find_symbol(a.get("symbol", "")),
    "find_callers":   lambda a: tool_find_callers(a.get("symbol", "")),
    "find_importers": lambda a: tool_find_importers(a.get("target", "")),
    "repo_map":       lambda a: tool_repo_map(a.get("query", ""), a.get("top_files", 30)),
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


# ── Seed-message construction ─────────────────────────────────────────────────
def build_seed_messages(
    user_query: str,
    conversation_history: list[dict] | None = None,
    route=None,
    vector_query_override: str | None = None,
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

    system_message = f"""You are an expert software engineer with full agentic access to the codebase.
{map_section}{ctx_section}{confidence_hint}{route_hint}
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
  tools until you do, up to {max_turns} tool turns."""

    messages: list[dict] = [{"role": "system", "content": system_message}]
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") in ("user", "assistant"):
                messages.append(msg)
    messages.append({"role": "user", "content": user_query})
    if route_note:
        logger.info(f"Route: {route_note} | seed={seed_strategy} | turns<={max_turns}")
    return messages
