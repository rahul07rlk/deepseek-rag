"""Semantic chunk classification.

Augments the existing AST/tree-sitter chunker output with a `chunk_type`
tag. Retrieval can then route by intent — debugging queries weight
``test`` and ``change`` chunks; refactor queries weight ``interface``
and ``module`` chunks; "how does X work" weights ``behavior`` chunks.

Six types, each lightweight to detect (no extra parsing):

  - ``module``    : top-of-file content. Module docstring, imports,
                    top-level constants. Captures architectural intent.
  - ``interface`` : signatures only — function/method declarations,
                    class headers, type aliases, decorators. No bodies.
  - ``behavior``  : symbol bodies — the actual logic of a function/method.
  - ``test``      : code in a test file (path matches test conventions)
                    or a function whose name starts with `test_`.
  - ``change``    : recently modified per git (last N commits / dirty).
                    Stamped at index time, refreshed by the watcher.
  - ``symbol``    : default for tagged AST chunks that don't fit elsewhere.

The dispatcher is pure-Python and ~100 LOC. It runs after the existing
chunker emits its dicts; it doesn't change chunk text or boundaries.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Filename patterns that signal a test file across most language ecosystems.
_TEST_PATH_PATTERNS = re.compile(
    r"(^|[\\/])("
    r"tests?|__tests__|spec|specs|e2e|integration|features"
    r")[\\/]"
    r"|"
    r"[._-](test|tests|spec|specs|e2e)\.[^.]+$"
    r"|"
    r"^test_[^.]+\.py$"
    r"|"
    r"_test\.go$",
    re.IGNORECASE,
)

# Symbol-name conventions that signal a test function.
_TEST_NAME_RE = re.compile(
    r"^(test_|test[A-Z]|it_should_|describe_|spec_)|_test$",
)

# A chunk is "interface only" when its body is mostly signatures.
# Heuristic: count how many lines look like declarations vs. statement-
# ending lines. A chunk with mostly `def foo(...): ...` and `class Bar:`
# but few executable statements is interface-shaped.
_DECL_LINE_RE = re.compile(
    r"^\s*(def\s+\w+|async\s+def\s+\w+|class\s+\w+|"
    r"function\s+\w+|interface\s+\w+|type\s+\w+|"
    r"export\s+(?:default\s+)?(?:async\s+)?function|"
    r"fn\s+\w+|struct\s+\w+|trait\s+\w+|enum\s+\w+|"
    r"@\w+|"
    r")",
)
_STMT_LINE_RE = re.compile(
    r"\b(return|if|for|while|try|throw|raise|yield|await|"
    r"console\.|print\(|self\.|this\.)\b",
)


def is_test_file(path: str | Path) -> bool:
    s = str(path).replace("\\", "/")
    return bool(_TEST_PATH_PATTERNS.search(s))


def is_test_symbol(symbol: str | None) -> bool:
    if not symbol:
        return False
    return bool(_TEST_NAME_RE.search(symbol.split(".")[-1]))


def _looks_interface_only(text: str) -> bool:
    """Heuristic: ratio of declaration lines to statement lines."""
    decl = stmt = 0
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", "//", "/*", "*")):
            continue
        if _DECL_LINE_RE.search(s):
            decl += 1
        elif _STMT_LINE_RE.search(s):
            stmt += 1
    if decl == 0:
        return False
    # Mostly declarations, very few statements → interface-like.
    return decl >= 2 and stmt <= max(1, decl // 3)


def _is_module_chunk(start_line: int, symbol: str | None) -> bool:
    """Top-of-file chunks with no symbol attached are module-level material."""
    return start_line <= 5 and not symbol


def classify_chunk(
    *,
    file_path: str,
    text: str,
    start_line: int,
    end_line: int,
    symbol: str | None,
    is_recent_change: bool = False,
) -> str:
    """Return one of: module / interface / behavior / test / change / symbol."""
    # `change` is a stamp added by the indexer when git tells us the file
    # was modified in the last few commits; takes precedence over content
    # type because debugging queries care most about freshly-edited code.
    if is_recent_change:
        return "change"
    if is_test_file(file_path) or is_test_symbol(symbol):
        return "test"
    if _is_module_chunk(start_line, symbol):
        return "module"
    if _looks_interface_only(text):
        return "interface"
    if symbol:
        # AST-tagged with a real symbol body.
        return "behavior"
    return "symbol"


def recent_change_files(repo_root: Path, lookback: int = 20) -> set[str]:
    """Return the absolute paths of files touched in the last ``lookback``
    commits, plus anything currently dirty. Best-effort — empty set when
    git is unavailable. Used by the indexer to stamp ``change`` chunks.
    """
    out: set[str] = set()
    try:
        # Recently committed files.
        r = subprocess.run(
            ["git", "-C", str(repo_root), "log",
             f"-{lookback}", "--name-only", "--pretty=format:"],
            capture_output=True, text=True, encoding='utf-8', timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                line = line.strip()
                if line:
                    out.add(str((repo_root / line).resolve()))
        # Dirty files (uncommitted changes).
        r = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                # Format: "XY filename" — slice off the 3-char status prefix.
                if len(line) > 3:
                    out.add(str((repo_root / line[3:].strip()).resolve()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


# Type weights used by confidence-aware retrieval — multipliers applied
# to fused scores when the query intent maps to a specific chunk type.
TYPE_BOOST_BY_INTENT: dict[str, dict[str, float]] = {
    "DEBUG":         {"change": 1.6, "test": 1.4, "behavior": 1.1},
    "HOW_X_WORKS":   {"behavior": 1.3, "module": 1.2, "interface": 1.1},
    "SYMBOL_LOOKUP": {"interface": 1.4, "behavior": 1.1},
    "OVERVIEW":      {"module": 1.5, "interface": 1.2},
    "FILE_LOOKUP":   {"module": 1.2},
    "EXACT_STRING":  {},
    "DEFAULT":       {},
}


def boost_for(intent: str, chunk_type: str) -> float:
    """Multiplier in [1.0, 1.6] applied at retrieval time."""
    return TYPE_BOOST_BY_INTENT.get(intent, {}).get(chunk_type, 1.0)
