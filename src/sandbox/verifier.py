r"""Sandbox verifier — run language-specific checks on generated code.

Workflow:

  1. ``verify_response(text)`` extracts every fenced code block from the
     model output (``\`\`\`lang\n...\n\`\`\``).
  2. For each block, dispatch to the matching ``Checker`` if its tool
     is available.
  3. Each checker writes the snippet to a temp file in an isolated
     directory, runs the check with a strict timeout, and returns
     a ``CheckReport``.
  4. The aggregate ``VerificationResult`` is returned to the caller —
     proxy_server can reject / annotate the answer based on it.

Checkers never crash the proxy: missing tools yield ``skipped=True``,
timeouts yield a clear error, and any other exception is captured.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+\-]+)?\s*\n(?P<body>.*?)```",
    re.DOTALL,
)

_LANG_ALIAS = {
    "py": "python", "python3": "python",
    "ts": "typescript", "tsx": "typescript",
    "js": "javascript", "jsx": "javascript",
    "golang": "go",
    "rs": "rust",
    "cc": "cpp", "cxx": "cpp", "c++": "cpp", "h": "c", "hpp": "cpp",
    "kt": "kotlin",
    "rb": "ruby",
    "cs": "csharp",
}


@dataclass
class CheckReport:
    tool: str
    language: str
    passed: bool
    skipped: bool = False
    duration_s: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class VerificationResult:
    blocks_seen: int
    blocks_checked: int
    reports: list[CheckReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.reports:
            return True
        return all(r.passed or r.skipped for r in self.reports)

    @property
    def failures(self) -> list[CheckReport]:
        return [r for r in self.reports if not r.passed and not r.skipped]

    def summary(self) -> str:
        if not self.reports:
            return "no checkable code blocks"
        ok = sum(1 for r in self.reports if r.passed)
        skipped = sum(1 for r in self.reports if r.skipped)
        failed = sum(1 for r in self.reports if not r.passed and not r.skipped)
        return f"{ok} ok, {failed} failed, {skipped} skipped"


# ── Per-language checker registry ────────────────────────────────────────────
def _have(*tools: str) -> bool:
    return all(shutil.which(t) is not None for t in tools)


def _run(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str, str, float]:
    import time as _time
    t0 = _time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr, _time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        return -1, "", f"timeout after {timeout}s: {e}", _time.perf_counter() - t0
    except FileNotFoundError as e:
        return -1, "", f"tool not found: {e}", _time.perf_counter() - t0


def check_python(code: str, work: Path, timeout: float) -> list[CheckReport]:
    out: list[CheckReport] = []
    f = work / "snippet.py"
    f.write_text(code, encoding="utf-8")

    # 1) syntax via stdlib (always available).
    try:
        import ast
        ast.parse(code)
        out.append(CheckReport(
            tool="python-syntax", language="python", passed=True,
        ))
    except SyntaxError as e:
        out.append(CheckReport(
            tool="python-syntax", language="python", passed=False,
            stderr=f"SyntaxError: {e}",
        ))
        return out  # downstream checks are noise on bad syntax

    # 2) ruff (fast lint) when available.
    if _have("ruff"):
        rc, so, se, dt = _run(
            ["ruff", "check", "--no-cache", "--select", "E,F", str(f)],
            work, timeout,
        )
        out.append(CheckReport(
            tool="ruff", language="python", passed=rc == 0,
            duration_s=dt, stdout=so, stderr=se,
        ))

    # 3) mypy (slower; opt-in via env).
    if _have("mypy") and os.getenv("SANDBOX_RUN_MYPY", "false").lower() == "true":
        rc, so, se, dt = _run(
            ["mypy", "--no-incremental", "--ignore-missing-imports", str(f)],
            work, timeout,
        )
        out.append(CheckReport(
            tool="mypy", language="python", passed=rc == 0,
            duration_s=dt, stdout=so, stderr=se,
        ))

    return out


def check_typescript(code: str, work: Path, timeout: float) -> list[CheckReport]:
    out: list[CheckReport] = []
    f = work / "snippet.ts"
    f.write_text(code, encoding="utf-8")
    if _have("tsc"):
        rc, so, se, dt = _run(
            ["tsc", "--noEmit", "--target", "es2020", "--module", "esnext",
             "--moduleResolution", "node", "--strict", "false", str(f)],
            work, timeout,
        )
        out.append(CheckReport(
            tool="tsc", language="typescript", passed=rc == 0,
            duration_s=dt, stdout=so, stderr=se,
        ))
    if _have("eslint"):
        rc, so, se, dt = _run(
            ["eslint", "--no-eslintrc", "--no-config-lookup", str(f)],
            work, timeout,
        )
        out.append(CheckReport(
            tool="eslint", language="typescript", passed=rc == 0,
            duration_s=dt, stdout=so, stderr=se,
        ))
    return out


def check_go(code: str, work: Path, timeout: float) -> list[CheckReport]:
    out: list[CheckReport] = []
    if "package " not in code:
        # Wrap into a buildable scratch package.
        code = "package main\n\n" + code
    f = work / "snippet.go"
    f.write_text(code, encoding="utf-8")
    if _have("gofmt"):
        rc, so, se, dt = _run(["gofmt", "-e", str(f)], work, timeout)
        out.append(CheckReport(
            tool="gofmt", language="go", passed=rc == 0,
            duration_s=dt, stdout=so, stderr=se,
        ))
    if _have("go"):
        rc, so, se, dt = _run(["go", "vet", "./..."], work, timeout)
        out.append(CheckReport(
            tool="go vet", language="go", passed=rc == 0,
            duration_s=dt, stdout=so, stderr=se,
        ))
    return out


def check_rust(code: str, work: Path, timeout: float) -> list[CheckReport]:
    out: list[CheckReport] = []
    if not _have("rustc"):
        return [CheckReport(tool="rustc", language="rust", passed=True, skipped=True,
                             error="rustc not on PATH")]
    f = work / "snippet.rs"
    f.write_text(code, encoding="utf-8")
    rc, so, se, dt = _run(
        ["rustc", "--edition", "2021", "-Zparse-only", str(f)],
        work, timeout,
    )
    if rc != 0:
        # Older rustc: fall back to type-only check via emit=metadata.
        rc, so, se, dt = _run(
            ["rustc", "--edition", "2021", "--emit=metadata",
             "-o", str(work / "snippet.rmeta"), str(f)],
            work, timeout,
        )
    out.append(CheckReport(
        tool="rustc", language="rust", passed=rc == 0,
        duration_s=dt, stdout=so, stderr=se,
    ))
    return out


def check_java(code: str, work: Path, timeout: float) -> list[CheckReport]:
    if not _have("javac"):
        return [CheckReport(tool="javac", language="java", passed=True, skipped=True,
                             error="javac not on PATH")]
    f = work / "Snippet.java"
    # Wrap loose code into a class so javac is happy.
    if "class " not in code:
        code = "public class Snippet {\n" + code + "\n}"
    f.write_text(code, encoding="utf-8")
    rc, so, se, dt = _run(["javac", "-d", str(work), str(f)], work, timeout)
    return [CheckReport(
        tool="javac", language="java", passed=rc == 0,
        duration_s=dt, stdout=so, stderr=se,
    )]


def check_cpp(code: str, work: Path, timeout: float) -> list[CheckReport]:
    tool = "clang++" if _have("clang++") else ("g++" if _have("g++") else None)
    if tool is None:
        return [CheckReport(tool="cpp", language="cpp", passed=True, skipped=True,
                             error="no C++ compiler on PATH")]
    f = work / "snippet.cpp"
    f.write_text(code, encoding="utf-8")
    rc, so, se, dt = _run([tool, "-fsyntax-only", "-std=c++17", str(f)], work, timeout)
    return [CheckReport(
        tool=tool, language="cpp", passed=rc == 0,
        duration_s=dt, stdout=so, stderr=se,
    )]


def check_c(code: str, work: Path, timeout: float) -> list[CheckReport]:
    tool = "clang" if _have("clang") else ("gcc" if _have("gcc") else None)
    if tool is None:
        return [CheckReport(tool="c", language="c", passed=True, skipped=True,
                             error="no C compiler on PATH")]
    f = work / "snippet.c"
    f.write_text(code, encoding="utf-8")
    rc, so, se, dt = _run([tool, "-fsyntax-only", str(f)], work, timeout)
    return [CheckReport(
        tool=tool, language="c", passed=rc == 0,
        duration_s=dt, stdout=so, stderr=se,
    )]


_CHECKERS: dict[str, callable] = {
    "python": check_python,
    "typescript": check_typescript,
    "javascript": check_typescript,  # tsc handles plain JS too
    "go": check_go,
    "rust": check_rust,
    "java": check_java,
    "cpp": check_cpp,
    "c": check_c,
}


def available_checkers() -> dict[str, list[str]]:
    """Return {language: [tools_on_path]} so callers can show capabilities."""
    out: dict[str, list[str]] = {}
    for lang, checks in (
        ("python", ["ruff", "mypy"]),
        ("typescript", ["tsc", "eslint"]),
        ("go", ["gofmt", "go"]),
        ("rust", ["rustc"]),
        ("java", ["javac"]),
        ("cpp", ["clang++", "g++"]),
        ("c", ["clang", "gcc"]),
    ):
        present = [t for t in checks if shutil.which(t)]
        if present:
            out[lang] = present
    return out


# ── Public API ───────────────────────────────────────────────────────────────
def verify_code(code: str, language: str,
                 timeout_s: float | None = None) -> list[CheckReport]:
    """Verify a single fenced code block. Returns one report per check run."""
    lang = _LANG_ALIAS.get(language.lower(), language.lower())
    checker = _CHECKERS.get(lang)
    if checker is None:
        return [CheckReport(
            tool="dispatch", language=lang, passed=True, skipped=True,
            error=f"no checker registered for language {lang!r}",
        )]
    timeout = timeout_s if timeout_s is not None else float(
        os.getenv("SANDBOX_TIMEOUT_S", "12")
    )
    with tempfile.TemporaryDirectory(prefix="rag-sandbox-") as td:
        work = Path(td)
        try:
            return checker(code, work, timeout)
        except Exception as e:
            return [CheckReport(
                tool="checker", language=lang, passed=False,
                error=f"{type(e).__name__}: {e}",
            )]


def verify_response(text: str,
                     timeout_s: float | None = None,
                     max_blocks: int = 6) -> VerificationResult:
    """Extract every fenced code block from ``text`` and verify each.

    ``max_blocks`` caps the number of blocks checked so an answer with
    20 examples doesn't burn 4 minutes of toolchain time.
    """
    result = VerificationResult(blocks_seen=0, blocks_checked=0)
    for m in _CODE_BLOCK_RE.finditer(text):
        result.blocks_seen += 1
        if result.blocks_checked >= max_blocks:
            continue
        lang = (m.group("lang") or "").strip()
        body = m.group("body").strip()
        if not lang or not body:
            continue
        if _LANG_ALIAS.get(lang.lower(), lang.lower()) not in _CHECKERS:
            continue
        reports = verify_code(body, lang, timeout_s)
        result.reports.extend(reports)
        result.blocks_checked += 1
    return result
