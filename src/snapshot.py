"""Repo snapshot identification.

Every retrieval and every cache entry should be tied to a verifiable repo
state. Without that, a cached answer from yesterday can silently apply to
today's code and quietly be wrong.

Snapshot ID composition (stable across processes):

    snapshot_id = sha1(
        "|".join(
            f"{repo.name}:{commit_or_tree_hash(repo)}"
            for repo in REPO_PATHS
        )
    )

For git repos we use the working-tree hash (`git rev-parse HEAD` plus the
hash of the dirty diff if there is one). For non-git directories we hash
file paths + mtimes + sizes — fast, stable enough for invalidation, and
doesn't require reading file contents.

A separate ``component_versions()`` returns a tuple of all the moving
parts that affect retrieval output (embedder model, reranker model,
prompt template, chunker version) so the cache key fully captures
"what would generate this answer."
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import REPO_PATHS

# Bump these when behavior changes in a way that should invalidate caches.
CHUNKER_VERSION = "2"          # bumped when semantic chunk types landed
RETRIEVER_VERSION = "3"        # bumped when confidence routing landed
PROMPT_TEMPLATE_VERSION = "2"  # bumped when streaming/agent prompts changed
GRAPH_VERSION = "2"            # bumped when multi-graph schema landed


@dataclass(frozen=True)
class Snapshot:
    """Immutable identifier for a point-in-time state of all configured repos."""
    id: str                      # 16-char hex digest
    per_repo: dict[str, str]     # repo_name -> commit/tree hash
    is_dirty: bool               # any repo has uncommitted changes
    component_versions: str      # combined version string of moving parts


def _git_head(repo: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding='utf-8', timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def _git_dirty_hash(repo: Path) -> str | None:
    """Hash the working-tree diff so uncommitted edits invalidate caches.

    Returns None if the tree is clean or git isn't available.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "diff", "--no-color"],
            capture_output=True, text=True, encoding='utf-8', timeout=10,
        )
        if r.returncode == 0 and r.stdout:
            return hashlib.sha1(r.stdout.encode("utf-8", "ignore")).hexdigest()[:16]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def _filesystem_hash(repo: Path, max_files: int = 5000) -> str:
    """Fast non-content fingerprint for non-git directories.

    Walks up to ``max_files`` paths, hashes (relpath, mtime, size) tuples.
    Cheap (no file reads) but stable enough to detect any save.
    """
    h = hashlib.sha1()
    count = 0
    for root, dirs, files in os.walk(repo):
        # Don't descend into vendored/build dirs.
        dirs[:] = [d for d in dirs if d not in {
            ".git", "node_modules", "__pycache__", "venv", ".venv",
            "dist", "build", "target", ".next", ".nuxt",
        }]
        for fn in files:
            if count >= max_files:
                return h.hexdigest()[:16]
            p = Path(root) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            rel = str(p.relative_to(repo)).replace("\\", "/")
            h.update(f"{rel}|{st.st_mtime_ns}|{st.st_size}\n".encode())
            count += 1
    return h.hexdigest()[:16]


def _repo_hash(repo: Path) -> tuple[str, bool]:
    """Return (hash, is_dirty) for a single repo path."""
    git_dir = repo / ".git"
    if git_dir.exists():
        head = _git_head(repo)
        dirty = _git_dirty_hash(repo)
        if head:
            if dirty:
                return f"{head[:12]}+{dirty}", True
            return head[:12], False
    return _filesystem_hash(repo), False


def current_snapshot() -> Snapshot:
    """Compute the current snapshot ID across all configured repos.

    O(repo_count) git calls + O(file_count) stat calls — typically <100ms
    even on large monorepos. Safe to call on the request hot path.
    """
    per_repo: dict[str, str] = {}
    is_dirty = False
    for repo in REPO_PATHS:
        h, dirty = _repo_hash(repo)
        per_repo[repo.name] = h
        is_dirty = is_dirty or dirty
    fingerprint = "|".join(f"{name}:{h}" for name, h in sorted(per_repo.items()))
    snap_id = hashlib.sha1(fingerprint.encode()).hexdigest()[:16]
    versions = component_versions()
    return Snapshot(
        id=snap_id,
        per_repo=per_repo,
        is_dirty=is_dirty,
        component_versions=versions,
    )


def component_versions() -> str:
    """Return a stable string capturing every moving part that affects
    retrieval output. Used in cache keys so that swapping the embedder or
    bumping the chunker version invalidates stale entries automatically."""
    from src.config import (
        EMBED_MODEL, EMBED_PROVIDER, RERANKER_MODEL, RERANKER_PROVIDER,
    )
    parts = [
        f"emb={EMBED_PROVIDER}/{EMBED_MODEL}",
        f"rer={RERANKER_PROVIDER}/{RERANKER_MODEL}",
        f"chunker={CHUNKER_VERSION}",
        f"retriever={RETRIEVER_VERSION}",
        f"prompt={PROMPT_TEMPLATE_VERSION}",
        f"graph={GRAPH_VERSION}",
    ]
    return "|".join(parts)


def file_fingerprint(file_path: str | Path) -> str:
    """Per-file fingerprint for cache key composition.

    Used by the cache to invalidate only entries whose evidence files
    actually changed, not the entire cache on every save.
    """
    p = Path(file_path)
    try:
        st = p.stat()
    except OSError:
        return "missing"
    return f"{st.st_size}|{st.st_mtime_ns}"


def files_fingerprint(file_paths: list[str | Path]) -> str:
    """Composite fingerprint for a set of evidence files."""
    h = hashlib.sha1()
    for fp in sorted(str(p) for p in file_paths):
        h.update(f"{fp}|{file_fingerprint(fp)}\n".encode())
    return h.hexdigest()[:16]
