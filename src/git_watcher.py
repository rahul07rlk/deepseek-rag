"""Git-aware indexer.

The watchdog-only watcher in ``src/watcher.py`` does the right thing for
single-file edits but is fragile around bulk events:

  - ``git checkout main`` rewrites thousands of files in milliseconds
  - ``npm install`` / ``cargo build`` create or refresh dist trees
  - ``git stash pop`` triggers many modifications at once

These bursts overwhelm the embedder + the BM25 rebuilder, lock up the
GTX 1650, and can corrupt the index if a write completes mid-flight.

This module adds a coordination layer:

  - polls ``git rev-parse HEAD`` per repo at a low interval. When HEAD
    changes (branch switch, pull, reset), it triggers a debounced
    snapshot-aware rebuild instead of per-file events.
  - detects "rename storms" (many modifications within a short window)
    and pauses the watchdog handler until the storm subsides.
  - exposes a ``pause()`` / ``resume()`` API so manual operations
    (re-index command, schema migration) can quiet the watcher.

The git-aware loop runs in a daemon thread alongside the watchdog
observer; it doesn't replace it. Single-file saves still go through
``watcher.py`` for sub-second reactivity.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from src.config import REPO_PATHS
from src.utils.logger import get_logger

logger = get_logger("git_watcher", "indexer.log")


# ── Tunables (env-friendly) ──────────────────────────────────────────────────
import os
_HEAD_POLL_INTERVAL_S = float(os.getenv("GIT_HEAD_POLL_INTERVAL_S", "10.0"))
_STORM_WINDOW_S = float(os.getenv("INDEX_STORM_WINDOW_S", "2.0"))
_STORM_THRESHOLD = int(os.getenv("INDEX_STORM_THRESHOLD", "30"))
_STORM_COOLDOWN_S = float(os.getenv("INDEX_STORM_COOLDOWN_S", "8.0"))


class IndexerControl:
    """Pause/resume gate the per-file watcher consults before reindexing."""

    def __init__(self):
        self._paused = False
        self._lock = threading.Lock()
        self._reason: str = ""
        # Sliding window of recent file events for storm detection.
        self._events: deque[float] = deque()

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def pause(self, reason: str = "manual") -> None:
        with self._lock:
            self._paused = True
            self._reason = reason
        logger.info(f"Indexer paused: {reason}")

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._reason = ""
        logger.info("Indexer resumed.")

    def record_event(self) -> bool:
        """Record a file event; return True if a storm was detected this call."""
        now = time.time()
        with self._lock:
            self._events.append(now)
            cutoff = now - _STORM_WINDOW_S
            while self._events and self._events[0] < cutoff:
                self._events.popleft()
            if len(self._events) >= _STORM_THRESHOLD and not self._paused:
                self._paused = True
                self._reason = (
                    f"file storm ({len(self._events)} events in "
                    f"{_STORM_WINDOW_S}s) — likely git/install/build"
                )
                logger.warning(f"Indexer auto-paused: {self._reason}")
                # Schedule auto-resume.
                threading.Timer(_STORM_COOLDOWN_S, self._auto_resume).start()
                return True
        return False

    def _auto_resume(self):
        with self._lock:
            if self._paused and self._reason.startswith("file storm"):
                self._paused = False
                self._reason = ""
                logger.info("Indexer auto-resumed after storm cooldown.")


_control: IndexerControl | None = None


def get_control() -> IndexerControl:
    global _control
    if _control is None:
        _control = IndexerControl()
    return _control


# ── Per-repo HEAD polling ────────────────────────────────────────────────────
class _RepoHeadTracker:
    def __init__(self, repo: Path):
        self.repo = repo
        self.head: str | None = None
        self.branch: str | None = None
        self._refresh()

    def _refresh(self) -> tuple[str | None, str | None]:
        try:
            r = subprocess.run(
                ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding='utf-8', timeout=5,
            )
            new_head = r.stdout.strip() if r.returncode == 0 else None
            r = subprocess.run(
                ["git", "-C", str(self.repo), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, encoding='utf-8', timeout=5,
            )
            new_branch = r.stdout.strip() if r.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            new_head = new_branch = None
        return new_head, new_branch

    def poll(self) -> tuple[bool, str | None]:
        """Returns (head_changed, new_branch_if_switched)."""
        new_head, new_branch = self._refresh()
        head_changed = new_head is not None and new_head != self.head
        branch_changed = (
            new_branch is not None and new_branch != self.branch
        )
        self.head = new_head
        self.branch = new_branch
        return head_changed, new_branch if branch_changed else None


def start_git_watcher(
    on_head_change=None,
    on_branch_switch=None,
) -> threading.Thread:
    """Start the daemon thread that polls each repo's HEAD.

    ``on_head_change(repo, new_head)`` fires whenever HEAD moves — pull,
    reset, commit. The proxy can use this to trigger a full graph
    rebuild + cache invalidation.

    ``on_branch_switch(repo, new_branch)`` fires when the branch name
    changes — a stronger signal that warrants pausing the file watcher
    while the new branch's index settles.
    """
    trackers = [_RepoHeadTracker(r) for r in REPO_PATHS if (r / ".git").exists()]
    if not trackers:
        logger.info("Git watcher: no .git directories under REPO_PATHS; idle.")

        # Return a no-op thread so callers can join() without crashing.
        def _noop():
            return
        t = threading.Thread(target=_noop, name="git-watcher-noop", daemon=True)
        t.start()
        return t

    control = get_control()
    stop = threading.Event()

    def _loop():
        logger.info(
            f"Git watcher started ({len(trackers)} repo(s), "
            f"poll={_HEAD_POLL_INTERVAL_S}s)."
        )
        while not stop.is_set():
            for tr in trackers:
                changed, new_branch = tr.poll()
                if changed:
                    logger.info(
                        f"Git: {tr.repo.name} HEAD → {tr.head[:12] if tr.head else '?'}"
                    )
                    if new_branch:
                        control.pause(reason=f"branch switch → {new_branch}")
                        # Resume after a window to let working tree settle.
                        threading.Timer(
                            _STORM_COOLDOWN_S, control.resume
                        ).start()
                        if on_branch_switch:
                            try:
                                on_branch_switch(tr.repo, new_branch)
                            except Exception as e:
                                logger.warning(f"on_branch_switch failed: {e}")
                    if on_head_change:
                        try:
                            on_head_change(tr.repo, tr.head)
                        except Exception as e:
                            logger.warning(f"on_head_change failed: {e}")
            stop.wait(_HEAD_POLL_INTERVAL_S)

    t = threading.Thread(target=_loop, name="git-watcher", daemon=True)
    t.start()
    return t
