"""Filesystem watcher that re-indexes on save, with debouncing.

Hybrid mode: the per-file ``watchdog`` observer handles single saves
sub-second. A separate ``git_watcher`` thread polls each repo's HEAD,
detects branch switches and rename storms, and pauses this watcher
while bulk operations settle. See ``src/git_watcher.py`` for details.

Per-save flow:
  1. file event arrives
  2. record event in IndexerControl (storm detector)
  3. if paused (storm OR branch switch in flight) → drop this event
  4. otherwise: re-index + invalidate code-graph rows + per-file
     cache invalidation (snapshot-aware: only entries citing this
     file are dropped, not the whole cache)
"""
from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.config import IGNORED_DIRS, INDEXED_EXTENSIONS, REPO_PATH, REPO_PATHS
from src.git_watcher import get_control, start_git_watcher
from src.indexer import index_single_file
from src.utils.logger import get_logger

logger = get_logger("watcher", "watcher.log")


class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, debounce_seconds: float = 2.0) -> None:
        self._last_seen: dict[str, float] = {}
        self._debounce = debounce_seconds
        self._control = get_control()

    def _should_process(self, path: str) -> bool:
        p = Path(path)
        if p.suffix not in INDEXED_EXTENSIONS:
            return False
        return not any(part in IGNORED_DIRS for part in p.parts)

    def _debounced(self, path: str) -> bool:
        now = time.time()
        if now - self._last_seen.get(path, 0.0) < self._debounce:
            return True
        self._last_seen[path] = now
        return False

    def _gated(self, path: str) -> bool:
        """Return True if the event should be DROPPED (paused / storm)."""
        # Always record for storm detection.
        self._control.record_event()
        if self._control.is_paused():
            logger.debug(f"Indexer paused ({self._control.reason()}) — dropping {path}")
            return True
        return False

    def _reindex(self, path: str) -> None:
        try:
            index_single_file(path)
        except Exception as e:
            logger.exception(f"Failed to re-index {path}: {e}")
        # Update the multi-graph for this file.
        try:
            from src.code_graph import update_for_file as graph_update
            graph_update(path)
        except Exception as e:
            logger.debug(f"Graph update skipped for {path}: {e}")
        # Per-file cache invalidation (snapshot-aware).
        try:
            from src.answer_cache import invalidate_for_file
            invalidate_for_file(path)
        except Exception as e:
            logger.debug(f"Cache invalidation skipped for {path}: {e}")

    def on_modified(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return
        if self._debounced(event.src_path) or self._gated(event.src_path):
            return
        logger.info(f"File changed: {Path(event.src_path).name}")
        self._reindex(event.src_path)

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return
        if self._gated(event.src_path):
            return
        logger.info(f"File deleted: {Path(event.src_path).name}")
        self._reindex(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            if not self._gated(event.src_path):
                logger.info(f"File moved/deleted: {Path(event.src_path).name}")
                self._reindex(event.src_path)
        if self._should_process(event.dest_path):
            if self._debounced(event.dest_path) or self._gated(event.dest_path):
                return
            logger.info(f"File moved/created: {Path(event.dest_path).name}")
            self._reindex(event.dest_path)


def _on_head_change(repo: Path, new_head: str | None) -> None:
    """Triggered when a repo's HEAD moves. Schedule a graph refresh."""
    if new_head is None:
        return
    logger.info(f"HEAD moved in {repo.name} → {new_head[:12]}; "
                "graph rebuild deferred until storm clears.")
    # Future enhancement: enqueue an async graph rebuild here. The
    # IndexerControl pause already prevents per-file thrashing.


def _on_branch_switch(repo: Path, new_branch: str) -> None:
    """Triggered on branch checkout. Cache must be wiped — different
    branch likely cites different code."""
    logger.info(f"Branch switch in {repo.name}: → {new_branch}; "
                "resetting semantic cache.")
    try:
        from src.answer_cache import reset as cache_reset
        cache_reset()
    except Exception as e:
        logger.warning(f"Cache reset failed: {e}")


def start_watcher(repo_paths: list[Path] | Path = REPO_PATHS) -> None:
    if isinstance(repo_paths, Path):
        repo_paths = [repo_paths]

    handler = CodeChangeHandler()
    observer = Observer()
    for rp in repo_paths:
        observer.schedule(handler, path=str(rp), recursive=True)
        logger.info(f"Watching: {rp}")

    # Start the git-aware coordinator thread alongside the observer.
    start_git_watcher(
        on_head_change=_on_head_change,
        on_branch_switch=_on_branch_switch,
    )

    observer.start()
    logger.info(
        f"Watching {len(repo_paths)} repo(s). Auto re-indexing on save. "
        "Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Watcher stopped.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher(REPO_PATHS)
