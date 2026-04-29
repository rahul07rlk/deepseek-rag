"""Filesystem watcher that re-indexes on save, with debouncing."""
from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.config import IGNORED_DIRS, INDEXED_EXTENSIONS, REPO_PATH, REPO_PATHS
from src.indexer import index_single_file
from src.utils.logger import get_logger

logger = get_logger("watcher", "watcher.log")


class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, debounce_seconds: float = 2.0) -> None:
        self._last_seen: dict[str, float] = {}
        self._debounce = debounce_seconds

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

    def on_modified(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return
        if self._debounced(event.src_path):
            return
        logger.info(f"File changed: {Path(event.src_path).name}")
        try:
            index_single_file(event.src_path)
        except Exception as e:
            logger.exception(f"Failed to re-index {event.src_path}: {e}")

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if event.is_directory or not self._should_process(event.src_path):
            return
        logger.info(f"File deleted: {Path(event.src_path).name}")
        try:
            index_single_file(event.src_path)
        except Exception as e:
            logger.exception(f"Failed to clean up {event.src_path}: {e}")

    def on_moved(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"File moved/deleted: {Path(event.src_path).name}")
            try:
                index_single_file(event.src_path)
            except Exception as e:
                logger.exception(f"Failed to clean up {event.src_path}: {e}")
        if self._should_process(event.dest_path):
            if self._debounced(event.dest_path):
                return
            logger.info(f"File moved/created: {Path(event.dest_path).name}")
            try:
                index_single_file(event.dest_path)
            except Exception as e:
                logger.exception(f"Failed to re-index {event.dest_path}: {e}")


def start_watcher(repo_paths: list[Path] | Path = REPO_PATHS) -> None:
    if isinstance(repo_paths, Path):
        repo_paths = [repo_paths]

    handler = CodeChangeHandler()
    observer = Observer()
    for rp in repo_paths:
        observer.schedule(handler, path=str(rp), recursive=True)
        logger.info(f"Watching: {rp}")

    observer.start()
    logger.info(f"Watching {len(repo_paths)} repo(s). Auto re-indexing on save. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Watcher stopped.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher(REPO_PATHS)
