"""Per-language LSP server lifecycle.

Owns one ``LSPClient`` per (server_spec, repo_root) pair. Servers are
launched lazily on first use and reused across queries. The manager
enforces a coarse global timeout per query so a hung server can't
poison the indexing pipeline.

Public surface:

  - ``manager.refs(file, line, col)``  → list[{path, line, char}]
  - ``manager.defs(file, line, col)``  → list[{path, line, char}]
  - ``manager.symbols(file)``          → list[{name, line, char, kind, container}]
  - ``manager.shutdown_all()``         → for clean exit

Failures are absorbed: if the server crashes or the call times out,
the manager logs and returns ``[]``. Callers fall back to tree-sitter.
"""
from __future__ import annotations

import threading
from pathlib import Path

from src.config import REPO_PATHS
from src.lsp.client import LSPClient
from src.lsp.servers import ServerSpec, detect_server_for, installed_servers
from src.utils.logger import get_logger

logger = get_logger("lsp_manager", "indexer.log")


class LSPManager:
    def __init__(self):
        # key: (server_name, repo_root_str) -> LSPClient
        self._clients: dict[tuple[str, str], LSPClient] = {}
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def shutdown_all(self) -> None:
        with self._lock:
            for cli in list(self._clients.values()):
                try:
                    cli.shutdown()
                except Exception:
                    pass
            self._clients.clear()

    def installed(self) -> list[str]:
        return [s.name for s in installed_servers()]

    # ── Routing ───────────────────────────────────────────────────────────────
    def _client_for(self, file: Path) -> LSPClient | None:
        spec = detect_server_for(file.suffix)
        if spec is None:
            return None
        repo = _resolve_repo(file)
        if repo is None:
            return None
        key = (spec.name, str(repo))
        with self._lock:
            cli = self._clients.get(key)
            if cli is not None:
                return cli
            cli = LSPClient(spec, repo)
            try:
                cli.start(timeout=30.0)
            except Exception as e:
                logger.warning(f"LSP {spec.name} failed to start: {e}")
                return None
            self._clients[key] = cli
            logger.info(f"LSP {spec.name} ready for repo {repo.name}")
            return cli

    # ── Queries ───────────────────────────────────────────────────────────────
    def refs(self, file: Path, line: int, character: int,
              include_decl: bool = False) -> list[dict]:
        cli = self._client_for(file)
        if cli is None:
            return []
        try:
            return cli.references(file, line, character, include_decl=include_decl)
        except Exception as e:
            logger.debug(f"LSP refs failed for {file}: {e}")
            return []

    def defs(self, file: Path, line: int, character: int) -> list[dict]:
        cli = self._client_for(file)
        if cli is None:
            return []
        try:
            return cli.definition(file, line, character)
        except Exception as e:
            logger.debug(f"LSP defs failed for {file}: {e}")
            return []

    def symbols(self, file: Path) -> list[dict]:
        cli = self._client_for(file)
        if cli is None:
            return []
        try:
            return cli.document_symbols(file)
        except Exception as e:
            logger.debug(f"LSP symbols failed for {file}: {e}")
            return []

    def hover(self, file: Path, line: int, character: int) -> str:
        cli = self._client_for(file)
        if cli is None:
            return ""
        try:
            return cli.hover(file, line, character)
        except Exception:
            return ""


_manager: LSPManager | None = None


def get_manager() -> LSPManager:
    global _manager
    if _manager is None:
        _manager = LSPManager()
    return _manager


def _resolve_repo(file: Path) -> Path | None:
    f = file.resolve()
    for r in REPO_PATHS:
        try:
            f.relative_to(r.resolve())
            return r.resolve()
        except ValueError:
            continue
    return None
