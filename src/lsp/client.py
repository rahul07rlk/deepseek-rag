"""Minimal LSP client over stdio.

Implements only the subset the indexer needs:

  - initialize / initialized / shutdown / exit
  - textDocument/didOpen, didClose
  - textDocument/definition
  - textDocument/references
  - textDocument/documentSymbol
  - textDocument/hover

Uses synchronous request/response with a per-request timeout — the
indexer is single-threaded per server and we don't need pipelining.
Notifications (window/logMessage, etc.) are silently dropped.

LSP framing is HTTP-style: ``Content-Length: <bytes>\\r\\n\\r\\n<json>``.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.lsp.servers import ServerSpec
from src.utils.logger import get_logger

logger = get_logger("lsp", "indexer.log")


class LSPClient:
    def __init__(self, spec: ServerSpec, root_path: Path):
        self.spec = spec
        self.root_path = root_path.resolve()
        self.proc: subprocess.Popen | None = None
        self._next_id = 1
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._open_docs: set[str] = set()
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self, timeout: float = 30.0) -> None:
        if self.proc is not None:
            return
        env = os.environ.copy()
        try:
            self.proc = subprocess.Popen(
                list(self.spec.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self.root_path),
                env=env,
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"LSP server not on PATH: {self.spec.executable}") from e

        # initialize
        root_uri = _path_to_uri(self.root_path)
        params: dict[str, Any] = {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "rootPath": str(self.root_path),
            "workspaceFolders": [{"uri": root_uri, "name": self.root_path.name}],
            "capabilities": {
                "workspace": {"workspaceFolders": True},
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "synchronization": {"dynamicRegistration": False},
                },
            },
        }
        if self.spec.init_options:
            params["initializationOptions"] = self.spec.init_options
        self._request("initialize", params, timeout=timeout)
        self._notify("initialized", {})
        self._initialized = True

    def shutdown(self, timeout: float = 5.0) -> None:
        if self.proc is None:
            return
        try:
            self._request("shutdown", None, timeout=timeout)
            self._notify("exit", None)
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None
        self._initialized = False
        self._open_docs.clear()

    # ── Document handling ─────────────────────────────────────────────────────
    def did_open(self, file_path: Path) -> None:
        if not self._initialized or self.proc is None:
            return
        uri = _path_to_uri(file_path)
        if uri in self._open_docs:
            return
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": self.spec.language_id,
                "version": 1,
                "text": text,
            }
        })
        self._open_docs.add(uri)

    def did_close(self, file_path: Path) -> None:
        if not self._initialized or self.proc is None:
            return
        uri = _path_to_uri(file_path)
        if uri not in self._open_docs:
            return
        self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        self._open_docs.discard(uri)

    # ── Queries ───────────────────────────────────────────────────────────────
    def definition(self, file: Path, line: int, character: int,
                    timeout: float = 5.0) -> list[dict]:
        self.did_open(file)
        result = self._request("textDocument/definition", {
            "textDocument": {"uri": _path_to_uri(file)},
            "position": {"line": line, "character": character},
        }, timeout=timeout)
        return _normalize_locations(result)

    def references(self, file: Path, line: int, character: int,
                    include_decl: bool = False,
                    timeout: float = 8.0) -> list[dict]:
        self.did_open(file)
        result = self._request("textDocument/references", {
            "textDocument": {"uri": _path_to_uri(file)},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_decl},
        }, timeout=timeout)
        return _normalize_locations(result)

    def document_symbols(self, file: Path,
                          timeout: float = 5.0) -> list[dict]:
        self.did_open(file)
        result = self._request("textDocument/documentSymbol", {
            "textDocument": {"uri": _path_to_uri(file)},
        }, timeout=timeout)
        if not result:
            return []
        # Two formats per spec: SymbolInformation[] (flat) or
        # DocumentSymbol[] (hierarchical). We flatten either to a list.
        return _flatten_symbols(result)

    def hover(self, file: Path, line: int, character: int,
               timeout: float = 4.0) -> str:
        self.did_open(file)
        result = self._request("textDocument/hover", {
            "textDocument": {"uri": _path_to_uri(file)},
            "position": {"line": line, "character": character},
        }, timeout=timeout)
        if not result:
            return ""
        contents = result.get("contents")
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, list):
            return "\n".join(
                c if isinstance(c, str) else (c.get("value", "") if isinstance(c, dict) else "")
                for c in contents
            )
        return ""

    # ── Wire protocol ─────────────────────────────────────────────────────────
    def _request(self, method: str, params: Any, timeout: float = 10.0) -> Any:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("LSP server not running")
        rid = self._next_id
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._write_message(msg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._read_message(timeout=max(0.1, deadline - time.time()))
            if resp is None:
                continue
            if resp.get("id") == rid:
                if "error" in resp:
                    err = resp["error"]
                    raise RuntimeError(f"LSP error {err.get('code')}: {err.get('message')}")
                return resp.get("result")
        raise TimeoutError(f"LSP timeout on {method}")

    def _notify(self, method: str, params: Any) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write_message(msg)

    def _write_message(self, msg: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            try:
                self.proc.stdin.write(header)
                self.proc.stdin.write(body)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f"LSP write failed: {e}") from e

    def _read_message(self, timeout: float = 5.0) -> dict | None:
        """Read one LSP message. Returns None on timeout to allow polling."""
        if self.proc is None or self.proc.stdout is None:
            return None
        with self._read_lock:
            # Read headers.
            headers: dict[str, str] = {}
            deadline = time.time() + timeout
            while True:
                if time.time() > deadline:
                    return None
                line = self.proc.stdout.readline()
                if not line:
                    return None
                line = line.decode("ascii", errors="ignore").strip()
                if not line:
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            length_str = headers.get("content-length")
            if not length_str:
                return None
            length = int(length_str)
            body = self.proc.stdout.read(length)
            if not body:
                return None
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return None


# ── Helpers ──────────────────────────────────────────────────────────────────
def _path_to_uri(path: Path) -> str:
    p = str(Path(path).resolve()).replace("\\", "/")
    if p.startswith("/"):
        return f"file://{p}"
    # Windows: file:///C:/path
    return f"file:///{p}"


def _uri_to_path(uri: str) -> str:
    if uri.startswith("file:///"):
        return uri[len("file:///"):].replace("/", os.sep)
    if uri.startswith("file://"):
        return uri[len("file://"):].replace("/", os.sep)
    return uri


def _normalize_locations(result) -> list[dict]:
    """LSP definition/references can return Location, Location[],
    LocationLink[], or null. Normalize to [{uri, range}]."""
    if not result:
        return []
    if isinstance(result, dict):
        return [_loc(result)]
    if isinstance(result, list):
        return [_loc(r) for r in result if r]
    return []


def _loc(item: dict) -> dict:
    # LocationLink fields: targetUri, targetRange, targetSelectionRange
    uri = item.get("uri") or item.get("targetUri") or ""
    rng = item.get("range") or item.get("targetRange") or {}
    start = (rng.get("start") or {})
    return {
        "uri": uri,
        "path": _uri_to_path(uri),
        "line": int(start.get("line") or 0),
        "character": int(start.get("character") or 0),
    }


def _flatten_symbols(result, prefix: str = "") -> list[dict]:
    out: list[dict] = []
    if not isinstance(result, list):
        return out
    for sym in result:
        if not isinstance(sym, dict):
            continue
        name = sym.get("name") or ""
        full = f"{prefix}.{name}" if prefix else name
        rng = sym.get("range") or sym.get("location", {}).get("range") or {}
        start = (rng.get("start") or {})
        out.append({
            "name": full,
            "kind": sym.get("kind"),
            "line": int(start.get("line") or 0),
            "character": int(start.get("character") or 0),
            "container": sym.get("containerName") or prefix,
        })
        children = sym.get("children")
        if isinstance(children, list):
            out.extend(_flatten_symbols(children, full))
    return out
