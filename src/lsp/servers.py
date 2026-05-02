"""Per-language LSP server registry.

Each entry tells the manager how to launch a language server and which
file extensions it owns. Detection is shutil.which-based so unavailable
servers are skipped silently.

Adding a new language is one entry — drop it here and the manager
picks it up automatically.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: tuple[str, ...]   # invocation; the executable must be on PATH
    extensions: frozenset[str]
    language_id: str           # LSP language id sent in didOpen notifications
    init_options: dict | None = None
    notes: str = ""

    @property
    def executable(self) -> str:
        return self.command[0]

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None


# The order matters only for the "first match" picker; extensions are
# unique-per-server in practice.
SERVERS: list[ServerSpec] = [
    ServerSpec(
        name="pyright",
        command=("pyright-langserver", "--stdio"),
        extensions=frozenset({".py", ".pyi"}),
        language_id="python",
        notes="npm i -g pyright",
    ),
    ServerSpec(
        name="typescript-language-server",
        command=("typescript-language-server", "--stdio"),
        extensions=frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}),
        language_id="typescript",
        notes="npm i -g typescript typescript-language-server",
    ),
    ServerSpec(
        name="gopls",
        command=("gopls",),
        extensions=frozenset({".go"}),
        language_id="go",
        notes="go install golang.org/x/tools/gopls@latest",
    ),
    ServerSpec(
        name="rust-analyzer",
        command=("rust-analyzer",),
        extensions=frozenset({".rs"}),
        language_id="rust",
        notes="rustup component add rust-analyzer",
    ),
    ServerSpec(
        name="jdtls",
        command=("jdtls",),
        extensions=frozenset({".java"}),
        language_id="java",
        notes="https://github.com/eclipse-jdtls/eclipse.jdt.ls",
    ),
    ServerSpec(
        name="clangd",
        command=("clangd",),
        extensions=frozenset({".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"}),
        language_id="cpp",
        notes="apt install clangd / brew install llvm",
    ),
    ServerSpec(
        name="omnisharp",
        command=("omnisharp", "-lsp"),
        extensions=frozenset({".cs"}),
        language_id="csharp",
        notes="dotnet tool install -g omnisharp",
    ),
    ServerSpec(
        name="kotlin-language-server",
        command=("kotlin-language-server",),
        extensions=frozenset({".kt", ".kts"}),
        language_id="kotlin",
    ),
    ServerSpec(
        name="solargraph",
        command=("solargraph", "stdio"),
        extensions=frozenset({".rb"}),
        language_id="ruby",
    ),
    ServerSpec(
        name="intelephense",
        command=("intelephense", "--stdio"),
        extensions=frozenset({".php"}),
        language_id="php",
    ),
    ServerSpec(
        name="sourcekit-lsp",
        command=("sourcekit-lsp",),
        extensions=frozenset({".swift"}),
        language_id="swift",
    ),
]


_BY_EXT: dict[str, ServerSpec] = {}
for s in SERVERS:
    for ext in s.extensions:
        _BY_EXT.setdefault(ext, s)


def detect_server_for(file_extension: str) -> ServerSpec | None:
    """Return the spec for the LSP server that owns ``ext`` if it's
    installed on PATH, else None."""
    spec = _BY_EXT.get(file_extension.lower())
    if spec is None:
        return None
    if not spec.is_available():
        return None
    return spec


def installed_servers() -> list[ServerSpec]:
    return [s for s in SERVERS if s.is_available()]
