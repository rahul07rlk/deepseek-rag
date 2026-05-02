"""Sandbox verifier for generated code.

Before returning model-generated code to the IDE, run a fast, contained
set of static checks against it: syntax, types, lints. A failed check
becomes a tool result the agent can use to fix its own output, rather
than letting a broken patch reach the developer.

Per-language verifiers (only run when the corresponding tool is on PATH):

  - Python    : syntax (ast.parse), mypy --no-incremental, ruff check
  - TS/JS     : tsc --noEmit, eslint --no-eslintrc
  - Go        : gofmt -e, go vet, staticcheck
  - Rust      : cargo check (when run inside a Cargo workspace)
  - Java      : javac -d /tmp/scratch (syntax only)
  - C/C++     : clang -fsyntax-only

Each check runs in a temp dir, has a timeout, and returns a structured
``CheckReport``. The proxy only invokes these when ``SANDBOX_ENABLED``
is true and the response contains a code block matching one of the
detected languages.
"""
from src.sandbox.verifier import (
    CheckReport,
    VerificationResult,
    available_checkers,
    verify_code,
    verify_response,
)

__all__ = [
    "CheckReport",
    "VerificationResult",
    "available_checkers",
    "verify_code",
    "verify_response",
]
