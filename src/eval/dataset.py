"""Gold-pair dataset format for eval.

A suite is a YAML file with a list of test cases. Each case has:

  - ``id``         : stable identifier (used in regression diffs)
  - ``query``      : the user question, exactly as it would arrive at the proxy
  - ``intent``     : optional expected route classification
  - ``must_cite``  : list of file paths that MUST appear in retrieval
                    (recall@k passes when *all* of these are cited)
  - ``should_cite``: list of file paths that *may* appear (boost MRR)
  - ``forbidden``  : phrases that, if present in the answer, fail the case
                    (used to catch hallucinations / wrong file references)
  - ``contains``   : phrases the answer should contain (substring match)
  - ``max_latency_s``: optional p100 latency cap for this case
  - ``tags``       : freeform labels for filtering ("debug", "symbol", …)

Example (place in ``evals/<suite>.yaml``):

    - id: watcher-debounce
      query: how does the watcher debounce work?
      intent: HOW_X_WORKS
      must_cite:
        - src/watcher.py
      should_cite:
        - src/indexer.py
      contains:
        - debounce
      forbidden:
        - "in C++"
      tags: [behavior, watcher]

The format is text-only on purpose — gold pairs are version-controlled
alongside the codebase they describe, so a senior dev can update them
in the same PR that changes the code's behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldCase:
    id: str
    query: str
    intent: str | None = None
    must_cite: list[str] = field(default_factory=list)
    should_cite: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    contains: list[str] = field(default_factory=list)
    max_latency_s: float | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldCase":
        return cls(
            id=str(d["id"]),
            query=str(d["query"]),
            intent=d.get("intent"),
            must_cite=list(d.get("must_cite") or []),
            should_cite=list(d.get("should_cite") or []),
            forbidden=list(d.get("forbidden") or []),
            contains=list(d.get("contains") or []),
            max_latency_s=d.get("max_latency_s"),
            tags=list(d.get("tags") or []),
        )


def load_suite(path: str | Path) -> list[GoldCase]:
    """Load a suite from YAML. Falls back to JSON if YAML isn't available."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval suite not found: {p}")
    text = p.read_text(encoding="utf-8")
    data: list[dict] | None = None
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(text)
        except ImportError:
            # Crude YAML→JSON fallback: only the simplest structure.
            import json
            data = json.loads(text)
    else:
        import json
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Suite {p} must be a list of cases at top level.")
    return [GoldCase.from_dict(c) for c in data]
