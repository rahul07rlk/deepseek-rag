"""Confidence-calibrated routing.

Every retrieval emits a calibrated confidence score in [0, 1] based on
multiple signals (rerank top score, exact-symbol hit, file locality,
graph proximity, recent change). A policy then decides what the
proxy should do with the result:

  - ``ANSWER_NOW``     : confidence high → skip the agentic loop, send
                         a single-shot answer with the seed context
  - ``ONE_MORE_ROUND`` : medium confidence → one extra retrieval pass
                         (multi-query, broader candidate pool)
  - ``AGENTIC_SEARCH`` : low confidence → engage the full agentic loop
                         so the model can investigate
  - ``ASK_CLARIFY``    : near-zero confidence + ambiguous query → return
                         a clarifying question instead of guessing

The policy lives in this module so every retrieval path uses the same
decision logic. Thresholds are env-configurable via ``CONFIDENCE_*``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Policy(str, Enum):
    ANSWER_NOW = "ANSWER_NOW"
    ONE_MORE_ROUND = "ONE_MORE_ROUND"
    AGENTIC_SEARCH = "AGENTIC_SEARCH"
    ASK_CLARIFY = "ASK_CLARIFY"


@dataclass
class ConfidenceSignals:
    """Raw signals collected during retrieval; weighted into a final score."""
    top_rerank: float = 0.0          # 0..1, sigmoid of cross-encoder
    second_rerank: float = 0.0       # margin signal
    exact_symbol_match: bool = False
    path_match: bool = False
    graph_hit: bool = False          # at least one chunk was added by graph expansion
    test_or_change_hit: bool = False
    n_files_cited: int = 0
    n_blocks_emitted: int = 0
    query_is_short: bool = False     # <40 chars → ambiguous risk

    def calibrate(self) -> float:
        """Weighted sum mapped to [0, 1].

        Weights tuned by hand against typical eval suites; revisit when
        the eval harness shows systematic over/underconfidence on a
        particular query class.
        """
        s = 0.0
        s += 0.50 * float(min(1.0, max(0.0, self.top_rerank)))
        margin = max(0.0, self.top_rerank - self.second_rerank)
        s += 0.10 * min(1.0, margin * 4.0)  # big top→2nd gap = confident
        if self.exact_symbol_match:
            s += 0.15
        if self.path_match:
            s += 0.08
        if self.graph_hit:
            s += 0.05
        if self.test_or_change_hit:
            s += 0.05
        if self.n_files_cited >= 3:
            s += 0.05
        if self.n_blocks_emitted == 0:
            s = min(s, 0.10)  # nothing retrieved → cap at 0.1
        if self.query_is_short:
            s -= 0.05
        return max(0.0, min(1.0, s))


def _f(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


# Thresholds (env-configurable).
ANSWER_NOW_FLOOR = _f("CONFIDENCE_ANSWER_NOW", 0.72)
AGENTIC_FLOOR = _f("CONFIDENCE_AGENTIC", 0.30)
CLARIFY_CEILING = _f("CONFIDENCE_CLARIFY", 0.10)


def decide(signals: ConfidenceSignals, intent: str | None = None) -> tuple[Policy, float]:
    """Map calibrated confidence + intent to a routing policy.

    Returns (policy, confidence). Some intents short-circuit the policy:
      - DEBUG / HOW_X_WORKS always benefit from the agent investigating,
        so they prefer AGENTIC_SEARCH unless confidence is unambiguously
        high.
      - SYMBOL_LOOKUP at high confidence answers immediately.
    """
    confidence = signals.calibrate()
    if confidence <= CLARIFY_CEILING and signals.n_blocks_emitted == 0:
        return Policy.ASK_CLARIFY, confidence
    if confidence >= ANSWER_NOW_FLOOR:
        # DEBUG: even high confidence still benefits from running the
        # tool loop so the model can verify by reading neighboring code.
        if intent == "DEBUG":
            return Policy.ONE_MORE_ROUND, confidence
        return Policy.ANSWER_NOW, confidence
    if confidence >= AGENTIC_FLOOR:
        return Policy.ONE_MORE_ROUND, confidence
    return Policy.AGENTIC_SEARCH, confidence


def hint_for(policy: Policy, confidence: float) -> str:
    """Short string injected into the system prompt so the LLM understands
    the retrieval state. Stable phrasing — eval cases match on this."""
    if policy is Policy.ANSWER_NOW:
        return f"[retrieval confidence: HIGH ({confidence:.2f}) — answer directly.]"
    if policy is Policy.ONE_MORE_ROUND:
        return (
            f"[retrieval confidence: MEDIUM ({confidence:.2f}) — verify with one "
            "extra investigation if you spot ambiguity.]"
        )
    if policy is Policy.AGENTIC_SEARCH:
        return (
            f"[retrieval confidence: LOW ({confidence:.2f}) — the seed context "
            "may be insufficient. Use tools to investigate before answering.]"
        )
    return (
        f"[retrieval confidence: NEAR-ZERO ({confidence:.2f}) — ask the user a "
        "clarifying question rather than guessing.]"
    )
