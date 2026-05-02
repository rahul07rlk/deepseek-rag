"""Eval metrics: recall@k, MRR, hit-rate, latency percentiles, cost."""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class CaseResult:
    case_id: str
    query: str
    intent_predicted: str | None
    intent_expected: str | None
    cited_files: list[str]
    must_cite: list[str]
    should_cite: list[str]
    answer: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cache_hit: bool
    confidence: float
    forbidden_hits: list[str]
    missing_contains: list[str]
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.forbidden_hits or self.missing_contains:
            return False
        # All must_cite files have to be present.
        cited_norm = {_norm_path(p) for p in self.cited_files}
        for needed in self.must_cite:
            if _norm_path(needed) not in cited_norm:
                return False
        return True

    @property
    def must_cite_recall(self) -> float:
        if not self.must_cite:
            return 1.0
        cited_norm = {_norm_path(p) for p in self.cited_files}
        hit = sum(1 for n in self.must_cite if _norm_path(n) in cited_norm)
        return hit / len(self.must_cite)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the FIRST must_cite file in the cited list."""
        if not self.must_cite:
            return 1.0
        wanted = {_norm_path(p) for p in self.must_cite}
        for i, f in enumerate(self.cited_files, start=1):
            if _norm_path(f) in wanted:
                return 1.0 / i
        return 0.0


@dataclass
class SuiteResult:
    cases: list[CaseResult]

    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n if self.n else 0.0

    @property
    def mean_recall(self) -> float:
        if not self.cases:
            return 0.0
        return statistics.fmean(c.must_cite_recall for c in self.cases)

    @property
    def mrr(self) -> float:
        if not self.cases:
            return 0.0
        return statistics.fmean(c.reciprocal_rank for c in self.cases)

    @property
    def latency_p50(self) -> float:
        return _percentile([c.latency_s for c in self.cases], 50)

    @property
    def latency_p95(self) -> float:
        return _percentile([c.latency_s for c in self.cases], 95)

    @property
    def cache_hit_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.cache_hit) / self.n

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens_in + c.tokens_out for c in self.cases)

    @property
    def avg_confidence(self) -> float:
        if not self.cases:
            return 0.0
        return statistics.fmean(c.confidence for c in self.cases)

    @property
    def hallucination_count(self) -> int:
        """Cases where forbidden phrases appeared in the answer."""
        return sum(1 for c in self.cases if c.forbidden_hits)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "passed": self.n_passed,
            "pass_rate": round(self.pass_rate, 3),
            "mean_recall": round(self.mean_recall, 3),
            "mrr": round(self.mrr, 3),
            "latency_p50_s": round(self.latency_p50, 3),
            "latency_p95_s": round(self.latency_p95, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "hallucinations": self.hallucination_count,
            "total_tokens": self.total_tokens,
        }


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/").lower().lstrip("./")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
