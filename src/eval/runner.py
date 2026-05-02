"""Eval runner — execute a suite against the live retrieval pipeline.

Two execution modes:

  - ``retrieval``  : drives only ``rag_engine.retrieve`` and grades on
                     ``must_cite``. Fast (~50ms-3s per case), no API cost.
                     Use this in CI / on every change.
  - ``full``       : drives the full proxy (HTTP) and grades the actual
                     answer text on ``contains``/``forbidden`` plus
                     citations. Slow, costs DeepSeek tokens.
                     Use this on release candidates.

Output: a ``SuiteResult`` plus a per-case JSON report under
``logs/eval/<suite>-<timestamp>.json``. Compare two reports with
``python -m src.eval.diff a.json b.json`` (added separately).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from src.eval.dataset import GoldCase, load_suite
from src.eval.metrics import CaseResult, SuiteResult


def run_retrieval_case(case: GoldCase) -> CaseResult:
    """Drive only the retrieval pipeline. Doesn't call DeepSeek."""
    from src.query_router import classify
    from src.rag_engine import retrieve

    t0 = time.perf_counter()
    intent_predicted: str | None = None
    error: str | None = None
    cited_files: list[str] = []
    confidence = 0.0
    tokens_in = 0
    try:
        try:
            route = classify(case.query)
            intent_predicted = getattr(route, "intent", None) or str(route)
        except Exception:
            pass
        ctx_str, tokens_in, metas = retrieve(case.query)
        # Preserve order of citation as emitted by retrieval (matters for MRR).
        seen: set[str] = set()
        for m in metas:
            f = m.get("file") or ""
            if f and f not in seen:
                cited_files.append(f)
                seen.add(f)
        if metas:
            confidence = float(metas[-1].get("confidence", 0.0))
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    if case.max_latency_s is not None and elapsed > case.max_latency_s:
        if not error:
            error = f"latency {elapsed:.2f}s > cap {case.max_latency_s:.2f}s"

    return CaseResult(
        case_id=case.id,
        query=case.query,
        intent_predicted=intent_predicted,
        intent_expected=case.intent,
        cited_files=cited_files,
        must_cite=case.must_cite,
        should_cite=case.should_cite,
        answer="",  # retrieval-only mode doesn't generate an answer
        latency_s=elapsed,
        tokens_in=tokens_in,
        tokens_out=0,
        cache_hit=False,
        confidence=confidence,
        forbidden_hits=[],
        missing_contains=[],
        error=error,
    )


def run_full_case(case: GoldCase, base_url: str = "http://localhost:8000/v1") -> CaseResult:
    """Drive the full proxy via HTTP. Requires the proxy to be running."""
    import httpx

    t0 = time.perf_counter()
    answer = ""
    error: str | None = None
    cited_files: list[str] = []
    cache_hit = False
    tokens_in = tokens_out = 0
    confidence = 0.0
    try:
        with httpx.Client(timeout=180) as cli:
            r = cli.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": case.query}],
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or 0)
            meta = data.get("rag_metadata") or {}
            cache_hit = bool(meta.get("cache_hit"))
            cited_files = list(meta.get("cited_files") or [])
            confidence = float(meta.get("confidence") or 0.0)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    if case.max_latency_s is not None and elapsed > case.max_latency_s:
        if not error:
            error = f"latency {elapsed:.2f}s > cap {case.max_latency_s:.2f}s"

    forbidden_hits = [p for p in case.forbidden if p.lower() in answer.lower()]
    missing = [p for p in case.contains if p.lower() not in answer.lower()]

    return CaseResult(
        case_id=case.id,
        query=case.query,
        intent_predicted=None,  # not exposed in HTTP mode
        intent_expected=case.intent,
        cited_files=cited_files,
        must_cite=case.must_cite,
        should_cite=case.should_cite,
        answer=answer,
        latency_s=elapsed,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=cache_hit,
        confidence=confidence,
        forbidden_hits=forbidden_hits,
        missing_contains=missing,
        error=error,
    )


def run_suite(
    suite_path: str | Path,
    mode: str = "retrieval",
    base_url: str = "http://localhost:8000/v1",
    tag_filter: list[str] | None = None,
) -> SuiteResult:
    cases = load_suite(suite_path)
    if tag_filter:
        wanted = set(tag_filter)
        cases = [c for c in cases if wanted & set(c.tags)]
    runner = run_full_case if mode == "full" else run_retrieval_case
    results: list[CaseResult] = []
    for c in cases:
        if mode == "full":
            results.append(runner(c, base_url=base_url))
        else:
            results.append(runner(c))
    return SuiteResult(cases=results)


def write_report(suite_name: str, result: SuiteResult, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{suite_name}-{ts}.json"
    payload = {
        "suite": suite_name,
        "summary": result.summary(),
        "cases": [asdict(c) for c in result.cases],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
