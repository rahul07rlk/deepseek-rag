"""CLI entry point for the eval harness.

Usage:
  python -m src.eval.run                                    # default suite, retrieval mode
  python -m src.eval.run --suite evals/default.yaml
  python -m src.eval.run --mode full --base http://localhost:8000/v1
  python -m src.eval.run --tags debug,symbol
  python -m src.eval.run --diff prev.json curr.json         # regression diff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.eval.runner import run_suite, write_report


DEFAULT_SUITE = Path("evals/default.yaml")


def cmd_run(args: argparse.Namespace) -> int:
    suite_path = Path(args.suite)
    if not suite_path.exists():
        print(f"[eval] Suite not found: {suite_path}")
        print(f"[eval] Generate one with: python -m src.eval.run --bootstrap")
        return 2
    print(f"[eval] Running {suite_path} (mode={args.mode})...")
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    result = run_suite(
        suite_path, mode=args.mode, base_url=args.base, tag_filter=tags,
    )
    out_dir = Path("logs/eval")
    report_path = write_report(suite_path.stem, result, out_dir)

    s = result.summary()
    print()
    print(f"  cases       : {s['n']}")
    print(f"  passed      : {s['passed']} ({s['pass_rate']*100:.1f}%)")
    print(f"  recall      : {s['mean_recall']*100:.1f}%")
    print(f"  MRR         : {s['mrr']:.3f}")
    print(f"  latency p50 : {s['latency_p50_s']*1000:.0f} ms")
    print(f"  latency p95 : {s['latency_p95_s']*1000:.0f} ms")
    print(f"  cache hits  : {s['cache_hit_rate']*100:.1f}%")
    print(f"  avg conf    : {s['avg_confidence']:.2f}")
    print(f"  hallucs     : {s['hallucinations']}")
    print(f"  tokens      : {s['total_tokens']}")
    print(f"  report      : {report_path}")
    print()

    if args.fail_below is not None and s["pass_rate"] < args.fail_below:
        print(f"[eval] FAIL: pass_rate {s['pass_rate']:.2f} < threshold {args.fail_below}")
        return 1
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.prev).read_text(encoding="utf-8"))
    b = json.loads(Path(args.curr).read_text(encoding="utf-8"))
    sa, sb = a["summary"], b["summary"]
    print(f"  metric           prev      curr      delta")
    print(f"  ---------------  --------  --------  --------")
    for key in ("pass_rate", "mean_recall", "mrr", "latency_p50_s",
                "latency_p95_s", "avg_confidence", "hallucinations"):
        va, vb = sa.get(key, 0), sb.get(key, 0)
        delta = vb - va
        print(f"  {key:<16} {va:<8.3f}  {vb:<8.3f}  {delta:+.3f}")
    # Per-case regressions: passed before, failing now.
    by_id_a = {c["case_id"]: c for c in a["cases"]}
    by_id_b = {c["case_id"]: c for c in b["cases"]}
    regressed: list[str] = []
    for cid, ca in by_id_a.items():
        cb = by_id_b.get(cid)
        if cb is None:
            continue
        passed_a = not (ca.get("error") or ca.get("forbidden_hits") or ca.get("missing_contains"))
        passed_b = not (cb.get("error") or cb.get("forbidden_hits") or cb.get("missing_contains"))
        if passed_a and not passed_b:
            regressed.append(cid)
    if regressed:
        print(f"\nREGRESSED ({len(regressed)}):")
        for cid in regressed:
            print(f"  - {cid}")
    return 1 if regressed else 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Generate a starter suite so the user has something to edit."""
    out = Path(args.suite)
    out.parent.mkdir(parents=True, exist_ok=True)
    starter = """\
# Starter eval suite — edit these to reflect questions about YOUR repos.
# Run: python -m src.eval.run --suite evals/default.yaml

- id: example-overview
  query: what does this repo do?
  intent: OVERVIEW
  must_cite:
    - README.md
  contains:
    - rag
  tags: [overview]

- id: example-symbol
  query: where is retrieve defined?
  intent: SYMBOL_LOOKUP
  must_cite:
    - src/rag_engine.py
  tags: [symbol]
"""
    out.write_text(starter, encoding="utf-8")
    print(f"[eval] Starter suite written to {out}")
    print(f"[eval] Edit it to add real test cases for your codebase, then run:")
    print(f"       python -m src.eval.run --suite {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek RAG eval harness")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE),
                        help="Path to YAML/JSON suite file")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval",
                        help="retrieval-only (fast, no API) or full proxy (slow, costs tokens)")
    parser.add_argument("--base", default="http://localhost:8000/v1",
                        help="Proxy base URL when --mode=full")
    parser.add_argument("--tags", default="",
                        help="Comma-separated tags to filter cases")
    parser.add_argument("--fail-below", type=float, default=None,
                        help="Exit non-zero if pass_rate falls below this (CI gate)")
    parser.add_argument("--diff", nargs=2, metavar=("PREV", "CURR"),
                        help="Diff two report JSON files")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Write a starter suite to --suite if none exists")
    args = parser.parse_args(argv)

    if args.diff:
        ns = argparse.Namespace(prev=args.diff[0], curr=args.diff[1])
        return cmd_diff(ns)
    if args.bootstrap:
        return cmd_bootstrap(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
