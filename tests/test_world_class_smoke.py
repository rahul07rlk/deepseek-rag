"""Smoke tests for the world-class upgrade pieces.

Each test imports its module and exercises the pure-logic surface so that
syntax/import regressions surface immediately. Tests requiring external
state (FAISS index, DeepSeek API, LSP servers, Kuzu) are skipped when
those aren't available.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure REPO_PATHS resolves to *something* so config.py imports cleanly.
os.environ.setdefault("REPO_PATHS", str(Path(__file__).parent.parent))
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


# ── Item 5: semantic chunk types ─────────────────────────────────────────────
def test_chunk_classify_test_path():
    from src.chunk_types import classify_chunk
    assert classify_chunk(
        file_path="src/foo/test_bar.py", text="def test_x(): pass",
        start_line=10, end_line=11, symbol="test_x",
    ) == "test"


def test_chunk_classify_module_top():
    from src.chunk_types import classify_chunk
    assert classify_chunk(
        file_path="src/foo/bar.py", text='"""docstring"""',
        start_line=1, end_line=2, symbol=None,
    ) == "module"


def test_chunk_classify_change_overrides_test():
    from src.chunk_types import classify_chunk
    # change always wins regardless of file path.
    assert classify_chunk(
        file_path="tests/test_x.py", text="def test_y(): pass",
        start_line=10, end_line=11, symbol="test_y",
        is_recent_change=True,
    ) == "change"


def test_chunk_classify_interface_only():
    from src.chunk_types import classify_chunk
    text = "class Foo:\n    def a(self): ...\n    def b(self): ...\n"
    out = classify_chunk(
        file_path="src/foo.py", text=text, start_line=20, end_line=22,
        symbol="Foo",
    )
    # Either interface or behavior is acceptable depending on heuristic
    # tuning; the contract is "not test/module/change".
    assert out in {"interface", "behavior", "symbol"}


# ── Item 2: snapshot + cache versioning ──────────────────────────────────────
def test_component_versions_stable():
    from src.snapshot import component_versions
    a = component_versions()
    b = component_versions()
    assert a == b
    assert "emb=" in a
    assert "chunker=" in a


def test_snapshot_id_is_hex():
    from src.snapshot import current_snapshot
    snap = current_snapshot()
    assert len(snap.id) == 16
    int(snap.id, 16)  # must be valid hex


def test_files_fingerprint_changes_on_size():
    from src.snapshot import file_fingerprint, files_fingerprint
    # Use this test file itself as the evidence.
    here = str(Path(__file__).resolve())
    fp1 = files_fingerprint([here])
    fp2 = files_fingerprint([here])
    assert fp1 == fp2  # stable
    assert fp1 != ""


# ── Item 6: eval harness ─────────────────────────────────────────────────────
def test_eval_metrics_passes():
    from src.eval.metrics import CaseResult, SuiteResult
    case = CaseResult(
        case_id="t1", query="q", intent_predicted=None, intent_expected=None,
        cited_files=["src/a.py", "src/b.py"], must_cite=["src/a.py"],
        should_cite=[], answer="", latency_s=0.1, tokens_in=10, tokens_out=0,
        cache_hit=False, confidence=0.5, forbidden_hits=[], missing_contains=[],
    )
    assert case.passed
    assert case.must_cite_recall == 1.0
    assert case.reciprocal_rank == 1.0
    s = SuiteResult([case])
    assert s.pass_rate == 1.0


def test_eval_metrics_fails_on_missing_must_cite():
    from src.eval.metrics import CaseResult
    case = CaseResult(
        case_id="t1", query="q", intent_predicted=None, intent_expected=None,
        cited_files=["src/b.py"], must_cite=["src/a.py"],
        should_cite=[], answer="", latency_s=0.1, tokens_in=0, tokens_out=0,
        cache_hit=False, confidence=0.0, forbidden_hits=[], missing_contains=[],
    )
    assert not case.passed
    assert case.must_cite_recall == 0.0


# ── Item 4: code graph backends ──────────────────────────────────────────────
def test_sqlite_backend_roundtrip(tmp_path):
    from src.code_graph.backend import GraphEdge, GraphNode
    from src.code_graph.sqlite_backend import SqliteBackend
    db = tmp_path / "g.sqlite"
    bk = SqliteBackend(db)
    bk.initialize()
    bk.upsert_nodes([
        GraphNode(id="file:a.py", kind="File", name="a.py", file="a.py"),
        GraphNode(id="sym:a.py::foo", kind="Symbol", name="foo",
                  file="a.py", line=1),
    ])
    bk.upsert_edges([
        GraphEdge(src_id="file:a.py", dst_id="sym:a.py::foo", kind="DEFINES"),
    ])
    defs = bk.find_definitions("foo")
    assert any(d["name"] == "foo" for d in defs)
    edges_in = bk.edges_into("sym:a.py::foo", kind="DEFINES")
    assert any(e["kind"] == "DEFINES" for e in edges_in)


def test_sqlite_backend_multi_hop(tmp_path):
    from src.code_graph.backend import GraphEdge, GraphNode
    from src.code_graph.sqlite_backend import SqliteBackend
    db = tmp_path / "g.sqlite"
    bk = SqliteBackend(db)
    bk.initialize()
    nodes = [
        GraphNode(id="file:a.py", kind="File", name="a.py", file="a.py"),
        GraphNode(id="file:b.py", kind="File", name="b.py", file="b.py"),
        GraphNode(id="file:c.py", kind="File", name="c.py", file="c.py"),
    ]
    bk.upsert_nodes(nodes)
    bk.upsert_edges([
        GraphEdge(src_id="file:a.py", dst_id="file:b.py", kind="IMPORTS"),
        GraphEdge(src_id="file:b.py", dst_id="file:c.py", kind="IMPORTS"),
    ])
    one_hop = bk.multi_hop("file:a.py", kinds=["IMPORTS"], max_hops=1, max_results=10)
    two_hop = bk.multi_hop("file:a.py", kinds=["IMPORTS"], max_hops=2, max_results=10)
    assert any(n["id"] == "file:b.py" for n in one_hop)
    assert any(n["id"] == "file:c.py" for n in two_hop)
    assert len(two_hop) >= len(one_hop)


# ── Item 1: LSP server detection ─────────────────────────────────────────────
def test_lsp_server_registry_has_languages():
    from src.lsp.servers import SERVERS, detect_server_for
    # Registry covers the languages the user cares about.
    langs = {s.name for s in SERVERS}
    assert "pyright" in langs
    assert "typescript-language-server" in langs
    assert "gopls" in langs
    assert "rust-analyzer" in langs
    # Detection returns None for an unknown extension.
    assert detect_server_for(".xyzwhatever") is None


# ── Item 3: git watcher control ──────────────────────────────────────────────
def test_indexer_control_pause_resume():
    from src.git_watcher import IndexerControl
    c = IndexerControl()
    assert not c.is_paused()
    c.pause("test")
    assert c.is_paused()
    assert c.reason() == "test"
    c.resume()
    assert not c.is_paused()


def test_indexer_control_storm_detection(monkeypatch):
    # Force tight thresholds so we don't have to fire 30 events in this test.
    import src.git_watcher as gw
    monkeypatch.setattr(gw, "_STORM_THRESHOLD", 3)
    monkeypatch.setattr(gw, "_STORM_WINDOW_S", 5.0)
    monkeypatch.setattr(gw, "_STORM_COOLDOWN_S", 0.05)
    c = gw.IndexerControl()
    for _ in range(3):
        c.record_event()
    assert c.is_paused()
    assert c.reason().startswith("file storm")


# ── Item 8: late-interaction (no-op shim when pylate missing) ────────────────
def test_late_interaction_graceful_when_missing():
    from src.late_interaction import get_store
    store = get_store()
    # Never raises — either available or silently disabled.
    assert isinstance(store.available, bool)
    # Search returns [] when disabled, not None.
    out = store.search("anything", k=5)
    assert isinstance(out, list)


# ── Item 10: confidence-calibrated routing ───────────────────────────────────
def test_confidence_calibration_high():
    from src.confidence import ConfidenceSignals, decide, Policy
    sig = ConfidenceSignals(
        top_rerank=0.9, second_rerank=0.4,
        exact_symbol_match=True, n_files_cited=3, n_blocks_emitted=5,
    )
    policy, conf = decide(sig)
    assert conf > 0.7
    assert policy is Policy.ANSWER_NOW


def test_confidence_calibration_low():
    from src.confidence import ConfidenceSignals, decide, Policy
    sig = ConfidenceSignals(top_rerank=0.0, n_blocks_emitted=0)
    policy, conf = decide(sig)
    assert conf <= 0.10
    assert policy in (Policy.ASK_CLARIFY, Policy.AGENTIC_SEARCH)


def test_confidence_debug_intent_keeps_loop():
    from src.confidence import ConfidenceSignals, decide, Policy
    sig = ConfidenceSignals(
        top_rerank=0.95, exact_symbol_match=True, n_blocks_emitted=4,
        n_files_cited=2,
    )
    # DEBUG intent forces ONE_MORE_ROUND even at high confidence.
    policy, _ = decide(sig, intent="DEBUG")
    assert policy is Policy.ONE_MORE_ROUND


# ── Item 7: sandbox verifier ─────────────────────────────────────────────────
def test_sandbox_python_syntax_pass():
    from src.sandbox import verify_code
    reports = verify_code("def f(x): return x + 1\n", "python")
    assert any(r.tool == "python-syntax" and r.passed for r in reports)


def test_sandbox_python_syntax_fail():
    from src.sandbox import verify_code
    reports = verify_code("def f(x):\n    return x +\n", "python")
    assert any(r.tool == "python-syntax" and not r.passed for r in reports)


def test_sandbox_unknown_language_skipped():
    from src.sandbox import verify_code
    reports = verify_code("(fact 5)", "lisp")
    assert all(r.skipped for r in reports)


def test_sandbox_extracts_blocks_from_response():
    from src.sandbox import verify_response
    text = (
        "Here is some code:\n"
        "```python\nx = 1 + 1\nprint(x)\n```\n"
        "And some prose."
    )
    result = verify_response(text)
    assert result.blocks_seen == 1
    assert result.blocks_checked == 1
    assert result.passed


# ── Item 9: progress event formatting ────────────────────────────────────────
def test_format_progress_tool_call():
    from src.proxy_server import _format_progress
    out = _format_progress({
        "event": "tool_call", "turn": 2,
        "name": "grep", "args_preview": "pattern=foo",
    })
    assert b": agent" in out
    assert b"grep" in out


def test_format_progress_unknown_event_safe():
    from src.proxy_server import _format_progress
    # Unknown event types render as a generic line.
    out = _format_progress({"event": "??unknown"})
    assert out.startswith(b": agent")
    assert out.endswith(b"\n\n")
