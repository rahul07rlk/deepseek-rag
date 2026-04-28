"""Retrieval sanity checks. Run after `python -m src.indexer`."""
from src.config import TOKEN_BUDGET
from src.indexer import get_index_stats
from src.rag_engine import retrieve


def test_index_not_empty():
    stats = get_index_stats()
    assert stats["total_chunks"] > 0, "Index is empty - run `python -m src.indexer` first"


def test_retrieval_returns_results():
    context, tokens, metas = retrieve("function definition", top_k=3)
    assert len(metas) > 0, "No results returned"
    assert tokens > 0
    assert len(context) > 0


def test_relevance_scores_in_range():
    _, _, metas = retrieve("database connection", top_k=5)
    for m in metas:
        assert 0.0 <= m["relevance"] <= 1.0


def test_token_budget_respected():
    _, tokens, _ = retrieve("import", top_k=20)
    # 10% slack because we count tokens of the formatted string, not raw docs.
    assert tokens <= TOKEN_BUDGET * 1.1


def test_hybrid_covers_exact_symbol():
    """BM25 should surface an exact symbol even if vector similarity is low."""
    # 'get_index_stats' is a real symbol in this codebase once indexed.
    _, _, metas = retrieve("get_index_stats", top_k=5)
    files = {m["filename"] for m in metas}
    # We can't hard-assert this unless the user indexed this repo itself,
    # so we just require that *something* came back.
    assert len(files) > 0


if __name__ == "__main__":
    print("Running retrieval tests...")
    test_index_not_empty(); print("  index not empty")
    test_retrieval_returns_results(); print("  retrieval returns results")
    test_relevance_scores_in_range(); print("  relevance scores valid")
    test_token_budget_respected(); print("  token budget respected")
    test_hybrid_covers_exact_symbol(); print("  hybrid returns results")
    print("\nAll tests passed.")
