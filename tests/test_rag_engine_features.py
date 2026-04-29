from src import rag_engine


def test_conversation_aware_query_enriches_vague_followup(monkeypatch):
    history = [
        {
            "role": "assistant",
            "content": (
                "The failure is in `src/auth.py`.\n\n"
                "```python\n"
                "def validate_token(token):\n"
                "    raise ValueError('expired token')\n"
                "```\n"
            ),
        }
    ]

    monkeypatch.setattr(rag_engine, "CONVERSATION_AWARE_RETRIEVAL", True)
    monkeypatch.setattr(rag_engine, "CONVERSATION_HISTORY_MAX_CHARS", 600)
    monkeypatch.setattr(rag_engine, "CONVERSATION_VAGUE_QUERY_CHARS", 40)

    enriched, changed = rag_engine._conversation_aware_query("fix it", history)

    assert changed is True
    assert enriched.startswith("fix it")
    assert "validate_token" in enriched
    assert "expired token" in enriched


def test_conversation_aware_query_leaves_specific_query_alone(monkeypatch):
    monkeypatch.setattr(rag_engine, "CONVERSATION_AWARE_RETRIEVAL", True)

    query = "explain how vector_store delete_by_file removes FAISS ids"
    enriched, changed = rag_engine._conversation_aware_query(query, [])

    assert changed is False
    assert enriched == query
