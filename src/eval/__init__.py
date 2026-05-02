"""Eval harness for the RAG pipeline.

Run with:  python -m src.eval.run --suite default

Without measurement, every "improvement" is vibes. This harness gives
the system a regression baseline for retrieval and answer quality so
changes to the embedder, reranker, prompt, or graph can be evaluated
objectively before being merged.
"""
