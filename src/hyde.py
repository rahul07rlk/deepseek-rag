"""HyDE — Hypothetical Document Embeddings.

For prose-heavy queries ("how does the cache eviction work"), real code uses
different vocabulary than questions: ``LRU.evict()``, ``maxAge``, ``ttl_s``.
The bi-encoder has to bridge that gap on its own and often fails.

HyDE works around this by asking a fast LLM to write a SHORT hypothetical
answer first, then embedding that answer (concatenated with the original
query, "Fused HyDE") and using the result as the vector query. The
hypothetical content is dense with code-shaped vocabulary, which lines up
much better with the embeddings of actual code chunks.

Why it's worth it
-----------------
Per the original paper (Gao et al. 2023) and a long stream of follow-ups,
HyDE matches or beats fine-tuned retrievers on out-of-domain prose, with
zero training. For us, the cost is a single ``deepseek-v4-flash`` call with
``thinking=disabled`` — ~1-2 seconds added latency, in exchange for
materially better top-10 recall on broad questions.

Failure handling
----------------
HyDE is "additive": on any failure (timeout, API error, empty content) we
return the original query and the rest of the pipeline keeps working.

This module is async-first because the proxy server is async; ``rag_engine``
gets the hypothetical pre-computed and passed in via ``vector_query_override``.
"""
from __future__ import annotations

from threading import RLock

import httpx

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_URL,
    HYDE_ENABLED,
    HYDE_MAX_TOKENS,
    HYDE_MODEL,
    HYDE_TIMEOUT,
)
from src.utils.logger import get_logger

logger = get_logger("hyde", "proxy.log")


_SYSTEM_PROMPT = (
    "You are an expert software engineer answering a code-search question. "
    "Write a SHORT hypothetical answer (3-5 sentences) describing what the "
    "relevant code probably looks like — likely function names, types, key "
    "operations, file structure, imports. Use code-shaped vocabulary "
    "(camelCase identifiers, types, function calls). Do NOT preface or "
    "explain; describe the code as if quoting documentation. If you do not "
    "know the answer, give your best guess — accuracy matters less than "
    "vocabulary density for this task."
)


_cache: dict[str, str] = {}
_cache_lock = RLock()


def _from_cache(query: str) -> str | None:
    with _cache_lock:
        return _cache.get(query)


def _to_cache(query: str, fused: str) -> None:
    with _cache_lock:
        # Cap process-local cache so a runaway loop doesn't grow forever.
        if len(_cache) > 256:
            _cache.pop(next(iter(_cache)))
        _cache[query] = fused


async def generate_hypothetical(query: str) -> str:
    """Return the fused HyDE query: ``"{query}\\n\\n{hypothetical}"``.

    Falls back to the original query on any failure so callers can use the
    return value unconditionally as the vector-search input.
    """
    if not HYDE_ENABLED or not query or not DEEPSEEK_API_KEY:
        return query

    cached = _from_cache(query)
    if cached is not None:
        return cached

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    # thinking=disabled → fast path. temperature 0.2 keeps the hypothetical
    # focused on plausible code shapes rather than creative tangents.
    payload = {
        "model": HYDE_MODEL,
        "stream": False,
        "max_tokens": HYDE_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=HYDE_TIMEOUT) as client:
            resp = await client.post(DEEPSEEK_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            logger.warning(
                f"HyDE LLM call returned {resp.status_code}; using raw query."
            )
            return query
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        if not content:
            return query
        # Fused HyDE: keep the original query so symbol-like tokens still
        # influence the embedding even if the hypothetical drifts.
        fused = f"{query}\n\n{content}"
        _to_cache(query, fused)
        logger.info(f"HyDE expanded query (+{len(content)} chars).")
        return fused
    except httpx.TimeoutException:
        logger.warning("HyDE call timed out; using raw query.")
        return query
    except Exception as e:
        logger.warning(f"HyDE call failed ({type(e).__name__}: {e}); using raw query.")
        return query
