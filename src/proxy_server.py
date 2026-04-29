"""OpenAI-compatible proxy that injects RAG context and forwards to DeepSeek.

Two modes:

  - Agentic (AGENTIC_ENABLED=true, default): the proxy seeds context (repo
    map + a slice of one-shot retrieval) and exposes retrieval primitives
    as tools. DeepSeek calls them iteratively until it has what it needs.
    Works for both streaming and non-streaming clients — the multi-turn
    tool conversation runs non-streamed internally, then the final answer
    is synthesized into an OpenAI SSE stream when the client requested one.

  - One-shot (AGENTIC_ENABLED=false): legacy behavior. ``build_messages``
    stuffs everything into the system prompt up front and DeepSeek answers
    in a single round-trip. Native streaming pass-through.

The last user message is always the retrieval query. Incoming system
messages are dropped — we always inject our own.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.agentic import TOOLS, build_seed_messages, dispatch
from src.config import (
    AGENTIC_ENABLED,
    AGENTIC_MAX_TOOL_TURNS,
    AGENTIC_PARALLEL_TOOL_CALLS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING,
    DEEPSEEK_TIMEOUT,
    DEEPSEEK_URL,
    HYDE_ENABLED,
    PROXY_HOST,
    PROXY_PORT,
    QUERY_ROUTER_ENABLED,
    SEMANTIC_CACHE_ENABLED,
)
from src.indexer import get_index_stats
from src.rag_engine import build_messages
from src.utils.logger import get_logger

logger = get_logger("proxy", "proxy.log")

app = FastAPI(
    title="DeepSeek RAG Proxy",
    description="Transparent RAG-enhanced proxy for DeepSeek API",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "status": "Running",
        "model": DEEPSEEK_MODEL,
        "agentic": AGENTIC_ENABLED,
        "index_stats": get_index_stats(),
        "endpoint": f"http://localhost:{PROXY_PORT}/v1/chat/completions",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict:
    base = get_index_stats()
    try:
        from src.answer_cache import stats as cache_stats
        base["semantic_cache"] = cache_stats()
    except Exception:
        base["semantic_cache"] = {"enabled": SEMANTIC_CACHE_ENABLED, "entries": 0}
    base["features"] = {
        "agentic": AGENTIC_ENABLED,
        "query_router": QUERY_ROUTER_ENABLED,
        "hyde": HYDE_ENABLED,
        "semantic_cache": SEMANTIC_CACHE_ENABLED,
        "parallel_tool_calls": AGENTIC_PARALLEL_TOOL_CALLS,
    }
    return base


@app.post("/cache/reset")
async def cache_reset() -> dict:
    """Wipe the semantic answer cache. Call after large code changes if you
    want to force fresh answers immediately (the watcher does this for you
    on save, but the manual hook is here for headless edits)."""
    try:
        from src.answer_cache import reset as cache_reset_fn
        cache_reset_fn()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extract_query_and_history(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Last user message becomes the query; everything before it (user+assistant)
    becomes history. Incoming system messages are dropped - we inject our own."""
    if not messages:
        return None, []

    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None, []

    query = messages[last_user_idx].get("content", "")
    history = [
        m for m in messages[:last_user_idx]
        if m.get("role") in ("user", "assistant")
    ]
    return query, history


def _base_payload(model: str, max_tokens: int, stream: bool) -> dict:
    """Common DeepSeek payload boilerplate (thinking mode, reasoning effort)."""
    payload: dict = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if DEEPSEEK_THINKING == "enabled":
        # temperature/top_p are silently ignored in thinking mode — omit them.
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    return payload


def _log_usage(usage: dict) -> None:
    """Surface DeepSeek prompt-cache numbers when the API returns them.
    Cache hits are how the agentic loop stays affordable across tool turns —
    every iteration after the first should hit cache on the seed prefix."""
    if not isinstance(usage, dict):
        return
    total = usage.get("total_tokens", "?")
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is not None or miss is not None:
        logger.info(
            f"Tokens: total={total} | cache_hit={hit} | cache_miss={miss}"
        )
    else:
        logger.info(f"Tokens: total={total}")


# ── Agentic loop ──────────────────────────────────────────────────────────────
async def _run_agentic_loop(
    seed_messages: list[dict],
    model: str,
    max_tokens: int,
    headers: dict,
    max_turns: int = AGENTIC_MAX_TOOL_TURNS,
) -> dict:
    """Drive a tool-use conversation with DeepSeek until it returns a final
    assistant message (no tool_calls) or ``max_turns`` is reached.

    Two performance features per session:
      - **Tool memoization**: ``memo`` dict captures every (tool, args) →
        result pair within this loop. Repeated calls (Q1's read_file flood)
        return instantly with a "(cached from earlier)" prefix so the LLM
        understands why the result was free.
      - **Parallel dispatch**: when one assistant turn returns multiple
        independent tool_calls, we run them concurrently with
        ``asyncio.to_thread`` + ``asyncio.gather``. Each tool function is
        thread-safe (pure read paths over the index).

    Returns the final OpenAI-format completion response. On loop exhaustion
    we ask the model for a final answer with ``tool_choice="none"`` so the
    client doesn't see an empty content reply.
    """
    messages = list(seed_messages)
    last_response: dict | None = None
    # Per-session tool memo. Lives only for this loop.
    tool_memo: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
        for turn in range(max_turns + 1):
            payload = _base_payload(model, max_tokens, stream=False)
            payload["messages"] = messages
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"
            # Allow DeepSeek to emit multiple tool calls in one turn so we
            # can dispatch them in parallel below.
            if AGENTIC_PARALLEL_TOOL_CALLS:
                payload["parallel_tool_calls"] = True

            resp = await client.post(DEEPSEEK_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error(f"DeepSeek {resp.status_code} (turn {turn}): {resp.text}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"DeepSeek API error: {resp.text}",
                )
            last_response = resp.json()
            _log_usage(last_response.get("usage", {}))

            choice = (last_response.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                logger.info(f"Agentic loop done after {turn} tool turn(s).")
                return last_response

            # Append the assistant's tool-call message verbatim, then run
            # each call and feed the results back in.
            #
            # DeepSeek thinking mode REQUIRES the assistant's reasoning_content
            # from each turn to be echoed back in subsequent calls — otherwise
            # the API rejects with: "The `reasoning_content` in the thinking
            # mode must be passed back to the API." Plain OpenAI clients drop
            # this field; we explicitly preserve it.
            assistant_msg: dict = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
            reasoning_content = msg.get("reasoning_content")
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_msg)

            # ── Dispatch (parallel when ≥2 independent calls) ─────────────────
            if AGENTIC_PARALLEL_TOOL_CALLS and len(tool_calls) >= 2:
                async def _run_one(tc: dict) -> tuple[dict, str]:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "") or ""
                    logger.info(f"[turn {turn}] tool: {name}({args[:120]})")
                    result = await asyncio.to_thread(dispatch, name, args, tool_memo)
                    return tc, result
                results = await asyncio.gather(*(_run_one(tc) for tc in tool_calls))
                for tc, result in results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    })
            else:
                for tc in tool_calls:
                    fn = (tc.get("function") or {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "") or ""
                    logger.info(f"[turn {turn}] tool: {name}({args[:120]})")
                    result = dispatch(name, args, tool_memo)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    })

        # ── Max-turn fallback: force a final content answer. ──────────────────
        # When the loop exits via budget exhaustion, last_response has
        # tool_calls but no usable content. Returning it makes Continue
        # render an empty reply (and downstream stream-synthesis can crash on
        # the empty content path). Make one more call with tool_choice="none"
        # so the model summarizes what it learned using the accumulated
        # context. The messages list is well-formed (ends in tool responses)
        # so the API accepts it.
        logger.warning(
            f"Agentic loop hit max turns ({max_turns}); "
            "forcing a final answer with tool_choice=none."
        )
        final_payload = _base_payload(model, max_tokens, stream=False)
        final_payload["messages"] = messages
        final_payload["tools"] = TOOLS
        final_payload["tool_choice"] = "none"
        try:
            resp = await client.post(DEEPSEEK_URL, headers=headers, json=final_payload)
        except Exception as e:
            logger.exception(f"Final-answer call failed: {e}")
            return last_response or {"error": "agentic loop produced no response"}
        if resp.status_code != 200:
            logger.error(f"Final-answer call DeepSeek {resp.status_code}: {resp.text}")
            return last_response or {"error": "agentic loop produced no response"}
        final_response = resp.json()
        _log_usage(final_response.get("usage", {}))
        return final_response


# ── Synthetic streaming for the agentic path ─────────────────────────────────
# The OpenAI streaming protocol can't cleanly express a multi-turn tool
# conversation, so we run the agentic loop fully non-streamed and then
# repackage its final completion as an SSE stream. To clients (Continue,
# OpenWebUI, curl --no-buffer) this looks identical to a native stream;
# they just receive the entire content in a small number of deltas at the
# end of the loop. SSE comments (`: heartbeat`) are emitted while the loop
# runs so reverse-proxies and HTTP clients don't drop the idle connection.
_HEARTBEAT_INTERVAL_S = 8.0


def _sse_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> bytes:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _synthesize_sse_from_completion(result: dict, model: str):
    """Yield SSE bytes that recreate a streaming completion from a finished one.
    Splits the final content into ~400-char deltas so clients see progressive
    rendering instead of one wall of text."""
    completion_id = result.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(result.get("created") or time.time())
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    finish = choice.get("finish_reason") or "stop"

    yield _sse_chunk(completion_id, created, model, {"role": "assistant"})
    if reasoning:
        # Many OpenAI-compatible clients (including Continue) accept
        # reasoning_content deltas; clients that don't simply ignore them.
        for i in range(0, len(reasoning), 400):
            yield _sse_chunk(
                completion_id, created, model,
                {"reasoning_content": reasoning[i : i + 400]},
            )
    if content:
        for i in range(0, len(content), 400):
            yield _sse_chunk(
                completion_id, created, model,
                {"content": content[i : i + 400]},
            )
    yield _sse_chunk(completion_id, created, model, {}, finish_reason=finish)
    yield b"data: [DONE]\n\n"


async def _stream_agentic(
    seed_messages: list[dict],
    model: str,
    max_tokens: int,
    headers: dict,
    max_turns: int = AGENTIC_MAX_TOOL_TURNS,
    on_complete=None,
):
    """Async generator yielding SSE bytes. Runs the agentic loop in a task
    while emitting periodic SSE comments so the client connection stays alive
    through long tool sessions, then synthesizes the final completion into
    OpenAI delta chunks.

    ``on_complete`` (sync callable) gets the final completion dict so the
    caller can persist the answer to the semantic cache.
    """
    loop_task = asyncio.create_task(
        _run_agentic_loop(seed_messages, model, max_tokens, headers, max_turns)
    )
    # First: an opening comment so the client gets bytes immediately and
    # accepts the connection as a real SSE stream.
    yield b": agentic-loop-started\n\n"
    while not loop_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(loop_task), timeout=_HEARTBEAT_INTERVAL_S)
        except asyncio.TimeoutError:
            yield b": heartbeat\n\n"
        except Exception:
            # Surfaced below via loop_task.result().
            break
    try:
        result = loop_task.result()
    except HTTPException as e:
        # Re-encode as an SSE error frame so the client sees a clean failure.
        err = json.dumps({"error": {"message": str(e.detail), "code": e.status_code}})
        yield f"data: {err}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        return
    except Exception as e:
        logger.exception(f"Agentic stream loop crashed: {e}")
        err = json.dumps({"error": {"message": f"{type(e).__name__}: {e}"}})
        yield f"data: {err}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        return
    # Persist to semantic cache before streaming back to the client.
    if on_complete is not None:
        try:
            on_complete(result)
        except Exception as e:
            logger.warning(f"on_complete hook failed: {e}")
    for chunk in _synthesize_sse_from_completion(result, model):
        yield chunk


# ── Cached-completion synthesis (semantic cache hits) ────────────────────────
def _synthesize_completion_from_cache(
    model: str, content: str, similarity: float, cached_q: str
) -> dict:
    """Build a non-streaming OpenAI-format completion from a cache hit.
    Tagged so the user can see it came from cache."""
    note = (
        f"_(cached answer — similarity {similarity:.2f} to earlier question: "
        f"{cached_q!r})_\n\n"
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": note + content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cached": True,
        },
    }


def _stream_completion_from_cache(
    model: str, content: str, similarity: float, cached_q: str
):
    """SSE stream from a cache hit, mirroring _synthesize_sse_from_completion."""
    note = (
        f"_(cached answer — similarity {similarity:.2f} to earlier question: "
        f"{cached_q!r})_\n\n"
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    yield _sse_chunk(completion_id, created, model, {"role": "assistant"})
    payload = note + content
    for i in range(0, len(payload), 400):
        yield _sse_chunk(completion_id, created, model, {"content": payload[i:i+400]})
    yield _sse_chunk(completion_id, created, model, {}, finish_reason="stop")
    yield b"data: [DONE]\n\n"


def _extract_answer_text(result: dict) -> str:
    """Pull the assistant content out of an OpenAI-format completion."""
    try:
        return ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
    except Exception:
        return ""


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    incoming = body.get("messages", [])
    stream = bool(body.get("stream", False))
    if not incoming:
        raise HTTPException(status_code=400, detail="No messages provided")

    query, history = _extract_query_and_history(incoming)
    if not query:
        raise HTTPException(status_code=400, detail="No user message found")

    preview = query if len(query) <= 120 else query[:117] + "..."
    logger.info(f"Query: {preview}")

    model = body.get("model", DEEPSEEK_MODEL)
    max_tokens = body.get("max_tokens", 8000)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    # ── Stage 0: semantic answer cache ───────────────────────────────────────
    # Look up the cache even when the IDE sends prior conversation turns —
    # IDEs (Continue, OpenWebUI) always include history, so skipping on
    # non-empty history meant the cache never fired for IDE users.
    # The 0.95 cosine threshold protects against context-dependent follow-ups
    # (short queries, "above", pronouns) hitting the wrong cached answer.
    # Cache *writes* still skip when history is present (see _cache_after).
    if SEMANTIC_CACHE_ENABLED:
        try:
            from src.answer_cache import lookup as cache_lookup
            hit = cache_lookup(query)
        except Exception as e:
            logger.warning(f"Cache lookup error: {e}")
            hit = None
        if hit:
            sim = hit.get("similarity", 0.0)
            cached_q = hit.get("q", "")
            answer = hit.get("a", "")
            logger.info(
                f"Semantic cache HIT (sim={sim:.3f}, route={hit.get('route','')}) "
                f"— short-circuiting pipeline."
            )
            if stream:
                return StreamingResponse(
                    _stream_completion_from_cache(model, answer, sim, cached_q),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return JSONResponse(
                _synthesize_completion_from_cache(model, answer, sim, cached_q)
            )

    # ── Stage 0.5: query routing ─────────────────────────────────────────────
    route = None
    if QUERY_ROUTER_ENABLED:
        try:
            from src.query_router import classify
            route = classify(query)
            logger.info(f"Router -> {route.route} ({route.note})")
        except Exception as e:
            logger.warning(f"Router failed ({e}); using default pipeline.")
            route = None

    # ── Stage 0.75: HyDE pre-computation ─────────────────────────────────────
    # Only compute HyDE when the route says it would help — saves API calls
    # on symbol/exact-string/file lookups where it's pure overhead.
    vector_query_override: str | None = None
    if HYDE_ENABLED and route is not None and route.use_hyde:
        try:
            from src.hyde import generate_hypothetical
            vector_query_override = await generate_hypothetical(query)
            if vector_query_override == query:
                vector_query_override = None  # HyDE failed; don't pretend.
        except Exception as e:
            logger.warning(f"HyDE failed ({e}); using raw query.")

    # ── Agentic path: tool-use loop. ─────────────────────────────────────────
    # For non-streaming clients we return JSON when the loop completes.
    # For streaming clients we run the loop non-streamed internally and emit
    # a synthesized SSE stream with periodic heartbeats — same UX, full
    # multi-turn tool-use under the hood.
    if AGENTIC_ENABLED:
        seed = build_seed_messages(
            user_query=query,
            conversation_history=history or None,
            route=route,
            vector_query_override=vector_query_override,
        )
        # Honor the route's max_tool_turns (capped by the global ceiling).
        max_turns = (
            min(AGENTIC_MAX_TOOL_TURNS, route.max_tool_turns)
            if route is not None else AGENTIC_MAX_TOOL_TURNS
        )
        # Build the cache-write callback.
        route_tag = route.route if route is not None else "DEFAULT"

        def _cache_after(completion: dict) -> None:
            if not SEMANTIC_CACHE_ENABLED or history:
                return
            answer = _extract_answer_text(completion)
            if answer:
                try:
                    from src.answer_cache import put as cache_put
                    cache_put(query, answer, route=route_tag)
                except Exception as e:
                    logger.warning(f"Cache put failed: {e}")

        if stream:
            return StreamingResponse(
                _stream_agentic(
                    seed, model, max_tokens, headers, max_turns, _cache_after
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await _run_agentic_loop(
                seed, model, max_tokens, headers, max_turns
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Agentic loop failed: {e} — falling back to one-shot.")
            # Fall through to one-shot below.
        else:
            _cache_after(result)
            return JSONResponse(result)

    # ── One-shot path (legacy / agentic-disabled / fallback). ────────────────
    enhanced = build_messages(user_query=query, conversation_history=history or None)
    payload = _base_payload(model, max_tokens, stream=stream)
    payload["messages"] = enhanced
    if not (DEEPSEEK_THINKING == "enabled"):
        payload["temperature"] = body.get("temperature", 0.0)

    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
                async with client.stream(
                    "POST", DEEPSEEK_URL, headers=headers, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        logger.error(f"DeepSeek {resp.status_code}: {err!r}")
                        yield f"data: {json.dumps({'error': err.decode('utf-8', 'replace')})}\n\n".encode()
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
        resp = await client.post(DEEPSEEK_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error(f"DeepSeek {resp.status_code}: {resp.text}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"DeepSeek API error: {resp.text}",
        )

    result = resp.json()
    _log_usage(result.get("usage", {}))
    return JSONResponse(result)


if __name__ == "__main__":
    logger.info(f"Starting DeepSeek RAG Proxy on http://localhost:{PROXY_PORT}")
    logger.info(f"Mode: {'agentic' if AGENTIC_ENABLED else 'one-shot'}")
    logger.info(f"Point your IDE to: http://localhost:{PROXY_PORT}/v1")
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")
