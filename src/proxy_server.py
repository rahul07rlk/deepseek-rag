"""OpenAI-compatible proxy that injects RAG context and forwards to DeepSeek.

Three modes:

  - Edit / Apply pass-through (auto-detected): when Continue's ``edit`` or
    ``apply`` role sends a request, the messages contain template markers
    (``<original_code>``, ``<new_code>``, ``<code_to_edit>``, …) and the
    incoming system message carries strict output-format instructions
    ("Output ONLY the new file…"). For these we bypass RAG entirely,
    preserve the original system+user messages verbatim, drop tool-use,
    and stream straight from DeepSeek. Thinking mode follows
    ``EDIT_APPLY_THINKING`` (default: inherit ``DEEPSEEK_THINKING``), so
    users who keep reasoning at max for chat get the same quality on
    edits. Without this fast-path, the proxy's "cite file paths" system
    prompt corrupts the output and Continue fails to apply the diff.

  - Agentic (AGENTIC_ENABLED=true, default): the proxy seeds context (repo
    map + a slice of one-shot retrieval) and exposes retrieval primitives
    as tools. DeepSeek calls them iteratively until it has what it needs.
    Works for both streaming and non-streaming clients — the multi-turn
    tool conversation runs non-streamed internally, then the final answer
    is synthesized into an OpenAI SSE stream when the client requested one.

  - One-shot (AGENTIC_ENABLED=false): legacy behavior. ``build_messages``
    stuffs everything into the system prompt up front and DeepSeek answers
    in a single round-trip. Native streaming pass-through.

For chat requests the last user message is the retrieval query and incoming
system messages are dropped (we inject our own). For edit/apply requests
incoming messages pass through unchanged.
"""
from __future__ import annotations

import asyncio
import json
import re
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
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING,
    DEEPSEEK_TIMEOUT,
    DEEPSEEK_URL,
    EDIT_APPLY_THINKING,
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


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compatible /v1/models endpoint.

    Continue (and other OpenAI-compatible IDEs) call this on startup to
    validate the connection and confirm the model name.  Without it they
    show a "connection error" or silently disable the integration.
    """
    import time
    return {
        "object": "list",
        "data": [
            {
                "id": DEEPSEEK_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek-rag-proxy",
            }
        ],
    }


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


# ── Edit / Apply detection ────────────────────────────────────────────────────
# Strong tag markers from Continue's edit/apply prompt templates. These are
# distinctive enough that a single hit means we are NOT looking at a chat
# turn — we're looking at an IDE edit/apply round-trip that needs raw output.
_EDIT_APPLY_TAGS = (
    "<original_code>",
    "</original_code>",
    "<new_code>",
    "</new_code>",
    "<code_to_edit>",
    "</code_to_edit>",
    "<file_prefix>",
    "<file_suffix>",
    "<file_content>",
    "</file_content>",
    "<|editable_region_start|>",
    "<|editable_region_end|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    # Continue 1.x apply/edit tags
    "<highlighted_code>",
    "</highlighted_code>",
    "<current_file>",
    "</current_file>",
    "<original>",
    "</original>",
    "<modified>",
    "</modified>",
)
# Phrasal markers from Continue's edit/apply prompt templates.
# Lower-cased for case-insensitive matching. Each phrase is distinct
# enough that it won't false-positive on normal chat questions.
# Keep this list broad — a miss here routes an edit through the agentic
# path, which corrupts the diff; a false positive just skips RAG on chat,
# which is harmless.
_EDIT_APPLY_PHRASES = (
    "rewrite the code_to_edit",
    "rewrite the code as specified",
    "output only the new file",
    "output only the updated file",
    "output the entire updated file",
    "apply the new code to",
    "apply the changes to the file",
    "do not wrap your response in triple-backticks",
    "merge the new code into",
    # Continue v0.9+ / v1.x variants
    "you are given a code snippet",
    "make the following edit",
    "the user wants to edit",
    "output the edited code",
    "output only the edited",
    "here is the code to edit",
    "please make the following changes",
    "apply the following edit",
    "apply the following changes",
    "only output the code",
    "only return the code",
    "return the modified",
    "return the updated",
    "return only the",
    "the following is the code",
    "edit the code below",
    # Continue 1.x apply-button phrases (sent when user clicks Apply on a chat block)
    "you are an ai code editor",
    "your task is to apply",
    "apply the suggested changes",
    "output the complete modified",
    "output the complete updated",
    "output only the complete",
    "output the full modified",
    "output the full updated",
    "output the full file",
    "output just the file",
    "output just the code",
    "do not include any explanation",
    "do not include explanations",
    "without any explanation",
    "no explanation needed",
    "no explanations or comments",
    "here are the contents of the file",
    "the file currently contains",
    "apply the changes shown",
    "generate the modified version",
    "produce the complete file",
    "rewrite the file with",
    "you are a code editing assistant",
    "your job is to edit",
)


def _looks_like_edit_or_apply(messages: list[dict]) -> bool:
    """Detect a Continue (or similar IDE) edit/apply round-trip by template
    markers. We must NOT run RAG retrieval on these — the proxy's own
    system prompt (cite paths, think step-by-step) corrupts the output
    and Continue's diff parser drops the response."""
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            # OpenAI-style multipart content (rare for these IDEs).
            for chunk in c:
                if isinstance(chunk, dict):
                    txt = chunk.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
    full_text = "\n".join(parts)
    if not full_text:
        return False
    if any(tag in full_text for tag in _EDIT_APPLY_TAGS):
        return True
    full_lower = full_text.lower()
    return any(p in full_lower for p in _EDIT_APPLY_PHRASES)


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


def _extract_history_context(messages: list[dict]) -> dict:
    """Scan recent assistant turns for repo/file context from prior turns.

    When a user has been asking about Aelvyris-Backend across several turns and
    then asks a vague follow-up ("how does error handling work?"), the current
    query has no explicit scope. This function extracts which repos and files
    appeared in recent assistant responses so the agentic seed can carry that
    context forward as a soft retrieval bias.

    Strategy:
    - Only look at the last 8 messages (≈ 4 turns) to avoid stale context.
    - Scan assistant turns for file path patterns (from read_file / grep results
      and the model's own citations).
    - Derive repos from those paths using REPO_PATHS.
    - Return empty dict for first turns (no history) so it's a no-op.
    """
    import re as _re

    # Limit to recent turns — old context should not bleed into new topics.
    recent = messages[-8:] if len(messages) > 8 else messages
    assistant_text = " ".join(
        str(m.get("content") or "")
        for m in recent
        if m.get("role") in ("assistant", "tool")
    )
    if not assistant_text:
        return {}

    # Extract file path references: backtick-quoted or plain paths with extension.
    file_refs: list[str] = list(dict.fromkeys(
        m.group(1)
        for m in _re.finditer(
            r'[`\s\[]([A-Za-z][\w./\\-]+\.[A-Za-z]{2,5})\b', assistant_text
        )
    ))[:10]

    # Map each file ref to a repo name (longest-name-first to avoid substrings).
    mentioned_repos: list[str] = []
    text_lower = assistant_text.lower()
    text_remaining = text_lower
    for rp in sorted(REPO_PATHS, key=lambda p: len(p.name), reverse=True):
        name = rp.name.lower()
        if name in text_remaining:
            mentioned_repos.append(rp.name)
            text_remaining = text_remaining.replace(name, " " * len(name))

    if not mentioned_repos and not file_refs:
        return {}

    return {"repos": mentioned_repos, "files": file_refs}


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


# ── Edit / Apply pass-through ─────────────────────────────────────────────────
def _resolve_edit_apply_thinking() -> bool:
    """Decide whether thinking should be on for the edit/apply pass-through.

    Default ("inherit") follows the global DEEPSEEK_THINKING — users who set
    DEEPSEEK_THINKING=enabled + DEEPSEEK_REASONING_EFFORT=max for higher
    chat quality get the same treatment for IDE edits, since Continue's
    SSE parser reads `content` deltas and ignores `reasoning_content`.

    Set EDIT_APPLY_THINKING=disabled only if a specific Continue build
    can't tolerate reasoning_content in its apply stream.
    """
    if EDIT_APPLY_THINKING == "enabled":
        return True
    if EDIT_APPLY_THINKING == "disabled":
        return False
    return DEEPSEEK_THINKING == "enabled"


def _sanitize_passthrough_payload(body: dict) -> dict:
    """Build the DeepSeek payload for an edit/apply pass-through.

    Forward the incoming messages verbatim — Continue ships its strict
    output-format instructions in the system message, and clobbering them
    with our RAG system prompt is exactly what was breaking apply.

    Thinking mode follows ``EDIT_APPLY_THINKING`` (default: inherit
    ``DEEPSEEK_THINKING``) so users who run reasoning at max for chat
    keep that quality on edits. Drop ``tools``/``tool_choice`` regardless
    — we never want the model emitting tool calls into the diff stream.
    """
    out = dict(body)  # shallow copy; messages list is reused as-is
    if _resolve_edit_apply_thinking():
        out["thinking"] = {"type": "enabled"}
        out["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
        # Thinking mode silently ignores temperature/top_p — strip them so
        # the request stays clean.
        out.pop("temperature", None)
        out.pop("top_p", None)
    else:
        out.pop("thinking", None)
        out.pop("reasoning_effort", None)
        # Force temperature=0.0 for deterministic edit/apply output.
        # Continue's config omits temperature (intentionally, for thinking
        # mode), so the key is absent from the request body. Without an
        # explicit 0.0, DeepSeek defaults to 1.0 — producing non-deterministic,
        # hallucinated code edits. This is the primary cause of "response seems
        # incomplete / wrong" on inline edits.
        out.setdefault("temperature", 0.0)
    out.pop("tools", None)
    out.pop("tool_choice", None)
    out.pop("parallel_tool_calls", None)
    # Always use the model configured in .env regardless of what the IDE
    # sent (the IDE config model name is a proxy alias, not the real model).
    # This prevents deprecated aliases like "deepseek-reasoner" from reaching
    # DeepSeek's API and silently downgrading to a different model tier.
    out["model"] = DEEPSEEK_MODEL
    # Cap tokens: never let the client under-provision. Continue's maxTokens
    # setting flows in via the body; fall back to the env-configured floor.
    requested_max = int(out.get("max_tokens") or 0)
    out["max_tokens"] = max(requested_max, DEEPSEEK_MAX_OUTPUT_TOKENS)
    return out


async def _passthrough_completion(body: dict, headers: dict, stream: bool):
    """Forward an edit/apply request straight to DeepSeek. Streaming uses
    native byte-level pass-through; non-streaming returns the JSON."""
    payload = _sanitize_passthrough_payload(body)
    payload["stream"] = stream

    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
                async with client.stream(
                    "POST", DEEPSEEK_URL, headers=headers, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        logger.error(
                            f"DeepSeek {resp.status_code} (passthrough): {err!r}"
                        )
                        yield (
                            f"data: {json.dumps({'error': err.decode('utf-8', 'replace')})}\n\n"
                        ).encode()
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
        resp = await client.post(DEEPSEEK_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        logger.error(f"DeepSeek {resp.status_code} (passthrough): {resp.text}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"DeepSeek API error: {resp.text}",
        )
    result = resp.json()
    _log_usage(result.get("usage", {}))
    return JSONResponse(result)


# ── Agentic loop ──────────────────────────────────────────────────────────────
async def _run_agentic_loop(
    seed_messages: list[dict],
    model: str,
    max_tokens: int,
    headers: dict,
    max_turns: int = AGENTIC_MAX_TOOL_TURNS,
    progress_queue: "asyncio.Queue | None" = None,
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

    def _emit(event: str, **fields) -> None:
        """Best-effort progress event; never blocks the loop."""
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait({"event": event, **fields})
        except Exception:
            pass

    _emit("loop_start", max_turns=max_turns)

    async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
        for turn in range(max_turns + 1):
            _emit("turn_start", turn=turn)
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
                _emit("loop_done", turn=turn)
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
                    _emit("tool_call", turn=turn, name=name, args_preview=args[:120])
                    result = await asyncio.to_thread(dispatch, name, args, tool_memo)
                    _emit("tool_result", turn=turn, name=name, bytes=len(result))
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
                    _emit("tool_call", turn=turn, name=name, args_preview=args[:120])
                    result = dispatch(name, args, tool_memo)
                    _emit("tool_result", turn=turn, name=name, bytes=len(result))
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


# Match plausible file paths in answer text or seed prompts. We require at
# least one slash and a recognized source-file extension so prose like
# ``the proxy.py file`` is captured but English words are not.
_EVIDENCE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|[./])?"                     # optional drive or "./"
    r"(?:[\w\-.]+[\\/])+"                            # at least one path segment
    r"[\w\-.]+"                                       # filename stem
    r"\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|cs|cpp|cc|c|h|hpp|"
    r"rb|swift|scala|php|md|json|yaml|yml|toml|sql|graphql|proto|sh|bat|ps1)"
    r"\b",
    re.IGNORECASE,
)


def _extract_evidence_files(answer: str, seed_messages: list[dict]) -> list[str]:
    """Pull plausible file paths out of the answer text plus seed prompts.

    Used to build the snapshot-aware cache key. We prefer paths that appear
    in BOTH the seed (so we know retrieval saw them) and the answer (so we
    know the answer relied on them), but fall back to the union if either
    set is empty.
    """
    seed_text = "\n".join(
        m.get("content") or "" for m in seed_messages
        if m.get("role") in ("system", "user")
    )
    answer_paths = {m.group(0).replace("\\", "/") for m in _EVIDENCE_PATH_RE.finditer(answer or "")}
    seed_paths = {m.group(0).replace("\\", "/") for m in _EVIDENCE_PATH_RE.finditer(seed_text)}
    intersection = answer_paths & seed_paths
    chosen = intersection if intersection else (answer_paths or seed_paths)
    # Cap at 30 — large lists make per-file invalidation slow.
    return sorted(chosen)[:30]


def _format_progress(evt: dict) -> bytes:
    """Render an agent progress event as an SSE comment line.

    Comments (lines starting with ``:``) are ignored by spec-compliant
    SSE parsers like Continue's, so emitting human-readable progress
    here is safe — strict clients still see only ``content`` deltas in
    the final synthesized stream, while curl / dashboards / progress
    panels can pick up the live trail.
    """
    name = evt.get("event", "?")
    if name == "tool_call":
        line = f": agent turn={evt.get('turn')} tool={evt.get('name')} args={evt.get('args_preview', '')}"
    elif name == "tool_result":
        line = f": agent turn={evt.get('turn')} done={evt.get('name')} bytes={evt.get('bytes', 0)}"
    elif name == "turn_start":
        line = f": agent turn={evt.get('turn')} thinking..."
    elif name == "loop_start":
        line = f": agent loop_start max_turns={evt.get('max_turns')}"
    elif name == "loop_done":
        line = f": agent loop_done turns={evt.get('turn')}"
    else:
        line = f": agent event={name}"
    return (line + "\n\n").encode("utf-8")


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

    Only ``content`` deltas are emitted — ``reasoning_content`` is intentionally
    dropped. Continue's chat renderer doesn't know what to do with raw
    DeepSeek reasoning tokens (they appear as garbled text in the chat pane)
    and its apply/edit parser would mistake them for the actual code.

    Splits content into 512-char chunks (up from 400) so large code responses
    need fewer round-trips through Continue's SSE parser.  Warns in the log
    when the model was cut off at the token limit so incomplete edits are
    visible in proxy.log rather than silently truncated.
    """
    completion_id = result.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(result.get("created") or time.time())
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    finish = choice.get("finish_reason") or "stop"

    if finish == "length":
        logger.warning(
            "Response truncated at token limit (finish_reason=length). "
            "The model ran out of output budget before finishing the code. "
            "Consider reducing AGENTIC_MAX_TOOL_TURNS or TOP_K_CHUNKS to "
            "leave more room for the final answer."
        )

    yield _sse_chunk(completion_id, created, model, {"role": "assistant"})
    if content:
        for i in range(0, len(content), 512):
            yield _sse_chunk(
                completion_id, created, model,
                {"content": content[i : i + 512]},
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
    stream_progress: bool = True,
):
    """Async generator yielding SSE bytes. Runs the agentic loop in a task
    while surfacing progress events as SSE comments so the user can watch
    the agent investigate (which tool ran, on which turn, with what args)
    instead of staring at silent heartbeats. Falls back to plain
    ``: heartbeat`` when progress streaming is disabled.

    ``on_complete`` (sync callable) gets the final completion dict so the
    caller can persist the answer to the semantic cache.
    """
    progress_queue: asyncio.Queue | None = asyncio.Queue() if stream_progress else None

    loop_task = asyncio.create_task(
        _run_agentic_loop(
            seed_messages, model, max_tokens, headers, max_turns,
            progress_queue=progress_queue,
        )
    )
    # First: an opening comment so the client gets bytes immediately and
    # accepts the connection as a real SSE stream.
    yield b": agentic-loop-started\n\n"
    while not loop_task.done():
        try:
            if progress_queue is not None:
                # Drain any pending progress events; emit them as SSE
                # comments so SSE-strict clients ignore them but anything
                # listening for raw bytes (curl, dashboards) sees them.
                try:
                    evt = await asyncio.wait_for(
                        progress_queue.get(), timeout=_HEARTBEAT_INTERVAL_S,
                    )
                    yield _format_progress(evt)
                    continue
                except asyncio.TimeoutError:
                    yield b": heartbeat\n\n"
            else:
                await asyncio.wait_for(
                    asyncio.shield(loop_task), timeout=_HEARTBEAT_INTERVAL_S,
                )
        except asyncio.TimeoutError:
            yield b": heartbeat\n\n"
        except Exception:
            # Surfaced below via loop_task.result().
            break
    # Flush any remaining progress events after the loop completes.
    if progress_queue is not None:
        while not progress_queue.empty():
            try:
                yield _format_progress(progress_queue.get_nowait())
            except Exception:
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

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    # ── Edit / Apply fast-path ──────────────────────────────────────────────
    # Continue's edit/apply roles ship template markers in the messages.
    # When detected, forward straight to DeepSeek with thinking off and the
    # original system prompt preserved — anything else corrupts the diff
    # the IDE is trying to apply.
    if _looks_like_edit_or_apply(incoming):
        logger.info(
            f"Edit/apply request detected (msgs={len(incoming)}, stream={stream}) "
            "— bypassing RAG and forwarding directly to DeepSeek."
        )
        return await _passthrough_completion(body, headers, stream)

    query, history = _extract_query_and_history(incoming)
    if not query:
        raise HTTPException(status_code=400, detail="No user message found")

    preview = query if len(query) <= 120 else query[:117] + "..."
    logger.info(f"Query: {preview}")

    # Always use the model from .env — the IDE sends its own alias (e.g.
    # "deepseek-reasoner") which may be deprecated. Ignoring the request
    # body model ensures the correct tier is used on every call.
    model = DEEPSEEK_MODEL
    requested_max_tokens = int(body.get("max_tokens") or 0)
    max_tokens = max(requested_max_tokens, DEEPSEEK_MAX_OUTPUT_TOKENS)

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
        session_context = _extract_history_context(incoming)
        seed = build_seed_messages(
            user_query=query,
            conversation_history=history or None,
            route=route,
            vector_query_override=vector_query_override,
            session_context=session_context or None,
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
            if not answer:
                return
            evidence = _extract_evidence_files(answer, seed)
            try:
                from src.answer_cache import put as cache_put
                cache_put(query, answer, route=route_tag, evidence_files=evidence)
            except Exception as e:
                logger.warning(f"Cache put failed: {e}")

        if stream:
            from src.config import STREAM_AGENT_THOUGHTS
            return StreamingResponse(
                _stream_agentic(
                    seed, model, max_tokens, headers, max_turns, _cache_after,
                    stream_progress=STREAM_AGENT_THOUGHTS,
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
