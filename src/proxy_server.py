"""OpenAI-compatible proxy that injects RAG context and forwards to DeepSeek.

The last user message in the incoming request is used as the retrieval query.
Any incoming system message is replaced by our RAG-augmented system prompt.
"""
from __future__ import annotations

import json

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING,
    DEEPSEEK_TIMEOUT,
    DEEPSEEK_URL,
    PROXY_HOST,
    PROXY_PORT,
)
from src.indexer import get_index_stats
from src.rag_engine import build_messages
from src.utils.logger import get_logger

logger = get_logger("proxy", "proxy.log")

app = FastAPI(
    title="DeepSeek RAG Proxy",
    description="Transparent RAG-enhanced proxy for DeepSeek API",
    version="1.0.0",
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
        "index_stats": get_index_stats(),
        "endpoint": f"http://localhost:{PROXY_PORT}/v1/chat/completions",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict:
    return get_index_stats()


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

    enhanced = build_messages(user_query=query, conversation_history=history or None)

    thinking_on = DEEPSEEK_THINKING == "enabled"
    payload = {
        "model": body.get("model", DEEPSEEK_MODEL),
        "messages": enhanced,
        "stream": stream,
        "max_tokens": body.get("max_tokens", 8000),
    }
    if thinking_on:
        # temperature / top_p are silently ignored in thinking mode — omit them.
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    else:
        payload["temperature"] = body.get("temperature", 0.0)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

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
    usage = result.get("usage", {}).get("total_tokens", "N/A")
    logger.info(f"Response received | Tokens used: {usage}")
    return JSONResponse(result)


if __name__ == "__main__":
    logger.info(f"Starting DeepSeek RAG Proxy on http://localhost:{PROXY_PORT}")
    logger.info(f"Point your IDE to: http://localhost:{PROXY_PORT}/v1")
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")
