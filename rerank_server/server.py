"""Standalone reranker microservice for the second laptop's GTX 1650.

Runs Qwen3-Reranker-0.6B on the local GPU and exposes a single endpoint
the main laptop's RAG proxy posts (query, docs) to. Listens on 0.0.0.0
so the main laptop on the same Wi-Fi can reach it.

Endpoints
---------
GET  /health                 -> {"status": "ok", "device": "cuda:0", "model": "..."}
POST /rerank                 -> {"scores": [float, ...]}
     body: {"query": str, "docs": [str, ...], "batch_size": int? = 8}

Run
---
    rerank_server\\setup.bat   (one-time, installs torch + transformers)
    rerank_server\\start.bat   (launches uvicorn on port 9000)

Find the second laptop's LAN IP with `ipconfig` (look for IPv4 Address
under your Wi-Fi adapter, e.g. 192.168.1.42), then on the main laptop set:

    REMOTE_RERANKER_URL=http://192.168.1.42:9000
    RERANKER_PROVIDER=remote
"""
from __future__ import annotations

import os
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qwen3_reranker import Qwen3Reranker

# ── Config (env-overridable) ──────────────────────────────────────────────────
MODEL_NAME = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
DEVICE = os.getenv("RERANK_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "768"))
HOST = os.getenv("RERANK_HOST", "0.0.0.0")
PORT = int(os.getenv("RERANK_PORT", "9000"))

# ── Lazy model load ───────────────────────────────────────────────────────────
_reranker: Qwen3Reranker | None = None
_model_lock = threading.Lock()


def _get_model() -> Qwen3Reranker:
    global _reranker
    if _reranker is None:
        print(f"[rerank-server] Loading {MODEL_NAME} on {DEVICE} (max_length={MAX_LENGTH})")
        _reranker = Qwen3Reranker(MODEL_NAME, device=DEVICE, max_length=MAX_LENGTH)
        if "cuda" in DEVICE:
            free, total = torch.cuda.mem_get_info(0)
            print(
                f"[rerank-server] VRAM: {(total - free) / 1024**2:.0f} MB used / "
                f"{total / 1024**2:.0f} MB total"
            )
        print("[rerank-server] Ready.")
    return _reranker


# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3 Reranker", version="1.0.0")


class RerankRequest(BaseModel):
    query: str
    docs: list[str]
    batch_size: int | None = 8


class RerankResponse(BaseModel):
    scores: list[float]
    elapsed_ms: int
    device: str
    model: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "loaded": _reranker is not None,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.query:
        raise HTTPException(status_code=400, detail="query must be non-empty")
    if not req.docs:
        return RerankResponse(scores=[], elapsed_ms=0, device=DEVICE, model=MODEL_NAME)
    pairs = [(req.query, d) for d in req.docs]
    t0 = time.time()
    # Single GPU + Qwen3-Reranker isn't safe under concurrent .predict() calls
    # (shared KV/buffers). Serialize at the API boundary.
    with _model_lock:
        model = _get_model()
        scores = model.predict(
            pairs, batch_size=int(req.batch_size or 8), show_progress_bar=False
        )
    return RerankResponse(
        scores=[float(s) for s in scores],
        elapsed_ms=int((time.time() - t0) * 1000),
        device=DEVICE,
        model=MODEL_NAME,
    )


if __name__ == "__main__":
    # Pre-load so the first /rerank call doesn't pay the load cost.
    _get_model()
    print(f"[rerank-server] Listening on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
