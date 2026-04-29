# DeepSeek RAG

OpenAI-compatible RAG proxy for DeepSeek V4 with an agentic tool-use loop.
Drop it in front of Continue, OpenWebUI, or any OpenAI-compatible client and
point at one or more local repositories. Every chat completion is grounded
in retrieved code; the model can iteratively investigate the codebase via
tool calls instead of receiving a single-shot stuffed prompt.

```
IDE / client  ──► localhost:8000/v1  ──► (RAG)  ──► api.deepseek.com
```

---

## Highlights

- **Hybrid retrieval** — FAISS + BM25, fused with Reciprocal Rank Fusion,
  adaptive `HYBRID_ALPHA` per query (symbol-heavy → BM25, prose → vector).
- **AST-aware chunking** — Python via stdlib `ast`; TS / JS / Go / Rust /
  Java / C / C++ / C# / PHP / Ruby / Swift / Kotlin via tree-sitter, with
  decorator + doc-comment expansion for NestJS / Angular / Spring patterns.
- **Symbol graph** — definitions, references, imports in SQLite. References
  are AST-extracted (comment / string-aware) so `find_callers("user")`
  doesn't get drowned in JSDoc and log strings.
- **Cross-encoder reranking** — Qwen3-Reranker-0.6B (or BGE / Voyage AI /
  remote LAN GPU) with CRAG-style adaptive relevance floor.
- **Agentic tool-use loop** — `retrieve`, `read_file`, `read_files`,
  `grep`, `find_symbol`, `find_callers`, `find_importers`, `repo_map`.
  Parallel dispatch and per-session memoization.
- **Query router** — classifies each query as OVERVIEW / SYMBOL_LOOKUP /
  EXACT_STRING / FILE_LOOKUP / HOW_X_WORKS / DEBUG / DEFAULT and tunes seed
  budget, HyDE, multi-query, and tool-turn budget per route.
- **HyDE** for prose queries, **multi-query expansion** (rule-based, free),
  **conversation-aware retrieval** for short pronoun-heavy follow-ups.
- **Contextual retrieval** — Anthropic-style 1-sentence prefix prepended
  before embedding (rules-mode is free; LLM-mode is opt-in).
- **Repo map** — Aider-style structural overview injected before chunk-level
  context; ranked by BM25 over file paths + top-level symbol names.
- **Semantic answer cache** — embed (query, answer); short-circuit on
  cosine ≥ 0.95 to identical-intent re-asks.
- **Watcher** — `watchdog` triggers per-file incremental re-index of FAISS,
  BM25, and the symbol graph on every save (debounced).
- **Provider abstraction** — local on-GPU (Qwen3 / Jina / CodeRankEmbed) or
  Voyage AI cloud (`voyage-code-3` + `rerank-2.5`); reranker optionally
  offloaded to a second LAN-attached laptop.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  IDE / client (Continue, OpenWebUI, curl)                        │
│  POST /v1/chat/completions   { messages, stream }                │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  proxy_server.py    (FastAPI, OpenAI-compatible)                 │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ semantic_cache hit?  ──► return cached answer (10 ms path)   ││
│  │ query_router.classify()  ──► RouteDecision                   ││
│  │ HyDE (route-gated)  ──► hypothetical document                ││
│  │ build_seed_messages (repo map + retrieved context)           ││
│  └──────────────────────────────────────────────────────────────┘│
│                              │                                    │
│             AGENTIC                  ONE-SHOT                     │
│             ▼                        ▼                            │
│   ┌──────────────────────┐    ┌─────────────────┐                 │
│   │ _run_agentic_loop    │    │ build_messages  │                 │
│   │  ⇄ DeepSeek tools    │    │  (single round) │                 │
│   │  parallel dispatch   │    └─────────────────┘                 │
│   │  per-session memo    │                                        │
│   └──────────┬───────────┘                                        │
│              ▼                                                    │
│   answer_cache.put(query, answer)                                 │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                    response (JSON or SSE)

┌────────────── Retrieval pipeline (used by /retrieve tool too) ───┐
│  query_analyzer (symbols / paths / alpha)                        │
│  multi_query.expand                                              │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐  ┌────────────┐                                │
│  │ FAISS (vec)  │  │ BM25       │                                │
│  └──────┬───────┘  └─────┬──────┘                                │
│         └──── RRF ───────┘                                       │
│         │                                                        │
│         ▼   symbol/path multiplicative boosts                    │
│         ▼   cross-encoder rerank (CRAG floor)                    │
│         ▼   call-graph 1-hop expansion (constrained)             │
│         ▼   density-aware whole-file / neighbor / chunk emission │
│         ▼   token-budget enforcement                             │
└──────────────────────────────────────────────────────────────────┘

┌────────────── Indexing pipeline ─────────────────────────────────┐
│  discover_files → chunker (AST-first, line-window fallback)      │
│       → contextual prefix (rules / llm / off)                    │
│       → embedder (Qwen3-Embedding-0.6B by default)               │
│       → VectorStore (FAISS IndexFlatIP, IDMap2)                  │
│       → BM25 cache (rank-bm25)                                   │
│       → symbol graph (tree-sitter refs + regex defs/imports)     │
│       → repo map (top-level symbols per file)                    │
└──────────────────────────────────────────────────────────────────┘

┌────────────── Watcher (background) ──────────────────────────────┐
│  watchdog → on save:                                             │
│    index_single_file (only changed file's chunks re-embed)       │
│    schedule_bm25_rebuild (debounced)                             │
│    update_for_file in symbol_graph                               │
│    answer_cache.reset()                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Component map

| Module | Role |
|---|---|
| `proxy_server.py` | FastAPI server. Owns the agentic loop and the SSE synthesizer. Drops incoming system messages and injects its own. |
| `rag_engine.py` | Hybrid retrieval, RRF fusion, multiplicative boosts, rerank, file-level emission tiers, low-confidence flagging. |
| `agentic.py` | Tool schemas + dispatcher. Per-session memoization, async-thread parallelism, hard cap on tool-result size. |
| `indexer.py` | Repo walking, chunking, embedding, BM25 cache, hash-keyed incremental re-index. |
| `chunker.py` | Per-language routing: Python `ast`, tree-sitter for the rest, line-window fallback. |
| `tree_sitter_chunker.py` | tree-sitter chunks + AST reference extractor (comment / string-aware). |
| `symbol_graph.py` | SQLite-backed (`definitions`, `refs`, `imports`). Used by tools and call-graph expansion. |
| `vector_store.py` | FAISS IndexIDMap2(IndexFlatIP) wrapper with sidecar file→IDs index for fast deletes. |
| `repo_map.py` | Aider-style structural overview, per-query BM25 ranking. |
| `query_router.py` | Rule-based intent classifier with per-route retrieval budgets. |
| `query_analyzer.py` | Per-query symbol / path detection and adaptive RRF alpha. |
| `hyde.py` | DeepSeek-flash hypothetical-document generation (cached). |
| `multi_query.py` | Free, rule-based query paraphrases for RRF over variants. |
| `answer_cache.py` | Persistent semantic cache (FIFO + TTL) keyed on query embedding. |
| `watcher.py` | `watchdog` observer wired to `index_single_file`. |
| `rerank_server/` | Optional companion HTTP rerank service for a second laptop's GPU. |

---

## Quickstart

### 1. Prerequisites

- Python 3.10+ (tested on 3.12)
- (Recommended) NVIDIA GPU with ≥ 4 GB VRAM for the embedder. CPU works but
  is ~10× slower for indexing. The reranker defaults to CPU.
- A DeepSeek API key (https://platform.deepseek.com).

### 2. Install

```bash
# Windows
scripts\setup.bat

# macOS / Linux
bash scripts/setup.sh
```

This creates a `venv/`, installs `requirements.txt`, and pulls the
appropriate PyTorch CUDA wheel.

### 3. Configure

```bash
cp .env.example .env
# Edit .env. Minimum:
#   DEEPSEEK_API_KEY=sk-...
#   REPO_PATHS=C:/My_Projects/yourrepo1,C:/My_Projects/yourrepo2
```

### 4. Build the index

```bash
# Windows
scripts\reindex.bat
# macOS / Linux
bash scripts/reindex.sh
```

Indexes every file matching `INDEXED_EXTENSIONS` under `REPO_PATHS`,
filters out lockfiles / minified bundles / oversize files, embeds chunks,
builds BM25 + symbol-graph + repo-map. Subsequent runs are incremental
(hash-skip unchanged files).

### 5. Start the proxy

```bash
# Windows
scripts\start.bat
# macOS / Linux
bash scripts/start.sh
```

Listens on `http://localhost:8000`. The watcher runs in the same process
and keeps the index in sync as you edit code.

### 6. Verify

```bash
curl http://localhost:8000/stats | jq
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"how does the watcher debounce work?"}]}'
```

### 7. Point your IDE at it

#### Continue (`~/.continue/config.yaml`)

```yaml
name: My DeepSeek RAG
schema: v1

contextProviders: []          # let the proxy own retrieval

models:
  - name: DeepSeek RAG
    provider: openai
    model: deepseek-v4-flash
    apiBase: http://localhost:8000/v1
    apiKey: rag-proxy         # placeholder; proxy doesn't validate
    roles: [chat, edit, apply]
    contextLength: 1000000
    completionOptions:
      maxTokens: 100000

  - name: DeepSeek Autocomplete
    provider: deepseek
    model: deepseek-v4-flash
    apiKey: ${{ secrets.DEEPSEEK_API_KEY }}
    roles: [autocomplete]
    completionOptions:
      maxTokens: 256
      temperature: 0.0

tabAutocomplete:
  model: DeepSeek Autocomplete
  enabled: true
```

**Why two models?** Tab autocomplete fires on every keystroke. Routing it
through the RAG proxy would trigger full retrieval + rerank + thinking
mode (>30 s latency per keypress). Autocomplete must hit DeepSeek
directly via FIM with thinking disabled — the `deepseek` provider handles
that switch automatically.

#### OpenWebUI

In *Settings → Connections → OpenAI* add a new connection:
- URL: `http://localhost:8000/v1`
- Key: any non-empty string

#### curl / your own script

The proxy is OpenAI-compatible; any OpenAI SDK works:

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="rag-proxy")
r = c.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role":"user","content":"explain how indexing handles file deletes"}],
    stream=False,
)
print(r.choices[0].message.content)
```

---

## Request flow

For a request `POST /v1/chat/completions`:

1. **Extract** the last user message as the retrieval query; everything
   prior becomes conversation history. Incoming system messages are dropped.
2. **Semantic answer cache lookup** — embed the query, cosine-match against
   ≤ 200 prior `(question, answer)` pairs. On hit (sim ≥ 0.95) return
   immediately as JSON or SSE.
3. **Query routing** — `query_router.classify` picks intent and retrieval
   strategy. Each route adjusts: seed budget, max tool turns, top-k,
   whether to run HyDE / multi-query, which seed sections to include.
4. **HyDE pre-computation** (route-gated) — one fast `deepseek-v4-flash`
   call (thinking off) generates a hypothetical answer; the original
   query + hypothetical becomes the vector-search input. BM25 keeps the
   raw query.
5. **Seed messages** — repo map (route-dependent) + initial retrieved
   context + tool instructions + conversation history + user query.
6. **Agentic loop** (when `AGENTIC_ENABLED=true`):
   - DeepSeek either emits a final answer or calls tools.
   - Tool calls run via `dispatch`. Independent calls in the same turn
     run concurrently with `asyncio.to_thread`. Results are cached per
     `(tool_name, normalized_args)` for the session.
   - For streaming clients, the loop runs non-streamed internally and
     emits SSE heartbeats while it works; the final response is
     synthesized into delta chunks.
   - Loop bound: `min(AGENTIC_MAX_TOOL_TURNS, route.max_tool_turns)`.
     On exhaustion, a forced `tool_choice="none"` summarization call
     prevents an empty reply.
7. **Cache the final answer** (semantic + route tag) for next time.

The retrieval pipeline used in step 5 (and by the `retrieve` tool) is:

1. Conversation-aware enrichment (short pronoun-heavy follow-ups get the
   prior assistant turn appended).
2. Hybrid recall: FAISS + BM25 over each query variant, RRF-fused.
3. Symbol & path multiplicative boosts (`score' = score · (1 + b_sym) ·
   (1 + b_path)`).
4. Cross-encoder rerank with adaptive floor `max(MIN_RERANK_SCORE,
   top_score · RERANK_RELATIVE_FLOOR)`.
5. Call-graph 1-hop expansion (callers, callees, importers).
6. Per-file emission tier — whole file (when small enough or hit-density
   high), neighbor-expanded line ranges, or raw chunks.
7. Token-budget enforcement.

---

## Configuration reference

All settings are environment variables (see `.env.example`). Core knobs:

### Repos and providers

| Var | Default | Notes |
|---|---|---|
| `REPO_PATHS` | *(required)* | Comma-separated absolute paths to index. |
| `EMBED_PROVIDER` | `local` | `local` (Qwen3 by default) or `voyage`. |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 1024-dim, instruction-aware queries. |
| `RERANKER_PROVIDER` | `local` | `local`, `voyage`, or `remote`. |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Used when provider is `local`. |
| `RERANKER_DEVICE` | `cpu` | The embedder owns the GTX 1650 by default. |
| `VOYAGE_API_KEY` | — | Required only when a Voyage provider is selected. |
| `REMOTE_RERANKER_URL` | — | Required only when `RERANKER_PROVIDER=remote`. |

### Retrieval

| Var | Default | Notes |
|---|---|---|
| `TOP_K_CHUNKS` | `30` | Max blocks emitted (a block = whole file, expanded range, or chunk). |
| `CANDIDATE_POOL` | `60` | Per-retriever pool before rerank. |
| `TOKEN_BUDGET` | `100000` | Hard cap on injected context. |
| `HYBRID_ALPHA` | `0.5` | Vector weight in RRF; overridden by analyzer per query. |
| `MIN_RELEVANCE_SCORE` | `0.22` | Fusion-only cutoff (when reranker is off). |
| `MIN_RERANK_SCORE` | `0.0` | Absolute rerank floor (0 = let RERANKER_TOP_N do filtering). |
| `RERANK_RELATIVE_FLOOR` | `0.4` | CRAG-style: cut anything < `top_score · this`. |
| `LOW_CONFIDENCE_THRESHOLD` | `0.25` | Below this top score, the seed flags low confidence to the agent. |
| `SYMBOL_BOOST` | `1.0` | **Multiplicative.** `score · (1 + this)` on exact-symbol match. |
| `PATH_BOOST` | `0.5` | Same shape, for query-mentioned file paths. |
| `WHOLE_FILE_THRESHOLD` | `2` | Min hits per file to emit whole file. |
| `WHOLE_FILE_MAX_CHARS` | `80000` | Cap before falling back to ranges. |
| `NEIGHBOR_PAD_LINES` | `30` | Lines around each retrieved chunk. |

### Pipeline toggles

| Var | Default | Notes |
|---|---|---|
| `AGENTIC_ENABLED` | `true` | Falls back to one-shot when off. |
| `AGENTIC_MAX_TOOL_TURNS` | `16` | Hard ceiling per request. |
| `AGENTIC_PARALLEL_TOOL_CALLS` | `true` | Async-thread dispatch. |
| `AGENTIC_TOOL_MEMOIZATION` | `true` | Per-session result memo. |
| `QUERY_ROUTER_ENABLED` | `true` | Per-intent budgets. |
| `QUERY_ANALYSIS_ENABLED` | `true` | Adaptive alpha + boost targets. |
| `HYDE_ENABLED` | `true` | Route-gated. |
| `MULTI_QUERY_ENABLED` | `true` | Rule-based variants. |
| `CONVERSATION_AWARE_RETRIEVAL` | `true` | Short pronoun follow-ups are enriched. |
| `CALL_GRAPH_EXPANSION_ENABLED` | `true` | 1-hop neighbor pull. |
| `CONTEXTUAL_RETRIEVAL_MODE` | `rules` | `off`, `rules`, or `llm`. |
| `REPO_MAP_ENABLED` | `true` | Aider-style overview injection. |
| `SEMANTIC_CACHE_ENABLED` | `true` | Persistent semantic answer cache. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Cosine threshold for short-circuit. |
| `SEMANTIC_CACHE_TTL_S` | `3600` | Bounds staleness even when files don't change. |

### DeepSeek

| Var | Default | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | *(required)* | |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Switch to `deepseek-v4-pro` for hard reasoning. |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` | `100000` | Match this in the IDE config. |
| `DEEPSEEK_THINKING` | `enabled` | Forced server-side. |
| `DEEPSEEK_REASONING_EFFORT` | `max` | Used when thinking is on. |

### Indexing

| Var | Default | Notes |
|---|---|---|
| `EMBED_BATCH_SIZE` | `8` | GTX 1650-friendly. |
| `MAX_FILE_BYTES` | `1048576` | Skip generated / vendored files larger than this. |
| `BM25_REBUILD_DEBOUNCE_S` | `5.0` | Coalesce burst edits. |

---

## Operations

### Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Status, index stats, model info. |
| `/health` | GET | Liveness probe. |
| `/stats` | GET | Detailed: chunk counts, symbol graph rows, semantic cache. |
| `/cache/reset` | POST | Wipe the semantic answer cache. |
| `/v1/chat/completions` | POST | OpenAI-compatible chat. |

### Common workflows

```bash
# Force-reindex from scratch
python -m src.indexer        # respects hashes; "force" requires editing the call

# Rebuild symbol graph alone
python -m src.symbol_graph

# Rebuild repo map alone
python -m src.repo_map

# Wipe semantic cache (if you suspect a stale hit)
curl -X POST http://localhost:8000/cache/reset

# Run the test suite
pytest tests/test_chunker.py tests/test_retrieval.py --basetemp=tests/_tmp -p no:cacheprovider
```

### What happens on a file save

1. `watchdog` fires → `index_single_file(path)`.
2. If file is gone or filtered → delete its chunks + symbol-graph rows.
3. Otherwise: re-chunk, re-embed, replace stored chunks.
4. Per-file symbol-graph refresh.
5. Schedule debounced rebuild of BM25 cache + repo map (collapses bursts).
6. On debounce fire: rebuild + invalidate semantic answer cache.

### Switching providers

| Goal | Change |
|---|---|
| Move embeddings to Voyage AI | `EMBED_PROVIDER=voyage`, set `VOYAGE_API_KEY`. Old FAISS index is auto-discarded on next launch (dim mismatch detection). Re-run `reindex`. |
| Move reranker to a second laptop | Run `rerank_server/start.bat` on the GPU laptop, set `RERANKER_PROVIDER=remote` and `REMOTE_RERANKER_URL=http://<laptop>:9001` in `.env`. |
| Disable reranker entirely | `RERANKER_ENABLED=false`. Falls back to fusion-only ranking with `MIN_RELEVANCE_SCORE` as the floor. |
| Disable agentic mode | `AGENTIC_ENABLED=false`. The proxy uses a single-round one-shot retrieval prompt instead. |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ValueError: No repos configured` | `REPO_PATHS` is empty in `.env`. Set comma-separated absolute paths. |
| `Repo path does not exist` | Typo or wrong drive letter. Forward slashes are fine on Windows. |
| `[VectorStore] Embedding model changed … Stale index discarded` | You changed `EMBED_MODEL` or `EMBED_PROVIDER`. Re-run `reindex`. |
| `Reranker failed to load: …` | Falls back to fusion-only automatically. Check logs in `logs/proxy.log`. Most common: VRAM OOM when embedder + reranker share the GTX 1650 (default puts reranker on CPU specifically to avoid this). |
| Empty / "(no results)" answers | Check `curl /stats` for `total_chunks > 0`. If zero, `reindex` first. |
| 30-second silences with no output | Almost always cold model load on first request. Subsequent requests are fast. |
| Tab autocomplete is slow / never fires | Continue is routing autocomplete through the RAG proxy. Make sure `tabAutocomplete.model` points at the `provider: deepseek` model, not the proxy. |
| "DeepSeek 4xx — reasoning_content must be passed back" | Client dropped `reasoning_content` between turns. Use the proxy itself; it preserves it. |
| Slow per-keystroke indexing | Heavy editor doing atomic-write rename storms. Increase `BM25_REBUILD_DEBOUNCE_S`. |

Logs live under `logs/`:

- `proxy.log` — request flow, agentic turns, cache hits, rerank stats.
- `indexer.log` — indexing, watcher events, symbol graph rebuilds.

---

## Performance notes

- The proxy logs DeepSeek's `prompt_cache_hit_tokens` / `miss_tokens` per
  call. The agentic loop is designed to keep the seed prefix cached across
  tool turns — every iteration after the first should be a cache hit.
- First request after server start triggers model load (Qwen3 embedder
  ~1.3 GB FP16, Qwen3 reranker on CPU). Expect ~10–20 s warmup.
- Per-query latency on a warm system, GTX 1650 + Ryzen 5 4600H:
  - Vector + BM25 + RRF: ~50 ms
  - Cross-encoder rerank (CPU, 60 candidates): ~1–3 s
  - HyDE call (when active): ~1–2 s
  - Agentic turns: ~3–8 s per turn (DeepSeek thinking).

---

## License

Internal project — see your repository's license file.
