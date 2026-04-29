"""Central configuration. All other modules read from here."""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
LOG_DIR = ROOT_DIR / "logs"
INDEX_DIR = ROOT_DIR / "codebase_index"
BM25_CACHE = INDEX_DIR / "bm25_cache.pkl"

LOG_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT_DIR / ".env")

# ── Repos being indexed ───────────────────────────────────────────────────────
# Supports a single path (REPO_PATH) or a comma-separated list (REPO_PATHS).
# REPO_PATHS takes priority when both are set.
# Example: REPO_PATHS=C:/My_Projects/Aelvyris,C:/My_Projects/Aelvyris-Backend
_raw_paths = os.getenv("REPO_PATHS", "").strip()
if _raw_paths:
    REPO_PATHS: list[Path] = [Path(p.strip()) for p in _raw_paths.split(",") if p.strip()]
else:
    _single = os.getenv("REPO_PATH", "").strip()
    REPO_PATHS = [Path(_single)] if _single else []

if not REPO_PATHS:
    raise ValueError(
        "No repos configured. Set REPO_PATHS (comma-separated) or REPO_PATH in .env"
    )

# Backward-compat alias used by older call sites.
REPO_PATH = REPO_PATHS[0]


# ── GPU selection ─────────────────────────────────────────────────────────────
def detect_gpu() -> str:
    """Pick the GTX 1650 if present, else cuda:0, else CPU."""
    if not torch.cuda.is_available():
        print("[Config] No CUDA GPU detected - using CPU (slower).")
        return "cpu"

    count = torch.cuda.device_count()
    print(f"[Config] Found {count} CUDA device(s):")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"         cuda:{i} -> {name} ({vram_gb:.1f}GB VRAM)")
        if "1650" in name:
            print(f"[Config] Pinned to GTX 1650 -> cuda:{i}")
            return f"cuda:{i}"

    print(f"[Config] GTX 1650 not found. Using cuda:0 ({torch.cuda.get_device_name(0)}).")
    return "cuda:0"


EMBED_DEVICE = detect_gpu()

# ── Provider selection ────────────────────────────────────────────────────────
# EMBED_PROVIDER:
#   "local"  -> on-device GPU (default, Qwen3-Embedding-0.6B)
#   "voyage" -> Voyage AI API (voyage-code-3)
#
# RERANKER_PROVIDER:
#   "local"  -> on-device CPU/GPU (default, Qwen3-Reranker-0.6B)
#   "voyage" -> Voyage AI API (rerank-2.5, no GPU needed)
#   "remote" -> HTTP rerank server on a second laptop's GPU
#               (companion server: deepseek-rag/rerank_server/)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "local")

# ── Remote reranker (LAN-attached second laptop) ─────────────────────────────
# Used when RERANKER_PROVIDER=remote. The server lives in
# deepseek-rag/rerank_server/ and is launched with start.bat there.
REMOTE_RERANKER_URL = os.getenv("REMOTE_RERANKER_URL", "")
REMOTE_RERANKER_TIMEOUT = float(os.getenv("REMOTE_RERANKER_TIMEOUT", "60"))

# ── Voyage AI ─────────────────────────────────────────────────────────────────
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-code-3")
VOYAGE_RERANKER_MODEL = os.getenv("VOYAGE_RERANKER_MODEL", "rerank-2.5")

# ── Local embedding model ─────────────────────────────────────────────────────
# Used when EMBED_PROVIDER=local. Switching models invalidates the FAISS index.
#
# Recommended for code retrieval (2026):
#   Qwen/Qwen3-Embedding-0.6B          1024  ~1.3GB FP16  Strongest open code embedder ← default
#   nomic-ai/CodeRankEmbed              768  ~280MB       Code-specialized, light
#   jinaai/jina-embeddings-v2-base-code 768  ~320MB       Older, kept for compat
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))

# Qwen3-Embedding accepts an instruction prefix on queries (asymmetric encoding)
# that meaningfully improves retrieval over plain encoding. Documents are not
# prefixed — symmetry is broken on purpose. Ignored for non-Qwen3 models.
QWEN3_QUERY_INSTRUCTION = os.getenv(
    "QWEN3_QUERY_INSTRUCTION",
    "Given a code search query, retrieve relevant code snippets, functions, "
    "classes, types, or configuration that satisfy the query.",
)

# ── Chunking ──────────────────────────────────────────────────────────────────
# Smaller chunks => more precise retrieval. 40 lines fits most functions whole.
CHUNK_LINES = 40
CHUNK_OVERLAP = 15
MIN_CHUNK_LEN = 30  # chars; skip anything shorter
MAX_CHUNK_CHARS = 6000  # hard cap for AST chunks (large funcs get sub-split)

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Hybrid search: BM25 + vector, fused with Reciprocal Rank Fusion.
# HYBRID_ALPHA weights vector similarity vs BM25 rank. 0.5 = even mix.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
# Max number of context BLOCKS emitted (a block = whole file, expanded chunk,
# or merged neighbor group). Bumped from 12 → 30 for V4 1M-ctx models.
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "30"))
# Candidates pulled from each retriever before fusion/filtering.
CANDIDATE_POOL = 60
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.22"))
# Bumped from 18000 → 100000 to take advantage of DeepSeek V4's 1M context.
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "100000"))

# ── Context expansion ─────────────────────────────────────────────────────────
# Pull surrounding lines around each retrieved chunk so the model sees imports,
# decorators, and adjacent helpers — the #1 cause of hallucinated edits.
NEIGHBOR_EXPANSION = os.getenv("NEIGHBOR_EXPANSION", "true").lower() == "true"
NEIGHBOR_PAD_LINES = int(os.getenv("NEIGHBOR_PAD_LINES", "30"))

# When ≥ N retrieved chunks land in the same file, include the WHOLE file
# instead of fragments — the model reasons better about full files. Disable
# by setting WHOLE_FILE_THRESHOLD=0.
WHOLE_FILE_THRESHOLD = int(os.getenv("WHOLE_FILE_THRESHOLD", "2"))
# Cap raised 50K → 80K to use more of the V4 100K budget on whole files.
WHOLE_FILE_MAX_CHARS = int(os.getenv("WHOLE_FILE_MAX_CHARS", "80000"))

# ── Query analysis (adaptive HYBRID_ALPHA + symbol/path boosts) ───────────────
# Inspect each query and:
#   - choose HYBRID_ALPHA dynamically (symbol queries → BM25, prose → vector)
#   - boost candidates whose stored symbol exactly matches a query identifier
#   - boost candidates whose file path matches a query path mention
QUERY_ANALYSIS_ENABLED = os.getenv("QUERY_ANALYSIS_ENABLED", "true").lower() == "true"
# Additive RRF score boosts (typical RRF scores are 0.01–0.10, so 0.05 is large).
SYMBOL_BOOST = float(os.getenv("SYMBOL_BOOST", "0.05"))
PATH_BOOST = float(os.getenv("PATH_BOOST", "0.03"))
# LRU cache for query embeddings — same query repeated (e.g. follow-up turn)
# skips the embedder entirely.
QUERY_EMBED_CACHE_SIZE = int(os.getenv("QUERY_EMBED_CACHE_SIZE", "256"))

# ── Reranker (cross-encoder, third stage) ─────────────────────────────────────
# After hybrid recall + RRF fusion, a cross-encoder rescores the top
# CANDIDATE_POOL pairs jointly — typically 5-10× more accurate than
# bi-encoder cosine. The single biggest quality lever in modern RAG.
#
# Toggle with RERANKER_ENABLED=false to fall back to the pure-fusion path.
#
# Model choices (English/multilingual, code-friendly):
#   Qwen/Qwen3-Reranker-0.6B     0.6B  causal LM, log-odds scoring  ← default
#   BAAI/bge-reranker-v2-m3      568M  cross-encoder, strong multilingual
#   BAAI/bge-reranker-base       278M  lightest cross-encoder
#
# Device defaults to CPU — GTX 1650 (4GB) is already used by the embedder.
# Set RERANKER_DEVICE=cuda:0 only if you switch to a smaller embedder.
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
# Default CPU: GTX 1650 (4GB) is owned by the Qwen3-Embedding model (~1.3GB
# resident) and a Qwen3-Reranker co-load OOMs at long inputs. Reranker on CPU
# scoring 30 candidates costs ~1-3s on Ryzen 5 4600H — acceptable.
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
# Lowered 50 → 30: rerank cost is the long pole for big queries (per audit).
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "30"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
# Lowered 1024 → 768: ~25% fewer reranker FLOPs; chunks rarely exceed this.
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "768"))
# Floor for normalized rerank scores (sigmoid-ed, so [0,1]). 0.0 disables
# the floor and lets RERANKER_TOP_N do all the filtering.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))

# ── DeepSeek API ──────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# deepseek-v4-flash: fast, thinking ON by default, 1M ctx ($0.14/1M in cache miss)
# deepseek-v4-pro:   best quality, 1M ctx ($1.74/1M in cache miss)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Thinking mode: "enabled" (default) or "disabled".
# When enabled: temperature/top_p are ignored; reasoning_content is returned alongside content.
# When disabled: temperature works normally; FIM completion is available.
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "enabled")

# Reasoning effort when thinking is enabled: "high" (default) or "max".
DEEPSEEK_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")

DEEPSEEK_TIMEOUT = 180

# ── Proxy server ──────────────────────────────────────────────────────────────
PROXY_HOST = "0.0.0.0"
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))

# ── What to index ─────────────────────────────────────────────────────────────
INDEXED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
    ".java", ".cs", ".php", ".rb", ".swift",
    ".kt", ".scala", ".r", ".jl",
    ".json", ".yaml", ".yml", ".toml", ".ini",
    ".env.example", ".xml",
    ".md", ".sql", ".graphql", ".proto",
    ".sh", ".bash", ".ps1", ".bat",
}

IGNORED_DIRS = {
    "node_modules", ".git", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt",
    "target", "bin", "obj", ".idea",
    ".vscode", "coverage", ".pytest_cache",
    "migrations", ".terraform",
    "codebase_index",
}

# ── Corpus de-noise (filename patterns to skip during indexing) ───────────────
# Audit verified: lockfiles were 56% of all chunks and showed up in 80%+ of
# retrievals before this filter. These files don't help code QA — they bloat
# BM25 vocabulary and dilute vector similarity.
IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "poetry.lock", "Pipfile.lock", "uv.lock", "Cargo.lock",
    "composer.lock", "Gemfile.lock", "go.sum",
    "tsconfig.tsbuildinfo", ".DS_Store",
}
# Suffixes that almost always indicate generated / minified / vendored content.
IGNORED_SUFFIXES = (
    ".min.js", ".min.css", ".bundle.js", ".chunk.js", ".map",
    ".d.ts.map", ".js.map", ".css.map",
    ".tsbuildinfo",
)
# Skip files larger than this (bytes). 1MB is generous for source — anything
# larger is generated, vendored, a notebook with embedded outputs, or data.
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(1024 * 1024)))

# ── BM25 cache rebuild debouncing ─────────────────────────────────────────────
# Without this, every save in a watched repo rebuilt the full BM25 index over
# all chunks (~2k → seconds). Coalesce events into one rebuild after the
# editor stops firing changes. 0 = rebuild immediately (legacy behavior).
BM25_REBUILD_DEBOUNCE_S = float(os.getenv("BM25_REBUILD_DEBOUNCE_S", "5.0"))

# ── Repo-map pre-pass (Aider-style structural summary) ────────────────────────
# Build a compressed signatures-only map of every file and inject the most
# relevant slice into the system prompt before chunk-level retrieval. Helps
# the model orient on the codebase before drilling into specific code.
REPO_MAP_ENABLED = os.getenv("REPO_MAP_ENABLED", "true").lower() == "true"
REPO_MAP_TOP_FILES = int(os.getenv("REPO_MAP_TOP_FILES", "40"))
REPO_MAP_TOKEN_BUDGET = int(os.getenv("REPO_MAP_TOKEN_BUDGET", "4000"))
REPO_MAP_PATH = INDEX_DIR / "repo_map.json"

# ── Agentic retrieval (DeepSeek tool-use loop) ────────────────────────────────
# When enabled, the proxy exposes retrieval primitives as tools the LLM can
# call iteratively (grep, read_file, find_symbol, find_callers, repo_map,
# retrieve_more) instead of stuffing one-shot context. Modeled on Cursor /
# Claude Code / Aider. Falls back to one-shot retrieval when disabled.
AGENTIC_ENABLED = os.getenv("AGENTIC_ENABLED", "true").lower() == "true"
# 8 was too tight for "audit the entire repo"-type queries — observed loops
# burned all 8 turns just on read_file calls. 16 leaves headroom; if it's
# still hit, the proxy now does a forced summarization call with
# tool_choice=none rather than returning a half-baked response.
AGENTIC_MAX_TOOL_TURNS = int(os.getenv("AGENTIC_MAX_TOOL_TURNS", "16"))
# Initial seed context sent before tools fire. Lower than TOKEN_BUDGET because
# the agent will pull more if needed — saves cache and lets the agent steer.
AGENTIC_SEED_TOKEN_BUDGET = int(os.getenv("AGENTIC_SEED_TOKEN_BUDGET", "30000"))
AGENTIC_TOOL_READ_MAX_LINES = int(os.getenv("AGENTIC_TOOL_READ_MAX_LINES", "400"))
AGENTIC_TOOL_GREP_MAX_HITS = int(os.getenv("AGENTIC_TOOL_GREP_MAX_HITS", "30"))

# ── Semantic answer cache ────────────────────────────────────────────────────
# After answering a query, embed the question and stash (q_emb, answer). On
# the next query, cosine-match against stored embeddings; if top match is
# above threshold, short-circuit the entire pipeline and return the cached
# answer. Threshold 0.95 is conservative — empirically ~0.93+ already means
# "same intent, different wording". Reset automatically when the index is
# re-built (file change), so cache never serves stale answers.
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
# 1 hour TTL: long enough to catch repeated questions in a session, short
# enough that day-old answers don't haunt you when code changes.
SEMANTIC_CACHE_TTL_S = float(os.getenv("SEMANTIC_CACHE_TTL_S", "3600"))
SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "200"))

# ── Query routing (intent classifier) ────────────────────────────────────────
# Cheap rule-based router that classifies queries into OVERVIEW / SYMBOL_LOOKUP
# / EXACT_STRING / FILE_LOOKUP / HOW_X_WORKS / DEBUG / DEFAULT and picks a
# retrieval strategy per route (seed budget, HyDE on/off, max tool turns).
QUERY_ROUTER_ENABLED = os.getenv("QUERY_ROUTER_ENABLED", "true").lower() == "true"

# ── HyDE (Hypothetical Document Embeddings) ──────────────────────────────────
# For prose-heavy queries ("how does X work"), ask the fast LLM to draft a
# hypothetical answer, then embed THAT and use it as the vector query. Real
# code uses different vocabulary than questions; HyDE bridges the gap.
# Adds ~1-2s latency but recall improvement on prose queries is large.
# Reuses the DeepSeek API key — no extra provider needed.
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "true").lower() == "true"
HYDE_MODEL = os.getenv("HYDE_MODEL", "deepseek-v4-flash")
HYDE_MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "260"))
HYDE_TIMEOUT = float(os.getenv("HYDE_TIMEOUT", "20"))

# ── Multi-query expansion ────────────────────────────────────────────────────
# Generate N rule-based query variants (rewrites that emphasize different
# facets) and RRF-fuse the retrieval results. Rule-based, so no LLM cost.
# Disabled by default since HyDE already covers most of the lift.
MULTI_QUERY_ENABLED = os.getenv("MULTI_QUERY_ENABLED", "false").lower() == "true"
MULTI_QUERY_VARIANTS = int(os.getenv("MULTI_QUERY_VARIANTS", "3"))

# ── CRAG-style adaptive rerank floor ─────────────────────────────────────────
# Adaptive minimum: floor = max(MIN_RERANK_SCORE, top_score * RERANK_RELATIVE_FLOOR).
# So if top is 0.9, we cut everything below 0.9*0.4=0.36 (clear signal).
# If top is 0.2, we keep MIN_RERANK_SCORE as the floor (low confidence — keep
# more candidates, the LLM will sort it out).
RERANK_RELATIVE_FLOOR = float(os.getenv("RERANK_RELATIVE_FLOOR", "0.4"))
# Below this absolute top score we mark the retrieval "low confidence" and
# inject a hint into the system prompt so the LLM validates before answering.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.25"))

# ── Agentic loop performance ─────────────────────────────────────────────────
# Run independent tool calls returned in the same assistant message
# concurrently with asyncio.to_thread. Pure latency win; no quality change.
AGENTIC_PARALLEL_TOOL_CALLS = (
    os.getenv("AGENTIC_PARALLEL_TOOL_CALLS", "true").lower() == "true"
)
# Memoize tool results within a single agentic session keyed by (tool, args).
# When the agent re-asks for the same file or grep, second call is O(1).
AGENTIC_TOOL_MEMOIZATION = (
    os.getenv("AGENTIC_TOOL_MEMOIZATION", "true").lower() == "true"
)

# ── Contextual Retrieval (Anthropic 2024 — +35% recall on code) ──────────────
# Each chunk gets a 1-sentence "this chunk is from X, in Y, which does Z"
# context prepended *before* embedding. The raw chunk text is what the LLM
# sees at answer time — context only influences retrieval, not the prompt.
#
# Modes:
#   "off"   — legacy behavior, embed raw chunk text (still recall-competitive)
#   "rules" — free, deterministic: file purpose (from top-of-file docstring/
#             header comment) + parent symbol path. Captures ~70% of the LLM
#             benefit for $0. ← default
#   "llm"   — delegate to deepseek-flash. ~$0.001 per 1k chunks for v4-flash;
#             best recall but slowest indexing. Use for large prose-heavy repos.
CONTEXTUAL_RETRIEVAL_MODE = os.getenv("CONTEXTUAL_RETRIEVAL_MODE", "rules").lower()
# Cap the contextual prefix to keep chunks under embedder context window.
CONTEXTUAL_PREFIX_MAX_CHARS = int(os.getenv("CONTEXTUAL_PREFIX_MAX_CHARS", "300"))
# When MODE=llm, only contextualize chunks above this size (small chunks
# rarely benefit from the context and would dominate API cost).
CONTEXTUAL_LLM_MIN_CHUNK_CHARS = int(os.getenv("CONTEXTUAL_LLM_MIN_CHUNK_CHARS", "200"))

# ── Call-graph expansion ─────────────────────────────────────────────────────
# After ranking, scan the top retrieved chunks for symbols and pull in 1-hop
# neighbors from symbol_graph (callers + callees + importers). Critical for
# coding RAG — half the answer to "how does X work" lives in X's callers.
CALL_GRAPH_EXPANSION_ENABLED = (
    os.getenv("CALL_GRAPH_EXPANSION_ENABLED", "true").lower() == "true"
)
# Max neighbor chunks to inject. Each neighbor is one or two chunks of context.
CALL_GRAPH_MAX_NEIGHBORS = int(os.getenv("CALL_GRAPH_MAX_NEIGHBORS", "6"))
# Token budget for the expansion section (kept separate from main retrieval
# so it can't push out core hits when budget gets tight).
CALL_GRAPH_TOKEN_BUDGET = int(os.getenv("CALL_GRAPH_TOKEN_BUDGET", "8000"))

# ── Conversation-aware retrieval ─────────────────────────────────────────────
# When the user's current query is short/self-referential ("fix it", "still
# wrong", "what about that"), enrich the retrieval query with content from the
# previous turn. Without this, repeated debug prompts retrieve the same chunks
# and produce the same wrong answer.
CONVERSATION_AWARE_RETRIEVAL = (
    os.getenv("CONVERSATION_AWARE_RETRIEVAL", "true").lower() == "true"
)
# Max chars pulled from the previous assistant turn to enrich the query.
CONVERSATION_HISTORY_MAX_CHARS = int(os.getenv("CONVERSATION_HISTORY_MAX_CHARS", "600"))
# Below this length (chars) and the query is treated as potentially vague.
CONVERSATION_VAGUE_QUERY_CHARS = int(os.getenv("CONVERSATION_VAGUE_QUERY_CHARS", "40"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Sanity check ──────────────────────────────────────────────────────────────
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

if EMBED_PROVIDER == "voyage" and not VOYAGE_API_KEY:
    raise ValueError(
        "EMBED_PROVIDER=voyage but VOYAGE_API_KEY is not set. Add it to .env."
    )

if RERANKER_PROVIDER == "voyage" and not VOYAGE_API_KEY:
    raise ValueError(
        "RERANKER_PROVIDER=voyage but VOYAGE_API_KEY is not set. Add it to .env."
    )

# Validate configured repo paths exist.
for _rp in REPO_PATHS:
    if not _rp.exists():
        raise ValueError(f"Repo path does not exist: {_rp}. Check REPO_PATHS in .env")
