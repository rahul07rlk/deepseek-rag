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
# EMBED_PROVIDER:    "local"  → jina-embeddings-v2-base-code on GPU (default)
#                   "voyage" → Voyage AI API (voyage-code-3, best code quality)
# RERANKER_PROVIDER: "local"  → bge/Qwen3 cross-encoder on CPU (default)
#                   "voyage" → Voyage AI API (rerank-2.5, fast, no GPU needed)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "local")

# ── Voyage AI ─────────────────────────────────────────────────────────────────
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-code-3")
VOYAGE_RERANKER_MODEL = os.getenv("VOYAGE_RERANKER_MODEL", "rerank-2.5")

# ── Local embedding model ─────────────────────────────────────────────────────
# Used when EMBED_PROVIDER=local. Switching models invalidates the FAISS index.
EMBED_MODEL = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

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
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "50"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
# Max tokens per (query, chunk) pair fed to the reranker. 1024 covers most
# code chunks (MAX_CHUNK_CHARS=6000 ≈ 1500 tokens). Lower = faster.
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
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
