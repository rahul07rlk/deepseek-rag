"""Repository indexer.

Walks REPO_PATHS, chunks every indexable file, embeds with the configured
model, and stores vectors + metadata in a FAISS-backed VectorStore.
Also maintains a BM25 cache for the hybrid retriever.
"""
from __future__ import annotations

import hashlib
import os
import os.path
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_file
from src.config import (
    BM25_CACHE,
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBED_PROVIDER,
    INDEX_DIR,
    INDEXED_EXTENSIONS,
    IGNORED_DIRS,
    RERANKER_DEVICE,
    RERANKER_ENABLED,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
    REPO_PATH,
    REPO_PATHS,
    VOYAGE_API_KEY,
    VOYAGE_EMBED_MODEL,
    VOYAGE_RERANKER_MODEL,
)
from src.utils.gpu_check import check_vram_after_model_load
from src.utils.logger import get_logger
from src.vector_store import VectorStore

logger = get_logger("indexer", "indexer.log")

# ── Lazy singletons ───────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_store: VectorStore | None = None
_reranker = None  # CrossEncoder | None — typed loosely to avoid eager import
_reranker_load_failed = False


def get_model():
    global _model
    if _model is None:
        if EMBED_PROVIDER == "voyage":
            from src.utils.voyage_embedder import VoyageEmbedder
            logger.info(f"Loading Voyage embedder: {VOYAGE_EMBED_MODEL}")
            _model = VoyageEmbedder(VOYAGE_EMBED_MODEL, api_key=VOYAGE_API_KEY)
        else:
            logger.info(f"Loading local embedding model: {EMBED_MODEL} on {EMBED_DEVICE}")
            # trust_remote_code=True required for jina-embeddings-v2 custom modeling code.
            _model = SentenceTransformer(
                EMBED_MODEL, device=EMBED_DEVICE, trust_remote_code=True
            )
            check_vram_after_model_load(EMBED_DEVICE)
        logger.info("Embedding model loaded.")
    return _model


def get_store() -> VectorStore:
    global _store
    if _store is None:
        model = get_model()
        dim = model.get_sentence_embedding_dimension()
        # Use the active model identifier so VectorStore detects provider switches.
        model_id = VOYAGE_EMBED_MODEL if EMBED_PROVIDER == "voyage" else EMBED_MODEL
        _store = VectorStore(INDEX_DIR, dim, embed_model=model_id)
        logger.info(f"Vector store ready. {_store.count()} chunks loaded.")
    return _store


def get_reranker():
    """Lazy-load the reranker. Returns None if disabled or if loading fails.

    Routing by RERANKER_PROVIDER:
      - "voyage" → VoyageReranker  (cloud API, rerank-2.5)
      - "local"  → Qwen3Reranker if "qwen3" in model name, else CrossEncoder

    On failure the caller silently falls back to the fusion-only ranking path.
    """
    global _reranker, _reranker_load_failed
    if not RERANKER_ENABLED or _reranker_load_failed:
        return None
    if _reranker is None:
        try:
            if RERANKER_PROVIDER == "voyage":
                from src.utils.voyage_reranker import VoyageReranker
                logger.info(f"Loading Voyage reranker: {VOYAGE_RERANKER_MODEL}")
                _reranker = VoyageReranker(VOYAGE_RERANKER_MODEL, api_key=VOYAGE_API_KEY)
            elif "qwen3" in RERANKER_MODEL.lower():
                logger.info(
                    f"Loading Qwen3 reranker: {RERANKER_MODEL} on {RERANKER_DEVICE} "
                    f"(max_length={RERANKER_MAX_LENGTH})"
                )
                from src.utils.qwen3_reranker import Qwen3Reranker
                _reranker = Qwen3Reranker(
                    RERANKER_MODEL,
                    device=RERANKER_DEVICE,
                    max_length=RERANKER_MAX_LENGTH,
                )
            else:
                logger.info(
                    f"Loading CrossEncoder reranker: {RERANKER_MODEL} on {RERANKER_DEVICE} "
                    f"(max_length={RERANKER_MAX_LENGTH})"
                )
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(
                    RERANKER_MODEL,
                    device=RERANKER_DEVICE,
                    max_length=RERANKER_MAX_LENGTH,
                )
            logger.info("Reranker loaded.")
        except Exception as e:
            logger.error(f"Reranker failed to load: {e}. Falling back to fusion-only.")
            _reranker_load_failed = True
            return None
    return _reranker


# ── File helpers ──────────────────────────────────────────────────────────────
def _file_hash(filepath: Path) -> str:
    try:
        return hashlib.md5(filepath.read_bytes()).hexdigest()
    except Exception:
        return ""


_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".go": "go", ".rs": "rust",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".java": "java", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".sql": "sql", ".md": "markdown", ".sh": "bash", ".bash": "bash",
    ".ps1": "powershell", ".bat": "batch",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".toml": "toml", ".xml": "xml",
}


def _language(filepath: Path) -> str:
    return _LANG.get(filepath.suffix, filepath.suffix.lstrip("."))


def discover_files(repo_paths: list[Path] | Path) -> list[Path]:
    if isinstance(repo_paths, Path):
        repo_paths = [repo_paths]
    files: list[Path] = []
    for repo_path in repo_paths:
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for fname in filenames:
                fpath = Path(root) / fname
                if fpath.suffix in INDEXED_EXTENSIONS:
                    files.append(fpath)
    return sorted(files)


# ── BM25 cache ────────────────────────────────────────────────────────────────
def _tokenize_for_bm25(text: str) -> list[str]:
    import re
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text)
    tokens: list[str] = []
    for w in words:
        parts = w.split("_")
        for p in parts:
            if not p:
                continue
            camel = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", p)
            tokens.extend(x.lower() for x in camel if x)
    return tokens


def rebuild_bm25_cache() -> None:
    store = get_store()
    total = store.count()
    if total == 0:
        logger.warning("No documents to build BM25 from.")
        return

    logger.info(f"Rebuilding BM25 cache over {total} chunks...")
    all_ids, all_docs, all_metas = store.get_all()
    tokenized = [_tokenize_for_bm25(d) for d in all_docs]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_CACHE, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "ids": all_ids, "docs": all_docs, "metas": all_metas},
            f,
        )
    logger.info(f"BM25 cache written ({total} chunks).")


# ── Indexing ──────────────────────────────────────────────────────────────────
def index_repo(repo_paths: list[Path] | Path = REPO_PATHS, force: bool = False) -> None:
    if isinstance(repo_paths, Path):
        repo_paths = [repo_paths]

    logger.info(f"Starting index across {len(repo_paths)} repo(s):")
    for rp in repo_paths:
        logger.info(f"  -> {rp}")
    logger.info(f"Force re-index: {force}")

    model = get_model()
    store = get_store()

    files = discover_files(repo_paths)
    logger.info(f"Discovered {len(files)} indexable files total")
    if not files:
        logger.error("No indexable files found. Check REPO_PATHS in .env")
        return

    repo_name_map: dict[str, str] = {str(rp): rp.name for rp in repo_paths}

    new_docs: list[str] = []
    new_ids: list[str] = []
    new_metas: list[dict] = []
    new_embs: list[np.ndarray] = []
    skipped = 0
    updated = 0

    for fpath in files:
        file_str = str(fpath)
        fhash = _file_hash(fpath)

        existing_ids = store.get_by_file(file_str)
        if not force and existing_ids:
            # Check hash of any stored chunk for this file.
            by_id = store.get_by_str_ids([existing_ids[0]])
            stored_hash = by_id.get(existing_ids[0], (None, {}))[1].get("hash", "")
            if stored_hash == fhash:
                skipped += 1
                continue
            store.delete_by_file(file_str)
            updated += 1
        elif force and existing_ids:
            store.delete_by_file(file_str)

        chunks = chunk_file(fpath)
        language = _language(fpath)
        repo_name = next(
            (name for root, name in repo_name_map.items() if file_str.startswith(root + os.sep)),
            fpath.parts[0] if fpath.parts else "unknown",
        )
        for i, ch in enumerate(chunks):
            new_docs.append(ch["text"])
            new_ids.append(f"{file_str}::chunk_{i}")
            new_metas.append({
                "file": file_str,
                "filename": fpath.name,
                "repo": repo_name,
                "language": language,
                "start_line": ch["start_line"],
                "end_line": ch["end_line"],
                "symbol": ch.get("symbol") or "",
                "hash": fhash,
            })

    new_count = len(files) - skipped - updated
    logger.info(
        f"Files: {len(files)} total | {skipped} unchanged | "
        f"{updated} updated | {new_count} new"
    )

    if not new_docs:
        logger.info("Nothing new to embed. Index is up to date.")
        if not BM25_CACHE.exists():
            rebuild_bm25_cache()
        return

    logger.info(f"Embedding {len(new_docs)} chunks on {EMBED_DEVICE}...")
    total_batches = (len(new_docs) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    all_embs: list[np.ndarray] = []
    for batch_start in range(0, len(new_docs), EMBED_BATCH_SIZE):
        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        end = min(batch_start + EMBED_BATCH_SIZE, len(new_docs))
        batch_docs = new_docs[batch_start:end]
        embs = model.encode(
            batch_docs,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embs.append(embs)
        logger.info(f"Batch {batch_num}/{total_batches} embedded ({end}/{len(new_docs)})")

    combined = np.vstack(all_embs)
    store.add(new_ids, new_docs, combined, new_metas)
    store.save()

    logger.info(f"Indexing complete. Total chunks in store: {store.count()}")
    rebuild_bm25_cache()


def index_single_file(filepath: str | Path) -> None:
    fpath = Path(filepath)
    if fpath.suffix not in INDEXED_EXTENSIONS:
        return
    if any(part in IGNORED_DIRS for part in fpath.parts):
        return

    store = get_store()
    file_str = str(fpath)

    if not fpath.exists():
        removed = store.delete_by_file(file_str)
        if removed:
            store.save()
            logger.info(f"Removed {removed} chunks for deleted {fpath.name}")
            rebuild_bm25_cache()
        return

    logger.info(f"Re-indexing: {fpath.name}")
    model = get_model()

    store.delete_by_file(file_str)

    chunks = chunk_file(fpath)
    if not chunks:
        logger.warning(f"No chunks produced for {fpath.name}")
        return

    fhash = _file_hash(fpath)
    language = _language(fpath)
    repo_name = next(
        (rp.name for rp in REPO_PATHS if file_str.startswith(str(rp) + os.sep)),
        fpath.parts[0] if fpath.parts else "unknown",
    )

    docs = [ch["text"] for ch in chunks]
    ids = [f"{file_str}::chunk_{i}" for i in range(len(chunks))]
    metas = [
        {
            "file": file_str,
            "filename": fpath.name,
            "repo": repo_name,
            "language": language,
            "start_line": ch["start_line"],
            "end_line": ch["end_line"],
            "symbol": ch.get("symbol") or "",
            "hash": fhash,
        }
        for ch in chunks
    ]

    embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    store.add(ids, docs, embs, metas)
    store.save()
    logger.info(f"Re-indexed {len(docs)} chunks from {fpath.name}")
    rebuild_bm25_cache()


def get_index_stats() -> dict:
    store = get_store()
    active_embed = VOYAGE_EMBED_MODEL if EMBED_PROVIDER == "voyage" else EMBED_MODEL
    active_reranker = (
        VOYAGE_RERANKER_MODEL if RERANKER_PROVIDER == "voyage" else RERANKER_MODEL
    )
    return {
        "total_chunks": store.count(),
        "index_path": str(INDEX_DIR),
        "embed_provider": EMBED_PROVIDER,
        "embed_model": active_embed,
        "embed_device": "cloud" if EMBED_PROVIDER == "voyage" else EMBED_DEVICE,
        "reranker_provider": RERANKER_PROVIDER,
        "reranker_model": active_reranker if RERANKER_ENABLED else "disabled",
        "bm25_cached": BM25_CACHE.exists(),
        "repos_indexed": [rp.name for rp in REPO_PATHS],
        "repo_count": len(REPO_PATHS),
    }


if __name__ == "__main__":
    from src.utils.gpu_check import run_gpu_diagnostics
    run_gpu_diagnostics()
    index_repo(REPO_PATHS, force=False)
