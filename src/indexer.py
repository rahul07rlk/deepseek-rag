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
import threading
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_file
from src.config import (
    BM25_CACHE,
    BM25_REBUILD_DEBOUNCE_S,
    CONTEXTUAL_LLM_MIN_CHUNK_CHARS,
    CONTEXTUAL_PREFIX_MAX_CHARS,
    CONTEXTUAL_RETRIEVAL_MODE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_URL,
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBED_PROVIDER,
    HYDE_MODEL,
    IGNORED_DIRS,
    IGNORED_FILENAMES,
    IGNORED_SUFFIXES,
    INDEX_DIR,
    INDEXED_EXTENSIONS,
    MAX_FILE_BYTES,
    QWEN3_QUERY_INSTRUCTION,
    REMOTE_RERANKER_TIMEOUT,
    REMOTE_RERANKER_URL,
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
        elif "qwen3-embedding" in EMBED_MODEL.lower():
            # Qwen3-Embedding ships an instruction-aware query path that the
            # plain SentenceTransformer wrapper doesn't expose. The Qwen3Embedder
            # adapter implements .embed_query (used by rag_engine for asymmetric
            # encoding) and .encode (used at index time, no instruction).
            from src.utils.qwen3_embedder import Qwen3Embedder
            logger.info(
                f"Loading Qwen3 embedder: {EMBED_MODEL} on {EMBED_DEVICE} "
                f"(instruction-aware queries)"
            )
            _model = Qwen3Embedder(
                EMBED_MODEL,
                device=EMBED_DEVICE,
                query_instruction=QWEN3_QUERY_INSTRUCTION,
            )
            check_vram_after_model_load(EMBED_DEVICE)
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
            elif RERANKER_PROVIDER == "remote":
                from src.utils.remote_reranker import RemoteReranker
                logger.info(f"Loading remote reranker: {REMOTE_RERANKER_URL}")
                _reranker = RemoteReranker(
                    REMOTE_RERANKER_URL, timeout=REMOTE_RERANKER_TIMEOUT
                )
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


def _context_version() -> str:
    return f"{CONTEXTUAL_RETRIEVAL_MODE}:{CONTEXTUAL_PREFIX_MAX_CHARS}"


def _repo_relative_path(fpath: Path) -> str:
    for repo in REPO_PATHS:
        try:
            return f"{repo.name}/{fpath.relative_to(repo).as_posix()}"
        except ValueError:
            continue
    return str(fpath)


def _rules_contextual_prefix(
    fpath: Path,
    chunk: dict,
    language: str,
    repo_name: str,
) -> str:
    rel = _repo_relative_path(fpath)
    symbol = chunk.get("symbol") or "a line-window chunk"
    start = chunk.get("start_line", "?")
    end = chunk.get("end_line", "?")
    prefix = (
        f"This chunk is from {rel} in the {repo_name} repo, "
        f"a {language} file, covering {symbol} at lines {start}-{end}."
    )
    return prefix[:CONTEXTUAL_PREFIX_MAX_CHARS].rstrip()


def _llm_contextual_prefix(
    fpath: Path,
    chunk: dict,
    language: str,
    repo_name: str,
) -> str | None:
    if len(chunk.get("text", "")) < CONTEXTUAL_LLM_MIN_CHUNK_CHARS:
        return None
    try:
        import httpx
    except Exception:
        return None

    rel = _repo_relative_path(fpath)
    symbol = chunk.get("symbol") or ""
    snippet = chunk.get("text", "")[:1800]
    prompt = (
        "Write one concise sentence, 35 words max, that contextualizes this "
        "code chunk for semantic code search. Include the file path, symbol, "
        "and likely role when inferable. Return only the sentence.\n\n"
        f"Repo: {repo_name}\nPath: {rel}\nLanguage: {language}\n"
        f"Symbol: {symbol}\nCode:\n{snippet}"
    )
    try:
        resp = httpx.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": HYDE_MODEL,
                "messages": [
                    {"role": "system", "content": "You create compact code-search context."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 80,
                "temperature": 0,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        text = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception:
        return None
    return text[:CONTEXTUAL_PREFIX_MAX_CHARS].rstrip() or None


def _contextualized_doc(
    fpath: Path,
    chunk: dict,
    language: str,
    repo_name: str,
) -> tuple[str, str]:
    """Return (text_to_embed, prefix). Stored docs stay raw for display."""
    raw = chunk["text"]
    if CONTEXTUAL_RETRIEVAL_MODE == "off":
        return raw, ""

    prefix: str | None = None
    if CONTEXTUAL_RETRIEVAL_MODE == "llm":
        prefix = _llm_contextual_prefix(fpath, chunk, language, repo_name)
    if not prefix:
        prefix = _rules_contextual_prefix(fpath, chunk, language, repo_name)
    return f"{prefix}\n\n{raw}", prefix


def _is_noise(fpath: Path) -> tuple[bool, str]:
    """Return (skip, reason). Reason is "" when the file should be indexed.

    Audit-driven filter: lockfiles + minified/generated bundles dominated the
    pre-filter index (56% of chunks). Removing them recovers ~95% of usable
    BM25 vocabulary and makes the reranker's job easier.
    """
    name = fpath.name
    if name in IGNORED_FILENAMES:
        return True, f"ignored filename: {name}"
    lower = name.lower()
    for sfx in IGNORED_SUFFIXES:
        if lower.endswith(sfx):
            return True, f"ignored suffix: {sfx}"
    try:
        size = fpath.stat().st_size
    except OSError:
        return True, "stat failed"
    if size > MAX_FILE_BYTES:
        return True, f"too large ({size} bytes > {MAX_FILE_BYTES})"
    if size == 0:
        return True, "empty"
    return False, ""


def discover_files(repo_paths: list[Path] | Path) -> list[Path]:
    if isinstance(repo_paths, Path):
        repo_paths = [repo_paths]
    files: list[Path] = []
    skipped_noise = 0
    for repo_path in repo_paths:
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for fname in filenames:
                fpath = Path(root) / fname
                if fpath.suffix not in INDEXED_EXTENSIONS:
                    continue
                skip, _reason = _is_noise(fpath)
                if skip:
                    skipped_noise += 1
                    continue
                files.append(fpath)
    if skipped_noise:
        logger.info(f"Skipped {skipped_noise} noisy files (lockfiles/minified/oversize)")
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


# ── Debounced BM25 rebuild (and repo-map) for high-frequency saves ────────────
# Editor saves often fire multiple FS events per save (atomic-write IDEs do
# write→rename→touch). Without coalescing, every save rebuilt BM25 over all
# 2k chunks. The debounce timer collapses bursts into one rebuild after the
# editor goes quiet for BM25_REBUILD_DEBOUNCE_S.
_bm25_rebuild_lock = threading.Lock()
_bm25_rebuild_timer: threading.Timer | None = None


def _execute_pending_rebuild() -> None:
    try:
        rebuild_bm25_cache()
        try:
            from src.repo_map import build_repo_map
            build_repo_map()
        except Exception as e:
            logger.debug(f"repo-map rebuild skipped: {e}")
        # Files changed → previous answers may now be stale. Wipe the
        # semantic answer cache. Conservative — we don't try to figure out
        # *which* answers depend on *which* files; the cost of being wrong
        # (serving a stale answer) is much higher than re-computing.
        try:
            from src.answer_cache import reset as _cache_reset
            _cache_reset()
        except Exception as e:
            logger.debug(f"semantic-cache invalidation skipped: {e}")
    except Exception as e:
        logger.exception(f"Debounced BM25 rebuild failed: {e}")


def schedule_bm25_rebuild() -> None:
    """Schedule a coalesced BM25 (+ repo-map) rebuild. Safe to call from
    arbitrary threads; bursts collapse into one rebuild."""
    global _bm25_rebuild_timer
    if BM25_REBUILD_DEBOUNCE_S <= 0:
        _execute_pending_rebuild()
        return
    with _bm25_rebuild_lock:
        if _bm25_rebuild_timer is not None:
            _bm25_rebuild_timer.cancel()
        _bm25_rebuild_timer = threading.Timer(
            BM25_REBUILD_DEBOUNCE_S, _execute_pending_rebuild
        )
        _bm25_rebuild_timer.daemon = True
        _bm25_rebuild_timer.start()


def _update_symbol_graph_for_file(fpath: Path) -> None:
    try:
        from src.symbol_graph import update_for_file
        summary = update_for_file(fpath)
        logger.debug(f"Symbol graph updated for {fpath.name}: {summary}")
    except Exception as e:
        logger.debug(f"symbol-graph per-file update skipped for {fpath}: {e}")


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
    discovered_file_strs = {str(fpath) for fpath in files}
    removed_missing = 0
    for indexed_file in store.get_indexed_files():
        if indexed_file not in discovered_file_strs:
            removed_missing += store.delete_by_file(indexed_file)
    if removed_missing:
        logger.info(f"Removed {removed_missing} chunks for files no longer on disk")
    removed_replaced = 0

    new_docs: list[str] = []
    new_embed_docs: list[str] = []
    new_ids: list[str] = []
    new_metas: list[dict] = []
    skipped = 0
    updated = 0
    context_version = _context_version()

    for fpath in files:
        file_str = str(fpath)
        fhash = _file_hash(fpath)

        existing_ids = store.get_by_file(file_str)
        if not force and existing_ids:
            # Check hash of any stored chunk for this file.
            by_id = store.get_by_str_ids([existing_ids[0]])
            stored_meta = by_id.get(existing_ids[0], (None, {}))[1]
            stored_hash = stored_meta.get("hash", "")
            stored_context = stored_meta.get("contextual_version", "")
            if stored_hash == fhash and stored_context == context_version:
                skipped += 1
                continue
            removed_replaced += store.delete_by_file(file_str)
            updated += 1
        elif force and existing_ids:
            removed_replaced += store.delete_by_file(file_str)

        chunks = chunk_file(fpath)
        language = _language(fpath)
        repo_name = next(
            (name for root, name in repo_name_map.items() if file_str.startswith(root + os.sep)),
            fpath.parts[0] if fpath.parts else "unknown",
        )
        for i, ch in enumerate(chunks):
            embed_doc, contextual_prefix = _contextualized_doc(
                fpath, ch, language, repo_name
            )
            new_docs.append(ch["text"])
            new_embed_docs.append(embed_doc)
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
                "contextual_version": context_version,
                "contextual_prefix": contextual_prefix,
            })

    new_count = len(files) - skipped - updated
    logger.info(
        f"Files: {len(files)} total | {skipped} unchanged | "
        f"{updated} updated | {new_count} new | "
        f"{removed_missing} missing-file chunks removed"
    )

    if not new_docs:
        logger.info("Nothing new to embed. Index is up to date.")
        store_changed = removed_missing > 0 or removed_replaced > 0
        if store_changed:
            store.save()
        if store_changed or not BM25_CACHE.exists():
            rebuild_bm25_cache()
            try:
                from src.repo_map import build_repo_map
                build_repo_map()
            except Exception:
                pass
            try:
                from src.symbol_graph import build_symbol_graph
                build_symbol_graph()
            except Exception:
                pass
        return

    logger.info(f"Embedding {len(new_docs)} chunks on {EMBED_DEVICE}...")
    total_batches = (len(new_docs) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    all_embs: list[np.ndarray] = []
    for batch_start in range(0, len(new_docs), EMBED_BATCH_SIZE):
        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        end = min(batch_start + EMBED_BATCH_SIZE, len(new_docs))
        batch_docs = new_embed_docs[batch_start:end]
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
    try:
        from src.repo_map import build_repo_map
        build_repo_map()
    except Exception as e:
        logger.warning(f"repo-map build skipped: {e}")
    try:
        from src.symbol_graph import build_symbol_graph
        build_symbol_graph()
    except Exception as e:
        logger.warning(f"symbol-graph build skipped: {e}")


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
        _update_symbol_graph_for_file(fpath)
        if removed:
            store.save()
            logger.info(f"Removed {removed} chunks for deleted {fpath.name}")
            schedule_bm25_rebuild()
        return

    skip, reason = _is_noise(fpath)
    if skip:
        removed = store.delete_by_file(file_str)
        _update_symbol_graph_for_file(fpath)
        if removed:
            store.save()
            logger.info(
                f"Removed {removed} chunks for no-longer-indexable "
                f"{fpath.name}: {reason}"
            )
            schedule_bm25_rebuild()
        else:
            logger.debug(f"Skipping {fpath.name}: {reason}")
        return

    logger.info(f"Re-indexing: {fpath.name}")
    removed = store.delete_by_file(file_str)

    chunks = chunk_file(fpath)
    if not chunks:
        _update_symbol_graph_for_file(fpath)
        if removed:
            store.save()
            schedule_bm25_rebuild()
        logger.warning(f"No chunks produced for {fpath.name}")
        return

    model = get_model()
    fhash = _file_hash(fpath)
    language = _language(fpath)
    repo_name = next(
        (rp.name for rp in REPO_PATHS if file_str.startswith(str(rp) + os.sep)),
        fpath.parts[0] if fpath.parts else "unknown",
    )

    context_version = _context_version()
    docs: list[str] = []
    embed_docs: list[str] = []
    contextual_prefixes: list[str] = []
    for ch in chunks:
        embed_doc, contextual_prefix = _contextualized_doc(
            fpath, ch, language, repo_name
        )
        docs.append(ch["text"])
        embed_docs.append(embed_doc)
        contextual_prefixes.append(contextual_prefix)
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
            "contextual_version": context_version,
            "contextual_prefix": contextual_prefix,
        }
        for ch, contextual_prefix in zip(chunks, contextual_prefixes)
    ]

    embs = model.encode(embed_docs, normalize_embeddings=True, show_progress_bar=False)
    store.add(ids, docs, embs, metas)
    store.save()
    _update_symbol_graph_for_file(fpath)
    logger.info(f"Re-indexed {len(docs)} chunks from {fpath.name}")
    schedule_bm25_rebuild()


def get_index_stats() -> dict:
    store = get_store()
    active_embed = VOYAGE_EMBED_MODEL if EMBED_PROVIDER == "voyage" else EMBED_MODEL
    active_reranker = (
        VOYAGE_RERANKER_MODEL if RERANKER_PROVIDER == "voyage" else RERANKER_MODEL
    )
    stats: dict = {
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
    try:
        from src.symbol_graph import graph_stats
        stats["symbol_graph"] = graph_stats()
    except Exception:
        stats["symbol_graph"] = {"error": "unavailable"}
    try:
        from src.config import REPO_MAP_PATH
        stats["repo_map_built"] = REPO_MAP_PATH.exists()
    except Exception:
        stats["repo_map_built"] = False
    return stats


if __name__ == "__main__":
    from src.utils.gpu_check import run_gpu_diagnostics
    run_gpu_diagnostics()
    index_repo(REPO_PATHS, force=False)
