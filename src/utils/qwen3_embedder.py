"""Qwen3-Embedding wrapper with asymmetric query/document encoding.

Qwen3-Embedding accepts an optional task instruction prefix on the *query*
side that meaningfully improves retrieval over plain encoding (per the
model card and CoIR/MTEB-Code numbers). Documents are not prefixed — that
asymmetry is the whole point.

This adapter implements the subset of SentenceTransformer's API used by the
indexer and rag_engine:

  - .encode(texts, normalize_embeddings=True, ...)  → document embeddings
  - .embed_query(text)                              → instruction-prefixed
  - .get_sentence_embedding_dimension()             → vector size

It keeps the heavy lifting (tokenization, last-token pooling, normalization)
in pure transformers so we don't pull a second framework into the deps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3-Embedding uses last-token pooling with left-padding awareness.

    For left-padded inputs every row's last position is the actual final
    token. For right-padded inputs we walk backwards to the last 1 in the
    attention mask. The tokenizer is configured for left-padding so the
    fast path is the common case.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lens]


class Qwen3Embedder:
    """Drop-in replacement for SentenceTransformer for Qwen3-Embedding models."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_length: int = 1024,
        query_instruction: str = "",
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.query_instruction = query_instruction.strip()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )
        # FP16 on CUDA (4GB VRAM-friendly), FP32 on CPU (BLAS path is faster).
        dtype = torch.float16 if "cuda" in device else torch.float32
        self.model = (
            AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        # Cached vector dim for downstream FAISS allocation.
        self._dim = int(self.model.config.hidden_size)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def _format_query(self, text: str) -> str:
        if not self.query_instruction:
            return text
        return f"Instruct: {self.query_instruction}\nQuery: {text}"

    @torch.no_grad()
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        emb = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().float().numpy()

    def encode(
        self,
        sentences,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Document-side encoding (no instruction prefix)."""
        if isinstance(sentences, str):
            sentences = [sentences]
        chunks: list[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            chunks.append(self._encode_batch(sentences[i : i + batch_size]))
        if not chunks:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.vstack(chunks).astype(np.float32)
        # _encode_batch already L2-normalizes; the flag is honored implicitly.
        return out

    def embed_query(self, query: str) -> np.ndarray:
        """Query-side encoding with the task instruction prefix."""
        formatted = self._format_query(query)
        return self._encode_batch([formatted])[0].astype(np.float32)
