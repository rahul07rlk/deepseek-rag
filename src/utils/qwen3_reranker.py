"""Qwen3-Reranker wrapper with a CrossEncoder-compatible .predict() interface.

Qwen3-Reranker models are causal LMs, not sequence classifiers. They score
relevance by generating a yes/no token given a chat-formatted (query, doc)
pair. The probability of the "yes" token is used as the relevance score.

The model also uses a brief chain-of-thought prefix (<think>\\n\\n</think>)
before the answer token — this is by design and improves recall.

Usage is identical to sentence_transformers.CrossEncoder:
    reranker = Qwen3Reranker("Qwen/Qwen3-Reranker-0.6B", device="cpu")
    scores = reranker.predict([("query", "doc"), ...])  # list[float] in [0,1]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# System prompt prescribed by the Qwen3-Reranker model card.
_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct, output 'yes' or 'no'."
)
_PREFIX = f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n"
# The empty <think> block is part of the official inference template —
# it signals the model to skip extended reasoning and answer directly.
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Code-specific instruction improves precision vs the generic "retrieve passages".
_CODE_INSTRUCTION = (
    "Given a code search query, retrieve relevant code snippets, functions, "
    "classes, or configuration that answer or implement what the query describes."
)


class Qwen3Reranker:
    """Qwen3-Reranker wrapper. Drop-in replacement for CrossEncoder."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_length: int = 1024,
        instruction: str = _CODE_INSTRUCTION,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )
        # float32 on CPU is fastest on x86 (optimised BLAS paths).
        # float16 on CUDA halves memory without quality loss.
        dtype = torch.float16 if "cuda" in device else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )

        # Resolve "yes"/"no" token IDs once at load time.
        self._true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self._false_id = self.tokenizer.convert_tokens_to_ids("no")

    def _format(self, query: str, doc: str) -> str:
        return (
            f"{_PREFIX}<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n<Document>: {doc}{_SUFFIX}"
        )

    def predict(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 8,
        show_progress_bar: bool = False,
    ) -> list[float]:
        """Return a relevance score in [0, 1] for each (query, doc) pair."""
        scores: list[float] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            texts = [self._format(q, d) for q, d in batch]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                # Take the last-position logits and pull yes/no columns.
                logits = self.model(**enc).logits[:, -1, :]
            # Return log-odds (yes_logit - no_logit) so the caller's sigmoid
            # normalization works the same as for bge CrossEncoder logits.
            log_odds = logits[:, self._true_id] - logits[:, self._false_id]
            scores.extend(log_odds.cpu().float().tolist())
        return scores
