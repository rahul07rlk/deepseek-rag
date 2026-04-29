"""Qwen3-Reranker wrapper with a CrossEncoder-compatible .predict() interface.

Standalone copy used by the rerank-server microservice. Identical to the
copy in deepseek-rag/src/utils/qwen3_reranker.py. Kept duplicated so this
folder can be copied to the second laptop without dragging the rest of
the project along.

Qwen3-Reranker models are causal LMs that score relevance by generating a
yes/no token. The probability of "yes" minus the probability of "no" gives
log-odds; the caller is expected to sigmoid-normalize.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct, output 'yes' or 'no'."
)
_PREFIX = f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

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
        dtype = torch.float16 if "cuda" in device else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )

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
                logits = self.model(**enc).logits[:, -1, :]
            log_odds = logits[:, self._true_id] - logits[:, self._false_id]
            scores.extend(log_odds.cpu().float().tolist())
        return scores
