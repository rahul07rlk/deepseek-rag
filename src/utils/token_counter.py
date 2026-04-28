"""Token accounting. cl100k_base is close enough to DeepSeek's tokenizer."""
import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def fits_in_budget(text: str, budget: int) -> bool:
    return count_tokens(text) <= budget


def truncate_to_budget(text: str, budget: int) -> str:
    tokens = _encoder.encode(text)
    if len(tokens) <= budget:
        return text
    return _encoder.decode(tokens[:budget])
