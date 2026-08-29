"""Token counting for chunking, isolated from everything else in this
package. The `Chunk` model (schema.py) only stores an integer
`token_count` — it has no idea which tokenizer produced it. This
module is the one place that knows.

TOKENIZER CHOICE: tiktoken, `cl100k_base` encoding.

Why: we need *some* real, consistent subword tokenizer to size chunks
by (character/whitespace counts are poor proxies for how much context
an embedding/generation model actually sees). `tiktoken` is:
  - lightweight (pure Python API over a small Rust extension; no
    torch/transformers pulled in),
  - fast and well-supported,
  - not tied to loading an actual model checkpoint just to count
    tokens.

This is a token-COUNTING choice for sizing chunks, not a commitment
that retrieval/generation must use this exact tokenizer later — the
planned Qwen3 models use their own tokenizer, which may split text
slightly differently. That's acceptable: chunk sizing only needs to
be a consistent, real approximation of "how much text is this", not
an exact match to a future model's vocabulary. Swapping the tokenizer
later only means touching this module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Tokenizer(Protocol):
    """Minimal interface chunking code depends on. Anything satisfying
    this (tiktoken, a HF tokenizer wrapper, ...) can be swapped in."""

    def count_tokens(self, text: str) -> int: ...


class TiktokenTokenizer:
    """Default tokenizer: tiktoken's cl100k_base encoding."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


@lru_cache(maxsize=1)
def get_default_tokenizer() -> TiktokenTokenizer:
    """Process-wide cached instance, since constructing a tiktoken
    encoding has non-trivial (if small) fixed overhead."""
    return TiktokenTokenizer()
