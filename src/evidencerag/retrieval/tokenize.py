"""Lexical tokenization for BM25 -- a retrieval-specific preprocessing
step, unrelated to M3's `evidencerag.chunking.tokenizer` (which counts
subword tokens to size chunks, not to build a BM25 vocabulary).

Deliberately simple and deterministic: lowercase, alphanumeric runs.
No stemming, no stopword removal -- a transparent baseline consistent
with M4 establishing a baseline, not a tuned lexical pipeline.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, in order. Never raises on empty
    or non-alphanumeric-only text -- just returns an empty list."""
    return _TOKEN_RE.findall(text.lower())
