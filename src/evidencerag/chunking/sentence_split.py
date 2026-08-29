"""Lightweight sentence boundary splitting, used only when a paragraph
is too large to keep whole (see chunker.py).

This is a heuristic regex-based splitter, NOT a full NLP sentence
tokenizer (no spaCy/nltk dependency, per the instruction not to add
unrelated NLP dependencies for chunking). It will occasionally get
abbreviations, decimals, or citations wrong (e.g. "Fig. 2" or "et
al."). That is an accepted limitation for this milestone: it is only
used as a fallback to avoid arbitrary character-level cuts inside
oversized paragraphs, not as a general-purpose sentence tokenizer.
"""

from __future__ import annotations

import re

# Split after ., !, or ? followed by whitespace and a capital letter or
# digit (typical sentence starts), while trying to avoid splitting on
# common abbreviations and single-letter initials. Not perfect -- see
# module docstring.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "et al.", "fig.", "eq.", "ref.", "vs.",
    "approx.", "cf.", "resp.", "no.",
}

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text: str) -> list[str]:
    """Best-effort sentence split. Never drops text: joining the
    result with single spaces reconstructs (whitespace-normalized)
    the original.
    """
    text = text.strip()
    if not text:
        return []

    raw_sentences = _SENTENCE_BOUNDARY_RE.split(text)

    # Merge back any split that occurred right after a known
    # abbreviation (best-effort cleanup of the regex's false positives).
    sentences: list[str] = []
    for piece in raw_sentences:
        if (
            sentences
            and sentences[-1].split()[-1:]
            and sentences[-1].rstrip().lower().endswith(tuple(_ABBREVIATIONS))
        ):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)

    return [s.strip() for s in sentences if s.strip()]
