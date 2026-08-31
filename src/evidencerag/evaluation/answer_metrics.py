"""QASPER-style Answer F1: SQuAD-style token normalization and
token-level F1, matching the official QASPER evaluator exactly
(including its normalization order and its "reported type is whichever
reference scored highest" convention -- not a majority vote, and not
necessarily the type of the "correct" reference).

Reference construction (which of `unanswerable` / `extractive_spans` /
`free_form_answer` / `yes_no` becomes the reference text, and its
type) lives in `gold.reference_answer_and_type`, which follows the
same official precedence.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization, in the official evaluator's exact
    order: lowercase, then strip punctuation (deleted, not replaced
    with a space -- matching the official script's behavior, including
    its edge cases around punctuation-adjacent words), then drop
    standalone articles, then collapse whitespace.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 between normalized `prediction` and `reference`,
    exactly as the official QASPER/SQuAD evaluator computes it
    (multiset token overlap, not set overlap -- repeated tokens count).
    """
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1_and_type(prediction: str, references: Sequence[tuple[str, str]]) -> tuple[float, str]:
    """Max `token_f1_score` across `references`, paired with *that*
    reference's answer type -- matching the official evaluator, which
    reports the type of whichever reference produced the highest
    score, not the type of some single "correct" reference.

    `references` must be non-empty -- a question with zero answer
    annotations is a data error the caller must not silently paper
    over.
    """
    if not references:
        raise ValueError("answer_f1_and_type requires at least one reference")
    scored = [(token_f1_score(prediction, text), answer_type) for text, answer_type in references]
    return max(scored, key=lambda scored_type: scored_type[0])
