"""Shared fixtures for M5 reranking tests, following the same
test-only-helper convention as tests/retrieval_fixtures.py.

`FakeReranker` is a lightweight, fully deterministic stand-in for
`CrossEncoderReranker`: no model download, no torch, no network. It
also records every call it receives, so tests can assert exactly which
`(query, chunk_text)` pairs were constructed and passed to it.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class FakeReranker:
    """Deterministic hash-based reranker for tests. Not a real
    cross-encoder -- it captures no semantic relevance, only exact
    pair identity/difference, which is all these unit tests need.

    Every call to `score()` is recorded in `self.calls`, so tests can
    verify the exact pairs a caller constructed without needing a real
    model to produce a meaningful score.
    """

    def __init__(self, model_name: str = "fake-reranker") -> None:
        self._model_name = model_name
        self.calls: list[list[tuple[str, str]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.calls.append(list(pairs))
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        return np.array([self._score_one(query, text) for query, text in pairs], dtype=np.float32)

    @staticmethod
    def _score_one(query: str, text: str) -> float:
        digest = hashlib.sha256(f"{query}\x00{text}".encode("utf-8")).digest()
        as_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return (as_int / (2**64 - 1)) * 2.0 - 1.0  # deterministic float in [-1, 1]


class ConstantReranker:
    """Assigns each candidate a score from a caller-supplied mapping,
    keyed by chunk text -- used when a test needs to dictate the exact
    ranking outcome rather than rely on FakeReranker's hash-derived
    (but arbitrary) scores.
    """

    def __init__(self, score_by_text: dict[str, float], model_name: str = "constant-reranker") -> None:
        self._score_by_text = score_by_text
        self._model_name = model_name
        self.calls: list[list[tuple[str, str]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.calls.append(list(pairs))
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        return np.array([self._score_by_text[text] for _, text in pairs], dtype=np.float32)
