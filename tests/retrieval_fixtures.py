"""Shared fixtures for M4 retrieval tests, following the same
test-only-helper convention as tests/chunking_fixtures.py.

`FakeEmbedder` is a lightweight, fully deterministic stand-in for
`QwenEmbedder`: no model download, no torch, no network. It hashes
each text into a fixed-size vector, so the SAME text always produces
the SAME vector (needed for build/save/load/retrieve-equivalence
tests), and different texts are (with overwhelming probability)
different vectors -- enough to exercise real FAISS search behavior
without needing a real embedding model.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from evidencerag.chunking.schema import Chunk


class FakeEmbedder:
    """Deterministic hash-based embedder for tests. Not a real
    embedding model -- it captures no semantic similarity, only exact
    text identity/difference, which is all these unit tests need."""

    def __init__(self, dimension: int = 16, model_name: str = "fake-embedder") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        # Derive several independent hash values from `text` so the
        # vector isn't just one repeated float.
        for i in range(self._dimension):
            digest = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
            # Map first 8 bytes of the digest to a float in [-1, 1].
            as_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
            vector[i] = (as_int / (2**64 - 1)) * 2.0 - 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        # Symmetric for this fake -- real asymmetric prompting is a
        # QwenEmbedder-specific concern (see embeddings.py).
        return self.embed_documents(texts)


def make_chunk(
    *,
    chunk_id: str,
    text: str,
    paper_id: str = "9999.00001",
    split: str = "train",
    section_index: int | None = 0,
    section_title: str = "Intro",
    paragraph_indices: tuple[int, ...] = (0,),
    token_count: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        split=split,
        section_index=section_index,
        section_title=section_title,
        paragraph_indices=paragraph_indices,
        text=text,
        token_count=token_count if token_count is not None else len(text.split()),
    )
