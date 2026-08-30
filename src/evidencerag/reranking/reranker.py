"""Cross-encoder reranking model abstraction.

Mirrors the `Embedder` Protocol pattern in
evidencerag.retrieval.embeddings: a minimal structural interface,
implemented by the real model (`CrossEncoderReranker`) and, in tests,
by a lightweight deterministic fake (see
tests/reranking_fixtures.py) -- ordinary unit tests must never need
to download or load the real model.

Import of `sentence_transformers` (and therefore torch) is deferred to
`CrossEncoderReranker.__init__`, exactly like `QwenEmbedder` defers
importing it -- importing this module never pulls in ML dependencies
unless the real reranker is actually constructed.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

# Fixed per M5 spec -- must not be silently substituted.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Minimal interface reranking depends on: score (query, chunk_text)
    pairs and return one float relevance score per pair, in the same
    order the pairs were given. Higher scores mean more relevant --
    the scale itself is implementation-specific (e.g. a cross-encoder's
    raw logit) and is never compared across different rerankers, only
    used to sort candidates from a single reranker's own call.
    """

    @property
    def model_name(self) -> str: ...

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class CrossEncoderReranker:
    """`cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers'
    `CrossEncoder` wrapper (the integration path documented on the
    model's own model card).

    Requires the `sentence-transformers` dependency (already declared
    for M4's `QwenEmbedder`) and network access to the Hugging Face Hub
    (or a local/cached copy of the model) -- neither is needed just to
    import this module.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self._model.predict([list(pair) for pair in pairs])
        return np.asarray(scores, dtype=np.float32)
