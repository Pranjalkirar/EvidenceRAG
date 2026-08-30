"""Embedding model abstraction for dense retrieval.

Mirrors the `Tokenizer` Protocol pattern in
evidencerag.chunking.tokenizer: a minimal structural interface,
implemented by the real model (`QwenEmbedder`) and, in tests, by a
lightweight deterministic fake (see tests/retrieval_fixtures.py) --
ordinary unit tests must never need to download or load the real
model.

Import of `sentence_transformers` (and therefore torch) is deferred to
`QwenEmbedder.__init__`, exactly like `TiktokenTokenizer` defers
importing `tiktoken` -- importing this module never pulls in ML
dependencies unless the real embedder is actually constructed.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

# Fixed per M4 spec -- must not be silently substituted.
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class Embedder(Protocol):
    """Minimal interface dense retrieval depends on. `embed_documents`
    and `embed_queries` may differ (some embedding models, including
    Qwen3-Embedding, recommend an instruction prefix for queries but
    not for documents) -- callers must use the right one for the
    right side. Both return an (n, dimension) float32 array, NOT
    necessarily L2-normalized; normalization is the caller's job (see
    dense.py), so an Embedder implementation never needs to know
    whether it's feeding cosine-via-inner-product search or something
    else.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class QwenEmbedder:
    """Qwen3-Embedding-0.6B via sentence-transformers (the integration
    path documented on the model's own model card, including its
    "query" prompt for asymmetric query/document embedding).

    Requires the `sentence-transformers` dependency and network access
    to the Hugging Face Hub (or a local/cached copy of the model) --
    neither is needed just to import this module.
    """

    def __init__(self, model_name: str = QWEN_EMBEDDING_MODEL, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        try:
            # Qwen3-Embedding's sentence-transformers config defines a
            # "query" prompt (an instruction prefix) for asymmetric
            # retrieval; use it whenever it's available.
            vectors = self._model.encode(
                list(texts), prompt_name="query", convert_to_numpy=True, show_progress_bar=False
            )
        except (KeyError, AttributeError):
            # Specific, narrow fallback: the loaded model/config simply
            # doesn't define a "query" prompt (sentence-transformers
            # raises KeyError for an unknown prompt name; a model/config
            # object with no `.prompts` at all -- e.g. a very old
            # sentence-transformers version -- would raise AttributeError).
            # This is the ONLY case that falls back to symmetric encoding.
            # Genuine runtime failures (CUDA, out-of-memory, tensor shape
            # errors, etc.) are NOT caught here and propagate as-is.
            vectors = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
