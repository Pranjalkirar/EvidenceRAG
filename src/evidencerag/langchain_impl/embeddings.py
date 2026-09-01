"""LangChain `Embeddings` adapter around our own `Embedder`.

Deliberately NOT a fresh LangChain embeddings integration (e.g.
`langchain_huggingface.HuggingFaceEmbeddings` loading Qwen3-Embedding
a second time) -- `EvidenceRAGEmbeddings` wraps the SAME `Embedder`
instance (typically the SAME `QwenEmbedder`, and, in the M8 runner, the
literal SAME object) the custom pipeline uses, so:

  - the model is loaded once, not twice (real GPU/RAM cost on Kaggle);
  - custom vs LangChain dense retrieval differ ONLY in the surrounding
    orchestration (vector store, retriever interface), never in which
    vectors get compared, which is the whole point of a controlled
    comparison.

L2-normalization happens here (mirroring
`evidencerag.retrieval.dense._l2_normalize` exactly) rather than
relying on the vector store to do it, so that whichever LangChain
`VectorStore` `retrievers.py` uses, cosine-via-inner-product ranking
matches `evidencerag.retrieval.dense.DenseRetriever`'s own
`IndexFlatIP`-over-normalized-vectors behavior as closely as the two
frameworks' APIs allow.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from evidencerag.langchain_impl import require_langchain

require_langchain()

from langchain_core.embeddings import Embeddings  # noqa: E402

from evidencerag.retrieval.embeddings import Embedder  # noqa: E402


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class EvidenceRAGEmbeddings(Embeddings):
    """Adapts our `Embedder` Protocol (a real `QwenEmbedder`, or a
    test `FakeEmbedder`) to LangChain's `Embeddings` interface.

    `embed_documents`/`embed_query` are LangChain's own naming (not
    `embed_queries`, plural) -- this class is the ONLY place that
    naming mismatch is bridged; `Embedder.embed_queries` is still
    called with a single-item list underneath.
    """

    def __init__(self, embedder: Embedder, normalize: bool = True) -> None:
        self._embedder = embedder
        self._normalize = normalize

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._embedder.embed_documents(list(texts))
        if self._normalize:
            vectors = _l2_normalize(vectors)
        return [row.tolist() for row in np.asarray(vectors, dtype=np.float32)]

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embedder.embed_queries([text])
        if self._normalize:
            vectors = _l2_normalize(vectors)
        return np.asarray(vectors, dtype=np.float32)[0].tolist()
