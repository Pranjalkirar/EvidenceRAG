"""Dense retrieval over M3 chunks: an `Embedder` (see embeddings.py) +
L2-normalized vectors + FAISS `IndexFlatIP`.

Because vectors are L2-normalized, inner product ranks identically to
cosine similarity -- an exact, brute-force (not approximate) index,
by design (see M4 spec: no IVF/HNSW/PQ here).

FAISS positions are an internal implementation detail. The corpus is
ordered once, deterministically, by `evidencerag.retrieval.corpus`,
and that same ordering is what gets added to the FAISS index -- so
position i in the index always corresponds to `self._chunk_ids[i]`,
and every result this module returns is translated back to a
`chunk_id` before it leaves. The FAISS integer itself never leaks out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from evidencerag.chunking.schema import Chunk
from evidencerag.retrieval.corpus import assert_matches_corpus, build_corpus
from evidencerag.retrieval.embeddings import Embedder
from evidencerag.retrieval.schema import RetrievalResult, validate_top_k

_METRIC = "cosine_via_normalized_inner_product"


@dataclass(frozen=True)
class DenseIndexMetadata:
    """Enough to validate/reconstruct a persisted dense index without
    re-embedding anything, and to catch it being paired with an
    incompatible embedder or a different chunk corpus."""

    embedding_model: str
    embedding_dimension: int
    metric: str
    normalized: bool
    num_chunks: int
    corpus_fingerprint: str


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # avoid dividing a zero vector by zero
    return vectors / norms


class DenseRetriever:
    """Dense retrieval over a fixed, deterministically ordered corpus
    of M3 chunks. Construct via `build()` or `load()`, not directly."""

    def __init__(
        self,
        *,
        index: "faiss.Index",
        chunk_ids: tuple[str, ...],
        metadata: DenseIndexMetadata,
        embedder: Embedder | None,
    ) -> None:
        self._index = index
        self._chunk_ids = chunk_ids
        self._metadata = metadata
        self._embedder = embedder

    @property
    def metadata(self) -> DenseIndexMetadata:
        return self._metadata

    @classmethod
    def build(cls, chunks: Iterable[Chunk], embedder: Embedder) -> "DenseRetriever":
        corpus = build_corpus(chunks)
        vectors = _l2_normalize(embedder.embed_documents(corpus.texts)) if corpus.texts else np.zeros(
            (0, embedder.dimension), dtype=np.float32
        )

        index = faiss.IndexFlatIP(embedder.dimension)
        if len(vectors):
            index.add(vectors)

        metadata = DenseIndexMetadata(
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.dimension,
            metric=_METRIC,
            normalized=True,
            num_chunks=len(corpus.chunk_ids),
            corpus_fingerprint=corpus.fingerprint,
        )
        return cls(index=index, chunk_ids=corpus.chunk_ids, metadata=metadata, embedder=embedder)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        validate_top_k(top_k)
        if self._embedder is None:
            raise RuntimeError(
                "This DenseRetriever has no embedder attached -- pass one to "
                "DenseRetriever.load(path, embedder=...) before calling retrieve()."
            )
        if not self._chunk_ids:
            return []

        query_vector = _l2_normalize(self._embedder.embed_queries([query]))
        top_k = min(top_k, len(self._chunk_ids))
        scores, positions = self._index.search(query_vector, top_k)

        results: list[RetrievalResult] = []
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
            if position < 0:  # FAISS pads with -1 if fewer than top_k matches exist
                continue
            results.append(
                RetrievalResult(chunk_id=self._chunk_ids[position], score=float(score), rank=rank, retriever="dense")
            )
        return results

    def verify_corpus(self, chunks: Iterable[Chunk]) -> None:
        """Raise ValueError if `chunks` isn't the corpus this index was
        built from (see evidencerag.retrieval.corpus)."""
        assert_matches_corpus(self._metadata.corpus_fingerprint, chunks)

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        payload = {"chunk_ids": list(self._chunk_ids), "metadata": asdict(self._metadata)}
        (dir_path / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path, embedder: Embedder | None = None, chunks: Iterable[Chunk] | None = None) -> "DenseRetriever":
        """Load a persisted dense index.

        Pass `chunks` -- the M3 chunk collection this index is expected
        to match -- whenever they're available; this is the recommended
        way to load an index, since it validates (via the same
        corpus-fingerprint mechanism as `verify_corpus()`) that the
        index wasn't built from a different chunk collection before
        it's used for retrieval. `chunks` is optional (e.g. when only
        inspecting `.metadata`, with no chunks on hand) but loading
        without it means that check doesn't happen until/unless you
        call `.verify_corpus()` yourself.
        """
        dir_path = Path(dir_path)
        index = faiss.read_index(str(dir_path / "index.faiss"))
        payload = json.loads((dir_path / "metadata.json").read_text(encoding="utf-8"))
        metadata = DenseIndexMetadata(**payload["metadata"])
        chunk_ids = tuple(payload["chunk_ids"])

        if embedder is not None:
            if embedder.model_name != metadata.embedding_model:
                raise ValueError(
                    f"Index was built with embedding model {metadata.embedding_model!r}, "
                    f"but the provided embedder is {embedder.model_name!r}."
                )
            if embedder.dimension != metadata.embedding_dimension:
                raise ValueError(
                    f"Index embedding dimension is {metadata.embedding_dimension}, "
                    f"but the provided embedder's dimension is {embedder.dimension}."
                )

        retriever = cls(index=index, chunk_ids=chunk_ids, metadata=metadata, embedder=embedder)
        if chunks is not None:
            retriever.verify_corpus(chunks)
        return retriever
