"""Second-stage cross-encoder reranking on top of M4 retrieval.

    M4 Hybrid retrieval (top `candidate_depth`)
                    │
                    ▼
        M5 cross-encoder reranking
                    │
                    ▼
              final top `top_k`

Reranking operates ONLY on the candidate list a base retriever already
returned -- it never touches the full corpus, never re-derives
chunk_ids, and never reconstructs chunks. `chunk_id` identity is
preserved exactly, matching the contract in
evidencerag.retrieval.schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evidencerag.chunking.schema import Chunk
from evidencerag.reranking.reranker import Reranker
from evidencerag.retrieval.base import Retriever
from evidencerag.retrieval.schema import RetrievalResult, validate_top_k


def rerank(
    query: str,
    candidates: Sequence[RetrievalResult],
    chunk_text_by_id: Mapping[str, str],
    reranker: Reranker,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rerank an already-retrieved candidate list with `reranker`.

    `candidates` are consumed exactly as given -- no candidate is
    dropped, added, or looked up against the wider corpus; only the
    candidates the caller supplied are ever scored. `chunk_text_by_id`
    supplies the text for each candidate's `(query, chunk_text)` pair;
    every `candidates[i].chunk_id` must have an entry (a missing one is
    a caller bug and raises `KeyError`, the same "caller's job" contract
    `DenseRetriever`/`BM25Retriever` place on their own callers).

    Deterministic ordering: rank by reranker score descending, then by
    chunk_id ascending to break ties reproducibly -- same convention as
    `BM25Retriever.retrieve` and `reciprocal_rank_fusion`.
    """
    validate_top_k(top_k)
    if not candidates:
        return []

    pairs = [(query, chunk_text_by_id[candidate.chunk_id]) for candidate in candidates]
    scores = reranker.score(pairs)

    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i].chunk_id))
    top = order[:top_k]
    return [
        RetrievalResult(chunk_id=candidates[i].chunk_id, score=float(scores[i]), rank=rank, retriever="reranker")
        for rank, i in enumerate(top, start=1)
    ]


@dataclass(frozen=True)
class RerankConfig:
    """`candidate_depth` is how many results the base retriever is
    asked for before reranking narrows them down to `retrieve()`'s
    requested `top_k` -- independent of, and typically larger than,
    that final `top_k`, mirroring `RRFConfig.candidate_depth`.
    """

    candidate_depth: int = 20


class RerankingRetriever:
    """M4 base retriever (e.g. `HybridRetriever`) + M5 `Reranker`,
    composed into a single `Retriever`-shaped pipeline stage.

    Wraps `base_retriever` unchanged -- it does not alter BM25, Dense,
    FAISS, RRF, or corpus-construction behavior at all. It only calls
    `base_retriever.retrieve()` for `config.candidate_depth` candidates
    and reranks that fixed set; it never reranks the full corpus.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        chunks: Iterable[Chunk],
        config: RerankConfig | None = None,
    ) -> None:
        self._base_retriever = base_retriever
        self._reranker = reranker
        self._chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
        self._config = config or RerankConfig()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        validate_top_k(top_k)
        depth = max(self._config.candidate_depth, top_k)
        candidates = self._base_retriever.retrieve(query, top_k=depth)
        return rerank(query, candidates, self._chunk_text_by_id, self._reranker, top_k=top_k)
