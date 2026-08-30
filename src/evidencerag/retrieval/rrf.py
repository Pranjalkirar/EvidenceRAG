"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF combines RANKS, never raw scores: BM25 scores and dense cosine
scores live on incomparable scales, so `bm25_score + dense_score`
would be meaningless. RRF sidesteps that entirely by only ever looking
at each retriever's rank ordering.

    RRF(d) = sum over rankings r containing d of  1 / (rrf_k + rank_r(d))
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from evidencerag.retrieval.bm25 import BM25Retriever
from evidencerag.retrieval.dense import DenseRetriever
from evidencerag.retrieval.schema import RetrievalResult, validate_top_k


@dataclass(frozen=True)
class RRFConfig:
    """`rrf_k` is the RRF smoothing constant (deliberately not named
    `top_k` -- it plays a different role: dampening the influence of
    very low ranks, not limiting how many results come back).

    `candidate_depth` is how many candidates each base retriever
    (BM25, dense) contributes to the fusion -- independent of, and
    typically larger than, the final `top_k` requested from
    `HybridRetriever.retrieve()`.
    """

    rrf_k: int = 60
    candidate_depth: int = 20


def reciprocal_rank_fusion(rankings: Sequence[Sequence[RetrievalResult]], rrf_k: int = 60) -> list[RetrievalResult]:
    """Fuse any number of per-retriever rank lists into one hybrid
    ranking, by chunk_id. Each input ranking's own `.rank` field is
    used (1-based); this does not re-derive rank from list position,
    so callers may safely pass a slice or a re-ordered list as long as
    `.rank` values already reflect the intended order.

    Edge cases handled:
      - a chunk in multiple rankings: contributions from each are
        summed;
      - a chunk in only one ranking: still eligible, using just that
        contribution;
      - rankings of different lengths: each contributes independently;
      - a duplicate chunk_id within a single ranking: only its first
        occurrence counts, so an accidental duplicate can't be
        double-counted;
      - an empty ranking: contributes nothing, no error;
      - no rankings at all, or all empty: returns [].
      - ties in the fused score: broken deterministically by chunk_id.
    """
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        seen_in_this_ranking: set[str] = set()
        for result in ranking:
            if result.chunk_id in seen_in_this_ranking:
                continue
            seen_in_this_ranking.add(result.chunk_id)
            scores[result.chunk_id] += 1.0 / (rrf_k + result.rank)

    ordered_ids = sorted(scores.keys(), key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        RetrievalResult(chunk_id=chunk_id, score=scores[chunk_id], rank=rank, retriever="hybrid")
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRetriever:
    """BM25 + Dense, fused by RRF.

    Construction verifies both were built from the exact same corpus
    (comparing `bm25.fingerprint` to `dense.metadata.corpus_fingerprint`
    -- the same corpus-fingerprint mechanism `verify_corpus()` uses) and
    raises `ValueError` immediately on a mismatch, rather than silently
    fusing rankings over two different chunk collections.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, config: RRFConfig | None = None) -> None:
        if bm25.fingerprint != dense.metadata.corpus_fingerprint:
            raise ValueError(
                "HybridRetriever requires BM25 and Dense to be built from the "
                "exact same M3 chunk corpus, but their corpus fingerprints "
                f"differ (bm25={bm25.fingerprint!r}, dense={dense.metadata.corpus_fingerprint!r}). "
                "Rebuild both from the same chunk collection."
            )
        self._bm25 = bm25
        self._dense = dense
        self._config = config or RRFConfig()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        validate_top_k(top_k)
        depth = max(self._config.candidate_depth, top_k)
        bm25_results = self._bm25.retrieve(query, top_k=depth)
        dense_results = self._dense.retrieve(query, top_k=depth)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], rrf_k=self._config.rrf_k)
        return fused[:top_k]
