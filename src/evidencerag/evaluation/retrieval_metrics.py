"""Chunk-level retrieval metrics: Recall@k and reciprocal rank.

Gold relevance is a `frozenset[str]` of M3 `chunk_id`s -- see `gold.py`
for how those are built from resolved M3 `EvidenceChunkMapping` rows.
Ranking is taken from `RetrievalResult.rank` / list order, never from
raw BM25/dense/RRF/reranker scores, so BM25 and dense scores (which
live on different scales) are never compared directly.

Every function here is pure and takes plain data -- no `Retriever`,
no `Chunk`, no I/O -- so they're trivial to unit test against
hand-built `RetrievalResult` lists.

Single-reference functions (`recall_at_k`, `reciprocal_rank`) score
against exactly one answer annotation's gold chunk set, and require a
non-empty one -- QASPER's multiple annotators are alternative
references, never merged (see `gold.py`), so "which reference(s) apply
to this question" is decided by the caller, not guessed here. The
`max_*` wrappers apply the official "max across references" rule and
are the ones `harness.py` actually calls; they return `None` (not
`0.0`) when a question has zero mappable references at all -- that is
reported as an explicit exclusion, never a silent zero score.
"""

from __future__ import annotations

from typing import AbstractSet, Optional, Sequence

from evidencerag.retrieval.schema import RetrievalResult


def recall_at_k(results: Sequence[RetrievalResult], relevant_chunk_ids: AbstractSet[str], k: int) -> float:
    """Fraction of `relevant_chunk_ids` present in `results[:k]`.

    `relevant_chunk_ids` must be non-empty -- recall against an empty
    gold set is undefined, not zero; callers must filter empty
    references out before calling (see `max_recall_at_k`).
    """
    if not relevant_chunk_ids:
        raise ValueError("recall_at_k requires a non-empty relevant_chunk_ids")
    retrieved = {result.chunk_id for result in results[:k]}
    return len(retrieved & relevant_chunk_ids) / len(relevant_chunk_ids)


def reciprocal_rank(results: Sequence[RetrievalResult], relevant_chunk_ids: AbstractSet[str]) -> float:
    """1 / rank of the first result in `relevant_chunk_ids`, or 0.0 if
    none of `results` is relevant.

    `relevant_chunk_ids` must be non-empty, for the same reason as
    `recall_at_k`.
    """
    if not relevant_chunk_ids:
        raise ValueError("reciprocal_rank requires a non-empty relevant_chunk_ids")
    for result in results:
        if result.chunk_id in relevant_chunk_ids:
            return 1.0 / result.rank
    return 0.0


def max_recall_at_k(
    results: Sequence[RetrievalResult], references: Sequence[AbstractSet[str]], k: int
) -> Optional[float]:
    """Max `recall_at_k` across a question's non-empty chunk
    references. `None` when `references` is empty -- i.e. every answer
    annotation for this question had zero chunk-mappable evidence --
    signaling the question should be excluded from any aggregate over
    this metric, not scored 0.0.
    """
    if not references:
        return None
    return max(recall_at_k(results, reference, k) for reference in references)


def max_reciprocal_rank(
    results: Sequence[RetrievalResult], references: Sequence[AbstractSet[str]]
) -> Optional[float]:
    """Max `reciprocal_rank` across a question's non-empty chunk
    references, with the same `None`-means-exclude convention as
    `max_recall_at_k`.
    """
    if not references:
        return None
    return max(reciprocal_rank(results, reference) for reference in references)
