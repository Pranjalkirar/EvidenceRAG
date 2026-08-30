"""Common retrieval result model, shared by every retriever
(BM25, dense, hybrid) so callers and tests can treat them uniformly.

The canonical document identity is always the M3 `chunk_id` (see
evidencerag.chunking.schema.Chunk) -- never a FAISS integer position,
a BM25-internal index, or an array offset. Those are all internal
implementation details of a specific retriever and must never leak
out as the application's notion of "which document".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved item for one query, from one retriever.

    `rank` is 1-based (rank=1 is the top result) and unique within a
    single retrieve() call's result list.
    """

    chunk_id: str
    score: float
    rank: int
    retriever: str


def validate_top_k(top_k: int) -> None:
    """Shared `top_k` validation for every retriever's `retrieve()`, so
    the error looks the same everywhere. `top_k` larger than the
    corpus size is fine (just returns as many results as exist) --
    only `top_k < 1` is rejected.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
