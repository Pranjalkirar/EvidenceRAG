"""Builds the four systems M7 compares -- BM25, Dense, Hybrid, and
Hybrid+Cross-Encoder-Reranker -- wiring M4/M5 exactly the way
`scripts/retrieve_smoke_test.py` / `scripts/rerank_smoke_test.py`
already do, from unchanged `evidencerag.config.SETTINGS` values. No
M4/M5 hyperparameter is overridden here.

`EvaluationSystems.retrievers` is what `harness.py` calls for each
system's final top-`top_k` output. `EvaluationSystems.candidate_sources`
is what it calls for each system's top-`candidate_depth` *pre-selection*
candidates (Recall@20): for `bm25`/`dense`/`hybrid` this is the system
itself, but for `hybrid_rerank` it is the same `hybrid` retriever --
Recall@20 for Hybrid+Reranker is defined as the pre-reranking Hybrid
candidate set, so it is computed from the same object, not from
`hybrid_rerank.retrieve()` (which would already have reranked and
narrowed to `top_k`). This is why `hybrid` and `hybrid_rerank` report
identical Recall@20 by construction -- not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evidencerag.chunking.schema import Chunk
from evidencerag.config import SETTINGS
from evidencerag.reranking.reranker import Reranker
from evidencerag.reranking.rerank import RerankConfig, RerankingRetriever
from evidencerag.retrieval.base import Retriever
from evidencerag.retrieval.bm25 import BM25Config, BM25Retriever
from evidencerag.retrieval.dense import DenseRetriever
from evidencerag.retrieval.embeddings import Embedder
from evidencerag.retrieval.rrf import HybridRetriever, RRFConfig

SYSTEM_NAMES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class EvaluationSystems:
    retrievers: dict[str, Retriever]
    candidate_sources: dict[str, Retriever]


def build_systems(chunks: Iterable[Chunk], embedder: Embedder, reranker: Reranker) -> EvaluationSystems:
    chunks = list(chunks)

    bm25 = BM25Retriever.build(chunks, config=BM25Config(k1=SETTINGS.bm25_k1, b=SETTINGS.bm25_b))
    dense = DenseRetriever.build(chunks, embedder=embedder)
    hybrid = HybridRetriever(
        bm25, dense, config=RRFConfig(rrf_k=SETTINGS.rrf_k, candidate_depth=SETTINGS.retrieval_candidate_depth)
    )
    hybrid_rerank = RerankingRetriever(
        base_retriever=hybrid,
        reranker=reranker,
        chunks=chunks,
        config=RerankConfig(candidate_depth=SETTINGS.retrieval_candidate_depth),
    )

    retrievers: dict[str, Retriever] = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid_rerank,
    }
    candidate_sources: dict[str, Retriever] = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid,
    }
    return EvaluationSystems(retrievers=retrievers, candidate_sources=candidate_sources)
