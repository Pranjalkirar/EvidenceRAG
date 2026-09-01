"""`LangChainPipeline`: the LangChain-side counterpart of
`evidencerag.comparison.custom_pipeline.CustomPipeline`.

Both classes expose the identical `run_question(question_text)` ->
`evidencerag.comparison.schema.StageResult` contract, so
`evidencerag.comparison.runner.run_comparison` can drive either one
without knowing which is which -- the whole point of M8's controlled
comparison.

Scope (see the M8 README section for the full rationale): this
pipeline implements the single primary chain the M8 diagram describes
-- BM25 + Dense -> RRF -> Cross-Encoder Reranker -> (optionally) LLM
generation -- not all four M7 systems. M7 already exhaustively compares
BM25/Dense/Hybrid/Hybrid+Reranker within the custom implementation;
M8's question is framework overhead for the best-performing
configuration, not a second full BM25-vs-dense study.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

from evidencerag.langchain_impl import require_langchain

require_langchain()

from evidencerag.chunking.schema import Chunk  # noqa: E402
from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS, Generator  # noqa: E402
from evidencerag.langchain_impl.documents import document_chunk_id  # noqa: E402
from evidencerag.langchain_impl.llm import EvidenceRAGLLM  # noqa: E402
from evidencerag.langchain_impl.prompt import build_langchain_prompt  # noqa: E402
from evidencerag.langchain_impl.reranking import rerank_documents  # noqa: E402
from evidencerag.langchain_impl.retrievers import (  # noqa: E402
    HybridRerankRetriever,
    build_bm25_retriever,
    build_dense_retriever,
    build_hybrid_retriever,
)
from evidencerag.reranking.reranker import Reranker  # noqa: E402
from evidencerag.retrieval.embeddings import Embedder  # noqa: E402


@dataclass(frozen=True)
class LangChainStageResult:
    """Per-question output of `LangChainPipeline.run_question`, mirroring
    `evidencerag.comparison.schema.StageResult` field-for-field (kept as
    a separate dataclass, not the same class, so this module never has
    to import `evidencerag.comparison` -- `comparison.runner` converts
    between the two, keeping the dependency direction one-way:
    `comparison` depends on `langchain_impl`, never the reverse)."""

    top5_chunk_ids: tuple[str, ...]
    candidate20_chunk_ids: tuple[str, ...]
    retrieval_latency_s: float
    rerank_latency_s: float
    generation_latency_s: Optional[float]
    total_latency_s: float
    answer: Optional[str]


class LangChainPipeline:
    """Builds the LangChain BM25 / Dense / Hybrid(RRF) / Hybrid+Reranker
    chain once from `chunks`, `embedder`, and `reranker` (the SAME
    objects the custom pipeline uses -- see `evidencerag.comparison.
    custom_pipeline.CustomPipeline`), then answers questions against
    it, one at a time, recording per-stage wall-clock latency.

    `generator` is optional: when `None`, `run_question` skips
    generation entirely (retrieval-only mode, mirroring M7's default
    `mode="retrieval"`) and `LangChainStageResult.answer` /
    `generation_latency_s` are both `None`.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embedder: Embedder,
        reranker: Reranker,
        generator: Optional[Generator] = None,
        candidate_depth: int = 20,
        top_k: int = 5,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        chunks = list(chunks)
        self._candidate_depth = candidate_depth
        self._top_k = top_k

        bm25 = build_bm25_retriever(chunks, k=candidate_depth)
        dense = build_dense_retriever(chunks, embedder=embedder, k=candidate_depth)
        self._hybrid = build_hybrid_retriever(bm25, dense, k=candidate_depth)
        self._hybrid_rerank = HybridRerankRetriever(
            base_retriever=self._hybrid, reranker=reranker, candidate_k=candidate_depth, top_n=top_k
        )
        self._reranker = reranker
        self._llm = EvidenceRAGLLM(generator, max_new_tokens=max_new_tokens) if generator is not None else None

    @property
    def hybrid_rerank_retriever(self) -> HybridRerankRetriever:
        """The composed hybrid+reranker retriever as a single
        `BaseRetriever`, for debugging/inspection (e.g. `.invoke(query)`
        in a notebook) -- `run_question` does not call this directly,
        since it needs the retrieval and rerank stages timed
        separately (see below); this is exposed purely so the
        LangChain chain can be inspected the same way a `Retriever`
        from `evidencerag.evaluation.systems` can.
        """
        return self._hybrid_rerank

    def run_question(self, question_text: str) -> LangChainStageResult:
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        candidates = self._hybrid.invoke(question_text)
        retrieval_latency = time.perf_counter() - retrieval_start
        candidate_ids = tuple(document_chunk_id(document) for document in candidates)

        rerank_start = time.perf_counter()
        top5_documents = rerank_documents(question_text, candidates, self._reranker, top_n=self._top_k)
        rerank_latency = time.perf_counter() - rerank_start
        top5_ids = tuple(document_chunk_id(document) for document in top5_documents)

        answer: Optional[str] = None
        generation_latency: Optional[float] = None
        if self._llm is not None:
            generation_start = time.perf_counter()
            prompt = build_langchain_prompt(question_text, top5_documents)
            answer = self._llm.invoke(prompt)
            generation_latency = time.perf_counter() - generation_start

        total_latency = time.perf_counter() - total_start

        return LangChainStageResult(
            top5_chunk_ids=top5_ids,
            candidate20_chunk_ids=candidate_ids,
            retrieval_latency_s=retrieval_latency,
            rerank_latency_s=rerank_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total_latency,
            answer=answer,
        )
