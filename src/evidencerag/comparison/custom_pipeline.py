"""`CustomPipeline`: the custom-implementation counterpart of
`evidencerag.langchain_impl.pipeline.LangChainPipeline`, timed at the
same three stages (retrieval, rerank, generation) for a fair M8
comparison.

Built directly from the unchanged M4/M5/M6 building blocks
(`BM25Retriever`, `DenseRetriever`, `HybridRetriever`,
`reranking.rerank.rerank`, `generation.generate.generate_answer`) --
NOT `evidencerag.evaluation.systems.build_systems`, because that
module's `RerankingRetriever` composes retrieval+rerank into one
`retrieve()` call (exactly right for M7, which only needs the combined
`top_k` result), while M8 needs the retrieval and rerank stages timed
separately, matching `LangChainPipeline.run_question`'s breakdown.
Every object this module constructs is one of those same M4/M5/M6
classes, used exactly as `evidencerag.evaluation.systems.build_systems`
already uses them -- no retrieval/reranking logic is reimplemented
here.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from evidencerag.chunking.schema import Chunk
from evidencerag.comparison.schema import StageResult
from evidencerag.config import SETTINGS
from evidencerag.generation.generate import generate_answer
from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS, Generator
from evidencerag.reranking.rerank import rerank
from evidencerag.reranking.reranker import Reranker
from evidencerag.retrieval.bm25 import BM25Config, BM25Retriever
from evidencerag.retrieval.dense import DenseRetriever
from evidencerag.retrieval.embeddings import Embedder
from evidencerag.retrieval.rrf import HybridRetriever, RRFConfig


class CustomPipeline:
    """Builds the custom BM25 / Dense / Hybrid(RRF) / Hybrid+Reranker
    stack once from `chunks`, `embedder`, and `reranker`, then answers
    questions against it one at a time, recording per-stage wall-clock
    latency -- the exact custom-side counterpart of
    `evidencerag.langchain_impl.pipeline.LangChainPipeline`.

    `generator` is optional, with the same retrieval-only-when-`None`
    behavior as `LangChainPipeline`.
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
        self._chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}

        bm25 = BM25Retriever.build(chunks, config=BM25Config(k1=SETTINGS.bm25_k1, b=SETTINGS.bm25_b))
        dense = DenseRetriever.build(chunks, embedder=embedder)
        self.hybrid = HybridRetriever(
            bm25, dense, config=RRFConfig(rrf_k=SETTINGS.rrf_k, candidate_depth=candidate_depth)
        )
        self._reranker = reranker
        self._generator = generator
        self._max_new_tokens = max_new_tokens

    def run_question(self, question_text: str) -> StageResult:
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        candidates = self.hybrid.retrieve(question_text, top_k=self._candidate_depth)
        retrieval_latency = time.perf_counter() - retrieval_start
        candidate_ids = tuple(result.chunk_id for result in candidates)

        rerank_start = time.perf_counter()
        top5_results = rerank(question_text, candidates, self._chunk_text_by_id, self._reranker, top_k=self._top_k)
        rerank_latency = time.perf_counter() - rerank_start
        top5_ids = tuple(result.chunk_id for result in top5_results)

        answer: Optional[str] = None
        generation_latency: Optional[float] = None
        if self._generator is not None:
            generation_start = time.perf_counter()
            result = generate_answer(
                question_text, top5_results, self._chunk_text_by_id, self._generator, max_new_tokens=self._max_new_tokens
            )
            answer = result.answer
            generation_latency = time.perf_counter() - generation_start

        total_latency = time.perf_counter() - total_start

        return StageResult(
            top5_chunk_ids=top5_ids,
            candidate20_chunk_ids=candidate_ids,
            retrieval_latency_s=retrieval_latency,
            rerank_latency_s=rerank_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total_latency,
            answer=answer,
        )
