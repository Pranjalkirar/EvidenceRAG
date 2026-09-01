"""Tests for evidencerag.langchain_impl.pipeline.LangChainPipeline --
orchestration and timing-field presence only (retrieval-quality
correctness of the individual stages is covered by
test_langchain_retrievers.py / test_langchain_reranking.py).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_community")
pytest.importorskip("langchain")

from evidencerag.langchain_impl.pipeline import LangChainPipeline  # noqa: E402
from tests.generation_fixtures import FakeGenerator  # noqa: E402
from tests.reranking_fixtures import FakeReranker  # noqa: E402
from tests.retrieval_fixtures import FakeEmbedder, make_chunk  # noqa: E402


def _chunks():
    return [make_chunk(chunk_id=f"c{i}", text=f"Chunk number {i} about topic {i % 3}.") for i in range(6)]


def test_retrieval_only_mode_never_calls_generator():
    pipeline = LangChainPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=None,
        candidate_depth=4, top_k=2,
    )
    result = pipeline.run_question("some question")

    assert result.answer is None
    assert result.generation_latency_s is None
    assert len(result.top5_chunk_ids) == 2
    assert len(result.candidate20_chunk_ids) <= 4


def test_end_to_end_mode_populates_answer():
    generator = FakeGenerator(canned_answer="a grounded answer")
    pipeline = LangChainPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=generator,
        candidate_depth=4, top_k=2,
    )
    result = pipeline.run_question("some question")

    assert result.answer == "a grounded answer"
    assert result.generation_latency_s is not None
    assert result.generation_latency_s >= 0.0
    assert len(generator.calls) == 1


def test_latency_fields_are_non_negative():
    pipeline = LangChainPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=None,
        candidate_depth=4, top_k=2,
    )
    result = pipeline.run_question("q")

    assert result.retrieval_latency_s >= 0.0
    assert result.rerank_latency_s >= 0.0
    assert result.total_latency_s >= 0.0
    assert result.total_latency_s >= result.retrieval_latency_s + result.rerank_latency_s - 1e-6


def test_hybrid_rerank_retriever_property_is_inspectable():
    pipeline = LangChainPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=None,
        candidate_depth=4, top_k=2,
    )
    from evidencerag.langchain_impl.retrievers import HybridRerankRetriever

    assert isinstance(pipeline.hybrid_rerank_retriever, HybridRerankRetriever)
