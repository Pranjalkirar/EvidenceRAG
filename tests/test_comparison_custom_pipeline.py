"""Tests for evidencerag.comparison.custom_pipeline.CustomPipeline --
orchestration and timing-field presence only (BM25/Dense/RRF/rerank
correctness is already covered by the M4/M5 test suites; this module
reuses those classes unchanged -- see custom_pipeline.py's docstring).
"""

from __future__ import annotations

from evidencerag.comparison.custom_pipeline import CustomPipeline
from tests.generation_fixtures import FakeGenerator
from tests.reranking_fixtures import FakeReranker
from tests.retrieval_fixtures import FakeEmbedder, make_chunk


def _chunks():
    return [make_chunk(chunk_id=f"c{i}", text=f"Chunk number {i} about topic {i % 3}.") for i in range(6)]


def test_retrieval_only_mode_never_calls_generator():
    pipeline = CustomPipeline(
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
    pipeline = CustomPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=generator,
        candidate_depth=4, top_k=2,
    )
    result = pipeline.run_question("some question")

    assert result.answer == "a grounded answer"
    assert result.generation_latency_s is not None
    assert len(generator.calls) == 1


def test_latency_fields_are_non_negative():
    pipeline = CustomPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=None,
        candidate_depth=4, top_k=2,
    )
    result = pipeline.run_question("q")

    assert result.retrieval_latency_s >= 0.0
    assert result.rerank_latency_s >= 0.0
    assert result.total_latency_s >= 0.0


def test_top5_chunk_ids_are_a_subset_of_candidates():
    pipeline = CustomPipeline(
        _chunks(), embedder=FakeEmbedder(dimension=8), reranker=FakeReranker(), generator=None,
        candidate_depth=4, top_k=2,
    )
    result = pipeline.run_question("q")
    assert set(result.top5_chunk_ids) <= set(result.candidate20_chunk_ids)
