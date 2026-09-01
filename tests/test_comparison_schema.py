"""Tests for evidencerag.comparison.schema -- construction and
field-presence only; these are plain frozen dataclasses with no
behavior of their own."""

from __future__ import annotations

import pytest

from evidencerag.comparison.schema import (
    ComparisonRecord,
    ComparisonRunMetadata,
    ComparisonRunSummary,
    EngineeringComplexity,
    ImplementationSummary,
    StageResult,
)


def test_stage_result_is_frozen():
    result = StageResult(
        top5_chunk_ids=("c1",),
        candidate20_chunk_ids=("c1", "c2"),
        retrieval_latency_s=0.1,
        rerank_latency_s=0.05,
        generation_latency_s=None,
        total_latency_s=0.15,
        answer=None,
    )
    with pytest.raises(Exception):
        result.answer = "mutated"  # type: ignore[misc]


def test_comparison_record_round_trips_fields():
    record = ComparisonRecord(
        question_id="Q1",
        question_index=0,
        paper_id="p1",
        split="validation",
        implementation="custom",
        question_text="What?",
        retrieved_chunk_ids=("c1", "c2"),
        candidate_chunk_ids_at_20=("c1", "c2", "c3"),
        gold_chunk_references=(("c1",),),
        recall_at_5=1.0,
        recall_at_20=1.0,
        reciprocal_rank_at_5=1.0,
        evidence_f1=0.8,
        answer="An answer.",
        answer_f1=0.5,
        answer_type="abstractive",
        retrieval_latency_s=0.01,
        rerank_latency_s=0.02,
        generation_latency_s=0.03,
        total_latency_s=0.06,
    )
    assert record.implementation == "custom"
    assert record.gold_chunk_references == (("c1",),)


def test_implementation_summary_allows_none_answer_f1():
    summary = ImplementationSummary(
        implementation="langchain",
        n_questions=2,
        n_excluded_no_gold_chunks=0,
        recall_at_5=0.5,
        recall_at_20=1.0,
        mrr_at_5=0.5,
        evidence_f1=0.6,
        answer_f1=None,
        mean_retrieval_latency_s=0.1,
        mean_rerank_latency_s=0.05,
        mean_generation_latency_s=None,
        mean_total_latency_s=0.15,
    )
    assert summary.answer_f1 is None


def test_comparison_run_summary_holds_both_kinds_of_snapshot():
    custom = EngineeringComplexity(
        implementation="custom", relevant_loc=100, file_count=5, custom_component_count=2,
        dependency_additions=(), notes=(),
    )
    langchain = EngineeringComplexity(
        implementation="langchain", relevant_loc=50, file_count=8, custom_component_count=4,
        dependency_additions=("langchain-core",), notes=(),
    )
    impl_summary = ImplementationSummary(
        implementation="custom", n_questions=1, n_excluded_no_gold_chunks=0,
        recall_at_5=1.0, recall_at_20=1.0, mrr_at_5=1.0, evidence_f1=1.0, answer_f1=None,
        mean_retrieval_latency_s=0.0, mean_rerank_latency_s=0.0, mean_generation_latency_s=None,
        mean_total_latency_s=0.0,
    )
    run_summary = ComparisonRunSummary(implementations=(impl_summary,), complexity=(custom, langchain))
    assert len(run_summary.complexity) == 2


def test_run_metadata_records_langchain_availability():
    metadata = ComparisonRunMetadata(
        run_id="r1", timestamp="2026-01-01T00:00:00Z", git_commit=None, split="validation",
        mode="retrieval", implementations=("custom",), retrieval_top_k=5, retrieval_candidate_depth=20,
        rrf_k=60, bm25_k1=1.5, bm25_b=0.75, embedding_model="fake", reranker_model="fake",
        generator_model=None, random_seed=42, max_questions=2, langchain_available=False,
    )
    assert metadata.langchain_available is False
