"""Tests for evidencerag.comparison.io -- round-trip serialization only
(mirrors tests/test_evaluation_io.py's approach for M7)."""

from __future__ import annotations

from pathlib import Path

from evidencerag.comparison.io import load_results, save_run
from evidencerag.comparison.schema import (
    ComparisonRecord,
    ComparisonRunMetadata,
    ComparisonRunSummary,
    EngineeringComplexity,
    ImplementationSummary,
)


def _make_record(implementation: str) -> ComparisonRecord:
    return ComparisonRecord(
        question_id="Q1",
        question_index=0,
        paper_id="p1",
        split="validation",
        implementation=implementation,
        question_text="What is the method?",
        retrieved_chunk_ids=("c1", "c2"),
        candidate_chunk_ids_at_20=("c1", "c2", "c3"),
        gold_chunk_references=(("c1",),),
        recall_at_5=1.0,
        recall_at_20=1.0,
        reciprocal_rank_at_5=1.0,
        evidence_f1=0.75,
        answer="An answer.",
        answer_f1=0.5,
        answer_type="abstractive",
        retrieval_latency_s=0.01,
        rerank_latency_s=0.02,
        generation_latency_s=0.03,
        total_latency_s=0.06,
    )


def _make_metadata() -> ComparisonRunMetadata:
    return ComparisonRunMetadata(
        run_id="pilot-1",
        timestamp="2026-01-01T00:00:00Z",
        git_commit="abc123",
        split="validation",
        mode="end_to_end",
        implementations=("custom", "langchain"),
        retrieval_top_k=5,
        retrieval_candidate_depth=20,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        embedding_model="fake-embedder",
        reranker_model="fake-reranker",
        generator_model="fake-generator",
        random_seed=42,
        max_questions=2,
        langchain_available=True,
    )


def _make_summary() -> ComparisonRunSummary:
    impl_summary = ImplementationSummary(
        implementation="custom", n_questions=1, n_excluded_no_gold_chunks=0,
        recall_at_5=1.0, recall_at_20=1.0, mrr_at_5=1.0, evidence_f1=0.75, answer_f1=0.5,
        mean_retrieval_latency_s=0.01, mean_rerank_latency_s=0.02, mean_generation_latency_s=0.03,
        mean_total_latency_s=0.06,
    )
    complexity = EngineeringComplexity(
        implementation="custom", relevant_loc=10, file_count=2, custom_component_count=1,
        dependency_additions=(), notes=("a note",),
    )
    return ComparisonRunSummary(implementations=(impl_summary,), complexity=(complexity,))


def test_save_run_writes_three_files(tmp_path: Path):
    records = [_make_record("custom"), _make_record("langchain")]
    n_written = save_run(tmp_path / "run1", _make_metadata(), records, _make_summary())

    assert n_written == 2
    assert (tmp_path / "run1" / "metadata.json").exists()
    assert (tmp_path / "run1" / "summary.json").exists()
    assert (tmp_path / "run1" / "results.jsonl").exists()


def test_load_results_round_trips_records(tmp_path: Path):
    original = [_make_record("custom"), _make_record("langchain")]
    save_run(tmp_path / "run1", _make_metadata(), original, _make_summary())

    loaded = list(load_results(tmp_path / "run1" / "results.jsonl"))

    assert loaded == original


def test_gold_chunk_references_round_trip_as_tuples(tmp_path: Path):
    save_run(tmp_path / "run1", _make_metadata(), [_make_record("custom")], _make_summary())
    [loaded] = list(load_results(tmp_path / "run1" / "results.jsonl"))
    assert loaded.gold_chunk_references == (("c1",),)
    assert isinstance(loaded.gold_chunk_references, tuple)
    assert isinstance(loaded.gold_chunk_references[0], tuple)


def test_save_run_creates_output_dir(tmp_path: Path):
    output_dir = tmp_path / "nested" / "does" / "not" / "exist" / "yet"
    save_run(output_dir, _make_metadata(), [], _make_summary())
    assert output_dir.exists()
