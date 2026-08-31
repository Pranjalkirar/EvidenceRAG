"""Tests for evidencerag.evaluation.io."""

from __future__ import annotations

import json

from evidencerag.evaluation.io import load_results, save_run
from evidencerag.evaluation.schema import EvalRecord, RunMetadata, RunSummary, SystemSummary


def _sample_record(system: str) -> EvalRecord:
    return EvalRecord(
        question_id="Q1",
        question_index=0,
        paper_id="9999.00001",
        split="train",
        system=system,
        retrieved_chunk_ids=("c1", "c2"),
        candidate_chunk_ids_at_20=("c1", "c2", "c3"),
        gold_chunk_references=(("c1",), ("c2", "c3")),
        recall_at_5=1.0,
        recall_at_20=1.0,
        reciprocal_rank_at_5=1.0,
        evidence_f1=0.8,
        answer="An answer.",
        answer_f1=0.75,
        answer_type="abstractive",
        generator_model="fake-generator",
    )


def _sample_metadata() -> RunMetadata:
    return RunMetadata(
        run_id="20260831T000000Z",
        timestamp="2026-08-31T00:00:00Z",
        git_commit="abc123",
        split="validation",
        mode="end_to_end",
        systems=("bm25", "dense", "hybrid", "hybrid_rerank"),
        retrieval_top_k=5,
        retrieval_candidate_depth=20,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        generator_model="Qwen/Qwen3-4B-Instruct-2507",
        random_seed=42,
        max_questions=None,
    )


def _sample_summary() -> RunSummary:
    return RunSummary(
        systems=(
            SystemSummary(
                system="bm25", n_questions=10, n_excluded_no_gold_chunks=1,
                recall_at_5=0.5, recall_at_20=0.7, mrr_at_5=0.4, evidence_f1=0.6,
                answer_f1=0.55, answer_f1_by_type={"abstractive": 0.55},
            ),
        )
    )


def test_save_run_writes_all_three_files(tmp_path):
    output_dir = tmp_path / "run1"
    records = [_sample_record("bm25"), _sample_record("hybrid_rerank")]
    n_written = save_run(output_dir, _sample_metadata(), records, _sample_summary())

    assert n_written == 2
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "results.jsonl").exists()


def test_save_run_metadata_json_round_trips_fields(tmp_path):
    output_dir = tmp_path / "run1"
    save_run(output_dir, _sample_metadata(), [], _sample_summary())
    data = json.loads((output_dir / "metadata.json").read_text())
    assert data["run_id"] == "20260831T000000Z"
    assert data["systems"] == ["bm25", "dense", "hybrid", "hybrid_rerank"]
    assert data["retrieval_top_k"] == 5
    assert data["retrieval_candidate_depth"] == 20


def test_load_results_round_trips_records(tmp_path):
    output_dir = tmp_path / "run1"
    original = [_sample_record("bm25"), _sample_record("hybrid_rerank")]
    save_run(output_dir, _sample_metadata(), original, _sample_summary())

    loaded = list(load_results(output_dir / "results.jsonl"))

    assert loaded == original
    # Tuple fields must come back as tuples, not lists, after the JSON round-trip.
    assert isinstance(loaded[0].retrieved_chunk_ids, tuple)
    assert isinstance(loaded[0].gold_chunk_references, tuple)
    assert isinstance(loaded[0].gold_chunk_references[0], tuple)
