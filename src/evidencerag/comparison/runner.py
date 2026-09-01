"""M8 evaluation harness: turns M2 `Paper`s + M3 `Chunk`/
`EvidenceChunkMapping`s + a `CustomPipeline` (+ optionally a
`LangChainPipeline`) into per-question, per-implementation
`ComparisonRecord`s and a `ComparisonRunSummary` -- the M8 analogue of
`evidencerag.evaluation.harness.run_evaluation`.

Reuses the UNCHANGED M7 scoring functions directly:

  - `evidencerag.evaluation.gold.build_gold` for gold references,
  - `evidencerag.evaluation.retrieval_metrics.max_recall_at_k` /
    `max_reciprocal_rank` for Recall@5/Recall@20/MRR@5 (a
    `CustomPipeline`/`LangChainPipeline` result's `top5_chunk_ids` /
    `candidate20_chunk_ids` tuples are wrapped into minimal
    `RetrievalResult` objects -- rank = 1-based position in the tuple,
    score unused -- purely so these functions' existing signature can
    be reused unchanged, not reimplemented for M8),
  - `evidencerag.evaluation.evidence_metrics.retrieved_paragraph_texts`
    / `evidence_f1` for Evidence F1,
  - `evidencerag.evaluation.answer_metrics.answer_f1_and_type` for
    Answer F1 in end-to-end mode.

Unlike `harness.run_evaluation`, this module does not accept a
pre-built `EvaluationSystems` -- M8 always runs exactly one chain
(Hybrid+Reranker) per implementation (see
`evidencerag.langchain_impl.pipeline`'s module docstring for that
scope decision), so each implementation is one `CustomPipeline`-shaped
object (`.run_question(text) -> StageResult`), not a `dict` of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal, Mapping, Optional, Protocol, Sequence

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.chunking.schema import Chunk
from evidencerag.comparison.complexity import measure_complexity
from evidencerag.comparison.schema import (
    ComparisonRecord,
    ComparisonRunSummary,
    ImplementationSummary,
    StageResult,
)
from evidencerag.evaluation import answer_metrics, evidence_metrics
from evidencerag.evaluation.gold import QuestionGold, build_gold
from evidencerag.evaluation.retrieval_metrics import max_recall_at_k, max_reciprocal_rank
from evidencerag.ingestion.schema import Paper
from evidencerag.retrieval.schema import RetrievalResult

Mode = Literal["retrieval", "end_to_end"]


class QuestionPipeline(Protocol):
    """Structural interface both `CustomPipeline` and (via a thin
    adapter) `LangChainPipeline` satisfy -- see
    `evidencerag.langchain_impl.pipeline.LangChainPipeline` and
    `to_stage_result` below."""

    def run_question(self, question_text: str) -> StageResult: ...


@dataclass(frozen=True)
class ComparisonConfig:
    """Mirrors `evidencerag.evaluation.harness.EvaluationConfig`.
    `max_questions` truncates across the whole split, pilot/smoke runs
    only -- same convention as M7."""

    mode: Mode
    split: str
    top_k: int
    candidate_depth: int
    max_questions: Optional[int] = None


def _as_retrieval_results(chunk_ids: Sequence[str], retriever_name: str) -> list[RetrievalResult]:
    """Wrap a plain, already-ranked `chunk_id` tuple (as produced by
    `StageResult.top5_chunk_ids` / `candidate20_chunk_ids`) into
    `RetrievalResult`s, purely so `retrieval_metrics.max_recall_at_k` /
    `max_reciprocal_rank` can be reused unchanged. `score` is not
    meaningful here (those functions never read it) -- only
    `chunk_id`/`rank` are.
    """
    return [
        RetrievalResult(chunk_id=chunk_id, score=0.0, rank=rank, retriever=retriever_name)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def _score_one(
    *,
    implementation: str,
    paper: Paper,
    gold: QuestionGold,
    stage: StageResult,
    config: ComparisonConfig,
    chunk_by_id: Mapping[str, Chunk],
) -> ComparisonRecord:
    top5_results = _as_retrieval_results(stage.top5_chunk_ids, implementation)
    candidate_results = _as_retrieval_results(stage.candidate20_chunk_ids, implementation)

    if gold.chunk_references:
        recall_5 = max_recall_at_k(top5_results, gold.chunk_references, k=config.top_k)
        recall_20 = max_recall_at_k(candidate_results, gold.chunk_references, k=config.candidate_depth)
        reciprocal_rank_5 = max_reciprocal_rank(top5_results, gold.chunk_references)
    else:
        recall_5 = recall_20 = reciprocal_rank_5 = None

    predicted_paragraphs = evidence_metrics.retrieved_paragraph_texts(stage.top5_chunk_ids, chunk_by_id, paper)
    evidence_f1_score = evidence_metrics.evidence_f1(predicted_paragraphs, gold.evidence_text_references)

    answer_f1 = answer_type = None
    if config.mode == "end_to_end" and stage.answer is not None:
        answer_f1, answer_type = answer_metrics.answer_f1_and_type(stage.answer, gold.answer_references)

    return ComparisonRecord(
        question_id=gold.question_id,
        question_index=gold.question_index,
        paper_id=gold.paper_id,
        split=paper.split,
        implementation=implementation,
        question_text=gold.question_text,
        retrieved_chunk_ids=stage.top5_chunk_ids,
        candidate_chunk_ids_at_20=stage.candidate20_chunk_ids,
        gold_chunk_references=tuple(tuple(sorted(reference)) for reference in gold.chunk_references),
        recall_at_5=recall_5,
        recall_at_20=recall_20,
        reciprocal_rank_at_5=reciprocal_rank_5,
        evidence_f1=evidence_f1_score,
        answer=stage.answer,
        answer_f1=answer_f1,
        answer_type=answer_type,
        retrieval_latency_s=stage.retrieval_latency_s,
        rerank_latency_s=stage.rerank_latency_s,
        generation_latency_s=stage.generation_latency_s,
        total_latency_s=stage.total_latency_s,
    )


def run_comparison(
    papers: Sequence[Paper],
    chunks: Sequence[Chunk],
    evidence_mappings_by_paper: Mapping[str, Sequence[EvidenceChunkMapping]],
    pipelines: Mapping[str, QuestionPipeline],
    config: ComparisonConfig,
) -> tuple[list[ComparisonRecord], ComparisonRunSummary]:
    """Run every implementation in `pipelines` (keyed by implementation
    name, e.g. `{"custom": custom_pipeline}` or
    `{"custom": ..., "langchain": ...}`) over every question in
    `papers` (or the first `config.max_questions`, split-order
    preserved), returning one `ComparisonRecord` per question x
    implementation plus a `ComparisonRunSummary`.

    Same `evidence_mappings_by_paper` / split-mismatch / unknown-paper
    validation contract as
    `evidencerag.evaluation.harness.run_evaluation` -- see that
    function's docstring; deliberately kept identical so a caller
    already familiar with the M7 harness needs to learn nothing new
    here.
    """
    for paper in papers:
        if paper.split != config.split:
            raise ValueError(f"paper {paper.paper_id!r} has split={paper.split!r}, expected {config.split!r}")

    known_paper_ids = {paper.paper_id for paper in papers}
    unknown_paper_ids = set(evidence_mappings_by_paper) - known_paper_ids
    if unknown_paper_ids:
        raise ValueError(
            f"evidence_mappings_by_paper references paper_id(s) not present in papers: {sorted(unknown_paper_ids)}"
        )

    if not pipelines:
        raise ValueError("run_comparison requires at least one implementation in `pipelines`")

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    records: list[ComparisonRecord] = []
    n_questions_seen = 0

    for paper in papers:
        if config.max_questions is not None and n_questions_seen >= config.max_questions:
            break

        gold_by_question = build_gold(paper, evidence_mappings_by_paper.get(paper.paper_id, ()))

        for q_idx, question in enumerate(paper.questions):
            if config.max_questions is not None and n_questions_seen >= config.max_questions:
                break
            n_questions_seen += 1

            gold = gold_by_question[q_idx]

            for implementation, pipeline in pipelines.items():
                stage = pipeline.run_question(question.question_text)
                records.append(
                    _score_one(
                        implementation=implementation,
                        paper=paper,
                        gold=gold,
                        stage=stage,
                        config=config,
                        chunk_by_id=chunk_by_id,
                    )
                )

    return records, _summarize(records, config.mode)


def _summarize(records: Sequence[ComparisonRecord], mode: Mode) -> ComparisonRunSummary:
    records_by_impl: dict[str, list[ComparisonRecord]] = {}
    for record in records:
        records_by_impl.setdefault(record.implementation, []).append(record)

    summaries = []
    for implementation, impl_records in records_by_impl.items():
        included = [record for record in impl_records if record.recall_at_5 is not None]
        excluded = len(impl_records) - len(included)

        answer_f1_mean: Optional[float] = None
        if mode == "end_to_end":
            scored = [record.answer_f1 for record in impl_records if record.answer_f1 is not None]
            answer_f1_mean = mean(scored) if scored else None

        generation_latencies = [
            record.generation_latency_s for record in impl_records if record.generation_latency_s is not None
        ]

        summaries.append(
            ImplementationSummary(
                implementation=implementation,
                n_questions=len(impl_records),
                n_excluded_no_gold_chunks=excluded,
                recall_at_5=mean(record.recall_at_5 for record in included) if included else None,
                recall_at_20=mean(record.recall_at_20 for record in included) if included else None,
                mrr_at_5=mean(record.reciprocal_rank_at_5 for record in included) if included else None,
                evidence_f1=mean(record.evidence_f1 for record in impl_records),
                answer_f1=answer_f1_mean,
                mean_retrieval_latency_s=mean(record.retrieval_latency_s for record in impl_records),
                mean_rerank_latency_s=mean(record.rerank_latency_s for record in impl_records),
                mean_generation_latency_s=mean(generation_latencies) if generation_latencies else None,
                mean_total_latency_s=mean(record.total_latency_s for record in impl_records),
            )
        )

    return ComparisonRunSummary(implementations=tuple(summaries), complexity=measure_complexity())
