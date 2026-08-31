"""M7 evaluation harness: turns M2 `Paper`s + M3 `Chunk`/
`EvidenceChunkMapping`s + the four `evidencerag.evaluation.systems`
retrievers (+ optionally an M6 `Generator`) into per-question,
per-system `EvalRecord`s and a `RunSummary`.

Reuses every M2-M6 interface unchanged:

  - `Retriever.retrieve()` (M4) for BM25/Dense/Hybrid, and
    `RerankingRetriever` (M5) for Hybrid+Reranker -- all four are
    called through the identical `retrieve(query, top_k)` signature,
    so this module never special-cases any one of them beyond the
    `candidate_sources` indirection already built into
    `evaluation.systems.EvaluationSystems`.
  - `generation.generate.generate_answer()` (M6) -- the exact function
    `GenerationPipeline.answer()` calls internally -- given the SAME
    top-5 `RetrievalResult` list already computed for Recall@5 /
    Evidence F1, rather than retrieving a second time.

Corpus identity is verified once up front via M4's own
`verify_corpus()` (on `bm25`/`dense`), and every `Paper`'s `split` is
checked against `config.split`, so a caller mistake (mismatched
chunks/systems, or a split mix-up) fails immediately and loudly, never
as a silent scoring anomaly -- per M7's correctness requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal, Mapping, Optional, Sequence

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.chunking.schema import Chunk
from evidencerag.evaluation import answer_metrics, evidence_metrics, retrieval_metrics
from evidencerag.evaluation.gold import QuestionGold, build_gold
from evidencerag.evaluation.schema import EvalRecord, RunSummary, SystemSummary
from evidencerag.evaluation.systems import EvaluationSystems
from evidencerag.generation.generate import generate_answer
from evidencerag.generation.generator import Generator
from evidencerag.ingestion.schema import Paper

Mode = Literal["retrieval", "end_to_end"]


@dataclass(frozen=True)
class EvaluationConfig:
    """`top_k` / `candidate_depth` are expected to be
    `SETTINGS.retrieval_top_k` / `SETTINGS.retrieval_candidate_depth`
    (see `scripts/evaluate_m7.py`) -- this dataclass does not enforce
    that itself, so tests can exercise other depths, but a "standard"
    M7 run always passes the Settings values through unchanged.

    `max_questions` truncates the run across the whole split (not per
    paper) -- intended for pilot/smoke runs only, never for a reported
    benchmark number.
    """

    mode: Mode
    split: str
    top_k: int
    candidate_depth: int
    max_questions: Optional[int] = None


def run_evaluation(
    papers: Sequence[Paper],
    chunks: Sequence[Chunk],
    evidence_mappings_by_paper: Mapping[str, Sequence[EvidenceChunkMapping]],
    systems: EvaluationSystems,
    config: EvaluationConfig,
    generator: Optional[Generator] = None,
) -> tuple[list[EvalRecord], RunSummary]:
    """Run all four systems over every question in `papers` (or the
    first `config.max_questions`, split-order preserved), returning
    one `EvalRecord` per question x system plus an aggregated
    `RunSummary`.

    `evidence_mappings_by_paper` must map each `paper.paper_id` in
    `papers` to exactly the `EvidenceChunkMapping`s
    `map_evidence_to_chunks` produced for that same paper's chunks --
    a paper with no entry is treated as "zero mappable evidence for
    every question in it", not an error, since a paper can legitimately
    have no chunk-resolvable evidence.

    Raises `ValueError` if `config.mode == "end_to_end"` and no
    `generator` is given, if any `paper.split != config.split`, if
    `evidence_mappings_by_paper` has a key not present in `papers`, or
    if `gold.build_gold` itself raises (e.g. a mapping belonging to a
    different paper, or referencing a question/answer index the paper
    doesn't have) -- all "fail clearly on a data problem" cases per
    M7's correctness requirement.
    """
    if config.mode == "end_to_end" and generator is None:
        raise ValueError("end_to_end mode requires a generator")

    for paper in papers:
        if paper.split != config.split:
            raise ValueError(f"paper {paper.paper_id!r} has split={paper.split!r}, expected {config.split!r}")

    known_paper_ids = {paper.paper_id for paper in papers}
    unknown_paper_ids = set(evidence_mappings_by_paper) - known_paper_ids
    if unknown_paper_ids:
        raise ValueError(
            f"evidence_mappings_by_paper references paper_id(s) not present in papers: "
            f"{sorted(unknown_paper_ids)}"
        )

    for name in ("bm25", "dense"):
        retriever = systems.retrievers.get(name)
        verify_corpus = getattr(retriever, "verify_corpus", None)
        if verify_corpus is not None:
            verify_corpus(chunks)

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}

    records: list[EvalRecord] = []
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

            for system_name, retriever in systems.retrievers.items():
                top_k_results = retriever.retrieve(question.question_text, top_k=config.top_k)
                candidate_results = systems.candidate_sources[system_name].retrieve(
                    question.question_text, top_k=config.candidate_depth
                )
                records.append(
                    _score_one(
                        system_name=system_name,
                        paper=paper,
                        gold=gold,
                        top_k_results=top_k_results,
                        candidate_results=candidate_results,
                        config=config,
                        chunk_by_id=chunk_by_id,
                        chunk_text_by_id=chunk_text_by_id,
                        generator=generator,
                    )
                )

    return records, _summarize(records, config.mode)


def _score_one(
    *,
    system_name: str,
    paper: Paper,
    gold: QuestionGold,
    top_k_results,
    candidate_results,
    config: EvaluationConfig,
    chunk_by_id,
    chunk_text_by_id,
    generator: Optional[Generator],
) -> EvalRecord:
    retrieved_chunk_ids = tuple(result.chunk_id for result in top_k_results)
    candidate_chunk_ids = tuple(result.chunk_id for result in candidate_results)

    if gold.chunk_references:
        recall_5 = retrieval_metrics.max_recall_at_k(top_k_results, gold.chunk_references, k=config.top_k)
        recall_20 = retrieval_metrics.max_recall_at_k(
            candidate_results, gold.chunk_references, k=config.candidate_depth
        )
        reciprocal_rank_5 = retrieval_metrics.max_reciprocal_rank(top_k_results, gold.chunk_references)
    else:
        recall_5 = recall_20 = reciprocal_rank_5 = None

    predicted_paragraphs = evidence_metrics.retrieved_paragraph_texts(retrieved_chunk_ids, chunk_by_id, paper)
    evidence_f1 = evidence_metrics.evidence_f1(predicted_paragraphs, gold.evidence_text_references)

    answer = answer_f1 = answer_type = generator_model = None
    if config.mode == "end_to_end":
        assert generator is not None  # enforced in run_evaluation
        result = generate_answer(gold.question_text, list(top_k_results), chunk_text_by_id, generator)
        answer = result.answer
        answer_f1, answer_type = answer_metrics.answer_f1_and_type(answer, gold.answer_references)
        generator_model = result.model_name

    return EvalRecord(
        question_id=gold.question_id,
        question_index=gold.question_index,
        paper_id=gold.paper_id,
        split=paper.split,
        system=system_name,
        retrieved_chunk_ids=retrieved_chunk_ids,
        candidate_chunk_ids_at_20=candidate_chunk_ids,
        gold_chunk_references=tuple(tuple(sorted(reference)) for reference in gold.chunk_references),
        recall_at_5=recall_5,
        recall_at_20=recall_20,
        reciprocal_rank_at_5=reciprocal_rank_5,
        evidence_f1=evidence_f1,
        answer=answer,
        answer_f1=answer_f1,
        answer_type=answer_type,
        generator_model=generator_model,
    )


def _summarize(records: Sequence[EvalRecord], mode: Mode) -> RunSummary:
    records_by_system: dict[str, list[EvalRecord]] = {}
    for record in records:
        records_by_system.setdefault(record.system, []).append(record)

    summaries = []
    for system_name, system_records in records_by_system.items():
        included = [record for record in system_records if record.recall_at_5 is not None]
        excluded = len(system_records) - len(included)

        answer_f1_mean: Optional[float] = None
        answer_f1_by_type: Optional[dict[str, float]] = None
        if mode == "end_to_end":
            scored = [record for record in system_records if record.answer_f1 is not None]
            answer_f1_mean = mean(record.answer_f1 for record in scored) if scored else None
            by_type: dict[str, list[float]] = {}
            for record in scored:
                by_type.setdefault(record.answer_type, []).append(record.answer_f1)
            answer_f1_by_type = {answer_type: mean(values) for answer_type, values in by_type.items()}

        summaries.append(
            SystemSummary(
                system=system_name,
                n_questions=len(system_records),
                n_excluded_no_gold_chunks=excluded,
                recall_at_5=mean(record.recall_at_5 for record in included) if included else None,
                recall_at_20=mean(record.recall_at_20 for record in included) if included else None,
                mrr_at_5=mean(record.reciprocal_rank_at_5 for record in included) if included else None,
                evidence_f1=mean(record.evidence_f1 for record in system_records),
                answer_f1=answer_f1_mean,
                answer_f1_by_type=answer_f1_by_type,
            )
        )

    return RunSummary(systems=tuple(summaries))
