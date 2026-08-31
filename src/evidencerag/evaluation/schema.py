"""Result, metadata, and summary shapes for M7 evaluation runs.

Mirrors the project's existing "one schema module per milestone"
convention (`chunking.schema`, `generation.schema`). Nothing here
computes anything -- see `retrieval_metrics.py`, `evidence_metrics.py`,
`answer_metrics.py`, and `harness.py` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvalRecord:
    """One system's scored answer to one question.

    `recall_at_5` / `recall_at_20` / `reciprocal_rank_at_5` are `None`
    exactly when the question has zero mappable chunk-level gold
    evidence across *every* answer reference (see `gold.py`) -- that
    is reported explicitly, never silently scored as 0.0.

    `reciprocal_rank_at_5` is named `_at_5` deliberately: it is always
    computed over the final top-5 ranking (post-reranking, for
    `hybrid_rerank`), never an unbounded ranking, so a later reader of
    this field can't mistake it for full-corpus MRR.

    `evidence_f1` is always defined (the official QASPER convention
    scores a correctly-empty prediction against empty gold evidence as
    1.0), unlike the chunk-level fields above.

    `candidate_chunk_ids_at_20` is the pre-selection candidate set
    Recall@20 was computed against -- for `hybrid_rerank` this is
    `hybrid`'s own top-20, the same set `hybrid`'s own record used, by
    design (see `systems.py`).

    `gold_chunk_references` is the exact per-answer mappable chunk-id
    sets `recall_at_5` / `recall_at_20` / `reciprocal_rank_at_5` were
    maximized over (empty tuple when the question was excluded) --
    kept here so a results.jsonl row is self-contained for failure
    analysis, without needing to re-run `gold.build_gold`.

    `answer` / `answer_f1` / `answer_type` / `generator_model` are all
    `None` in retrieval-only mode.
    """

    question_id: Optional[str]
    question_index: int
    paper_id: str
    split: str
    system: str

    retrieved_chunk_ids: tuple[str, ...]
    candidate_chunk_ids_at_20: tuple[str, ...]
    gold_chunk_references: tuple[tuple[str, ...], ...]

    recall_at_5: Optional[float]
    recall_at_20: Optional[float]
    reciprocal_rank_at_5: Optional[float]

    evidence_f1: float

    answer: Optional[str]
    answer_f1: Optional[float]
    answer_type: Optional[str]
    generator_model: Optional[str]


@dataclass(frozen=True)
class RunMetadata:
    """Effective configuration for one evaluation run.

    `retrieval_top_k` / `retrieval_candidate_depth` / `rrf_k` /
    `bm25_k1` / `bm25_b` / `embedding_model` are always the current
    `evidencerag.config.SETTINGS` values -- `scripts/evaluate_m7.py`
    does not expose flags to change them, so every run this metadata
    describes is the same agreed M7 experiment (top_k=5,
    candidate_depth=20) by construction, not merely by convention.
    """

    run_id: str
    timestamp: str
    git_commit: Optional[str]
    split: str
    mode: str  # "retrieval" | "end_to_end"
    systems: tuple[str, ...]
    retrieval_top_k: int
    retrieval_candidate_depth: int
    rrf_k: int
    bm25_k1: float
    bm25_b: float
    embedding_model: str
    reranker_model: str
    generator_model: Optional[str]
    random_seed: int
    max_questions: Optional[int]


@dataclass(frozen=True)
class SystemSummary:
    """Aggregated metrics for one system over one run.

    `n_excluded_no_gold_chunks` counts questions excluded from
    `recall_at_5` / `recall_at_20` / `mrr_at_5` because every answer
    reference had zero mappable chunk-level gold evidence -- those
    fields are averaged over `n_questions - n_excluded_no_gold_chunks`
    questions, never zero-filled. `evidence_f1` is averaged over all
    `n_questions` (the official empty/empty=1.0 convention already
    handles the "nothing to find" case).

    `answer_f1` / `answer_f1_by_type` are `None` in retrieval-only
    mode.
    """

    system: str
    n_questions: int
    n_excluded_no_gold_chunks: int
    recall_at_5: Optional[float]
    recall_at_20: Optional[float]
    mrr_at_5: Optional[float]
    evidence_f1: float
    answer_f1: Optional[float]
    answer_f1_by_type: Optional[dict[str, float]]


@dataclass(frozen=True)
class RunSummary:
    systems: tuple[SystemSummary, ...]
