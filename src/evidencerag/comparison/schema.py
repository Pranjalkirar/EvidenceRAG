"""Result, metadata, and summary shapes for M8 (Custom vs LangChain)
comparison runs.

Mirrors `evidencerag.evaluation.schema`'s "one schema module per
milestone" convention and, where a field means the same thing, its
exact naming (`recall_at_5`, `evidence_f1`, ...) -- a `ComparisonRecord`
is best understood as an `evidencerag.evaluation.schema.EvalRecord`
plus an `implementation` tag and per-stage latency, not a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StageResult:
    """One pipeline's (custom or LangChain) raw output for one
    question, before any gold-comparison scoring happens -- produced by
    `evidencerag.comparison.custom_pipeline.CustomPipeline.run_question`
    and (after conversion from `LangChainStageResult`) by
    `evidencerag.langchain_impl.pipeline.LangChainPipeline.run_question`.

    `candidate20_chunk_ids` is the pre-rerank Hybrid (BM25+Dense RRF)
    candidate set, used for Recall@20 -- the same
    "hybrid_rerank's own Recall@20 comes from the pre-rerank hybrid
    candidates" convention `evidencerag.evaluation.systems` documents
    for M7.

    `rerank_latency_s` is 0.0 (not `None`) when reranking is a no-op
    stage of the pipeline being timed; `generation_latency_s`/`answer`
    are `None` together, exactly when the run is retrieval-only.
    """

    top5_chunk_ids: tuple[str, ...]
    candidate20_chunk_ids: tuple[str, ...]
    retrieval_latency_s: float
    rerank_latency_s: float
    generation_latency_s: Optional[float]
    total_latency_s: float
    answer: Optional[str]


@dataclass(frozen=True)
class ComparisonRecord:
    """One implementation's (custom or LangChain) scored answer to one
    question -- the M8 analogue of
    `evidencerag.evaluation.schema.EvalRecord`, with `implementation`
    added and per-stage latency in place of `system` (M8 always runs
    the single Hybrid+Reranker chain on both sides, so there is no
    per-system dimension to record here -- see
    `evidencerag.langchain_impl.pipeline`'s module docstring for that
    scope decision).

    `recall_at_5` / `recall_at_20` / `reciprocal_rank_at_5` are `None`
    under the exact same "zero mappable gold evidence" convention as
    `EvalRecord` (see `evidencerag.evaluation.schema.EvalRecord`) --
    reported as an explicit exclusion, never a silent zero.
    """

    question_id: Optional[str]
    question_index: int
    paper_id: str
    split: str
    implementation: str  # "custom" | "langchain"

    question_text: str
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

    retrieval_latency_s: float
    rerank_latency_s: float
    generation_latency_s: Optional[float]
    total_latency_s: float


@dataclass(frozen=True)
class ComparisonRunMetadata:
    """Effective configuration for one M8 comparison run -- mirrors
    `evidencerag.evaluation.schema.RunMetadata`, plus the model ids
    (embedding/reranker/generator) recorded once here rather than
    per-implementation, since M8's whole point is that both
    implementations use the identical models."""

    run_id: str
    timestamp: str
    git_commit: Optional[str]
    split: str
    mode: str  # "retrieval" | "end_to_end"
    implementations: tuple[str, ...]
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
    langchain_available: bool


@dataclass(frozen=True)
class ImplementationSummary:
    """Aggregated metrics + latency for one implementation over one
    run -- mirrors `evidencerag.evaluation.schema.SystemSummary`, plus
    mean latency fields."""

    implementation: str
    n_questions: int
    n_excluded_no_gold_chunks: int
    recall_at_5: Optional[float]
    recall_at_20: Optional[float]
    mrr_at_5: Optional[float]
    evidence_f1: float
    answer_f1: Optional[float]
    mean_retrieval_latency_s: float
    mean_rerank_latency_s: float
    mean_generation_latency_s: Optional[float]
    mean_total_latency_s: float


@dataclass(frozen=True)
class EngineeringComplexity:
    """A snapshot of implementation-complexity indicators for one side
    of the comparison, computed from the repository's actual files
    (see `evidencerag.comparison.complexity`) -- never hand-estimated,
    so these numbers change automatically if the code they describe
    changes.
    """

    implementation: str
    relevant_loc: int
    file_count: int
    custom_component_count: int
    dependency_additions: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonRunSummary:
    implementations: tuple[ImplementationSummary, ...]
    complexity: tuple[EngineeringComplexity, ...]
