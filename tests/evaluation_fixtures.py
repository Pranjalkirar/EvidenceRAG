"""Shared fixtures for M7 evaluation tests.

Reuses the existing per-milestone fixture modules directly rather than
duplicating them:

  - `tests.chunking_fixtures` for `Paper`/`Question`/`Answer`/
    `EvidenceSpan` builders (`make_paper`, `make_question`,
    `make_resolved_evidence`, `make_unresolved_evidence`,
    `make_float_selected_evidence`) and `make_answer` (abstractive).
  - `tests.retrieval_fixtures` for `make_chunk` and `FakeEmbedder`.
  - `tests.reranking_fixtures` for `FakeReranker`/`ConstantReranker`.
  - `tests.generation_fixtures` for `FakeGenerator`/`FailingGenerator`.

Only what's genuinely new for M7 lives here: answer builders for the
extractive/boolean/unanswerable cases `make_answer` doesn't cover, and
`QueryKeyedFakeRetriever`, since `generation_fixtures.FakeRetriever`
returns one fixed ranking regardless of query -- harness tests need
distinct rankings for distinct questions.
"""

from __future__ import annotations

from typing import Sequence

from evidencerag.ingestion.schema import Answer, EvidenceSpan
from evidencerag.retrieval.schema import RetrievalResult


def make_extractive_answer(
    evidence: tuple[EvidenceSpan, ...], extractive_spans: tuple[str, ...]
) -> Answer:
    return Answer(
        annotation_id="ann1",
        worker_id="worker1",
        unanswerable=False,
        yes_no=None,
        free_form_answer="",
        extractive_spans=extractive_spans,
        evidence=evidence,
    )


def make_boolean_answer(evidence: tuple[EvidenceSpan, ...], yes_no: bool) -> Answer:
    return Answer(
        annotation_id="ann1",
        worker_id="worker1",
        unanswerable=False,
        yes_no=yes_no,
        free_form_answer="",
        extractive_spans=(),
        evidence=evidence,
    )


def make_unanswerable_answer(evidence: tuple[EvidenceSpan, ...] = ()) -> Answer:
    return Answer(
        annotation_id="ann1",
        worker_id="worker1",
        unanswerable=True,
        yes_no=None,
        free_form_answer="",
        extractive_spans=(),
        evidence=evidence,
    )


def make_result(chunk_id: str, rank: int, score: float = 1.0, retriever: str = "fake") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, rank=rank, retriever=retriever)


class QueryKeyedFakeRetriever:
    """Deterministic stand-in for a `Retriever` that returns a
    caller-supplied ranking per distinct query text (unlike
    `tests.generation_fixtures.FakeRetriever`, which returns one fixed
    ranking regardless of query) -- needed for harness tests that
    exercise several questions with distinct expected rankings.

    Has no `verify_corpus` method, matching how `harness.run_evaluation`
    treats corpus verification as optional (`getattr(..., "verify_corpus", None)`)
    so plain fakes don't need to implement it.
    """

    def __init__(self, results_by_query: dict[str, Sequence[RetrievalResult]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return list(self._results_by_query[query])[:top_k]
