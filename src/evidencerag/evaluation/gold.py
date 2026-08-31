"""Builds per-question gold references for M7 evaluation directly from
M2 `Paper` data and M3 `EvidenceChunkMapping` -- no evidence resolution
is re-derived or guessed here; M2/M3 already did that work.

Two distinct gold representations are built, because they serve two
metrics with different official semantics:

  * `chunk_references` -- one entry per answer annotation that has at
    least one chunk-mappable evidence span, used for
    Recall@5/Recall@20/MRR@5 against M3 `chunk_id`s. Only resolved,
    non-float-selected evidence can contribute (unresolved evidence
    cannot be mapped to a chunk without guessing, per
    `evidence_map.py`). An answer with zero mappable evidence
    contributes no entry at all -- it is not an empty set to be
    unioned away, it is simply absent, per the "alternative
    references, never merged" rule below.

  * `evidence_text_references` -- one entry per answer annotation
    (always present, possibly an empty `frozenset`), used for
    QASPER-style Evidence F1. This mirrors the official evaluator
    directly: raw evidence text strings, excluding FLOAT SELECTED
    entries, independent of whether M2 could resolve a paragraph
    location for them -- the official metric compares text, never
    positions, so "unresolved" is not a reason to drop it here.

Each answer annotation is kept as its own separate reference in both
cases. QASPER's multiple annotators are alternative valid answers to
the same question -- this module never merges them into one combined
gold set (that would penalize a system for retrieving one annotator's
valid evidence instead of every annotator's alternative evidence).
Combining across references, where the official semantics call for it
(max-across-references for Evidence F1, Answer F1, and Recall/MRR
here), is the caller's job -- see `retrieval_metrics.max_recall_at_k`,
`retrieval_metrics.max_reciprocal_rank`, and `evidence_metrics.evidence_f1`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.ingestion.schema import Answer, Paper


@dataclass(frozen=True)
class QuestionGold:
    """Gold references for one question, keyed by the caller via
    `question_index` (see `build_gold`)."""

    paper_id: str
    question_id: Optional[str]
    question_index: int
    question_text: str
    chunk_references: tuple[frozenset[str], ...]
    evidence_text_references: tuple[frozenset[str], ...]
    answer_references: tuple[tuple[str, str], ...]


def reference_answer_and_type(answer: Answer) -> tuple[str, str]:
    """The (answer_text, answer_type) pair for one QASPER answer
    annotation, following the official evaluator's precedence exactly:
    unanswerable, then extractive_spans, then free_form_answer, then
    yes_no. Exactly one of these is meaningful per QASPER's documented
    convention (see `evidencerag.ingestion.schema.Answer`) -- an
    annotation matching none of them is a data error, not a case to
    silently default.
    """
    if answer.unanswerable:
        return "Unanswerable", "none"
    if answer.extractive_spans:
        return ", ".join(answer.extractive_spans), "extractive"
    if answer.free_form_answer:
        return answer.free_form_answer, "abstractive"
    if answer.yes_no is True:
        return "Yes", "boolean"
    if answer.yes_no is False:
        return "No", "boolean"
    raise ValueError(
        f"Answer {answer.annotation_id!r} has no unanswerable/extractive_spans/"
        "free_form_answer/yes_no content -- cannot construct a reference answer."
    )


def build_gold(paper: Paper, mappings: Sequence[EvidenceChunkMapping]) -> dict[int, QuestionGold]:
    """Build one `QuestionGold` per question in `paper`, keyed by
    `question_index` (stable within one paper, unlike `question_id`
    which QASPER allows to be `None`).

    `mappings` must be exactly the `EvidenceChunkMapping`s produced by
    `evidencerag.chunking.evidence_map.map_evidence_to_chunks(paper, chunks)`
    for this same `paper` -- every mapping's `paper_id`/`split` is
    checked against `paper` and a mismatch raises immediately, rather
    than silently building gold references against the wrong paper.
    """
    mappings_by_qa: dict[tuple[int, int], list[EvidenceChunkMapping]] = defaultdict(list)
    for mapping in mappings:
        if mapping.paper_id != paper.paper_id or mapping.split != paper.split:
            raise ValueError(
                f"EvidenceChunkMapping for paper={mapping.paper_id!r}/split={mapping.split!r} "
                f"does not match paper={paper.paper_id!r}/split={paper.split!r}"
            )
        mappings_by_qa[(mapping.question_index, mapping.answer_index)].append(mapping)

    valid_qa_keys = {
        (q_idx, a_idx)
        for q_idx, question in enumerate(paper.questions)
        for a_idx in range(len(question.answers))
    }
    unknown_qa_keys = set(mappings_by_qa) - valid_qa_keys
    if unknown_qa_keys:
        raise ValueError(
            f"EvidenceChunkMapping for paper {paper.paper_id!r} references "
            f"(question_index, answer_index) pairs not present in this paper: {sorted(unknown_qa_keys)}"
        )

    gold: dict[int, QuestionGold] = {}
    for q_idx, question in enumerate(paper.questions):
        chunk_references: list[frozenset[str]] = []
        evidence_text_references: list[frozenset[str]] = []
        answer_references: list[tuple[str, str]] = []

        for a_idx, answer in enumerate(question.answers):
            chunk_ids = {
                chunk_id
                for mapping in mappings_by_qa.get((q_idx, a_idx), ())
                for chunk_id in mapping.chunk_ids
            }
            if chunk_ids:
                chunk_references.append(frozenset(chunk_ids))

            evidence_text_references.append(
                frozenset(evidence.text for evidence in answer.evidence if not evidence.is_float_selected)
            )
            answer_references.append(reference_answer_and_type(answer))

        gold[q_idx] = QuestionGold(
            paper_id=paper.paper_id,
            question_id=question.question_id,
            question_index=q_idx,
            question_text=question.question_text,
            chunk_references=tuple(chunk_references),
            evidence_text_references=tuple(evidence_text_references),
            answer_references=tuple(answer_references),
        )

    return gold
