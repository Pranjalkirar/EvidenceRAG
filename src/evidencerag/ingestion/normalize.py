"""Normalization: raw QASPER rows -> our internal representation
(schema.py's dataclasses).

AMBIGUITIES DISCOVERED AND HOW WE HANDLE THEM
-----------------------------------------------

1. `highlighted_evidence` vs `evidence` pairing.
   QASPER's documentation says `highlighted_evidence` sentences "map...
   to the paragraph level" of `evidence`, but never states the two
   lists are positionally parallel (index i of one corresponding to
   index i of the other), and in practice their lengths can differ.
   We therefore do NOT attempt to zip them together or attach specific
   highlighted sentences to specific `EvidenceSpan`s. We preserve
   `evidence` as individual `EvidenceSpan`s (never concatenated) and
   `highlighted_evidence` as its own flat tuple on `Answer`. A future
   milestone that wants sentence-level highlighting can re-derive the
   mapping deliberately (e.g. by checking sentence-in-paragraph
   containment) rather than relying on an unverified positional
   assumption made here.

2. `figures_and_tables` is not part of the originally documented
   QASPER schema (the paper/README only describe `id`, `title`,
   `abstract`, `full_text`, `qas`). It exists in the dataset as
   currently hosted on the Hub. We preserve it (see schema.py) but do
   not treat its absence as a validation failure — some source rows
   (or hand-built fixtures) may not include it.

3. Evidence-to-paragraph provenance is resolved by exact text match
   (after whitespace normalization) against the paper's own
   paragraphs, because QASPER's evidence strings are documented to be
   verbatim, entire paragraphs. This is expected to succeed in the
   large majority of cases, but is NOT guaranteed (e.g. if upstream
   text extraction introduced any non-whitespace differences between
   the paragraph copy stored in `full_text` and the copy stored in
   `evidence`, which does happen occasionally in this dataset). When a
   match cannot be found, we record `resolved=False` rather than
   guessing — see `EvidenceSpan.resolved`.

4. "FLOAT SELECTED" (figure/table) evidence is matched against
   `figures_and_tables` captions on a best-effort containment basis
   (the evidence string is typically the caption prefixed with
   "FLOAT SELECTED: "). This is heuristic, not exact-match, and is
   flagged as such via `EvidenceSpan.resolved`.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from evidencerag.ingestion.schema import (
    Answer,
    EvidenceSpan,
    FigureOrTable,
    Paper,
    Question,
    Section,
)

FLOAT_SELECTED_PREFIX = "FLOAT SELECTED"

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_sections(full_text: dict[str, Any]) -> tuple[Section, ...]:
    section_names = full_text["section_name"]
    paragraphs = full_text["paragraphs"]
    return tuple(
        Section(section_index=i, title=title or "", paragraphs=tuple(paras))
        for i, (title, paras) in enumerate(zip(section_names, paragraphs))
    )


def normalize_figures_and_tables(
    figures_and_tables: Optional[dict[str, Any]]
) -> tuple[FigureOrTable, ...]:
    if not figures_and_tables:
        return ()
    captions = figures_and_tables.get("caption", [])
    files = figures_and_tables.get("file", [None] * len(captions))
    return tuple(
        FigureOrTable(caption=caption, file=file_) for caption, file_ in zip(captions, files)
    )


class _EvidenceResolver:
    """Resolves evidence strings back to (section, paragraph) or
    figure/table locations within a single paper. Built once per paper
    and reused across all of that paper's answers for efficiency.
    """

    def __init__(self, sections: tuple[Section, ...], figures_and_tables: tuple[FigureOrTable, ...]):
        self._paragraph_lookup: dict[str, tuple[int, int]] = {}
        for section in sections:
            for para_idx, paragraph in enumerate(section.paragraphs):
                key = _normalize_whitespace(paragraph)
                # First occurrence wins on duplicate paragraph text;
                # documented as a known limitation.
                self._paragraph_lookup.setdefault(key, (section.section_index, para_idx))

        self._figures_and_tables = figures_and_tables

    def resolve(self, evidence_text: str) -> EvidenceSpan:
        is_float_selected = evidence_text.startswith(FLOAT_SELECTED_PREFIX)

        if is_float_selected:
            figure_index = self._resolve_figure_or_table(evidence_text)
            return EvidenceSpan(
                text=evidence_text,
                is_float_selected=True,
                resolved=figure_index is not None,
                figure_or_table_index=figure_index,
            )

        key = _normalize_whitespace(evidence_text)
        location = self._paragraph_lookup.get(key)
        if location is None:
            return EvidenceSpan(text=evidence_text, is_float_selected=False, resolved=False)

        section_index, paragraph_index = location
        return EvidenceSpan(
            text=evidence_text,
            is_float_selected=False,
            resolved=True,
            section_index=section_index,
            paragraph_index=paragraph_index,
        )

    def _resolve_figure_or_table(self, evidence_text: str) -> Optional[int]:
        remainder = _normalize_whitespace(evidence_text[len(FLOAT_SELECTED_PREFIX) :].lstrip(": "))
        if not remainder:
            return None
        for idx, fig in enumerate(self._figures_and_tables):
            caption_norm = _normalize_whitespace(fig.caption)
            if not caption_norm:
                continue
            if caption_norm in remainder or remainder in caption_norm:
                return idx
        return None


def normalize_answer(raw_answer: dict[str, Any], resolver: _EvidenceResolver) -> Answer:
    evidence = tuple(resolver.resolve(text) for text in raw_answer.get("evidence", []))
    return Answer(
        annotation_id=raw_answer.get("annotation_id"),
        worker_id=raw_answer.get("worker_id"),
        unanswerable=bool(raw_answer.get("unanswerable", False)),
        yes_no=raw_answer.get("yes_no"),
        free_form_answer=raw_answer.get("free_form_answer") or None,
        extractive_spans=tuple(raw_answer.get("extractive_spans", [])),
        evidence=evidence,
        highlighted_evidence=tuple(raw_answer.get("highlighted_evidence", [])),
    )


def normalize_questions(
    qas: dict[str, Any], paper_id: str, resolver: _EvidenceResolver
) -> tuple[Question, ...]:
    questions_text = qas["question"]
    question_ids = qas.get("question_id", [None] * len(questions_text))
    answer_groups = qas["answers"]
    question_writers = qas.get("question_writer", [None] * len(questions_text))
    nlp_backgrounds = qas.get("nlp_background", [None] * len(questions_text))
    topic_backgrounds = qas.get("topic_background", [None] * len(questions_text))
    paper_read_statuses = qas.get("paper_read", [None] * len(questions_text))
    search_queries = qas.get("search_query", [None] * len(questions_text))

    questions = []
    for i, question_text in enumerate(questions_text):
        answer_group = answer_groups[i]
        raw_answers = answer_group.get("answer", [])
        annotation_ids = answer_group.get("annotation_id", [None] * len(raw_answers))
        worker_ids = answer_group.get("worker_id", [None] * len(raw_answers))

        answers = []
        for a_idx, raw_answer in enumerate(raw_answers):
            answer = normalize_answer(raw_answer, resolver)
            # annotation_id / worker_id live one level up in the raw
            # schema (parallel to `answer`, not inside each answer
            # dict) — attach them onto the Answer we just built.
            answer = Answer(
                annotation_id=_safe_index(annotation_ids, a_idx) or answer.annotation_id,
                worker_id=_safe_index(worker_ids, a_idx) or answer.worker_id,
                unanswerable=answer.unanswerable,
                yes_no=answer.yes_no,
                free_form_answer=answer.free_form_answer,
                extractive_spans=answer.extractive_spans,
                evidence=answer.evidence,
                highlighted_evidence=answer.highlighted_evidence,
            )
            answers.append(answer)

        questions.append(
            Question(
                question_id=_safe_index(question_ids, i),
                paper_id=paper_id,
                question_text=question_text,
                answers=tuple(answers),
                question_writer=_safe_index(question_writers, i),
                nlp_background=_safe_index(nlp_backgrounds, i),
                topic_background=_safe_index(topic_backgrounds, i),
                paper_read=_safe_index(paper_read_statuses, i),
                search_query=_safe_index(search_queries, i),
            )
        )
    return tuple(questions)


def _safe_index(values: Any, index: int) -> Optional[Any]:
    try:
        return values[index]
    except (IndexError, TypeError):
        return None


def normalize_paper(row: dict[str, Any], split: str) -> Paper:
    """Convert one raw QASPER row (as returned by the `datasets` library,
    or an equivalently-shaped dict such as a test fixture) into a `Paper`.

    Does not validate — call `validate.validate_paper_row` first if you
    need to guarantee well-formedness.
    """
    sections = normalize_sections(row["full_text"])
    figures_and_tables = normalize_figures_and_tables(row.get("figures_and_tables"))
    resolver = _EvidenceResolver(sections, figures_and_tables)
    questions = normalize_questions(row["qas"], paper_id=row["id"], resolver=resolver)

    return Paper(
        paper_id=row["id"],
        title=row["title"],
        abstract=row["abstract"],
        split=split,
        sections=sections,
        figures_and_tables=figures_and_tables,
        questions=questions,
    )


def normalize_split(rows: list[dict[str, Any]], split: str) -> tuple[Paper, ...]:
    return tuple(normalize_paper(row, split=split) for row in rows)
