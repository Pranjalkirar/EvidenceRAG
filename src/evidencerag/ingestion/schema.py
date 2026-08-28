"""Internal, typed representation of QASPER data.

This is the "our internal data representation" referred to in the
ingestion pipeline: everything downstream (chunking, retrieval,
evaluation) should consume these dataclasses rather than raw QASPER
dicts, so the raw dataset's quirks (parallel-array encoding, optional
fields, split naming) are dealt with exactly once, here.

Relationships are explicit and match the real structure of QASPER:

    Paper
      └── Question (many)
            └── Answer (one or more workers answered each question)
                  └── EvidenceSpan (each evidence piece kept separate,
                                    with best-effort provenance back to
                                    a paper section/paragraph or figure)

Nothing in this module downloads, validates, or transforms data — see
loader.py, validate.py and normalize.py for that. This module only
defines the shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Section:
    """One section of a paper's full text.

    QASPER's `full_text` is two parallel arrays (`section_name`,
    `paragraphs`) rather than a list of section objects. We restructure
    that into one `Section` per entry, keeping the original order via
    `section_index` since section titles are not unique or reliable
    identifiers (they can be empty strings, e.g. an un-named preamble).
    """

    section_index: int
    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class FigureOrTable:
    """One figure/table entry from QASPER's `figures_and_tables` field.

    NOTE: this field is present in the current HuggingFace-hosted
    version of QASPER but is not documented in the original QASPER
    paper/README schema description (which only mentions figures and
    tables indirectly, via "FLOAT SELECTED ..." markers inside
    `evidence`). We preserve it because it exists in the actual data,
    without claiming it is part of the "official" documented schema.
    """

    caption: str
    file: Optional[str] = None


@dataclass(frozen=True)
class EvidenceSpan:
    """One individual piece of evidence supporting an answer.

    QASPER's `evidence` field is a list of strings — each either an
    entire paragraph of the paper, or a table/figure reference prefixed
    with "FLOAT SELECTED". We keep each entry as its own `EvidenceSpan`
    (never concatenated into one blob) and attempt to resolve it back
    to a specific location in the paper, so that a later chunking step
    can derive ground-truth (question -> evidence -> chunk) mappings.

    Resolution is best-effort: `resolved` is False when we could not
    confidently locate the evidence text in the paper's sections or
    figures/tables (e.g. due to minor text normalization differences
    upstream). We do not guess in that case — we record the failure.
    """

    text: str
    is_float_selected: bool
    resolved: bool
    section_index: Optional[int] = None
    paragraph_index: Optional[int] = None
    figure_or_table_index: Optional[int] = None


@dataclass(frozen=True)
class Answer:
    """One worker's answer to a question.

    A question can have multiple `Answer`s (multiple workers answered
    independently). Exactly one of `unanswerable`, `yes_no`,
    `free_form_answer`, `extractive_spans` is meaningful per QASPER's
    documented convention — we preserve all fields as given rather than
    collapsing them into a single "answer text".

    `highlighted_evidence` is kept as its own flat tuple of sentences,
    separate from `evidence`. QASPER's documentation states these
    sentences map onto the paragraph-level `evidence` entries, but does
    NOT guarantee a positional one-to-one correspondence (a paragraph
    can contribute zero, one, or several highlighted sentences). We do
    not invent that pairing — see normalize.py's module docstring for
    details on this ambiguity.
    """

    annotation_id: Optional[str]
    worker_id: Optional[str]
    unanswerable: bool
    yes_no: Optional[bool]
    free_form_answer: Optional[str]
    extractive_spans: tuple[str, ...]
    evidence: tuple[EvidenceSpan, ...]
    highlighted_evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Question:
    """One question asked about a paper, with all of its answers.

    `question_id` is preserved when present in the raw data (it is, in
    the current v0.3 release) rather than being invented.
    """

    question_id: Optional[str]
    paper_id: str
    question_text: str
    answers: tuple[Answer, ...]
    question_writer: Optional[str] = None
    nlp_background: Optional[str] = None
    topic_background: Optional[str] = None
    paper_read: Optional[str] = None
    search_query: Optional[str] = None


@dataclass(frozen=True)
class Paper:
    """One QASPER paper: metadata, full text, figures/tables, and questions.

    `split` preserves which official QASPER split ("train", "validation"
    or "test") this paper came from — splits are never merged.
    """

    paper_id: str
    title: str
    abstract: str
    split: str
    sections: tuple[Section, ...]
    figures_and_tables: tuple[FigureOrTable, ...]
    questions: tuple[Question, ...]
