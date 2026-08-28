"""Serialization of the normalized internal representation (schema.py)
to and from disk.

Format: one JSON Lines (.jsonl) file per split, each line a single
serialized `Paper` (including all of its nested questions/answers/
evidence). JSON Lines rather than one big JSON array so that large
splits can later be streamed line-by-line without loading everything
into memory at once, and so a partially-written file is still mostly
usable.

This module only converts between `Paper` objects and plain
dicts/JSON — it does not decide *where* to write files. Callers pass
an explicit path (see scripts/ingest_qasper.py, which uses
`evidencerag.config.PATHS.processed_data_dir`).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from evidencerag.ingestion.schema import (
    Answer,
    EvidenceSpan,
    FigureOrTable,
    Paper,
    Question,
    Section,
)


def paper_to_dict(paper: Paper) -> dict[str, Any]:
    """Convert a `Paper` (and everything nested inside it) to a plain dict.

    `dataclasses.asdict` already does this recursively for frozen
    dataclasses, including tuples-of-dataclasses; we use it directly
    rather than hand-rolling the recursion.
    """
    return asdict(paper)


def paper_from_dict(data: dict[str, Any]) -> Paper:
    """Reconstruct a `Paper` from a dict produced by `paper_to_dict`
    (or an equivalently-shaped dict loaded from JSON)."""
    sections = tuple(
        Section(
            section_index=s["section_index"],
            title=s["title"],
            paragraphs=tuple(s["paragraphs"]),
        )
        for s in data["sections"]
    )
    figures_and_tables = tuple(
        FigureOrTable(caption=f["caption"], file=f.get("file")) for f in data["figures_and_tables"]
    )
    questions = tuple(_question_from_dict(q) for q in data["questions"])

    return Paper(
        paper_id=data["paper_id"],
        title=data["title"],
        abstract=data["abstract"],
        split=data["split"],
        sections=sections,
        figures_and_tables=figures_and_tables,
        questions=questions,
    )


def _question_from_dict(data: dict[str, Any]) -> Question:
    return Question(
        question_id=data.get("question_id"),
        paper_id=data["paper_id"],
        question_text=data["question_text"],
        answers=tuple(_answer_from_dict(a) for a in data["answers"]),
        question_writer=data.get("question_writer"),
        nlp_background=data.get("nlp_background"),
        topic_background=data.get("topic_background"),
        paper_read=data.get("paper_read"),
        search_query=data.get("search_query"),
    )


def _answer_from_dict(data: dict[str, Any]) -> Answer:
    return Answer(
        annotation_id=data.get("annotation_id"),
        worker_id=data.get("worker_id"),
        unanswerable=data["unanswerable"],
        yes_no=data.get("yes_no"),
        free_form_answer=data.get("free_form_answer"),
        extractive_spans=tuple(data["extractive_spans"]),
        evidence=tuple(_evidence_from_dict(e) for e in data["evidence"]),
        highlighted_evidence=tuple(data.get("highlighted_evidence", [])),
    )


def _evidence_from_dict(data: dict[str, Any]) -> EvidenceSpan:
    return EvidenceSpan(
        text=data["text"],
        is_float_selected=data["is_float_selected"],
        resolved=data["resolved"],
        section_index=data.get("section_index"),
        paragraph_index=data.get("paragraph_index"),
        figure_or_table_index=data.get("figure_or_table_index"),
    )


def save_papers(papers: Iterable[Paper], path: Path) -> int:
    """Write papers as JSON Lines to `path`. Returns the number written.

    Creates parent directories if needed. Overwrites `path` if it
    already exists (ingestion is meant to be re-run idempotently).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper_to_dict(paper), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def load_papers(path: Path) -> Iterator[Paper]:
    """Lazily read papers back from a JSON Lines file written by `save_papers`."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield paper_from_dict(json.loads(line))
