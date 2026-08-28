"""Validation of raw QASPER rows against the structure we actually
observed on the Hugging Face Hub (see loader.py's module docstring
for the source).

This is deliberately structural validation (right keys, right types,
right nesting) — not semantic validation of QASPER's research content.
It exists so that normalization (normalize.py) can assume a well-formed
input and fail loudly, with a clear message, on anything unexpected
rather than silently producing a malformed internal representation.

Each `validate_*` function returns a list of human-readable problem
strings. An empty list means "no problems found". Nothing raises here;
callers (e.g. scripts/ingest_qasper.py) decide whether to abort.
"""

from __future__ import annotations

from typing import Any

REQUIRED_PAPER_KEYS = ("id", "title", "abstract", "full_text", "qas")
REQUIRED_FULL_TEXT_KEYS = ("section_name", "paragraphs")
REQUIRED_QAS_KEYS = ("question", "question_id", "answers")
REQUIRED_ANSWER_ENTRY_KEYS = ("answer", "annotation_id", "worker_id")
REQUIRED_ANSWER_KEYS = (
    "unanswerable",
    "extractive_spans",
    "yes_no",
    "free_form_answer",
    "evidence",
    "highlighted_evidence",
)


def validate_paper_row(row: dict[str, Any]) -> list[str]:
    """Validate a single raw QASPER paper row. Returns a list of issues."""
    issues: list[str] = []

    for key in REQUIRED_PAPER_KEYS:
        if key not in row:
            issues.append(f"paper row missing required key '{key}'")
    if issues:
        # Without the basic keys there's nothing more we can safely check.
        return issues

    if not isinstance(row["id"], str) or not row["id"]:
        issues.append("paper 'id' must be a non-empty string")

    issues.extend(_validate_full_text(row["full_text"], paper_id=row.get("id", "?")))
    issues.extend(_validate_qas(row["qas"], paper_id=row.get("id", "?")))

    if "figures_and_tables" in row:
        issues.extend(
            _validate_figures_and_tables(row["figures_and_tables"], paper_id=row.get("id", "?"))
        )

    return issues


def _validate_full_text(full_text: Any, paper_id: str) -> list[str]:
    issues: list[str] = []
    if not isinstance(full_text, dict):
        return [f"paper {paper_id}: 'full_text' must be a dict, got {type(full_text).__name__}"]

    for key in REQUIRED_FULL_TEXT_KEYS:
        if key not in full_text:
            issues.append(f"paper {paper_id}: 'full_text' missing key '{key}'")
    if issues:
        return issues

    section_names = full_text["section_name"]
    paragraphs = full_text["paragraphs"]
    if len(section_names) != len(paragraphs):
        issues.append(
            f"paper {paper_id}: 'full_text.section_name' has {len(section_names)} entries "
            f"but 'full_text.paragraphs' has {len(paragraphs)} — these must be parallel arrays"
        )
    return issues


def _validate_qas(qas: Any, paper_id: str) -> list[str]:
    issues: list[str] = []
    if not isinstance(qas, dict):
        return [f"paper {paper_id}: 'qas' must be a dict, got {type(qas).__name__}"]

    for key in REQUIRED_QAS_KEYS:
        if key not in qas:
            issues.append(f"paper {paper_id}: 'qas' missing key '{key}'")
    if issues:
        return issues

    questions = qas["question"]
    answers_list = qas["answers"]
    if len(questions) != len(answers_list):
        issues.append(
            f"paper {paper_id}: 'qas.question' has {len(questions)} entries but "
            f"'qas.answers' has {len(answers_list)} — these must be parallel arrays"
        )

    for q_idx, answer_group in enumerate(answers_list):
        issues.extend(_validate_answer_group(answer_group, paper_id=paper_id, q_idx=q_idx))

    return issues


def _validate_answer_group(answer_group: Any, paper_id: str, q_idx: int) -> list[str]:
    issues: list[str] = []
    if not isinstance(answer_group, dict):
        return [
            f"paper {paper_id} question #{q_idx}: answer group must be a dict, "
            f"got {type(answer_group).__name__}"
        ]

    for key in REQUIRED_ANSWER_ENTRY_KEYS:
        if key not in answer_group:
            issues.append(
                f"paper {paper_id} question #{q_idx}: answer group missing key '{key}'"
            )
    if issues:
        return issues

    for a_idx, answer in enumerate(answer_group["answer"]):
        for key in REQUIRED_ANSWER_KEYS:
            if key not in answer:
                issues.append(
                    f"paper {paper_id} question #{q_idx} answer #{a_idx}: "
                    f"missing key '{key}'"
                )
    return issues


def _validate_figures_and_tables(figures_and_tables: Any, paper_id: str) -> list[str]:
    if not isinstance(figures_and_tables, dict):
        return [
            f"paper {paper_id}: 'figures_and_tables' must be a dict, "
            f"got {type(figures_and_tables).__name__}"
        ]
    if "caption" not in figures_and_tables:
        return [f"paper {paper_id}: 'figures_and_tables' missing key 'caption'"]
    return []


def validate_split(rows: list[dict[str, Any]], split_name: str) -> list[str]:
    """Validate every row in one split. Issues are prefixed with the split name."""
    issues: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_issues = validate_paper_row(row)
        issues.extend(f"[{split_name}] {issue}" for issue in row_issues)

        paper_id = row.get("id")
        if paper_id in seen_ids:
            issues.append(f"[{split_name}] duplicate paper id '{paper_id}'")
        elif paper_id is not None:
            seen_ids.add(paper_id)

    return issues
