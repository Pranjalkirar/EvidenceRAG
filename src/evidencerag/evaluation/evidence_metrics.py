"""QASPER-style Evidence F1, following the official evaluator's
set-overlap semantics on evidence *text* -- not chunk identifiers or
paragraph positions.

This is deliberately a different representation from the chunk-level
Recall/MRR gold in `retrieval_metrics.py`/`gold.py`: those use resolved
`(section_index, paragraph_index)` provenance to identify M3 chunks,
because "was the right chunk retrieved" is a positional question.
Evidence F1 is a QASPER-defined text-overlap metric, so it is scored
on text, exactly like the original evaluator does, independent of
whether M2 could resolve a position for a given gold span.

Gold references come from `gold.QuestionGold.evidence_text_references`
(raw `EvidenceSpan.text`, FLOAT SELECTED already excluded there).
Predicted evidence is recovered from the M3 provenance of the
retrieved chunks: each chunk's `paragraph_indices` identify which
paragraphs of the *same* `Paper` it was chunked from, and
`paper.sections[section_index].paragraphs[paragraph_index]` recovers
the identical text QASPER's own `evidence` field would contain, since
neither M2 nor M3 rewrites paragraph text anywhere along that chain.
"""

from __future__ import annotations

from typing import AbstractSet, Mapping, Sequence

from evidencerag.chunking.schema import Chunk
from evidencerag.ingestion.schema import Paper


def retrieved_paragraph_texts(
    chunk_ids: Sequence[str], chunk_by_id: Mapping[str, Chunk], paper: Paper
) -> frozenset[str]:
    """The set of original paragraph texts covered by `chunk_ids`,
    recovered from `paper`.

    Every `chunk_id` must belong to `paper` (same `paper_id`/`split`)
    -- a mismatch is a caller bug and raises, rather than silently
    reading the wrong paper's paragraphs. Abstract chunks
    (`section_index is None`) contribute nothing, matching
    `evidence_map.py`'s own rule that no evidence ever resolves onto
    the abstract. Locations are deduplicated before text lookup, so a
    paragraph split across multiple M3 chunks (see `chunker.py`)
    contributes its text only once.
    """
    locations: set[tuple[int, int]] = set()
    for chunk_id in chunk_ids:
        chunk = chunk_by_id[chunk_id]
        if chunk.paper_id != paper.paper_id or chunk.split != paper.split:
            continue
        if chunk.section_index is None:
            continue
        for paragraph_index in chunk.paragraph_indices:
            locations.add((chunk.section_index, paragraph_index))

    return frozenset(paper.sections[section_index].paragraphs[paragraph_index] for section_index, paragraph_index in locations)


def paragraph_set_f1(predicted: AbstractSet[str], reference: AbstractSet[str]) -> float:
    """Official QASPER `paragraph_f1_score`, on sets of paragraph text.

    Both empty -> 1.0 (the question needed no evidence and none was
    predicted). Otherwise set-intersection precision/recall/F1; zero
    overlap -> 0.0.
    """
    if not predicted and not reference:
        return 1.0
    num_same = len(predicted & reference)
    if num_same == 0:
        return 0.0
    precision = num_same / len(predicted)
    recall = num_same / len(reference)
    return 2 * precision * recall / (precision + recall)


def evidence_f1(predicted: AbstractSet[str], references: Sequence[AbstractSet[str]]) -> float:
    """Max `paragraph_set_f1` across a question's answer references,
    matching the official evaluator's "multiple valid annotations,
    take the max" rule.

    `references` must be non-empty -- a question with zero answer
    annotations is a data error the caller must not silently paper
    over (an answer with zero *evidence* is fine and represented as an
    empty-set reference, which `paragraph_set_f1` already handles).
    """
    if not references:
        raise ValueError("evidence_f1 requires at least one reference")
    return max(paragraph_set_f1(predicted, reference) for reference in references)
