"""Maps M2's evidence provenance (paragraph/section-level) onto M3's
chunks, preserving the full chain:

    question -> answer -> evidence -> paragraph -> chunk(s)

This module does not guess. An evidence span only maps to chunk(s)
when M2 already resolved it to a specific `(section_index,
paragraph_index)` (`EvidenceSpan.resolved is True` and
`is_float_selected is False` -- figure/table evidence has no paragraph
to map onto, since figures/tables are not chunked in this milestone).
Unresolved evidence maps to an empty tuple of chunk_ids, explicitly,
rather than being silently dropped -- the mapping entry still exists
and is inspectable.

If an oversized paragraph was split into multiple chunks (see
chunker.py), all of those chunks share `paragraph_index` for that
paragraph in their `paragraph_indices`, so all of them are correctly
included as "containing" that evidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from evidencerag.chunking.schema import Chunk
from evidencerag.ingestion.schema import Paper


@dataclass(frozen=True)
class EvidenceChunkMapping:
    """One evidence span's mapping onto the chunk(s) that contain its
    source paragraph, with the full provenance chain preserved.

    `question_index` / `answer_index` / `evidence_index` are positions
    within `paper.questions[question_index].answers[answer_index]
    .evidence[evidence_index]` -- included alongside `question_id`
    (which can be None, though it is present for all current QASPER
    data) so this mapping is always usable for lookups even if
    `question_id` is missing.
    """

    paper_id: str
    split: str
    question_index: int
    question_id: Optional[str]
    answer_index: int
    evidence_index: int
    is_float_selected: bool
    resolved: bool
    section_index: Optional[int]
    paragraph_index: Optional[int]
    chunk_ids: tuple[str, ...]


def map_evidence_to_chunks(paper: Paper, chunks: tuple[Chunk, ...]) -> tuple[EvidenceChunkMapping, ...]:
    """Build the evidence -> chunk(s) mapping for one paper.

    `chunks` must be the chunks produced by `chunker.chunk_paper` for
    this exact `paper` (same paper_id/split) -- this function does not
    re-chunk anything, it only looks up which already-produced chunks
    contain each evidence span's source paragraph.
    """
    # (section_index, paragraph_index) -> chunk_ids containing it.
    # Abstract chunks have paragraph_indices=() (see chunker.py), so
    # they never contribute an entry here and no evidence ever maps
    # onto them -- this dict's keys are always (int, int) in practice,
    # but the annotation is widened to match Chunk.section_index now
    # being Optional[int].
    location_to_chunk_ids: dict[tuple[Optional[int], int], list[str]] = defaultdict(list)
    for chunk in chunks:
        for para_idx in chunk.paragraph_indices:
            location_to_chunk_ids[(chunk.section_index, para_idx)].append(chunk.chunk_id)

    mappings: list[EvidenceChunkMapping] = []
    for q_idx, question in enumerate(paper.questions):
        for a_idx, answer in enumerate(question.answers):
            for e_idx, evidence in enumerate(answer.evidence):
                if (
                    evidence.resolved
                    and not evidence.is_float_selected
                    and evidence.section_index is not None
                    and evidence.paragraph_index is not None
                ):
                    chunk_ids = tuple(
                        location_to_chunk_ids.get((evidence.section_index, evidence.paragraph_index), ())
                    )
                else:
                    chunk_ids = ()

                mappings.append(
                    EvidenceChunkMapping(
                        paper_id=paper.paper_id,
                        split=paper.split,
                        question_index=q_idx,
                        question_id=question.question_id,
                        answer_index=a_idx,
                        evidence_index=e_idx,
                        is_float_selected=evidence.is_float_selected,
                        resolved=evidence.resolved,
                        section_index=evidence.section_index,
                        paragraph_index=evidence.paragraph_index,
                        chunk_ids=chunk_ids,
                    )
                )

    return tuple(mappings)
