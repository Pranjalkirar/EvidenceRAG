"""The Chunk data model: a retrieval-ready unit of text, produced from
one QASPER paper's normalized (M2) representation.

This module ONLY defines the shape. It does not:
  - generate chunk_ids (the chunking pipeline in chunker.py does),
  - count tokens (tokenizer.py does),
  - decide how to split text (chunker.py does).

Keeping this file free of that logic is deliberate: `Chunk` should stay
tokenizer-agnostic and algorithm-agnostic so it can be reused if the
chunking strategy or tokenizer changes later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrieval-ready chunk of paper text.

    Provenance is explicit and sufficient to trace a chunk back to its
    exact source paragraphs within one section of one paper:

        paper_id + split + section_index  -> which section
        paragraph_indices                 -> which paragraph(s) of that
                                              section contributed text
                                              (in section-local order,
                                              matching M2's
                                              EvidenceSpan.paragraph_index)

    ABSTRACT CHUNKS ARE THE ONE EXCEPTION to that provenance scheme.
    M2's `Paper.abstract` is a standalone string, not part of
    `Paper.sections`, so it has no section index and no
    paragraph-index structure for evidence to point into (see
    chunker.py). Abstract chunks are marked with:

        section_index      = None
        section_title       = "Abstract"
        paragraph_indices  = ()  (always empty -- never (0,))

    Every other (section-derived) chunk always has an int
    `section_index` and a non-empty `paragraph_indices`.

    `chunk_id` is assigned by the chunking pipeline (see chunker.py)
    from deterministic source information -- never randomly generated,
    and never computed here.

    `token_count` is supplied by whatever tokenizer the chunking
    pipeline used (see tokenizer.py) -- this class does not know or
    care which tokenizer that was.
    """

    chunk_id: str
    paper_id: str
    split: str
    section_index: int | None
    section_title: str
    paragraph_indices: tuple[int, ...]
    text: str
    token_count: int
