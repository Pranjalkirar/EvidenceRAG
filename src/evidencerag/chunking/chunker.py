"""Paragraph-aware, section-aware chunking of normalized QASPER papers
(M2's `evidencerag.ingestion.schema.Paper`) into `Chunk`s.

CHUNKING POLICY (fixed project decisions -- not reinvented here):
  - target chunk size:  ~300 tokens (a flexible target, not a ceiling)
  - overlap:             64 tokens between consecutive chunks
  - no hard maximum chunk size
  - complete paragraphs are preferred over cutting to hit the target
  - chunks never cross section boundaries
  - oversized paragraphs (see OVERSIZED_PARAGRAPH_TOKENS below) may be
    split, preferring sentence boundaries
  - the abstract IS chunked (see ABSTRACT CHUNKING below), and its
    chunk(s) are returned before a paper's section chunks

ABSTRACT CHUNKING.
  M2's `Paper.abstract` is a standalone string field, not part of
  `Paper.sections` -- it has no section index, and (unlike a section)
  no existing paragraph structure to preserve, since M2 stores it as
  one string rather than a list of paragraphs. We do not invent a
  paragraph split for it. Instead the whole abstract is treated as one
  unit and run through the same target/overlap/no-hard-maximum policy
  as a single paragraph would be: left as one chunk unless it exceeds
  OVERSIZED_PARAGRAPH_TOKENS, in which case it is split the same way
  an oversized section paragraph is (see `_split_oversized_paragraph`),
  preferring sentence boundaries and applying the same ~300-token
  target / 64-token overlap at sentence granularity. An empty (or
  whitespace-only) abstract produces zero chunks.

  Because the abstract has no section index and no paragraph
  structure, its chunk(s) use the explicit exception documented on
  `Chunk` (see schema.py): `section_index=None`,
  `section_title="Abstract"`, `paragraph_indices=()`. We do NOT invent
  a synthetic `section_index=-1` or a `paragraph_indices=(0,)` --
  those would add provenance conventions M2 never defined, and that
  evidence-to-chunk mapping could never actually use (no evidence ever
  resolves into the abstract; see evidence_map.py). Every
  section-derived chunk keeps its real, non-empty `paragraph_indices`
  unchanged -- this exception applies only to abstract chunks.

  Abstract chunk_ids use a distinct `abs` marker in place of the
  `s{section_index:03d}` segment (which has no meaning without a
  section index): `{split}:{paper_id}:abs:c{index:04d}`. Normal
  section chunk_ids are unchanged.

OVERSIZED PARAGRAPH THRESHOLD.
  The brief says a paragraph should be split when it is "substantially
  larger" than the target, without giving an exact number. We define
  "substantially larger" as more than 2x the target
  (`OVERSIZED_PARAGRAPH_TOKENS = 600`). This is a concrete, documented
  interpretation of an underspecified threshold, not a change to the
  target/overlap/no-hard-maximum policy itself.

OVERLAP SCOPE.
  Overlap is applied paragraph-by-paragraph between consecutive
  "normal" (non-oversized-paragraph) chunks within the same section,
  and sentence-by-sentence between consecutive pieces of the same
  oversized paragraph being split. Overlap is deliberately NOT applied
  across the boundary between an oversized-paragraph split and an
  adjacent normal chunk, because bridging that boundary would mix
  paragraph-level and sentence-level provenance in a way the brief
  does not specify. This keeps every chunk's `paragraph_indices`
  unambiguous and traceable.
"""

from __future__ import annotations

from dataclasses import dataclass

from evidencerag.chunking.schema import Chunk
from evidencerag.chunking.sentence_split import split_into_sentences
from evidencerag.chunking.tokenizer import Tokenizer, get_default_tokenizer
from evidencerag.ingestion.schema import Paper, Section

TARGET_CHUNK_TOKENS = 300
OVERLAP_TOKENS = 64
OVERSIZED_PARAGRAPH_TOKENS = 2 * TARGET_CHUNK_TOKENS  # 600; see module docstring


@dataclass
class _PendingChunk:
    """A chunk's content before it has a chunk_id (assigned later,
    deterministically, once we know its position among its siblings).
    """

    paragraph_indices: tuple[int, ...]
    text: str
    token_count: int


def chunk_paper(paper: Paper, tokenizer: Tokenizer | None = None) -> tuple[Chunk, ...]:
    """Chunk `paper.abstract` and every section of `paper`, and return
    all resulting chunks: the abstract's chunk(s) first (see ABSTRACT
    CHUNKING in the module docstring), then section chunks in section
    then in-section order.
    """
    tokenizer = tokenizer or get_default_tokenizer()

    chunks: list[Chunk] = []

    for i, pending in enumerate(_chunk_abstract(paper.abstract, tokenizer)):
        chunk_id = _make_abstract_chunk_id(split=paper.split, paper_id=paper.paper_id, index_in_abstract=i)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper.paper_id,
                split=paper.split,
                section_index=None,
                section_title="Abstract",
                paragraph_indices=(),
                text=pending.text,
                token_count=pending.token_count,
            )
        )

    for section in paper.sections:
        pending_chunks = _chunk_section(section, tokenizer)
        for i, pending in enumerate(pending_chunks):
            chunk_id = _make_chunk_id(
                split=paper.split, paper_id=paper.paper_id, section_index=section.section_index, index_in_section=i
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper.paper_id,
                    split=paper.split,
                    section_index=section.section_index,
                    section_title=section.title,
                    paragraph_indices=pending.paragraph_indices,
                    text=pending.text,
                    token_count=pending.token_count,
                )
            )
    return tuple(chunks)


def _make_chunk_id(*, split: str, paper_id: str, section_index: int, index_in_section: int) -> str:
    """Deterministic chunk_id from source information: same paper +
    same algorithm always produces the same IDs, with no randomness.
    """
    return f"{split}:{paper_id}:s{section_index:03d}:c{index_in_section:04d}"


def _make_abstract_chunk_id(*, split: str, paper_id: str, index_in_abstract: int) -> str:
    """Deterministic chunk_id for an abstract chunk. Uses a distinct
    `abs` marker in place of the section-index segment, since abstract
    chunks have no section index (see ABSTRACT CHUNKING above).
    """
    return f"{split}:{paper_id}:abs:c{index_in_abstract:04d}"


def _chunk_abstract(abstract: str, tokenizer: Tokenizer) -> list[_PendingChunk]:
    """Chunk `paper.abstract` as a single unit (see ABSTRACT CHUNKING
    in the module docstring): one chunk unless it is oversized, in
    which case it is split the same way an oversized section paragraph
    is. Returns `_PendingChunk`s with `paragraph_indices=()` -- the
    abstract has no paragraph structure to record.
    """
    if not abstract.strip():
        return []

    token_count = tokenizer.count_tokens(abstract)
    if token_count <= OVERSIZED_PARAGRAPH_TOKENS:
        return [_PendingChunk(paragraph_indices=(), text=abstract, token_count=token_count)]

    return [
        _PendingChunk(paragraph_indices=(), text=piece_text, token_count=piece_tokens)
        for piece_text, piece_tokens in _split_oversized_paragraph(abstract, tokenizer)
    ]


def _chunk_section(section: Section, tokenizer: Tokenizer) -> list[_PendingChunk]:
    if not section.paragraphs:
        return []

    para_tokens = [tokenizer.count_tokens(p) for p in section.paragraphs]

    pending: list[_PendingChunk] = []

    # Rolling "current chunk under construction" state.
    current_indices: list[int] = []
    current_texts: list[str] = []
    current_tokens = 0
    has_new_paragraph = False  # False while current_* holds only a carried-over overlap seed

    def flush(seed_next_with_overlap: bool) -> None:
        nonlocal current_indices, current_texts, current_tokens, has_new_paragraph
        if current_indices and has_new_paragraph:
            text = "\n\n".join(current_texts)
            pending.append(
                _PendingChunk(
                    paragraph_indices=tuple(current_indices),
                    text=text,
                    token_count=tokenizer.count_tokens(text),
                )
            )

        if seed_next_with_overlap and current_indices and has_new_paragraph:
            # Carry forward trailing paragraphs whose cumulative token
            # count is >= OVERLAP_TOKENS, to seed the next chunk.
            seed_indices: list[int] = []
            seed_texts: list[str] = []
            seed_tokens = 0
            for idx, txt in zip(reversed(current_indices), reversed(current_texts)):
                seed_indices.insert(0, idx)
                seed_texts.insert(0, txt)
                seed_tokens += para_tokens[idx]
                if seed_tokens >= OVERLAP_TOKENS:
                    break
            current_indices, current_texts, current_tokens = seed_indices, seed_texts, seed_tokens
        else:
            current_indices, current_texts, current_tokens = [], [], 0
        has_new_paragraph = False

    for idx, paragraph in enumerate(section.paragraphs):
        p_tokens = para_tokens[idx]

        if p_tokens > OVERSIZED_PARAGRAPH_TOKENS:
            # Flush whatever normal content is pending (no overlap seed
            # carried into the oversized-paragraph split -- see module
            # docstring "OVERLAP SCOPE").
            flush(seed_next_with_overlap=False)

            for piece_text, piece_tokens in _split_oversized_paragraph(paragraph, tokenizer):
                pending.append(
                    _PendingChunk(paragraph_indices=(idx,), text=piece_text, token_count=piece_tokens)
                )
            continue

        # Prefer complete paragraphs: only flush before adding this
        # paragraph if we already have real content AND adding it
        # would push us past the target. There is no hard ceiling, so
        # a single paragraph alone can still exceed the target.
        if has_new_paragraph and current_tokens + p_tokens > TARGET_CHUNK_TOKENS:
            flush(seed_next_with_overlap=True)

        current_indices.append(idx)
        current_texts.append(paragraph)
        current_tokens += p_tokens
        has_new_paragraph = True

    flush(seed_next_with_overlap=False)  # final flush, nothing left to seed

    return pending


def _split_oversized_paragraph(paragraph: str, tokenizer: Tokenizer) -> list[tuple[str, int]]:
    """Split one oversized paragraph into (text, token_count) pieces,
    preferring sentence boundaries and applying the same ~300-token
    target / 64-token overlap policy at sentence granularity.

    Falls back to a word-boundary split for the rare case of a single
    sentence that is itself oversized (see sentence_split.py's
    docstring on why we don't attempt a smarter sentence splitter).
    """
    sentences = split_into_sentences(paragraph)
    if not sentences:
        return []

    # Expand any individually-oversized "sentence" (e.g. no punctuation
    # at all) into smaller word-boundary pieces so packing below always
    # has small enough units to work with.
    units: list[str] = []
    for sentence in sentences:
        if tokenizer.count_tokens(sentence) > OVERSIZED_PARAGRAPH_TOKENS:
            units.extend(_split_by_words(sentence, tokenizer))
        else:
            units.append(sentence)

    unit_tokens = [tokenizer.count_tokens(u) for u in units]

    pieces: list[tuple[str, int]] = []
    current_units: list[str] = []
    current_unit_tokens: list[int] = []
    current_tokens = 0
    has_new_unit = False

    def flush(seed_next_with_overlap: bool) -> None:
        nonlocal current_units, current_unit_tokens, current_tokens, has_new_unit
        if current_units and has_new_unit:
            text = " ".join(current_units)
            pieces.append((text, tokenizer.count_tokens(text)))

        if seed_next_with_overlap and current_units and has_new_unit:
            seed_units: list[str] = []
            seed_toks: list[int] = []
            seed_total = 0
            for u, t in zip(reversed(current_units), reversed(current_unit_tokens)):
                seed_units.insert(0, u)
                seed_toks.insert(0, t)
                seed_total += t
                if seed_total >= OVERLAP_TOKENS:
                    break
            current_units, current_unit_tokens, current_tokens = seed_units, seed_toks, seed_total
        else:
            current_units, current_unit_tokens, current_tokens = [], [], 0
        has_new_unit = False

    for unit, u_tokens in zip(units, unit_tokens):
        if has_new_unit and current_tokens + u_tokens > TARGET_CHUNK_TOKENS:
            flush(seed_next_with_overlap=True)
        current_units.append(unit)
        current_unit_tokens.append(u_tokens)
        current_tokens += u_tokens
        has_new_unit = True

    flush(seed_next_with_overlap=False)

    return pieces


def _split_by_words(text: str, tokenizer: Tokenizer) -> list[str]:
    """Last-resort fallback: split a single (sentence-boundary-free)
    oversized unit into ~target-sized pieces on whitespace. Never
    drops text.
    """
    words = text.split(" ")
    if len(words) <= 1:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        w_tokens = tokenizer.count_tokens(word + " ")
        if current and current_tokens + w_tokens > TARGET_CHUNK_TOKENS:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += w_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces
