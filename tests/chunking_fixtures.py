"""Fixtures for M3 chunking tests.

Includes `WordCountTokenizer`: a deterministic, network-free stand-in
for the real tiktoken-based tokenizer (see
`evidencerag.chunking.tokenizer.TiktokenTokenizer`), used ONLY in
tests. Real tiktoken requires downloading its BPE encoding file over
the network on first use; that must not be a requirement for running
this test suite. `WordCountTokenizer` counts whitespace-split words as
"tokens", which lets tests construct text with an exact, predictable
token count and exercise the real threshold constants in chunker.py
(TARGET_CHUNK_TOKENS=300, OVERLAP_TOKENS=64,
OVERSIZED_PARAGRAPH_TOKENS=600) precisely, without any network access.

`evidencerag.ingestion.schema` (M2) dataclasses are reused directly to
build synthetic `Paper` objects, matching the project's existing
convention of building normalized `Paper`s straight from fixtures
rather than going through the full raw-QASPER pipeline for these tests.
"""

from __future__ import annotations

from evidencerag.ingestion.schema import Answer, EvidenceSpan, Paper, Question, Section


class WordCountTokenizer:
    """Counts whitespace-split words as tokens. Deterministic and
    network-free; NOT used in the shipped chunking pipeline (see
    evidencerag.chunking.tokenizer for the real implementation).
    """

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


def make_words_paragraph(n_words: int, prefix: str = "word") -> str:
    """A single 'paragraph' with no sentence-ending punctuation and an
    exact word count (== token count under WordCountTokenizer).
    """
    return " ".join(f"{prefix}{i}" for i in range(n_words))


def make_sentence(n_words: int, label: str) -> str:
    """One sentence with `n_words` words (including the label token),
    capitalized start, ending in a period -- matches the sentence
    splitter's boundary regex.
    """
    words = [label] + [f"w{i}" for i in range(max(n_words - 1, 0))]
    return " ".join(words) + "."


def make_sentences_paragraph(sentence_word_counts: list[int]) -> str:
    """A paragraph made of multiple distinct sentences, each with a
    known word count, joined with single spaces.
    """
    sentences = [
        make_sentence(n, label=f"Sentence{i}") for i, n in enumerate(sentence_word_counts)
    ]
    return " ".join(sentences)


def make_paper(
    *,
    paper_id: str = "9999.00001",
    split: str = "train",
    abstract: str = "",
    sections: tuple[Section, ...] = (),
    questions: tuple[Question, ...] = (),
) -> Paper:
    # Defaults to an empty abstract so tests that only care about
    # section-based chunking aren't affected by the abstract also
    # being chunked (an empty abstract produces zero chunks -- see
    # chunker.py's `_chunk_abstract`). Tests that specifically exercise
    # abstract chunking pass a non-empty `abstract=` explicitly.
    return Paper(
        paper_id=paper_id,
        title="Fixture Paper",
        abstract=abstract,
        split=split,
        sections=sections,
        figures_and_tables=(),
        questions=questions,
    )


def make_resolved_evidence(text: str, section_index: int, paragraph_index: int) -> EvidenceSpan:
    return EvidenceSpan(
        text=text,
        is_float_selected=False,
        resolved=True,
        section_index=section_index,
        paragraph_index=paragraph_index,
    )


def make_unresolved_evidence(text: str) -> EvidenceSpan:
    return EvidenceSpan(text=text, is_float_selected=False, resolved=False)


def make_float_selected_evidence(text: str, figure_index: int) -> EvidenceSpan:
    return EvidenceSpan(
        text=text, is_float_selected=True, resolved=True, figure_or_table_index=figure_index
    )


def make_question(
    *,
    paper_id: str,
    question_id: str = "Q1",
    question_text: str = "What is the method?",
    answers: tuple[Answer, ...] = (),
) -> Question:
    return Question(
        question_id=question_id,
        paper_id=paper_id,
        question_text=question_text,
        answers=answers,
    )


def make_answer(evidence: tuple[EvidenceSpan, ...]) -> Answer:
    return Answer(
        annotation_id="ann1",
        worker_id="worker1",
        unanswerable=False,
        yes_no=None,
        free_form_answer="An answer.",
        extractive_spans=(),
        evidence=evidence,
    )
