from evidencerag.chunking.chunker import chunk_paper
from evidencerag.chunking.evidence_map import map_evidence_to_chunks
from evidencerag.ingestion.schema import Section
from tests.chunking_fixtures import (
    WordCountTokenizer,
    make_answer,
    make_float_selected_evidence,
    make_paper,
    make_question,
    make_resolved_evidence,
    make_sentences_paragraph,
    make_unresolved_evidence,
    make_words_paragraph,
)

TOKENIZER = WordCountTokenizer()


def test_resolved_evidence_maps_to_the_chunk_containing_its_paragraph():
    section = Section(
        section_index=0,
        title="Method",
        paragraphs=(make_words_paragraph(50, "a"), make_words_paragraph(50, "b")),
    )
    evidence = make_resolved_evidence(
        text=make_words_paragraph(50, "b"), section_index=0, paragraph_index=1
    )
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    assert len(mappings) == 1
    mapping = mappings[0]
    assert mapping.resolved is True
    assert mapping.chunk_ids != ()
    # The mapped chunk(s) must actually contain paragraph_index=1.
    mapped_chunks = [c for c in chunks if c.chunk_id in mapping.chunk_ids]
    assert all(1 in c.paragraph_indices for c in mapped_chunks)


def test_unresolved_evidence_maps_to_no_chunks_without_guessing():
    section = Section(section_index=0, title="Method", paragraphs=(make_words_paragraph(50),))
    evidence = make_unresolved_evidence(text="some text that was never located")
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    assert len(mappings) == 1
    assert mappings[0].resolved is False
    assert mappings[0].chunk_ids == ()


def test_float_selected_evidence_maps_to_no_chunks():
    # Figures/tables aren't chunked in M3, so even a "resolved"
    # float-selected evidence item must map to zero chunks.
    section = Section(section_index=0, title="Results", paragraphs=(make_words_paragraph(50),))
    evidence = make_float_selected_evidence(text="FLOAT SELECTED: Table 1", figure_index=0)
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    assert len(mappings) == 1
    assert mappings[0].is_float_selected is True
    assert mappings[0].chunk_ids == ()


def test_oversized_paragraph_evidence_maps_to_multiple_chunks():
    paragraph = make_sentences_paragraph([80] * 10)  # oversized, will be split
    section = Section(section_index=0, title="Related Work", paragraphs=(paragraph,))
    evidence = make_resolved_evidence(text=paragraph, section_index=0, paragraph_index=0)
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    assert len(chunks) > 1  # sanity: paragraph really was split
    assert len(mappings) == 1
    # All chunks derived from this (split) paragraph should be listed.
    expected_chunk_ids = {c.chunk_id for c in chunks if 0 in c.paragraph_indices}
    assert set(mappings[0].chunk_ids) == expected_chunk_ids
    assert len(mappings[0].chunk_ids) > 1


def test_preserves_full_question_answer_evidence_chain():
    section = Section(section_index=0, title="Method", paragraphs=(make_words_paragraph(30),))
    evidence = make_resolved_evidence(text=make_words_paragraph(30), section_index=0, paragraph_index=0)
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", question_id="Q42", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    m = mappings[0]
    assert m.paper_id == "1000.00001"
    assert m.question_id == "Q42"
    assert m.question_index == 0
    assert m.answer_index == 0
    assert m.evidence_index == 0
    assert m.section_index == 0
    assert m.paragraph_index == 0


def test_abstract_chunks_never_receive_evidence_mappings():
    # No evidence ever resolves into the abstract (M2 never assigns
    # EvidenceSpan.section_index/paragraph_index pointing to it), and
    # abstract chunks have paragraph_indices=() -- so they must never
    # appear in an evidence mapping's chunk_ids, even when evidence
    # resolves correctly into a real section.
    section = Section(section_index=0, title="Method", paragraphs=(make_words_paragraph(50),))
    evidence = make_resolved_evidence(
        text=make_words_paragraph(50), section_index=0, paragraph_index=0
    )
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(
        paper_id="1000.00001", abstract="A short abstract.", sections=(section,), questions=(question,)
    )

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    abstract_chunk_ids = {c.chunk_id for c in chunks if c.section_title == "Abstract"}
    assert abstract_chunk_ids  # sanity: an abstract chunk was actually produced
    assert len(mappings) == 1
    assert not (set(mappings[0].chunk_ids) & abstract_chunk_ids)


def test_multiple_evidence_items_produce_separate_mapping_entries():
    section = Section(
        section_index=0,
        title="Method",
        paragraphs=(make_words_paragraph(30, "a"), make_words_paragraph(30, "b")),
    )
    e1 = make_resolved_evidence(make_words_paragraph(30, "a"), section_index=0, paragraph_index=0)
    e2 = make_resolved_evidence(make_words_paragraph(30, "b"), section_index=0, paragraph_index=1)
    answer = make_answer(evidence=(e1, e2))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    mappings = map_evidence_to_chunks(paper, chunks)

    assert len(mappings) == 2
    assert mappings[0].evidence_index == 0
    assert mappings[1].evidence_index == 1
    # Not concatenated/merged: each keeps its own paragraph_index.
    assert mappings[0].paragraph_index == 0
    assert mappings[1].paragraph_index == 1
