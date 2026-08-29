from evidencerag.chunking.chunker import (
    OVERLAP_TOKENS,
    OVERSIZED_PARAGRAPH_TOKENS,
    TARGET_CHUNK_TOKENS,
    chunk_paper,
)
from evidencerag.ingestion.schema import Section
from tests.chunking_fixtures import (
    WordCountTokenizer,
    make_paper,
    make_sentences_paragraph,
    make_words_paragraph,
)

TOKENIZER = WordCountTokenizer()


def test_policy_constants_are_exactly_the_specified_values():
    # Guards against silently drifting from the fixed project decisions.
    assert TARGET_CHUNK_TOKENS == 300
    assert OVERLAP_TOKENS == 64


def test_paragraph_aware_packing_combines_small_paragraphs():
    section = Section(
        section_index=0,
        title="Introduction",
        paragraphs=(
            make_words_paragraph(50, "a"),
            make_words_paragraph(50, "b"),
            make_words_paragraph(50, "c"),
        ),
    )
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    # All three small paragraphs (150 tokens total) fit comfortably
    # under the 300-token target, so they should be packed together.
    assert len(chunks) == 1
    assert chunks[0].paragraph_indices == (0, 1, 2)
    assert chunks[0].token_count == 150


def test_packing_stops_before_exceeding_target_and_prefers_complete_paragraphs():
    # Two paragraphs of 200 tokens each: the second would push the
    # first chunk to 400 (> 300 target), so packing must start a new
    # chunk rather than truncating the second paragraph.
    section = Section(
        section_index=0,
        title="Method",
        paragraphs=(make_words_paragraph(200, "a"), make_words_paragraph(200, "b")),
    )
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 2
    assert chunks[0].paragraph_indices == (0,)
    # Second chunk includes paragraph 1 (and, per the overlap policy,
    # re-includes trailing paragraph(s) from the first chunk).
    assert 1 in chunks[1].paragraph_indices
    # Every paragraph's full text must appear intact somewhere.
    assert make_words_paragraph(200, "a") in chunks[0].text
    assert make_words_paragraph(200, "b") in chunks[1].text


def test_no_hard_ceiling_a_single_paragraph_may_exceed_target():
    # A single paragraph of 450 tokens is over the 300-token target but
    # under the 600-token "oversized" threshold, so it must remain a
    # single, uncut chunk (no hard maximum).
    section = Section(section_index=0, title="Results", paragraphs=(make_words_paragraph(450),))
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    assert chunks[0].token_count == 450
    assert chunks[0].paragraph_indices == (0,)


def test_overlap_repeats_trailing_paragraph_across_consecutive_chunks():
    # paragraph 0: 280 tokens: paragraph 1: 40 tokens (280+40=320 > 300,
    # so paragraph 1 starts a new chunk). Paragraph 1 alone is only 40
    # tokens (< 64 overlap target), so the overlap seed must also pull
    # in paragraph 0's tail to reach >= 64 tokens carried forward is
    # not required by paragraph granularity -- instead paragraph 0
    # itself is re-included whole, since overlap operates at whole
    # paragraph granularity.
    section = Section(
        section_index=0,
        title="Discussion",
        paragraphs=(make_words_paragraph(280, "a"), make_words_paragraph(40, "b")),
    )
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 2
    assert chunks[0].paragraph_indices == (0,)
    # Chunk 1 re-includes paragraph 0 (overlap) ahead of new paragraph 1,
    # since paragraph 0 alone (280 tokens) already exceeds the 64-token
    # overlap requirement.
    assert chunks[1].paragraph_indices == (0, 1)
    assert make_words_paragraph(280, "a") in chunks[1].text
    assert make_words_paragraph(40, "b") in chunks[1].text


def test_overlap_does_not_duplicate_when_no_preceding_chunk_exists():
    section = Section(section_index=0, title="Intro", paragraphs=(make_words_paragraph(50),))
    paper = make_paper(sections=(section,))
    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    assert len(chunks) == 1
    assert chunks[0].paragraph_indices == (0,)


def test_chunks_never_cross_section_boundaries():
    section_a = Section(section_index=0, title="A", paragraphs=(make_words_paragraph(50, "a"),))
    section_b = Section(section_index=1, title="B", paragraphs=(make_words_paragraph(50, "b"),))
    paper = make_paper(sections=(section_a, section_b))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 2
    assert chunks[0].section_index == 0
    assert chunks[1].section_index == 1
    # No chunk mixes text from both sections.
    for chunk in chunks:
        assert make_words_paragraph(50, "a" if chunk.section_index == 0 else "b") in chunk.text


def test_oversized_paragraph_is_split_using_sentence_boundaries():
    # 10 sentences of 80 words each = 800 tokens, well over the
    # 600-token oversized threshold.
    paragraph = make_sentences_paragraph([80] * 10)
    assert TOKENIZER.count_tokens(paragraph) > OVERSIZED_PARAGRAPH_TOKENS

    section = Section(section_index=0, title="Related Work", paragraphs=(paragraph,))
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) > 1
    # Every resulting chunk still traces back to the same source paragraph.
    assert all(chunk.paragraph_indices == (0,) for chunk in chunks)
    # No chunk mid-splits a sentence: each chunk's text is a clean
    # concatenation of whole sentences (each ends with terminal punctuation).
    for chunk in chunks:
        assert chunk.text.strip().endswith((".", "!", "?"))


def test_oversized_paragraph_split_preserves_all_words():
    paragraph = make_sentences_paragraph([80] * 10)
    section = Section(section_index=0, title="Related Work", paragraphs=(paragraph,))
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    # Every distinct sentence label must appear in at least one chunk
    # (no silent text loss), even accounting for overlap duplication.
    for i in range(10):
        assert any(f"Sentence{i} " in chunk.text for chunk in chunks)


def test_oversized_paragraph_pieces_stay_near_target_when_possible():
    paragraph = make_sentences_paragraph([80] * 10)  # 800 tokens total
    section = Section(section_index=0, title="Related Work", paragraphs=(paragraph,))
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    # No single piece should be absurdly larger than the target when
    # sentence-sized units (80 tokens) are available to pack with.
    for chunk in chunks:
        assert chunk.token_count <= TARGET_CHUNK_TOKENS + 80  # one sentence of slack


def test_deterministic_chunk_ids_stable_across_runs():
    section = Section(
        section_index=0,
        title="Intro",
        paragraphs=(make_words_paragraph(50, "a"), make_words_paragraph(50, "b")),
    )
    paper = make_paper(paper_id="1000.00042", split="validation", sections=(section,))

    chunks_1 = chunk_paper(paper, tokenizer=TOKENIZER)
    chunks_2 = chunk_paper(paper, tokenizer=TOKENIZER)

    assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]
    assert chunks_1[0].chunk_id == "validation:1000.00042:s000:c0000"


def test_chunk_ids_are_unique_within_a_paper():
    section_a = Section(section_index=0, title="A", paragraphs=(make_words_paragraph(400),))
    section_b = Section(section_index=1, title="B", paragraphs=(make_words_paragraph(400),))
    paper = make_paper(sections=(section_a, section_b))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_section_produces_no_chunks():
    section = Section(section_index=0, title="", paragraphs=())
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    assert chunks == ()


def test_very_short_paragraph_still_produces_a_chunk():
    section = Section(section_index=0, title="Intro", paragraphs=("Hi.",))
    paper = make_paper(sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    assert len(chunks) == 1
    assert chunks[0].text == "Hi."
    assert chunks[0].token_count == 1


def test_mixed_empty_and_nonempty_sections():
    empty_section = Section(section_index=0, title="", paragraphs=())
    real_section = Section(section_index=1, title="Body", paragraphs=(make_words_paragraph(30),))
    paper = make_paper(sections=(empty_section, real_section))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    assert len(chunks) == 1
    assert chunks[0].section_index == 1


def test_chunk_carries_correct_section_title_and_paper_metadata():
    section = Section(section_index=2, title="Conclusion", paragraphs=(make_words_paragraph(20),))
    paper = make_paper(paper_id="7.7", split="test", sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    assert len(chunks) == 1
    assert chunks[0].paper_id == "7.7"
    assert chunks[0].split == "test"
    assert chunks[0].section_index == 2
    assert chunks[0].section_title == "Conclusion"


def test_same_paper_id_in_different_splits_never_collides():
    # Splits are kept separate throughout the project (M2 policy);
    # chunk_ids must reflect that rather than colliding across splits.
    section = Section(section_index=0, title="Intro", paragraphs=(make_words_paragraph(30),))
    train_paper = make_paper(paper_id="1000.00001", split="train", sections=(section,))
    val_paper = make_paper(paper_id="1000.00001", split="validation", sections=(section,))

    train_chunks = chunk_paper(train_paper, tokenizer=TOKENIZER)
    val_chunks = chunk_paper(val_paper, tokenizer=TOKENIZER)

    assert train_chunks[0].chunk_id != val_chunks[0].chunk_id
    assert train_chunks[0].split == "train"
    assert val_chunks[0].split == "validation"


# --- Abstract chunking -------------------------------------------------


def test_empty_abstract_produces_no_chunks():
    section = Section(section_index=0, title="Intro", paragraphs=(make_words_paragraph(30),))
    paper = make_paper(abstract="", sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    assert chunks[0].section_index == 0


def test_normal_sized_abstract_becomes_one_chunk_with_none_provenance():
    abstract = make_words_paragraph(100, "abs")
    section = Section(section_index=0, title="Intro", paragraphs=(make_words_paragraph(30),))
    paper = make_paper(paper_id="1000.00001", split="train", abstract=abstract, sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 2
    abstract_chunk = chunks[0]
    assert abstract_chunk.section_index is None
    assert abstract_chunk.section_title == "Abstract"
    assert abstract_chunk.paragraph_indices == ()
    assert abstract_chunk.text == abstract
    assert abstract_chunk.chunk_id == "train:1000.00001:abs:c0000"


def test_abstract_chunks_precede_section_chunks():
    abstract = make_words_paragraph(20, "abs")
    section = Section(section_index=0, title="Intro", paragraphs=(make_words_paragraph(30),))
    paper = make_paper(abstract=abstract, sections=(section,))

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 2
    assert chunks[0].section_title == "Abstract"
    assert chunks[1].section_index == 0


def test_oversized_abstract_is_split_on_sentence_boundaries():
    # 10 sentences of 80 words each = 800 tokens, over the 600-token
    # oversized threshold -- same rule as an oversized section paragraph.
    abstract = make_sentences_paragraph([80] * 10)
    assert TOKENIZER.count_tokens(abstract) > OVERSIZED_PARAGRAPH_TOKENS
    paper = make_paper(abstract=abstract, sections=())

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) > 1
    assert all(c.section_index is None and c.section_title == "Abstract" for c in chunks)
    assert all(c.paragraph_indices == () for c in chunks)
    # No text lost: every sentence label appears in some chunk.
    for i in range(10):
        assert any(f"Sentence{i} " in c.text for c in chunks)


def test_non_oversized_abstract_is_not_split_even_if_over_target():
    # 450 tokens: over the 300-token target but under the 600-token
    # oversized threshold -- must remain a single chunk (no hard ceiling).
    abstract = make_words_paragraph(450, "abs")
    paper = make_paper(abstract=abstract, sections=())

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    assert chunks[0].token_count == 450


def test_paper_with_no_sections_and_an_abstract_yields_only_abstract_chunks():
    abstract = make_words_paragraph(50, "abs")
    paper = make_paper(abstract=abstract, sections=())

    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    assert chunks[0].section_title == "Abstract"
