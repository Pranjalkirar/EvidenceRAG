import pytest

from evidencerag.retrieval.corpus import assert_matches_corpus, build_corpus
from tests.retrieval_fixtures import make_chunk


def test_corpus_is_ordered_deterministically_by_chunk_id_regardless_of_input_order():
    a = make_chunk(chunk_id="train:p:s000:c0000", text="alpha")
    b = make_chunk(chunk_id="train:p:s000:c0001", text="beta")
    c = make_chunk(chunk_id="train:p:s001:c0000", text="gamma")

    corpus1 = build_corpus([c, a, b])
    corpus2 = build_corpus([b, c, a])

    assert corpus1.chunk_ids == corpus2.chunk_ids == (a.chunk_id, b.chunk_id, c.chunk_id)
    assert corpus1.texts == corpus2.texts == ("alpha", "beta", "gamma")
    assert corpus1.fingerprint == corpus2.fingerprint


def test_fingerprint_changes_if_a_chunk_id_changes():
    a = make_chunk(chunk_id="train:p:s000:c0000", text="alpha")
    b = make_chunk(chunk_id="train:p:s000:c0001", text="beta")
    corpus1 = build_corpus([a, b])

    b_renamed = make_chunk(chunk_id="train:p:s000:c0002", text="beta")
    corpus2 = build_corpus([a, b_renamed])

    assert corpus1.fingerprint != corpus2.fingerprint


def test_fingerprint_changes_if_text_changes_but_chunk_id_does_not():
    a = make_chunk(chunk_id="train:p:s000:c0000", text="alpha")
    corpus1 = build_corpus([a])

    a_edited = make_chunk(chunk_id="train:p:s000:c0000", text="alpha, edited")
    corpus2 = build_corpus([a_edited])

    assert corpus1.fingerprint != corpus2.fingerprint


def test_duplicate_chunk_id_raises():
    a = make_chunk(chunk_id="train:p:s000:c0000", text="alpha")
    a_dupe = make_chunk(chunk_id="train:p:s000:c0000", text="a different text but same id")

    with pytest.raises(ValueError):
        build_corpus([a, a_dupe])


def test_assert_matches_corpus_passes_for_the_same_chunks():
    chunks = [make_chunk(chunk_id="train:p:s000:c0000", text="alpha")]
    fingerprint = build_corpus(chunks).fingerprint
    assert_matches_corpus(fingerprint, chunks)  # must not raise


def test_assert_matches_corpus_raises_for_a_different_corpus():
    original = [make_chunk(chunk_id="train:p:s000:c0000", text="alpha")]
    fingerprint = build_corpus(original).fingerprint

    different = [make_chunk(chunk_id="train:p:s000:c0000", text="a completely different chunk")]
    with pytest.raises(ValueError):
        assert_matches_corpus(fingerprint, different)


def test_empty_corpus():
    corpus = build_corpus([])
    assert corpus.chunk_ids == ()
    assert corpus.texts == ()
    assert corpus.fingerprint  # still a valid (non-empty) hash string
