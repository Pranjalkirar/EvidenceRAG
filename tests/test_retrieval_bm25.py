import pytest

from evidencerag.retrieval.bm25 import BM25Config, BM25Retriever
from tests.retrieval_fixtures import make_chunk

CHUNKS = [
    make_chunk(chunk_id="train:p:s000:c0000", text="the cat sat on the mat"),
    make_chunk(chunk_id="train:p:s000:c0001", text="dogs are loyal animals"),
    make_chunk(chunk_id="train:p:s001:c0000", text="cats and dogs can be friends"),
]


def test_lexical_relevance_ranks_matching_chunk_first():
    bm25 = BM25Retriever.build(CHUNKS)
    results = bm25.retrieve("cat", top_k=3)
    assert results[0].chunk_id == "train:p:s000:c0000"
    assert all(r.retriever == "bm25" for r in results)


def test_top_k_limits_result_count():
    bm25 = BM25Retriever.build(CHUNKS)
    results = bm25.retrieve("dogs", top_k=1)
    assert len(results) == 1


def test_chunk_id_is_preserved_not_array_position():
    bm25 = BM25Retriever.build(CHUNKS)
    results = bm25.retrieve("cats dogs friends", top_k=3)
    returned_ids = {r.chunk_id for r in results}
    assert returned_ids <= {c.chunk_id for c in CHUNKS}
    assert all(isinstance(r.chunk_id, str) for r in results)


def test_ranks_are_1_indexed_and_sequential():
    bm25 = BM25Retriever.build(CHUNKS)
    results = bm25.retrieve("dogs", top_k=3)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_deterministic_ordering_across_repeated_calls():
    bm25 = BM25Retriever.build(CHUNKS)
    first = bm25.retrieve("cats dogs", top_k=3)
    second = bm25.retrieve("cats dogs", top_k=3)
    assert first == second


def test_ties_broken_deterministically_by_chunk_id():
    # Two chunks with identical text score identically -- the lower
    # chunk_id must consistently come first.
    tied = [
        make_chunk(chunk_id="train:p:s000:c0001", text="identical text here"),
        make_chunk(chunk_id="train:p:s000:c0000", text="identical text here"),
    ]
    bm25 = BM25Retriever.build(tied)
    results = bm25.retrieve("identical text", top_k=2)
    assert [r.chunk_id for r in results] == ["train:p:s000:c0000", "train:p:s000:c0001"]


def test_empty_query_does_not_crash():
    bm25 = BM25Retriever.build(CHUNKS)
    results = bm25.retrieve("", top_k=3)
    assert isinstance(results, list)


def test_empty_corpus_returns_no_results():
    bm25 = BM25Retriever.build([])
    assert bm25.retrieve("anything", top_k=5) == []


def test_config_k1_and_b_are_applied():
    bm25 = BM25Retriever.build(CHUNKS, config=BM25Config(k1=2.0, b=0.5))
    assert bm25.config.k1 == 2.0
    assert bm25.config.b == 0.5


# --- top_k validation (Fix 4) --------------------------------------------


def test_top_k_zero_raises():
    bm25 = BM25Retriever.build(CHUNKS)
    with pytest.raises(ValueError):
        bm25.retrieve("cat", top_k=0)


def test_top_k_negative_raises():
    bm25 = BM25Retriever.build(CHUNKS)
    with pytest.raises(ValueError):
        bm25.retrieve("cat", top_k=-1)


def test_top_k_larger_than_corpus_returns_all_available_results():
    bm25 = BM25Retriever.build(CHUNKS)
    results = bm25.retrieve("cat dog", top_k=1000)
    assert len(results) == len(CHUNKS)
