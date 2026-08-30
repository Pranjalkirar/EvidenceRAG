import numpy as np
import pytest

from evidencerag.retrieval.dense import DenseRetriever, _l2_normalize
from tests.retrieval_fixtures import FakeEmbedder, make_chunk

CHUNKS = [
    make_chunk(chunk_id="train:p:s000:c0000", text="alpha document about cats"),
    make_chunk(chunk_id="train:p:s000:c0001", text="beta document about dogs"),
    make_chunk(chunk_id="train:p:s001:c0000", text="gamma document about birds"),
]


def test_index_creation_holds_all_chunks():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    assert retriever.metadata.num_chunks == len(CHUNKS)


def test_l2_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    normalized = _l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms[:2], 1.0)
    assert np.allclose(normalized[2], 0.0)  # zero vector stays zero, no div-by-zero crash


def test_exact_text_match_retrieves_itself_as_top_result():
    embedder = FakeEmbedder()
    retriever = DenseRetriever.build(CHUNKS, embedder=embedder)
    # Querying with a chunk's own exact text must retrieve that exact
    # chunk as the top hit (identical text -> identical fake vector).
    results = retriever.retrieve("alpha document about cats", top_k=1)
    assert results[0].chunk_id == "train:p:s000:c0000"
    assert results[0].retriever == "dense"


def test_top_k_limits_result_count_and_ranks_are_sequential():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    results = retriever.retrieve("dogs", top_k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


def test_faiss_positions_never_leak_only_chunk_ids_do():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    results = retriever.retrieve("birds", top_k=3)
    returned_ids = {r.chunk_id for r in results}
    assert returned_ids <= {c.chunk_id for c in CHUNKS}
    for r in results:
        assert isinstance(r.chunk_id, str) and ":" in r.chunk_id


def test_deterministic_across_repeated_calls():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    first = retriever.retrieve("cats", top_k=3)
    second = retriever.retrieve("cats", top_k=3)
    assert first == second


def test_empty_corpus_returns_no_results():
    retriever = DenseRetriever.build([], embedder=FakeEmbedder())
    assert retriever.retrieve("anything", top_k=5) == []


def test_retrieve_without_embedder_raises():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    retriever_no_embedder = DenseRetriever(index=retriever._index, chunk_ids=retriever._chunk_ids, metadata=retriever.metadata, embedder=None)
    try:
        retriever_no_embedder.retrieve("cats", top_k=1)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# --- top_k validation (Fix 4) --------------------------------------------


def test_top_k_zero_raises():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        retriever.retrieve("cats", top_k=0)


def test_top_k_negative_raises():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        retriever.retrieve("cats", top_k=-1)


def test_top_k_larger_than_corpus_returns_all_available_results():
    retriever = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    results = retriever.retrieve("cats dogs birds", top_k=1000)
    assert len(results) == len(CHUNKS)
