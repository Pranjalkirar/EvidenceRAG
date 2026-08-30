import pytest

from evidencerag.retrieval.bm25 import BM25Retriever
from evidencerag.retrieval.dense import DenseRetriever
from evidencerag.retrieval.rrf import HybridRetriever, RRFConfig
from tests.retrieval_fixtures import FakeEmbedder, make_chunk

CHUNKS = [
    make_chunk(chunk_id="train:p:s000:c0000", text="the cat sat on the mat"),
    make_chunk(chunk_id="train:p:s000:c0001", text="dogs are loyal animals"),
    make_chunk(chunk_id="train:p:s001:c0000", text="cats and dogs can be friends"),
    make_chunk(chunk_id="train:p:s001:c0001", text="birds fly south for winter"),
]


def _build_hybrid(config: RRFConfig | None = None) -> HybridRetriever:
    bm25 = BM25Retriever.build(CHUNKS)
    dense = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    return HybridRetriever(bm25, dense, config=config)


def test_hybrid_returns_hybrid_tagged_results():
    hybrid = _build_hybrid()
    results = hybrid.retrieve("cats and dogs", top_k=3)
    assert all(r.retriever == "hybrid" for r in results)
    assert len(results) <= 3


def test_hybrid_result_chunk_ids_are_canonical_m3_ids():
    hybrid = _build_hybrid()
    results = hybrid.retrieve("cats", top_k=4)
    valid_ids = {c.chunk_id for c in CHUNKS}
    assert all(r.chunk_id in valid_ids for r in results)


def test_candidate_depth_can_exceed_final_top_k():
    # candidate_depth=20 (larger than the 4-chunk corpus) with a small
    # final top_k must not error, and must still return <= top_k results.
    hybrid = _build_hybrid(RRFConfig(rrf_k=60, candidate_depth=20))
    results = hybrid.retrieve("cats dogs birds", top_k=2)
    assert len(results) <= 2


def test_deterministic_across_repeated_calls():
    hybrid = _build_hybrid()
    first = hybrid.retrieve("cats dogs", top_k=3)
    second = hybrid.retrieve("cats dogs", top_k=3)
    assert first == second


# --- Corpus consistency (Fix 1) -----------------------------------------


def test_hybrid_construction_succeeds_when_fingerprints_match():
    bm25 = BM25Retriever.build(CHUNKS)
    dense = DenseRetriever.build(CHUNKS, embedder=FakeEmbedder())
    HybridRetriever(bm25, dense)  # must not raise


def test_hybrid_construction_raises_on_fingerprint_mismatch():
    bm25 = BM25Retriever.build(CHUNKS)
    different_chunks = [make_chunk(chunk_id="train:p:s000:c0000", text="a totally different corpus")]
    dense = DenseRetriever.build(different_chunks, embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        HybridRetriever(bm25, dense)


# --- top_k validation (Fix 4) --------------------------------------------


def test_hybrid_top_k_zero_raises():
    hybrid = _build_hybrid()
    with pytest.raises(ValueError):
        hybrid.retrieve("cats", top_k=0)


def test_hybrid_top_k_negative_raises():
    hybrid = _build_hybrid()
    with pytest.raises(ValueError):
        hybrid.retrieve("cats", top_k=-1)


def test_hybrid_top_k_larger_than_corpus_returns_available_results():
    hybrid = _build_hybrid()
    results = hybrid.retrieve("cats dogs birds", top_k=1000)
    assert 0 < len(results) <= len(CHUNKS)
