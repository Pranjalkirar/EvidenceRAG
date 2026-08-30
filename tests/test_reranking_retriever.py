import pytest

from evidencerag.reranking.rerank import RerankConfig, RerankingRetriever
from evidencerag.retrieval.schema import RetrievalResult
from tests.reranking_fixtures import ConstantReranker, FakeReranker
from tests.retrieval_fixtures import make_chunk


class RecordingBaseRetriever:
    """Fake `Retriever` (satisfies the `retrieve()` protocol) that
    records every call and returns a fixed, caller-supplied ranking --
    lets tests dictate exactly what M4 "would have" returned without
    building a real BM25/Dense/Hybrid index.
    """

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self._results[:top_k]


CHUNKS = [
    make_chunk(chunk_id="A", text="alpha text"),
    make_chunk(chunk_id="B", text="beta text"),
    make_chunk(chunk_id="C", text="gamma text"),
]
BASE_RESULTS = [
    RetrievalResult(chunk_id="A", score=0.9, rank=1, retriever="hybrid"),
    RetrievalResult(chunk_id="B", score=0.5, rank=2, retriever="hybrid"),
    RetrievalResult(chunk_id="C", score=0.1, rank=3, retriever="hybrid"),
]


def test_base_retriever_is_called_with_the_configured_candidate_depth():
    base = RecordingBaseRetriever(BASE_RESULTS)
    pipeline = RerankingRetriever(base, FakeReranker(), CHUNKS, config=RerankConfig(candidate_depth=20))
    pipeline.retrieve("q", top_k=5)
    assert base.calls == [("q", 20)]


def test_candidate_depth_is_never_smaller_than_the_requested_top_k():
    base = RecordingBaseRetriever(BASE_RESULTS)
    pipeline = RerankingRetriever(base, FakeReranker(), CHUNKS, config=RerankConfig(candidate_depth=2))
    pipeline.retrieve("q", top_k=5)
    assert base.calls == [("q", 5)]  # top_k(5) > candidate_depth(2), so 5 is used


def test_final_results_are_limited_to_the_requested_top_k():
    base = RecordingBaseRetriever(BASE_RESULTS)
    pipeline = RerankingRetriever(base, FakeReranker(), CHUNKS, config=RerankConfig(candidate_depth=20))
    results = pipeline.retrieve("q", top_k=2)
    assert len(results) == 2


def test_pipeline_reranks_only_the_base_retrievers_candidates():
    # Only A and B come back from the base retriever -- C must never
    # appear in the final results even though its chunk text is known.
    base = RecordingBaseRetriever(BASE_RESULTS[:2])
    scores = {"alpha text": 0.1, "beta text": 0.9, "gamma text": 1.0}
    pipeline = RerankingRetriever(base, ConstantReranker(scores), CHUNKS, config=RerankConfig(candidate_depth=20))
    results = pipeline.retrieve("q", top_k=5)
    assert {r.chunk_id for r in results} == {"A", "B"}


def test_final_results_are_reranked_not_just_passed_through():
    base = RecordingBaseRetriever(BASE_RESULTS)
    # Base order is A, B, C by hybrid score; reranker reverses it.
    scores = {"alpha text": 0.1, "beta text": 0.5, "gamma text": 0.9}
    pipeline = RerankingRetriever(base, ConstantReranker(scores), CHUNKS, config=RerankConfig(candidate_depth=20))
    results = pipeline.retrieve("q", top_k=5)
    assert [r.chunk_id for r in results] == ["C", "B", "A"]
    assert all(r.retriever == "reranker" for r in results)


def test_default_config_matches_m4_candidate_depth():
    assert RerankConfig().candidate_depth == 20


def test_invalid_top_k_is_rejected():
    base = RecordingBaseRetriever(BASE_RESULTS)
    pipeline = RerankingRetriever(base, FakeReranker(), CHUNKS)
    with pytest.raises(ValueError):
        pipeline.retrieve("q", top_k=0)


def test_empty_base_results_returns_empty_and_never_calls_the_reranker():
    base = RecordingBaseRetriever([])
    reranker = FakeReranker()
    pipeline = RerankingRetriever(base, reranker, CHUNKS)
    results = pipeline.retrieve("q", top_k=5)
    assert results == []
    assert reranker.calls == []
