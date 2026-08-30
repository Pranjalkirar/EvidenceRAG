import pytest

from evidencerag.reranking.rerank import rerank
from evidencerag.retrieval.schema import RetrievalResult
from tests.reranking_fixtures import ConstantReranker, FakeReranker


def _candidate(chunk_id: str, rank: int, retriever: str = "hybrid", score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, rank=rank, retriever=retriever)


CANDIDATES = [
    _candidate("A", 1),
    _candidate("B", 2),
    _candidate("C", 3),
]
TEXT_BY_ID = {"A": "alpha text", "B": "beta text", "C": "gamma text"}


def test_pairs_are_constructed_from_query_and_candidate_chunk_text():
    reranker = FakeReranker()
    rerank("what is alpha?", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert reranker.calls == [[("what is alpha?", "alpha text"), ("what is alpha?", "beta text"), ("what is alpha?", "gamma text")]]


def test_scores_are_mapped_back_to_the_correct_chunk_id():
    scores = {"alpha text": 0.9, "beta text": 0.1, "gamma text": 0.5}
    reranker = ConstantReranker(scores)
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    result_scores = {r.chunk_id: r.score for r in results}
    # float32 round-trip through ConstantReranker/np.array, hence approx.
    assert result_scores == pytest.approx({"A": 0.9, "B": 0.1, "C": 0.5})


def test_candidates_are_sorted_by_descending_reranker_score():
    scores = {"alpha text": 0.1, "beta text": 0.9, "gamma text": 0.5}
    reranker = ConstantReranker(scores)
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert [r.chunk_id for r in results] == ["B", "C", "A"]


def test_ranks_are_reassigned_starting_at_one():
    scores = {"alpha text": 0.1, "beta text": 0.9, "gamma text": 0.5}
    reranker = ConstantReranker(scores)
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert [r.rank for r in results] == [1, 2, 3]


def test_top_k_is_respected():
    reranker = FakeReranker()
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


def test_fewer_candidates_than_top_k_returns_all_of_them():
    reranker = FakeReranker()
    results = rerank("q", CANDIDATES[:1], TEXT_BY_ID, reranker, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "A"


def test_empty_candidates_returns_empty_and_never_calls_the_reranker():
    reranker = FakeReranker()
    results = rerank("q", [], TEXT_BY_ID, reranker, top_k=5)
    assert results == []
    assert reranker.calls == []


def test_invalid_top_k_is_rejected():
    reranker = FakeReranker()
    with pytest.raises(ValueError):
        rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=0)
    with pytest.raises(ValueError):
        rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=-1)


def test_original_candidate_chunk_ids_are_preserved_exactly():
    reranker = FakeReranker()
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert {r.chunk_id for r in results} == {c.chunk_id for c in CANDIDATES}


def test_results_are_tagged_as_reranker():
    reranker = FakeReranker()
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert all(r.retriever == "reranker" for r in results)


def test_deterministic_across_repeated_calls():
    reranker = FakeReranker()
    first = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    second = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert first == second


def test_ties_broken_deterministically_by_chunk_id():
    scores = {"alpha text": 0.5, "beta text": 0.5, "gamma text": 0.5}
    reranker = ConstantReranker(scores)
    results = rerank("q", CANDIDATES, TEXT_BY_ID, reranker, top_k=5)
    assert [r.chunk_id for r in results] == ["A", "B", "C"]


def test_reranker_only_operates_on_the_supplied_candidate_list():
    # A chunk_id ("D") not among the candidates must never appear in the
    # output, even though its text is present in the lookup mapping --
    # the reranker must not silently widen the candidate set.
    text_by_id_with_extra = dict(TEXT_BY_ID, D="delta text, not a candidate")
    reranker = FakeReranker()
    results = rerank("q", CANDIDATES, text_by_id_with_extra, reranker, top_k=5)
    assert {r.chunk_id for r in results} == {"A", "B", "C"}
    all_scored_texts = {text for call in reranker.calls for _, text in call}
    assert "delta text, not a candidate" not in all_scored_texts


def test_missing_chunk_text_for_a_candidate_raises_keyerror():
    reranker = FakeReranker()
    incomplete_text_by_id = {"A": "alpha text", "B": "beta text"}  # missing "C"
    with pytest.raises(KeyError):
        rerank("q", CANDIDATES, incomplete_text_by_id, reranker, top_k=5)
