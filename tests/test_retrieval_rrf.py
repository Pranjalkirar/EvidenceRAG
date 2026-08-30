from evidencerag.retrieval.rrf import reciprocal_rank_fusion
from evidencerag.retrieval.schema import RetrievalResult


def _result(chunk_id: str, rank: int, retriever: str, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, rank=rank, retriever=retriever)


def test_overlapping_document_gets_contributions_from_both_rankings():
    # BM25: A, B, C   Dense: C, A, D  (from the M4 spec's own example)
    bm25 = [_result("A", 1, "bm25"), _result("B", 2, "bm25"), _result("C", 3, "bm25")]
    dense = [_result("C", 1, "dense"), _result("A", 2, "dense"), _result("D", 3, "dense")]

    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    scores = {r.chunk_id: r.score for r in fused}

    assert scores["A"] == 1 / 61 + 1 / 62  # rank 1 in bm25, rank 2 in dense
    assert scores["C"] == 1 / 63 + 1 / 61  # rank 3 in bm25, rank 1 in dense
    assert scores["B"] == 1 / 62  # only in bm25
    assert scores["D"] == 1 / 63  # only in dense


def test_fused_ranking_is_sorted_by_descending_score():
    bm25 = [_result("A", 1, "bm25"), _result("B", 2, "bm25"), _result("C", 3, "bm25")]
    dense = [_result("C", 1, "dense"), _result("A", 2, "dense"), _result("D", 3, "dense")]
    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in fused] == list(range(1, len(fused) + 1))


def test_one_sided_result_remains_eligible():
    bm25 = [_result("A", 1, "bm25")]
    dense: list[RetrievalResult] = []
    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    assert len(fused) == 1
    assert fused[0].chunk_id == "A"


def test_different_list_lengths_handled():
    bm25 = [_result("A", 1, "bm25"), _result("B", 2, "bm25"), _result("C", 3, "bm25"), _result("D", 4, "bm25")]
    dense = [_result("A", 1, "dense")]
    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    assert {r.chunk_id for r in fused} == {"A", "B", "C", "D"}
    assert fused[0].chunk_id == "A"  # top of both


def test_ties_produce_deterministic_ordering_by_chunk_id():
    bm25 = [_result("B", 1, "bm25"), _result("A", 2, "bm25")]
    dense = [_result("A", 1, "dense"), _result("B", 2, "dense")]
    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    # Both get 1/(k+1) + 1/(k+2) -- exact tie -- so chunk_id breaks it.
    assert fused[0].score == fused[1].score
    assert [r.chunk_id for r in fused] == ["A", "B"]


def test_duplicate_chunk_id_within_a_single_ranking_is_not_double_counted():
    bm25 = [_result("A", 1, "bm25"), _result("A", 2, "bm25")]  # accidental duplicate
    dense: list[RetrievalResult] = []
    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    assert len(fused) == 1
    assert fused[0].score == 1 / 61  # only the first occurrence (rank 1) counted


def test_empty_ranking_handled_gracefully():
    fused = reciprocal_rank_fusion([[], []], rrf_k=60)
    assert fused == []


def test_no_rankings_at_all():
    fused = reciprocal_rank_fusion([], rrf_k=60)
    assert fused == []


def test_rrf_k_is_configurable_and_changes_the_fused_scores():
    bm25 = [_result("A", 1, "bm25")]
    dense: list[RetrievalResult] = []
    fused_small_k = reciprocal_rank_fusion([bm25, dense], rrf_k=1)
    fused_large_k = reciprocal_rank_fusion([bm25, dense], rrf_k=1000)
    assert fused_small_k[0].score != fused_large_k[0].score
    assert fused_small_k[0].score == 1 / 2
    assert fused_large_k[0].score == 1 / 1001


def test_fused_results_are_tagged_as_hybrid():
    bm25 = [_result("A", 1, "bm25")]
    dense = [_result("A", 1, "dense")]
    fused = reciprocal_rank_fusion([bm25, dense], rrf_k=60)
    assert fused[0].retriever == "hybrid"
