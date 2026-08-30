from evidencerag.retrieval.schema import RetrievalResult


def test_retrieval_result_holds_chunk_id_score_rank_retriever():
    result = RetrievalResult(chunk_id="train:1000.00001:s000:c0000", score=1.23, rank=1, retriever="bm25")
    assert result.chunk_id == "train:1000.00001:s000:c0000"
    assert result.score == 1.23
    assert result.rank == 1
    assert result.retriever == "bm25"


def test_retrieval_result_is_immutable():
    result = RetrievalResult(chunk_id="a", score=1.0, rank=1, retriever="dense")
    try:
        result.rank = 2  # type: ignore[misc]
        assert False, "expected FrozenInstanceError"
    except Exception:
        pass
