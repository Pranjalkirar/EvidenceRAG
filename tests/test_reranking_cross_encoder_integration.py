"""Real-model integration test for reranking with the actual
`cross-encoder/ms-marco-MiniLM-L-6-v2` model.

Deliberately separate from tests/test_reranking_cross_encoder.py
(which uses a fake model and must always run, fast, with no
network/model). This test is automatically skipped -- not failed --
if `sentence-transformers` isn't installed, or if the model can't
actually be loaded (no network access, no cached weights, etc.), so
ordinary unit-test runs never require downloading anything. Once the
model IS available, genuine runtime errors are not swallowed.
"""

from __future__ import annotations

import pytest

from evidencerag.reranking.rerank import rerank
from evidencerag.reranking.reranker import CrossEncoderReranker

CANDIDATE_TEXT_BY_ID = {
    "on-topic": "The cat sat on the mat in the sun.",
    "off-topic": "Quarterly revenue grew by twelve percent year over year.",
}


@pytest.fixture(scope="module")
def cross_encoder_reranker():
    try:
        return CrossEncoderReranker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cross-encoder/ms-marco-MiniLM-L-6-v2 unavailable in this environment: {exc}")


def test_real_cross_encoder_scores_pairs_and_returns_one_float_each(cross_encoder_reranker):
    scores = cross_encoder_reranker.score(
        [("What animal sat on the mat?", text) for text in CANDIDATE_TEXT_BY_ID.values()]
    )
    assert scores.shape == (2,)


def test_real_cross_encoder_ranks_the_relevant_candidate_first(cross_encoder_reranker):
    from evidencerag.retrieval.schema import RetrievalResult

    candidates = [
        RetrievalResult(chunk_id="on-topic", score=0.0, rank=1, retriever="hybrid"),
        RetrievalResult(chunk_id="off-topic", score=0.0, rank=2, retriever="hybrid"),
    ]
    results = rerank(
        "What animal sat on the mat?", candidates, CANDIDATE_TEXT_BY_ID, cross_encoder_reranker, top_k=2
    )
    assert results[0].chunk_id == "on-topic"
