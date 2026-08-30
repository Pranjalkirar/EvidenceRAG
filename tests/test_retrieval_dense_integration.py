"""Real-model integration test for dense retrieval with the actual
Qwen3-Embedding-0.6B model.

Deliberately separate from tests/test_retrieval_dense.py (which uses
FakeEmbedder and must always run, fast, with no network/model). This
test is automatically skipped -- not failed -- if `sentence-transformers`
isn't installed, or if the model can't actually be loaded (no network
access, no cached weights, etc.), so ordinary unit-test runs never
require downloading anything.
"""

from __future__ import annotations

import pytest

from evidencerag.retrieval.dense import DenseRetriever
from tests.retrieval_fixtures import make_chunk

CHUNKS = [
    make_chunk(chunk_id="train:p:s000:c0000", text="the cat sat on the mat"),
    make_chunk(chunk_id="train:p:s000:c0001", text="dogs are loyal animals"),
]


@pytest.fixture(scope="module")
def qwen_embedder():
    try:
        from evidencerag.retrieval.embeddings import QwenEmbedder

        return QwenEmbedder()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qwen3-Embedding-0.6B unavailable in this environment: {exc}")


def test_real_qwen_embedder_reports_a_positive_dimension(qwen_embedder):
    assert qwen_embedder.dimension > 0


def test_real_dense_retrieval_ranks_lexically_related_chunk_first(qwen_embedder):
    retriever = DenseRetriever.build(CHUNKS, embedder=qwen_embedder)
    results = retriever.retrieve("What animal sat on the mat?", top_k=1)
    assert results[0].chunk_id == "train:p:s000:c0000"
