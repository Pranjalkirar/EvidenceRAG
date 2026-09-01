"""Tests for evidencerag.langchain_impl.retrievers.

Requires the full `langchain` extra (langchain-core, langchain,
langchain-community) AND `faiss` (already a required, non-optional M4
dependency -- see requirements.txt) since `build_dense_retriever` uses
a real FAISS index (with a `FakeEmbedder`, never a real model).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_community")
langchain = pytest.importorskip("langchain")

from evidencerag.langchain_impl.documents import document_chunk_id  # noqa: E402
from evidencerag.langchain_impl.retrievers import (  # noqa: E402
    HybridRerankRetriever,
    build_bm25_retriever,
    build_dense_retriever,
    build_hybrid_retriever,
)
from tests.reranking_fixtures import ConstantReranker  # noqa: E402
from tests.retrieval_fixtures import FakeEmbedder, make_chunk  # noqa: E402


def _chunks():
    return [
        make_chunk(chunk_id="c1", text="Transformers are effective for natural language processing tasks."),
        make_chunk(chunk_id="c2", text="Convolutional networks are common in computer vision."),
        make_chunk(chunk_id="c3", text="Recurrent networks were historically used for sequence modeling."),
    ]


def test_bm25_retriever_favors_lexical_overlap():
    retriever = build_bm25_retriever(_chunks(), k=3)
    results = retriever.invoke("transformers natural language processing")
    assert document_chunk_id(results[0]) == "c1"


def test_bm25_retriever_respects_k():
    retriever = build_bm25_retriever(_chunks(), k=1)
    results = retriever.invoke("networks")
    assert len(results) == 1


def test_dense_retriever_returns_k_documents():
    retriever = build_dense_retriever(_chunks(), embedder=FakeEmbedder(dimension=16), k=2)
    results = retriever.invoke("some query")
    assert len(results) == 2
    for document in results:
        assert document_chunk_id(document) in {"c1", "c2", "c3"}


def test_hybrid_retriever_fuses_and_truncates():
    chunks = _chunks()
    bm25 = build_bm25_retriever(chunks, k=3)
    dense = build_dense_retriever(chunks, embedder=FakeEmbedder(dimension=16), k=3)
    hybrid = build_hybrid_retriever(bm25, dense, k=2)

    results = hybrid.invoke("transformers language processing")
    assert len(results) <= 2
    assert all(document_chunk_id(d) in {"c1", "c2", "c3"} for d in results)


def test_hybrid_rerank_retriever_returns_top_n_by_reranker_score():
    chunks = _chunks()
    bm25 = build_bm25_retriever(chunks, k=3)
    dense = build_dense_retriever(chunks, embedder=FakeEmbedder(dimension=16), k=3)
    hybrid = build_hybrid_retriever(bm25, dense, k=3)

    reranker = ConstantReranker(
        {
            "Transformers are effective for natural language processing tasks.": 0.1,
            "Convolutional networks are common in computer vision.": 0.9,
            "Recurrent networks were historically used for sequence modeling.": 0.5,
        }
    )
    hybrid_rerank = HybridRerankRetriever(base_retriever=hybrid, reranker=reranker, candidate_k=3, top_n=2)

    results = hybrid_rerank.invoke("query")

    assert len(results) == 2
    assert document_chunk_id(results[0]) == "c2"
    assert document_chunk_id(results[1]) == "c3"
