"""Tests for evidencerag.langchain_impl.reranking.rerank_documents."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from evidencerag.langchain_impl.documents import chunk_to_document, document_chunk_id  # noqa: E402
from evidencerag.langchain_impl.reranking import rerank_documents  # noqa: E402
from tests.reranking_fixtures import ConstantReranker  # noqa: E402
from tests.retrieval_fixtures import make_chunk  # noqa: E402


def _documents(*chunk_ids_and_texts: tuple[str, str]):
    return [chunk_to_document(make_chunk(chunk_id=cid, text=text)) for cid, text in chunk_ids_and_texts]


def test_rerank_orders_by_score_descending():
    documents = _documents(("c1", "low"), ("c2", "high"), ("c3", "mid"))
    reranker = ConstantReranker({"low": 0.1, "high": 0.9, "mid": 0.5})

    reranked = rerank_documents("query", documents, reranker, top_n=3)

    assert [document_chunk_id(d) for d in reranked] == ["c2", "c3", "c1"]


def test_rerank_truncates_to_top_n():
    documents = _documents(("c1", "a"), ("c2", "b"), ("c3", "c"))
    reranker = ConstantReranker({"a": 0.1, "b": 0.2, "c": 0.3})

    reranked = rerank_documents("query", documents, reranker, top_n=2)

    assert len(reranked) == 2
    assert [document_chunk_id(d) for d in reranked] == ["c3", "c2"]


def test_rerank_breaks_ties_by_chunk_id():
    documents = _documents(("c2", "same"), ("c1", "same"))
    reranker = ConstantReranker({"same": 0.5})

    reranked = rerank_documents("query", documents, reranker, top_n=2)

    assert [document_chunk_id(d) for d in reranked] == ["c1", "c2"]


def test_rerank_adds_score_and_rank_metadata():
    documents = _documents(("c1", "only"))
    reranker = ConstantReranker({"only": 0.42})

    [reranked] = rerank_documents("query", documents, reranker, top_n=1)

    assert reranked.metadata["rerank_score"] == pytest.approx(0.42)
    assert reranked.metadata["rerank_rank"] == 1


def test_rerank_empty_candidates_returns_empty():
    reranker = ConstantReranker({})
    assert rerank_documents("query", [], reranker, top_n=5) == []


def test_rerank_passes_query_and_page_content_as_pairs():
    documents = _documents(("c1", "passage text"))
    reranker = ConstantReranker({"passage text": 1.0})

    rerank_documents("my question", documents, reranker, top_n=1)

    assert reranker.calls == [[("my question", "passage text")]]
