"""Tests for evidencerag.langchain_impl.embeddings.EvidenceRAGEmbeddings."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("langchain_core")

from evidencerag.langchain_impl.embeddings import EvidenceRAGEmbeddings  # noqa: E402
from tests.retrieval_fixtures import FakeEmbedder  # noqa: E402


def test_embed_documents_normalizes_and_matches_dimension():
    embedder = EvidenceRAGEmbeddings(FakeEmbedder(dimension=8), normalize=True)
    vectors = embedder.embed_documents(["hello world", "goodbye"])

    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == 8
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-5


def test_embed_query_matches_embed_documents_for_same_text():
    embedder = EvidenceRAGEmbeddings(FakeEmbedder(dimension=8), normalize=True)
    query_vector = embedder.embed_query("same text")
    [doc_vector] = embedder.embed_documents(["same text"])
    assert query_vector == pytest.approx(doc_vector)


def test_normalize_false_skips_normalization():
    raw_embedder = FakeEmbedder(dimension=8)
    embedder = EvidenceRAGEmbeddings(raw_embedder, normalize=False)
    [vector] = embedder.embed_documents(["some text"])
    [raw_vector] = raw_embedder.embed_documents(["some text"])
    assert vector == pytest.approx(list(raw_vector), abs=1e-5)


def test_model_name_delegates_to_wrapped_embedder():
    embedder = EvidenceRAGEmbeddings(FakeEmbedder(model_name="my-fake-model"))
    assert embedder.model_name == "my-fake-model"


def test_different_texts_produce_different_vectors():
    embedder = EvidenceRAGEmbeddings(FakeEmbedder(dimension=8))
    a = embedder.embed_query("alpha")
    b = embedder.embed_query("beta")
    assert a != b
