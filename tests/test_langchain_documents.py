"""Tests for evidencerag.langchain_impl.documents.

Skips the whole module (via pytest.importorskip) when the optional
`langchain` extra isn't installed -- see evidencerag.langchain_impl's
module docstring / require_langchain().
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from evidencerag.langchain_impl.documents import (  # noqa: E402
    build_chunk_lookup,
    chunk_to_document,
    chunks_to_documents,
    document_chunk_id,
)
from tests.retrieval_fixtures import make_chunk  # noqa: E402


def test_chunk_to_document_preserves_text_and_chunk_id():
    chunk = make_chunk(chunk_id="c1", text="Some paragraph text.", section_index=2, paragraph_indices=(3, 4))
    document = chunk_to_document(chunk)

    assert document.page_content == "Some paragraph text."
    assert document.metadata["chunk_id"] == "c1"
    assert document.metadata["section_index"] == 2
    assert document.metadata["paragraph_indices"] == [3, 4]


def test_document_chunk_id_round_trips():
    chunk = make_chunk(chunk_id="c42", text="text")
    document = chunk_to_document(chunk)
    assert document_chunk_id(document) == "c42"


def test_document_chunk_id_raises_on_missing_metadata():
    from langchain_core.documents import Document

    bare_document = Document(page_content="no chunk_id here")
    with pytest.raises(ValueError):
        document_chunk_id(bare_document)


def test_chunks_to_documents_preserves_order():
    chunks = [make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(3)]
    documents = chunks_to_documents(chunks)
    assert [document_chunk_id(d) for d in documents] == ["c0", "c1", "c2"]


def test_build_chunk_lookup_keys_by_chunk_id():
    chunks = [make_chunk(chunk_id="a", text="A"), make_chunk(chunk_id="b", text="B")]
    lookup = build_chunk_lookup(chunks)
    assert lookup["a"].text == "A"
    assert lookup["b"].text == "B"
    assert len(lookup) == 2
