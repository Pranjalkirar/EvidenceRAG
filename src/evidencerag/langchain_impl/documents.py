"""Conversion between M3 `Chunk`s and LangChain `Document`s.

`chunk_id` is the canonical document identity everywhere else in this
project (see `evidencerag.retrieval.schema`) -- that convention
continues here: every `Document` this module produces carries its
source chunk's `chunk_id` in `metadata["chunk_id"]`, and
`document_chunk_id()` is the one place that reads it back out, so a
LangChain retriever's output can always be translated back to the same
`chunk_id`-keyed world the custom pipeline and M7 metrics use.

This module does not re-derive, rewrite, or truncate chunk text --
`Document.page_content` is exactly `Chunk.text`, unchanged.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from evidencerag.langchain_impl import require_langchain

require_langchain()

from langchain_core.documents import Document  # noqa: E402

from evidencerag.chunking.schema import Chunk  # noqa: E402

CHUNK_ID_METADATA_KEY = "chunk_id"


def chunk_to_document(chunk: Chunk) -> Document:
    """One M3 `Chunk` -> one LangChain `Document`, `page_content` =
    `chunk.text` unchanged, full M3 provenance preserved in metadata
    for debugging (never used to reconstruct identity -- `chunk_id`
    alone is the identity)."""
    return Document(
        page_content=chunk.text,
        metadata={
            CHUNK_ID_METADATA_KEY: chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "split": chunk.split,
            "section_index": chunk.section_index,
            "section_title": chunk.section_title,
            "paragraph_indices": list(chunk.paragraph_indices),
            "token_count": chunk.token_count,
        },
    )


def chunks_to_documents(chunks: Iterable[Chunk]) -> list[Document]:
    """`chunk_to_document` over a collection, preserving input order
    (callers that need a specific deterministic order, e.g. M4's
    corpus ordering, must sort `chunks` themselves before calling
    this -- this function performs no reordering of its own, matching
    the "caller decides order" convention already used by
    `evidencerag.generation.prompt.build_prompt`)."""
    return [chunk_to_document(chunk) for chunk in chunks]


def document_chunk_id(document: Document) -> str:
    """The `chunk_id` a `Document` was built from. Raises `ValueError`
    if the document has no `chunk_id` metadata -- e.g. it did not come
    from `chunk_to_document()` -- rather than silently returning
    `None` or an empty string, since a caller treating a missing
    identity as a valid `chunk_id` would silently corrupt every
    downstream `RetrievalResult`/`EvalRecord`.
    """
    chunk_id = document.metadata.get(CHUNK_ID_METADATA_KEY)
    if not chunk_id:
        raise ValueError(
            "Document is missing 'chunk_id' metadata -- was it built by "
            "evidencerag.langchain_impl.documents.chunk_to_document()?"
        )
    return chunk_id


def build_chunk_lookup(chunks: Iterable[Chunk]) -> Mapping[str, Chunk]:
    """`{chunk_id: Chunk}` for every chunk in `chunks` -- a small
    convenience shared by the LangChain retriever builders (see
    `retrievers.py`), which each need to look chunk text back up by
    `chunk_id` (for BM25/rerank scoring) without depending on a
    specific `Document` having survived the whole pipeline unchanged.
    """
    return {chunk.chunk_id: chunk for chunk in chunks}
