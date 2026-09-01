"""Reranks LangChain `Document`s with our own `Reranker`.

LangChain does offer a compressor abstraction for this
(`ContextualCompressionRetriever` + a `BaseDocumentCompressor`), but
its exact import path and constructor signature have moved between
`langchain_core.documents.compressor.BaseDocumentCompressor` and
`langchain.retrievers.document_compressors.base.BaseDocumentCompressor`
across recent LangChain releases, and it does not offer meaningfully
different composition behavior over calling a reranker function
directly -- see `retrievers.py`'s `HybridRerankRetriever`, a plain
`BaseRetriever` subclass, for where this is actually plugged in.

This module reuses `evidencerag.reranking.reranker.Reranker` (the same
Protocol, and -- in the M8 runner -- the literal same
`CrossEncoderReranker` instance) unchanged, exactly like
`evidencerag.langchain_impl.embeddings.EvidenceRAGEmbeddings` reuses
`Embedder` -- no reranking model is loaded twice.
"""

from __future__ import annotations

from typing import Sequence

from evidencerag.langchain_impl import require_langchain

require_langchain()

from langchain_core.documents import Document  # noqa: E402

from evidencerag.langchain_impl.documents import document_chunk_id  # noqa: E402
from evidencerag.reranking.reranker import Reranker  # noqa: E402


def rerank_documents(query: str, documents: Sequence[Document], reranker: Reranker, top_n: int = 5) -> list[Document]:
    """Rerank `documents` with `reranker`, returning the top `top_n`,
    best-first.

    Deterministic ordering: rank by reranker score descending, then by
    `chunk_id` ascending to break ties -- the SAME tie-break convention
    as `evidencerag.reranking.rerank.rerank()`, so identical candidate
    sets scored by the identical model produce identical orderings in
    both the custom and LangChain pipelines.

    Each returned `Document`'s metadata gains `rerank_score` (the raw
    reranker score) and `rerank_rank` (1-based final rank) -- useful
    for M8 debugging/inspection, never used as the document's identity
    (that is still `chunk_id`, unchanged).
    """
    documents = list(documents)
    if not documents:
        return []

    pairs = [(query, document.page_content) for document in documents]
    scores = reranker.score(pairs)

    order = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), document_chunk_id(documents[i])))
    top = order[:top_n]

    reranked: list[Document] = []
    for rank, i in enumerate(top, start=1):
        document = documents[i]
        metadata = dict(document.metadata)
        metadata["rerank_score"] = float(scores[i])
        metadata["rerank_rank"] = rank
        reranked.append(Document(page_content=document.page_content, metadata=metadata))
    return reranked
