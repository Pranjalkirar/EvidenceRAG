"""Builds the LangChain-side retrievers M8 compares against
`evidencerag.evaluation.systems.build_systems`'s custom BM25 / Dense /
Hybrid / Hybrid+Reranker, from the SAME M3 chunks, the SAME embedder,
and the SAME reranker.

Component choices (documented here because each is a deliberate
"does LangChain have a genuine equivalent?" decision, per the M8
spec):

  - BM25: `langchain_community.retrievers.BM25Retriever` -- a genuine
    LangChain component, itself built on `rank_bm25.BM25Okapi` (the
    SAME library `evidencerag.retrieval.bm25.BM25Retriever` uses).
    Constructed with `evidencerag.retrieval.tokenize.tokenize` as its
    `preprocess_func` and the SAME `k1`/`b` (via `bm25_params`, when
    the installed version supports it) so tokenization and
    hyperparameters match the custom BM25 index as closely as the two
    libraries' APIs allow -- only the indexing/scoring implementation
    itself differs.
  - Dense: LangChain's `Embeddings` interface (see `embeddings.py`) +
    `langchain_community.vectorstores.FAISS`, configured for
    inner-product search over L2-normalized vectors -- the same
    cosine-via-normalized-inner-product metric
    `evidencerag.retrieval.dense.DenseRetriever` uses, via the SAME
    underlying `faiss` library, just LangChain's `VectorStore` wrapper
    around it instead of a raw `faiss.IndexFlatIP`.
  - Hybrid: `langchain.retrievers.ensemble.EnsembleRetriever` -- a
    genuine LangChain component implementing Reciprocal Rank Fusion
    (the SAME algorithm as `evidencerag.retrieval.rrf`), with its `c`
    RRF-smoothing-constant parameter set from
    `evidencerag.config.SETTINGS.rrf_k`.
  - Hybrid+Reranker: LangChain does not ship an off-the-shelf
    retriever that reranks with an arbitrary pre-loaded, non-LangChain
    cross-encoder object without loading a second model -- see
    `reranking.py`'s module docstring for why this is a small adapter
    (`HybridRerankRetriever`, a plain `BaseRetriever` subclass) around
    the SAME `Reranker` instance, rather than a LangChain-native
    compressor pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from evidencerag.langchain_impl import require_langchain

require_langchain()

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_community.retrievers import BM25Retriever as LCBM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS as LCFAISS  # noqa: E402
from langchain_community.vectorstores.utils import DistanceStrategy  # noqa: E402

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover -- exercised only when langchain proper is absent
    EnsembleRetriever = None  # type: ignore[assignment,misc]

from evidencerag.chunking.schema import Chunk  # noqa: E402
from evidencerag.config import SETTINGS  # noqa: E402
from evidencerag.langchain_impl.documents import chunks_to_documents, document_chunk_id  # noqa: E402
from evidencerag.langchain_impl.embeddings import EvidenceRAGEmbeddings  # noqa: E402
from evidencerag.langchain_impl.reranking import rerank_documents  # noqa: E402
from evidencerag.retrieval.embeddings import Embedder  # noqa: E402
from evidencerag.retrieval.tokenize import tokenize  # noqa: E402


def build_bm25_retriever(chunks: Iterable[Chunk], k: int) -> LCBM25Retriever:
    """LangChain community BM25 retriever over `chunks`, tokenized and
    parameterized to match `evidencerag.retrieval.bm25.BM25Retriever`
    as closely as its constructor allows."""
    documents = chunks_to_documents(chunks)
    try:
        return LCBM25Retriever.from_documents(
            documents,
            k=k,
            preprocess_func=tokenize,
            bm25_params={"k1": SETTINGS.bm25_k1, "b": SETTINGS.bm25_b},
        )
    except TypeError:
        # Older langchain-community releases don't accept
        # `bm25_params` -- fall back to matching tokenization only,
        # rather than failing the whole pipeline over a hyperparameter
        # LangChain's own BM25Retriever otherwise defaults for us.
        return LCBM25Retriever.from_documents(documents, k=k, preprocess_func=tokenize)


def build_dense_retriever(chunks: Iterable[Chunk], embedder: Embedder, k: int) -> BaseRetriever:
    """LangChain FAISS vector store retriever over `chunks`, using
    `EvidenceRAGEmbeddings` (L2-normalized) and inner-product search --
    the same cosine-via-normalized-inner-product ranking as
    `evidencerag.retrieval.dense.DenseRetriever`."""
    documents = chunks_to_documents(chunks)
    embeddings = EvidenceRAGEmbeddings(embedder, normalize=True)
    store = LCFAISS.from_documents(documents, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    return store.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(bm25_retriever: BaseRetriever, dense_retriever: BaseRetriever, k: int) -> BaseRetriever:
    """`EnsembleRetriever` fusing `bm25_retriever` and `dense_retriever`
    by Reciprocal Rank Fusion, `c=SETTINGS.rrf_k` (the same smoothing
    constant `evidencerag.retrieval.rrf.RRFConfig.rrf_k` defaults to),
    weighted equally (matching `reciprocal_rank_fusion`'s unweighted
    sum over rankings). Truncated to `k` via `_TopKRetriever` below,
    since `EnsembleRetriever` itself returns its full fused list.
    """
    if EnsembleRetriever is None:
        raise ImportError(
            "langchain.retrievers.EnsembleRetriever is unavailable -- install the "
            "'langchain' package (not just langchain-core/langchain-community)."
        )
    try:
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5],
            c=SETTINGS.rrf_k,
            id_key="chunk_id",
        )
    except TypeError:
        # `id_key` (dedup-by-metadata-field) was added in a later
        # langchain release -- without it, EnsembleRetriever dedupes by
        # Document content/metadata equality instead, which is
        # equivalent here since every Document's content+metadata is
        # already unique per chunk_id.
        ensemble = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.5, 0.5], c=SETTINGS.rrf_k)
    return _TopKRetriever(inner=ensemble, k=k)


class _TopKRetriever(BaseRetriever):
    """Truncates a wrapped retriever's results to its first `k` --
    `EnsembleRetriever` has no `k`/`top_k` of its own (see
    `build_hybrid_retriever`); this keeps `HybridRerankRetriever` (and
    any direct caller) working with a bounded candidate list, matching
    `evidencerag.retrieval.rrf.HybridRetriever.retrieve(top_k=...)`'s
    own truncation.
    """

    inner: Any
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        results = self.inner.invoke(query) if hasattr(self.inner, "invoke") else self.inner.get_relevant_documents(query)
        return list(results)[: self.k]


class HybridRerankRetriever(BaseRetriever):
    """M8's LangChain equivalent of
    `evidencerag.reranking.rerank.RerankingRetriever`: wraps a base
    retriever (typically `build_hybrid_retriever`'s ensemble) unchanged
    and reranks only the candidates it returns, via
    `evidencerag.langchain_impl.reranking.rerank_documents` and the
    SAME `Reranker` instance the custom pipeline uses.

    A plain `BaseRetriever` subclass rather than a
    `ContextualCompressionRetriever` + `BaseDocumentCompressor` pair --
    see `reranking.py`'s module docstring for why.
    """

    base_retriever: Any
    reranker: Any
    candidate_k: int = 20
    top_n: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        candidates = (
            self.base_retriever.invoke(query)
            if hasattr(self.base_retriever, "invoke")
            else self.base_retriever.get_relevant_documents(query)
        )
        return rerank_documents(query, list(candidates)[: self.candidate_k], self.reranker, top_n=self.top_n)


__all__ = [
    "build_bm25_retriever",
    "build_dense_retriever",
    "build_hybrid_retriever",
    "HybridRerankRetriever",
    "document_chunk_id",
]
