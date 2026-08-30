"""Cross-encoder reranking (Milestone 5).

Second-stage reranking applied on top of M4 hybrid retrieval results:

    evidencerag.retrieval.rrf.HybridRetriever
                    │  (top `candidate_depth` candidates)
                    ▼
    evidencerag.reranking.reranker.CrossEncoderReranker
    (cross-encoder/ms-marco-MiniLM-L-6-v2, via sentence-transformers)
                    │
                    ▼
        final top `top_k` RetrievalResult list

See `reranker.py` for the model abstraction (`Reranker` protocol +
`CrossEncoderReranker`) and `rerank.py` for the reranking pipeline
(`rerank()` and `RerankingRetriever`).
"""
