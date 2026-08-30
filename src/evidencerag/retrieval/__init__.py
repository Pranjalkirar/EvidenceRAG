"""Retrieval methods over M3 chunks: BM25 sparse retrieval, dense
retrieval (Qwen3-Embedding-0.6B + FAISS IndexFlatIP), and hybrid
retrieval via Reciprocal Rank Fusion (RRF).

- schema.py      -- RetrievalResult (chunk_id, score, rank, retriever)
- base.py        -- Retriever Protocol (structural, not inheritance)
- corpus.py      -- one canonical, deterministically ordered corpus
                     + fingerprint, shared by BM25 and Dense
- tokenize.py    -- lexical tokenization for BM25 (unrelated to M3's
                     token-COUNTING tokenizer in chunking/tokenizer.py)
- bm25.py        -- BM25Retriever (rank_bm25), build/save/load/retrieve
- embeddings.py  -- Embedder protocol + QwenEmbedder
                     (Qwen/Qwen3-Embedding-0.6B via sentence-transformers)
- dense.py       -- DenseRetriever (FAISS IndexFlatIP over L2-normalized
                     vectors), build/save/load/retrieve
- rrf.py         -- reciprocal_rank_fusion() + HybridRetriever

Reranking, generation, and evaluation are NOT implemented here or
anywhere else yet.
"""
