"""Paragraph-aware, section-aware chunking of QASPER papers into
retrieval-ready `Chunk`s, with explicit evidence-to-chunk provenance.

- schema.py       -- immutable Chunk data model (tokenizer-agnostic)
- tokenizer.py     -- isolated token-counting (tiktoken, cl100k_base)
- sentence_split.py -- lightweight sentence-boundary splitting for
                        oversized paragraphs
- chunker.py         -- the chunking algorithm (chunk_paper)
- evidence_map.py     -- question -> answer -> evidence -> chunk(s) mapping
- serialize.py         -- JSON Lines save/load for chunks and evidence maps

Retrieval, embeddings, BM25, FAISS, reranking, and generation are NOT
implemented here or anywhere else yet.
"""
