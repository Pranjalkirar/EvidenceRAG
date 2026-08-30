#!/usr/bin/env python
"""Smoke test for M5 reranking: build M4 Hybrid retrieval over M3
chunks for one split, get its top-`candidate-depth` candidates for one
query, rerank them with the cross-encoder, and print both rankings
side by side for manual inspection.

This does not use the question's paper_id (search runs over the whole
corpus) and does not use QASPER gold evidence (that belongs to a later
evaluation milestone) -- it only demonstrates the M5 flow end to end:
M4 HybridRetriever -> CrossEncoderReranker -> ranked chunk_ids.

Requires the Qwen3-Embedding-0.6B and cross-encoder/ms-marco-MiniLM-L-6-v2
models (downloaded via sentence-transformers on first run) and
`scripts/chunk_qasper.py` having already been run for the requested split.

Usage:
    python scripts/rerank_smoke_test.py --query "How is the model evaluated?"
    python scripts/rerank_smoke_test.py --split validation --max-chunks 500 \
        --query "..." --show-text
"""

from __future__ import annotations

import argparse
import sys

from evidencerag.chunking.serialize import load_chunks
from evidencerag.config import PATHS, SETTINGS
from evidencerag.reranking.rerank import RerankConfig, RerankingRetriever
from evidencerag.reranking.reranker import CrossEncoderReranker
from evidencerag.retrieval.bm25 import BM25Config, BM25Retriever
from evidencerag.retrieval.dense import DenseRetriever
from evidencerag.retrieval.embeddings import QwenEmbedder
from evidencerag.retrieval.rrf import HybridRetriever, RRFConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", help="Which M3 chunk split to search (default: train).")
    parser.add_argument("--query", required=True, help="Query to run against the whole corpus.")
    parser.add_argument("--top-k", type=int, default=SETTINGS.retrieval_top_k)
    parser.add_argument("--candidate-depth", type=int, default=SETTINGS.retrieval_candidate_depth)
    parser.add_argument(
        "--max-chunks", type=int, default=None, help="Only load this many chunks -- useful for a quick run."
    )
    parser.add_argument("--show-text", action="store_true", help="Also print each retrieved chunk's text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    chunks_path = PATHS.processed_data_dir / "chunks" / f"{args.split}.jsonl"
    if not chunks_path.exists():
        print(f"{chunks_path} not found -- run scripts/chunk_qasper.py first.")
        return 1

    chunks = list(load_chunks(chunks_path))
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    bm25 = BM25Retriever.build(chunks, config=BM25Config(k1=SETTINGS.bm25_k1, b=SETTINGS.bm25_b))
    embedder = QwenEmbedder(model_name=SETTINGS.embedding_model)
    dense = DenseRetriever.build(chunks, embedder=embedder)
    hybrid = HybridRetriever(
        bm25, dense, config=RRFConfig(rrf_k=SETTINGS.rrf_k, candidate_depth=args.candidate_depth)
    )

    reranker = CrossEncoderReranker()
    pipeline = RerankingRetriever(
        hybrid, reranker, chunks, config=RerankConfig(candidate_depth=args.candidate_depth)
    )

    chunk_by_id = {c.chunk_id: c for c in chunks}
    for name, retriever in (("Hybrid (pre-rerank)", hybrid), ("Reranked (M5)", pipeline)):
        print(f"\n=== {name} top-{args.top_k} for query: {args.query!r} ===")
        for result in retriever.retrieve(args.query, top_k=args.top_k):
            print(f"  rank={result.rank:2d}  score={result.score:.4f}  chunk_id={result.chunk_id}")
            if args.show_text:
                print(f"    {chunk_by_id[result.chunk_id].text[:200]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
