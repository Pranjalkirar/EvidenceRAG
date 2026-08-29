#!/usr/bin/env python
"""Run the M3 chunking pipeline over M2's normalized QASPER output:

    data/processed/{split}.jsonl (M2 output, read-only here)
        -> chunk_paper() per paper
        -> map_evidence_to_chunks() per paper
        -> data/processed/chunks/{split}.jsonl
        -> data/processed/chunks/{split}_evidence_map.jsonl

This script does not modify M2's output files, and does not download
or re-normalize anything -- it assumes `scripts/ingest_qasper.py` has
already been run.

Usage:
    python scripts/chunk_qasper.py
    python scripts/chunk_qasper.py --splits train validation
    python scripts/chunk_qasper.py --max-papers-per-split 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

from evidencerag.chunking.chunker import chunk_paper
from evidencerag.chunking.evidence_map import map_evidence_to_chunks
from evidencerag.chunking.serialize import save_chunks, save_evidence_mappings
from evidencerag.config import PATHS
from evidencerag.ingestion.loader import QASPER_SPLITS
from evidencerag.ingestion.serialize import load_papers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(QASPER_SPLITS),
        choices=list(QASPER_SPLITS),
        help="Which splits to chunk (default: all three, whichever have M2 output present).",
    )
    parser.add_argument(
        "--max-papers-per-split",
        type=int,
        default=None,
        help="If set, only chunk this many papers per split (useful for a quick smoke test).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for split in args.splits:
        input_path = PATHS.processed_data_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"[{split}] {input_path} not found -- run scripts/ingest_qasper.py first. Skipping.")
            continue

        papers = list(load_papers(input_path))
        if args.max_papers_per_split is not None:
            papers = papers[: args.max_papers_per_split]
        print(f"[{split}] loaded {len(papers)} papers from {input_path}")

        all_chunks = []
        all_mappings = []
        for paper in papers:
            chunks = chunk_paper(paper)
            all_chunks.extend(chunks)
            all_mappings.extend(map_evidence_to_chunks(paper, chunks))

        chunks_path = PATHS.processed_data_dir / "chunks" / f"{split}.jsonl"
        mapping_path = PATHS.processed_data_dir / "chunks" / f"{split}_evidence_map.jsonl"

        n_chunks = save_chunks(all_chunks, chunks_path)
        n_mappings = save_evidence_mappings(all_mappings, mapping_path)

        print(f"[{split}] wrote {n_chunks} chunks to {chunks_path}")
        print(f"[{split}] wrote {n_mappings} evidence mappings to {mapping_path}")

        n_evidence_with_chunks = sum(1 for m in all_mappings if m.chunk_ids)
        n_evidence_total = len(all_mappings)
        print(
            f"[{split}] evidence resolved to >=1 chunk: {n_evidence_with_chunks}/{n_evidence_total}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
