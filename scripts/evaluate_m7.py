#!/usr/bin/env python
"""Run the M7 evaluation harness over an already-chunked QASPER split:

    data/processed/{split}.jsonl                       (M2 output)
    data/processed/chunks/{split}.jsonl                 (M3 output)
    data/processed/chunks/{split}_evidence_map.jsonl    (M3 output)

If any of these are missing, this script fails with a clear message
naming the script to run first -- it never regenerates M2/M3 output
itself and never invents evaluation data.

Compares BM25 / Dense / Hybrid / Hybrid+Reranker using the SAME
Settings-defined top_k=5, candidate_depth=20, RRF/BM25/reranker
configuration M4/M5 already established. This script does not expose
flags to change those values -- a "standard" M7 run always means the
same experiment, regardless of who runs it.

`--mode end-to-end` additionally loads the M6 generator and computes
QASPER Answer F1; this is the expensive path (loads an LLM). The
default mode is `retrieval`, which only needs the embedder and
reranker.

`--max-questions` truncates the run for a quick pilot/smoke check --
never use it for a reported benchmark number. The final validation and
test evaluations must run over the complete split.

Usage:
    python scripts/evaluate_m7.py --split validation --mode retrieval
    python scripts/evaluate_m7.py --split validation --mode retrieval --max-questions 20
    python scripts/evaluate_m7.py --split test --mode end-to-end --run-id 2026-08-31-final
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.chunking.serialize import load_chunks, load_evidence_mappings
from evidencerag.config import PATHS, SETTINGS
from evidencerag.evaluation.harness import EvaluationConfig, run_evaluation
from evidencerag.evaluation.io import save_run
from evidencerag.evaluation.schema import RunMetadata
from evidencerag.evaluation.systems import build_systems
from evidencerag.generation.generator import HFGenerator
from evidencerag.ingestion.serialize import load_papers
from evidencerag.reranking.reranker import CrossEncoderReranker
from evidencerag.retrieval.embeddings import QwenEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--mode", default="retrieval", choices=("retrieval", "end-to-end"))
    parser.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp.")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Pilot/smoke runs only -- do not use for a reported benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Defaults to {PATHS.experiments_dir / 'm7'}.",
    )
    return parser.parse_args()


def _git_commit() -> Optional[str]:
    """Best-effort -- returns None if git isn't available or this
    isn't a git checkout, never raises. Metadata treats this as
    "record if practical", not a requirement."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PATHS.root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _fmt(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


def main() -> int:
    args = parse_args()
    mode = "end_to_end" if args.mode == "end-to-end" else "retrieval"

    papers_path = PATHS.processed_data_dir / f"{args.split}.jsonl"
    chunks_path = PATHS.processed_data_dir / "chunks" / f"{args.split}.jsonl"
    mapping_path = PATHS.processed_data_dir / "chunks" / f"{args.split}_evidence_map.jsonl"

    for path, script in (
        (papers_path, "scripts/ingest_qasper.py"),
        (chunks_path, "scripts/chunk_qasper.py"),
        (mapping_path, "scripts/chunk_qasper.py"),
    ):
        if not path.exists():
            print(f"{path} not found -- run `python {script} --split {args.split}` first.")
            return 1

    papers = list(load_papers(papers_path))
    chunks = list(load_chunks(chunks_path))
    mappings = list(load_evidence_mappings(mapping_path))

    mappings_by_paper: dict[str, list[EvidenceChunkMapping]] = {}
    for mapping in mappings:
        mappings_by_paper.setdefault(mapping.paper_id, []).append(mapping)

    print(
        f"Loaded {len(papers)} papers, {len(chunks)} chunks, {len(mappings)} evidence mappings "
        f"for split={args.split!r}"
    )

    embedder = QwenEmbedder(model_name=SETTINGS.embedding_model)
    reranker = CrossEncoderReranker()
    generator = HFGenerator() if mode == "end_to_end" else None

    systems = build_systems(chunks, embedder=embedder, reranker=reranker)

    config = EvaluationConfig(
        mode=mode,
        split=args.split,
        top_k=SETTINGS.retrieval_top_k,
        candidate_depth=SETTINGS.retrieval_candidate_depth,
        max_questions=args.max_questions,
    )

    records, summary = run_evaluation(
        papers=papers,
        chunks=chunks,
        evidence_mappings_by_paper=mappings_by_paper,
        systems=systems,
        config=config,
        generator=generator,
    )

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (args.output_dir or (PATHS.experiments_dir / "m7")) / run_id

    metadata = RunMetadata(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_git_commit(),
        split=args.split,
        mode=mode,
        systems=tuple(systems.retrievers),
        retrieval_top_k=config.top_k,
        retrieval_candidate_depth=config.candidate_depth,
        rrf_k=SETTINGS.rrf_k,
        bm25_k1=SETTINGS.bm25_k1,
        bm25_b=SETTINGS.bm25_b,
        embedding_model=embedder.model_name,
        reranker_model=reranker.model_name,
        generator_model=generator.model_name if generator is not None else None,
        random_seed=SETTINGS.random_seed,
        max_questions=args.max_questions,
    )

    n_written = save_run(output_dir, metadata, records, summary)
    print(f"\nWrote {n_written} records to {output_dir}")

    if args.max_questions is not None:
        print("(--max-questions was set -- this is a pilot/smoke run, not a reportable benchmark.)")

    for system_summary in summary.systems:
        line = (
            f"  {system_summary.system:14s} n={system_summary.n_questions:4d} "
            f"excluded={system_summary.n_excluded_no_gold_chunks:3d} "
            f"recall@5={_fmt(system_summary.recall_at_5)} "
            f"recall@20={_fmt(system_summary.recall_at_20)} "
            f"mrr@5={_fmt(system_summary.mrr_at_5)} "
            f"evidence_f1={system_summary.evidence_f1:.3f}"
        )
        if system_summary.answer_f1 is not None:
            line += f" answer_f1={system_summary.answer_f1:.3f}"
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
