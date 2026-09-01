#!/usr/bin/env python
"""Run the M8 Custom-vs-LangChain comparison over an already-chunked
QASPER split:

    data/processed/{split}.jsonl                       (M2 output)
    data/processed/chunks/{split}.jsonl                 (M3 output)
    data/processed/chunks/{split}_evidence_map.jsonl    (M3 output)

If any of these are missing, this script fails with a clear message
naming the script to run first -- same convention as
`scripts/evaluate_m7.py`, which this script otherwise mirrors closely.

Both implementations run the SAME single chain -- Hybrid (BM25+Dense
RRF) + Cross-Encoder Reranker, optionally + LLM generation -- built
from the SAME `evidencerag.config.SETTINGS`-defined chunks, embedder,
reranker, and (in end-to-end mode) generator. This script does not
expose flags to change those model choices; only `--mode`,
`--max-questions`, and `--implementations` vary between runs.

The LangChain implementation is optional. If `langchain`/
`langchain-core`/`langchain-community` are not installed, this script
automatically falls back to `--implementations custom` (printing a
note) rather than failing -- the custom pipeline never depends on
LangChain being available.

Usage:
    python scripts/evaluate_m8.py --split validation --max-questions 2 --run-id pilot-1
    python scripts/evaluate_m8.py --split validation --mode end-to-end --max-questions 2
    python scripts/evaluate_m8.py --split test --run-id 2026-08-31-final
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
from evidencerag.comparison.custom_pipeline import CustomPipeline
from evidencerag.comparison.io import save_run
from evidencerag.comparison.runner import ComparisonConfig, run_comparison
from evidencerag.comparison.schema import ComparisonRunMetadata
from evidencerag.config import PATHS, SETTINGS
from evidencerag.generation.generator import HFGenerator
from evidencerag.ingestion.serialize import load_papers
from evidencerag.reranking.reranker import CrossEncoderReranker
from evidencerag.retrieval.embeddings import QwenEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--mode", default="retrieval", choices=("retrieval", "end-to-end"))
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["custom", "langchain"],
        choices=("custom", "langchain"),
        help="Which implementation(s) to run. Defaults to both; falls back to "
        "'custom' only if LangChain is not installed.",
    )
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
        help=f"Defaults to {PATHS.experiments_dir / 'm8'}.",
    )
    return parser.parse_args()


def _git_commit() -> Optional[str]:
    """Best-effort, matching `scripts/evaluate_m7.py`'s `_git_commit`."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PATHS.root, capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _fmt(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


def _fmt_ms(value: Optional[float]) -> str:
    return f"{value * 1000:6.1f}ms" if value is not None else "    n/a"


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

    pipelines: dict[str, object] = {}

    if "custom" in args.implementations:
        pipelines["custom"] = CustomPipeline(
            chunks,
            embedder=embedder,
            reranker=reranker,
            generator=generator,
            candidate_depth=SETTINGS.retrieval_candidate_depth,
            top_k=SETTINGS.retrieval_top_k,
        )

    langchain_available = True
    if "langchain" in args.implementations:
        try:
            from evidencerag.langchain_impl.pipeline import LangChainPipeline
        except ImportError as exc:
            langchain_available = False
            print(f"LangChain implementation unavailable, skipping ({exc}).")
        else:
            langchain_pipeline = LangChainPipeline(
                chunks,
                embedder=embedder,
                reranker=reranker,
                generator=generator,
                candidate_depth=SETTINGS.retrieval_candidate_depth,
                top_k=SETTINGS.retrieval_top_k,
            )
            pipelines["langchain"] = _LangChainPipelineAdapter(langchain_pipeline)

    if not pipelines:
        print("No implementations available to run -- see the message(s) above.")
        return 1

    config = ComparisonConfig(
        mode=mode,
        split=args.split,
        top_k=SETTINGS.retrieval_top_k,
        candidate_depth=SETTINGS.retrieval_candidate_depth,
        max_questions=args.max_questions,
    )

    records, summary = run_comparison(
        papers=papers,
        chunks=chunks,
        evidence_mappings_by_paper=mappings_by_paper,
        pipelines=pipelines,
        config=config,
    )

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (args.output_dir or (PATHS.experiments_dir / "m8")) / run_id

    metadata = ComparisonRunMetadata(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_git_commit(),
        split=args.split,
        mode=mode,
        implementations=tuple(pipelines),
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
        langchain_available=langchain_available,
    )

    n_written = save_run(output_dir, metadata, records, summary)
    print(f"\nWrote {n_written} records to {output_dir}")

    if args.max_questions is not None:
        print("(--max-questions was set -- this is a pilot/smoke run, not a reportable benchmark.)")

    for impl_summary in summary.implementations:
        line = (
            f"  {impl_summary.implementation:10s} n={impl_summary.n_questions:4d} "
            f"excluded={impl_summary.n_excluded_no_gold_chunks:3d} "
            f"recall@5={_fmt(impl_summary.recall_at_5)} "
            f"recall@20={_fmt(impl_summary.recall_at_20)} "
            f"mrr@5={_fmt(impl_summary.mrr_at_5)} "
            f"evidence_f1={impl_summary.evidence_f1:.3f} "
            f"retrieval={_fmt_ms(impl_summary.mean_retrieval_latency_s)} "
            f"rerank={_fmt_ms(impl_summary.mean_rerank_latency_s)} "
            f"generation={_fmt_ms(impl_summary.mean_generation_latency_s)} "
            f"total={_fmt_ms(impl_summary.mean_total_latency_s)}"
        )
        if impl_summary.answer_f1 is not None:
            line += f" answer_f1={impl_summary.answer_f1:.3f}"
        print(line)

    print()
    for complexity in summary.complexity:
        deps = ", ".join(complexity.dependency_additions) or "(none)"
        print(
            f"  {complexity.implementation:10s} relevant_loc={complexity.relevant_loc:5d} "
            f"files={complexity.file_count:2d} components={complexity.custom_component_count:2d} "
            f"deps_added=[{deps}]"
        )

    return 0


class _LangChainPipelineAdapter:
    """Adapts `evidencerag.langchain_impl.pipeline.LangChainPipeline`'s
    `LangChainStageResult` to `evidencerag.comparison.schema.StageResult`
    so `run_comparison` can drive both pipelines through the identical
    `QuestionPipeline` protocol -- see `evidencerag.comparison.runner`.
    Kept in this script (not in `evidencerag.comparison`) so that
    package never needs to import `evidencerag.langchain_impl` at
    module load time.
    """

    def __init__(self, pipeline) -> None:  # pipeline: LangChainPipeline
        self._pipeline = pipeline

    def run_question(self, question_text: str):
        from evidencerag.comparison.schema import StageResult

        result = self._pipeline.run_question(question_text)
        return StageResult(
            top5_chunk_ids=result.top5_chunk_ids,
            candidate20_chunk_ids=result.candidate20_chunk_ids,
            retrieval_latency_s=result.retrieval_latency_s,
            rerank_latency_s=result.rerank_latency_s,
            generation_latency_s=result.generation_latency_s,
            total_latency_s=result.total_latency_s,
            answer=result.answer,
        )


if __name__ == "__main__":
    sys.exit(main())
