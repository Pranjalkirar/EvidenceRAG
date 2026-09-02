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

RESUMING AN INTERRUPTED RUN: every question x system result is written
to `results.jsonl` immediately, as it's computed -- not only once the
whole run finishes. If a run is interrupted (a Kaggle session limit, an
OOM, a manual stop) and rerun with the SAME `--run-id`, it picks up
where it left off instead of redoing (and re-paying the generation
cost of) work that already finished. Use a fresh `--run-id` to force a
run from scratch instead of resuming.

`--max-new-tokens` caps how many tokens the generator produces per
answer in end-to-end mode (M6's own default is used if omitted).
QASPER's reference answers are typically short, so a much smaller
budget than the default meaningfully cuts end-to-end wall-clock time
without materially changing Answer F1 -- this is an M7 evaluation
choice threaded through `generate_answer()`'s existing parameter, not
a change to M6 itself.

`--generator-model` swaps the end-to-end generator for a different
Hugging Face causal LM (e.g. a smaller/faster one), for comparing
generation quality/speed/memory trade-offs. M6's own default
(Qwen/Qwen3-4B-Instruct-2507) is used when this is omitted, so the
standard benchmark is unaffected.

Usage:
    python scripts/evaluate_m7.py --split validation --mode retrieval
    python scripts/evaluate_m7.py --split validation --mode retrieval --max-questions 20
    python scripts/evaluate_m7.py --split test --mode end-to-end --run-id 2026-08-31-final
    python scripts/evaluate_m7.py --split validation --mode end-to-end --run-id validation-e2e \
        --max-new-tokens 64
    python scripts/evaluate_m7.py --split validation --mode end-to-end --run-id validation-e2e
        # (rerun with the same --run-id to resume after an interruption)
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
from evidencerag.evaluation.harness import EvaluationConfig, run_evaluation, summarize_records
from evidencerag.evaluation.io import append_record, load_completed_keys, load_results, save_metadata, save_summary
from evidencerag.evaluation.schema import RunMetadata
from evidencerag.evaluation.systems import build_systems
from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS, HFGenerator
from evidencerag.ingestion.serialize import load_papers
from evidencerag.reranking.reranker import CrossEncoderReranker
from evidencerag.retrieval.embeddings import QwenEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--mode", default="retrieval", choices=("retrieval", "end-to-end"))
    parser.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp. Reuse to resume a run.")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Pilot/smoke runs only -- do not use for a reported benchmark.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"End-to-end generation budget per answer (default: M6's own default, {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Hugging Face causal LM to use in end-to-end mode instead of M6's default generator.",
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

    # Build the four retrieval systems -- including the one-time bulk
    # dense-embedding pass over the whole corpus in build_systems() --
    # BEFORE loading the generator. That embedding pass is the single
    # biggest transient GPU consumer in retrieval setup; running it
    # first means it doesn't have to share the GPU with an idle
    # multi-GB generator on top of it.
    systems = build_systems(chunks, embedder=embedder, reranker=reranker)

    generator = None
    if mode == "end_to_end":
        # bfloat16 is requested explicitly rather than relying on
        # torch_dtype="auto": some transformers versions resolve
        # "auto" through the now-deprecated torch_dtype path and
        # silently fall back to float32, roughly doubling generator
        # memory on a GPU that may already be close to full after the
        # steps above. model_name is only overridden when
        # --generator-model is given, so the standard benchmark still
        # uses M6's own default (Qwen/Qwen3-4B-Instruct-2507)
        # unchanged.
        generator_kwargs = {"torch_dtype": "bfloat16"}
        if args.generator_model:
            generator_kwargs["model_name"] = args.generator_model
        generator = HFGenerator(**generator_kwargs)

    config = EvaluationConfig(
        mode=mode,
        split=args.split,
        top_k=SETTINGS.retrieval_top_k,
        candidate_depth=SETTINGS.retrieval_candidate_depth,
        max_new_tokens=args.max_new_tokens,
        max_questions=args.max_questions,
    )

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (args.output_dir or (PATHS.experiments_dir / "m7")) / run_id
    results_path = output_dir / "results.jsonl"

    # Resuming: anything already in results.jsonl from a previous,
    # interrupted invocation of this same --run-id is skipped rather
    # than recomputed.
    skip_keys = load_completed_keys(results_path)
    if skip_keys:
        print(
            f"Resuming run {run_id!r}: {len(skip_keys)} question x system record(s) "
            f"already present in {results_path}, skipping them."
        )

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
    # Written up front, before the (possibly many-hour) loop below, so
    # a run's effective configuration is on disk even if it's
    # interrupted before finishing.
    save_metadata(output_dir, metadata)

    new_records, _ = run_evaluation(
        papers=papers,
        chunks=chunks,
        evidence_mappings_by_paper=mappings_by_paper,
        systems=systems,
        config=config,
        generator=generator,
        on_record=lambda record: append_record(output_dir, record),
        skip_keys=skip_keys,
    )

    # Recomputed from the full on-disk results.jsonl (previously
    # completed + newly computed this run), not just what this
    # process computed -- so a resumed run's summary.json reflects
    # the whole run, not only its final resumed segment.
    all_records = list(load_results(results_path))
    summary = summarize_records(all_records, mode)
    save_summary(output_dir, summary)

    print(f"\nWrote {len(new_records)} new record(s) this run ({len(all_records)} total) to {output_dir}")

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
