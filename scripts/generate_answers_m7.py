#!/usr/bin/env python
"""Fill in Answer F1 for an already-completed retrieval-mode M7 run,
WITHOUT re-running retrieval.

Reads an existing retrieval-mode results.jsonl (each record already has
`retrieved_chunk_ids` -- exactly what generation needs) plus M2 papers.jsonl
and M3 chunks.jsonl, builds the same prompts `harness._score_one` would have
built in end-to-end mode, generates answers, and writes a NEW results.jsonl
with answer/answer_f1/answer_type/generator_model filled in.

Deliberately does NOT touch: BM25Retriever, DenseRetriever, QwenEmbedder,
CrossEncoderReranker, evidence mappings. Only M2 Paper/Answer data and M3
chunk text are needed for Answer F1 (see gold.build_gold -- answer_references
comes from paper.questions[i].answers directly, not from evidence mappings).

Reuses the SAME resumability primitives as evaluate_m7.py: --run-id,
append_record, load_completed_keys. Run this with the SAME --run-id you used
for the retrieval-mode run's --source-run-id if you want the two to be
directly comparable (recommended), or a new --run-id if you want a
standalone generation-only experiment directory.

Usage:
    python scripts/generate_answers_m7.py \\
        --split validation \\
        --source-run-id validation-retrieval-full \\
        --run-id validation-e2e-generation \\
        --batch-size 4 \\
        --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import json

from evidencerag.chunking.serialize import load_chunks
from evidencerag.config import PATHS
from evidencerag.evaluation.gold import build_gold
from evidencerag.evaluation.harness import summarize_records
from evidencerag.evaluation.io import append_record, load_completed_keys, load_results, save_metadata, save_summary
from evidencerag.evaluation.schema import EvalRecord, RunMetadata
from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS
from evidencerag.generation.prompt import ContextChunk, build_prompt
from evidencerag.ingestion.serialize import load_papers
from evidencerag.evaluation import answer_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--source-run-id", required=True, help="Existing retrieval-mode run to read chunk IDs from.")
    parser.add_argument("--run-id", required=True, help="Output run-id. Reuse to resume after an interruption.")
    parser.add_argument("--generator-model", default=None, help="Defaults to M6's Qwen3-4B-Instruct-2507.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=1, help="1 = unbatched (safest baseline).")
    parser.add_argument("--device-map", default=None, help="e.g. '{\"\": 0}' to force one GPU explicitly.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paper-start", type=int, default=None, help="For sharding across GPUs: paper index start.")
    parser.add_argument("--paper-end", type=int, default=None, help="For sharding across GPUs: paper index end.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    papers_path = PATHS.processed_data_dir / f"{args.split}.jsonl"
    chunks_path = PATHS.processed_data_dir / "chunks" / f"{args.split}.jsonl"
    source_results_path = (PATHS.experiments_dir / "m7" / args.source_run_id) / "results.jsonl"

    if not source_results_path.exists():
        print(f"{source_results_path} not found -- run evaluate_m7.py --mode retrieval first.")
        return 1

    # Retrieval config didn't change (we're not re-running retrieval) --
    # read it from the source run's own metadata.json rather than
    # fabricating placeholder values, per the project's "no invented
    # stats" convention.
    source_metadata = json.loads((source_results_path.parent / "metadata.json").read_text(encoding="utf-8"))

    papers = list(load_papers(papers_path))
    if args.paper_start is not None or args.paper_end is not None:
        papers = papers[args.paper_start or 0 : args.paper_end]
    chunks = list(load_chunks(chunks_path))
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
    paper_by_id = {paper.paper_id: paper for paper in papers}

    gold_by_paper = {paper.paper_id: build_gold(paper, ()) for paper in papers}

    # One retrieval-mode record per (paper_id, question_index, system) --
    # exactly the top-k chunk IDs generation needs. No re-retrieval.
    source_records = {
        (r.paper_id, r.question_index, r.system): r
        for r in load_results(source_results_path)
        if r.paper_id in paper_by_id
    }
    print(f"Loaded {len(source_records)} retrieval-mode records for {len(papers)} papers.")

    output_dir = (args.output_dir or (PATHS.experiments_dir / "m7")) / args.run_id
    results_path = output_dir / "results.jsonl"
    skip_keys = load_completed_keys(results_path)
    if skip_keys:
        print(f"Resuming: {len(skip_keys)} record(s) already done, skipping them.")

    pending = [key for key in source_records if key not in skip_keys]
    print(f"{len(pending)} record(s) to generate.")
    if not pending:
        return 0

    # Only import/load the generator once we know there's real work to do.
    from evidencerag.generation.generator import HFGenerator

    generator_kwargs = {"torch_dtype": "bfloat16"}
    if args.generator_model:
        generator_kwargs["model_name"] = args.generator_model
    if args.device_map:
        import json as _json

        generator_kwargs["device"] = _json.loads(args.device_map)
    generator = HFGenerator(**generator_kwargs)

    save_metadata(
        output_dir,
        RunMetadata(
            run_id=args.run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            git_commit=None,
            split=args.split,
            mode="end_to_end",
            systems=tuple(sorted({key[2] for key in pending})),
            retrieval_top_k=source_metadata["retrieval_top_k"],
            retrieval_candidate_depth=source_metadata["retrieval_candidate_depth"],
            rrf_k=source_metadata["rrf_k"],
            bm25_k1=source_metadata["bm25_k1"],
            bm25_b=source_metadata["bm25_b"],
            embedding_model=source_metadata["embedding_model"],
            reranker_model=source_metadata["reranker_model"],
            generator_model=generator.model_name,
            random_seed=source_metadata["random_seed"],
            max_questions=None,
        ),
    )

    # Build (key, prompt) pairs up front -- pure string work, no GPU --
    # so batching can sort by prompt length to minimize padding waste.
    prepared: list[tuple[tuple[str, int, str], str, str]] = []  # (key, question, prompt)
    for key in pending:
        paper_id, q_idx, system = key
        record = source_records[key]
        gold = gold_by_paper[paper_id][q_idx]
        context = [
            ContextChunk(chunk_id=cid, text=chunk_text_by_id[cid]) for cid in record.retrieved_chunk_ids
        ]
        prompt = build_prompt(gold.question_text, context)
        prepared.append((key, gold.question_text, prompt))

    prepared.sort(key=lambda item: len(item[2]))  # shortest prompts first -- minimizes padding in each batch

    t_start = time.time()
    n_done = 0
    batch_size = max(1, args.batch_size)

    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        prompts = [item[2] for item in batch]

        if batch_size == 1:
            answers = [generator.generate(prompts[0], max_new_tokens=args.max_new_tokens)]
        else:
            answers = generator.generate_batch(prompts, max_new_tokens=args.max_new_tokens)

        for (key, question, _prompt), answer in zip(batch, answers):
            paper_id, q_idx, system = key
            gold = gold_by_paper[paper_id][q_idx]
            answer_f1, answer_type = answer_metrics.answer_f1_and_type(answer, gold.answer_references)
            record = source_records[key]
            new_record = EvalRecord(
                question_id=record.question_id,
                question_index=record.question_index,
                paper_id=record.paper_id,
                split=record.split,
                system=record.system,
                retrieved_chunk_ids=record.retrieved_chunk_ids,
                candidate_chunk_ids_at_20=record.candidate_chunk_ids_at_20,
                gold_chunk_references=record.gold_chunk_references,
                recall_at_5=record.recall_at_5,
                recall_at_20=record.recall_at_20,
                reciprocal_rank_at_5=record.reciprocal_rank_at_5,
                evidence_f1=record.evidence_f1,
                answer=answer,
                answer_f1=answer_f1,
                answer_type=answer_type,
                generator_model=generator.model_name,
            )
            append_record(output_dir, new_record)
            n_done += 1

        elapsed = time.time() - t_start
        rate = elapsed / n_done if n_done else 0
        eta_hours = rate * (len(prepared) - n_done) / 3600
        print(
            f"[{n_done}/{len(prepared)}] elapsed={elapsed/60:.1f}m "
            f"rate={rate:.1f}s/record eta={eta_hours:.1f}h",
            file=sys.stderr,
        )

    all_records = list(load_results(results_path))
    summary = summarize_records(all_records, "end_to_end")
    save_summary(output_dir, summary)
    print(f"\nDone. {len(all_records)} total records in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
