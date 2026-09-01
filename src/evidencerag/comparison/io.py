"""Serialization for M8 comparison runs: `metadata.json`, `summary.json`,
and `results.jsonl` (one line per `ComparisonRecord`) -- mirrors
`evidencerag.evaluation.io`'s explicit, field-by-field to-dict/from-dict
pattern exactly, for the same reason: JSON has no tuple type, so
`json.load` always hands back lists, and this keeps the on-disk shape
explicit rather than relying on `dataclasses.asdict`/`**kwargs`
round-tripping to happen to work.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from evidencerag.comparison.schema import ComparisonRecord, ComparisonRunMetadata, ComparisonRunSummary


def _record_to_dict(record: ComparisonRecord) -> dict[str, Any]:
    return asdict(record)


def _record_from_dict(data: dict[str, Any]) -> ComparisonRecord:
    return ComparisonRecord(
        question_id=data["question_id"],
        question_index=data["question_index"],
        paper_id=data["paper_id"],
        split=data["split"],
        implementation=data["implementation"],
        question_text=data["question_text"],
        retrieved_chunk_ids=tuple(data["retrieved_chunk_ids"]),
        candidate_chunk_ids_at_20=tuple(data["candidate_chunk_ids_at_20"]),
        gold_chunk_references=tuple(tuple(reference) for reference in data["gold_chunk_references"]),
        recall_at_5=data["recall_at_5"],
        recall_at_20=data["recall_at_20"],
        reciprocal_rank_at_5=data["reciprocal_rank_at_5"],
        evidence_f1=data["evidence_f1"],
        answer=data["answer"],
        answer_f1=data["answer_f1"],
        answer_type=data["answer_type"],
        retrieval_latency_s=data["retrieval_latency_s"],
        rerank_latency_s=data["rerank_latency_s"],
        generation_latency_s=data["generation_latency_s"],
        total_latency_s=data["total_latency_s"],
    )


def save_run(
    output_dir: Path, metadata: ComparisonRunMetadata, records: Iterable[ComparisonRecord], summary: ComparisonRunSummary
) -> int:
    """Writes `metadata.json`, `summary.json`, and `results.jsonl`
    under `output_dir` (created if needed). Returns the number of
    result records written."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")

    count = 0
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record)) + "\n")
            count += 1
    return count


def load_results(path: Path) -> Iterator[ComparisonRecord]:
    """Reads `results.jsonl` back into `ComparisonRecord`s, for later
    failure analysis / custom-vs-langchain diffing without re-running
    the comparison."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield _record_from_dict(json.loads(line))
