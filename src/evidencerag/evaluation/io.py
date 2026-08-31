"""Serialization for M7 evaluation runs: `metadata.json`, `summary.json`,
and `results.jsonl` (one line per `EvalRecord`) -- mirroring
`evidencerag.chunking.serialize`'s explicit, field-by-field
to-dict/from-dict pattern (`chunk_to_dict`/`chunk_from_dict`) rather
than blindly splatting dataclasses through `dict`/`**kwargs`, so tuple
fields round-trip as tuples (JSON has no tuple type -- `json.load`
always hands back lists) and the on-disk shape stays explicit even if
the dataclasses evolve later.

Does not duplicate full `Chunk`/`Paper`/embedding/model objects --
only the lightweight `EvalRecord`/`RunMetadata`/`RunSummary` shapes
already designed for that in `schema.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from evidencerag.evaluation.schema import EvalRecord, RunMetadata, RunSummary


def _record_to_dict(record: EvalRecord) -> dict[str, Any]:
    return asdict(record)


def _record_from_dict(data: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        question_id=data["question_id"],
        question_index=data["question_index"],
        paper_id=data["paper_id"],
        split=data["split"],
        system=data["system"],
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
        generator_model=data["generator_model"],
    )


def save_run(output_dir: Path, metadata: RunMetadata, records: Iterable[EvalRecord], summary: RunSummary) -> int:
    """Writes `metadata.json`, `summary.json`, and `results.jsonl`
    under `output_dir` (created if needed). Returns the number of
    result records written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")

    count = 0
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record)) + "\n")
            count += 1
    return count


def load_results(path: Path) -> Iterator[EvalRecord]:
    """Reads `results.jsonl` back into `EvalRecord`s, for later
    failure analysis without re-running evaluation.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield _record_from_dict(json.loads(line))
