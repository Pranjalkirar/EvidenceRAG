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

Two ways to write results, for two different situations:

  * `save_run` -- write everything (metadata + summary + all results)
    at once, from an already-complete in-memory run. Simple, and what
    the test suite uses for a fake/fast run with no realistic risk of
    interruption.

  * `append_record` / `save_metadata` / `save_summary` -- write
    incrementally, as a long real run progresses. `append_record`
    flushes and fsyncs each line immediately, so a `results.jsonl`
    under `experiments/m7/<run_id>/` always reflects every question x
    system pair actually scored so far, even if the process is killed
    (a Kaggle session timeout, an OOM elsewhere, a manual interrupt)
    before the run finishes. `load_completed_keys` reads that partial
    file back so a rerun with the same `--run-id` can skip what's
    already done instead of re-paying for expensive generation calls.
    This is what `scripts/evaluate_m7.py` uses for real runs.
"""

from __future__ import annotations

import json
import os
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
    under `output_dir` (created if needed), all at once, from an
    already-complete `records`/`summary`. Returns the number of result
    records written.

    For a long real run where interruption is a real possibility,
    prefer `save_metadata` once up front and `append_record` per
    record instead -- see the module docstring.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(output_dir, metadata)
    save_summary(output_dir, summary)

    count = 0
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record)) + "\n")
            count += 1
    return count


def save_metadata(output_dir: Path, metadata: RunMetadata) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")


def save_summary(output_dir: Path, summary: RunSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")


def append_record(output_dir: Path, record: EvalRecord) -> None:
    """Append one `EvalRecord` to `results.jsonl`, flushed and fsynced
    immediately -- so a run's progress is durable on disk the moment
    each question x system pair is scored, not only once the entire
    run finishes. Creates `output_dir` and `results.jsonl` if they
    don't exist yet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record_to_dict(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed_keys(results_path: Path) -> set[tuple[str, int, str]]:
    """The `(paper_id, question_index, system)` triples already present
    in an existing `results.jsonl` -- used to resume an interrupted run
    without recomputing (and, worse, re-paying the generation cost of)
    what a previous invocation already finished. Returns an empty set
    if `results_path` doesn't exist yet, so a fresh run and a resumed
    run go through the same code path.
    """
    if not results_path.exists():
        return set()
    return {(record.paper_id, record.question_index, record.system) for record in load_results(results_path)}


def load_results(path: Path) -> Iterator[EvalRecord]:
    """Reads `results.jsonl` back into `EvalRecord`s -- for later
    failure analysis without re-running evaluation, and for
    `load_completed_keys` / summary recomputation on a resumed run.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield _record_from_dict(json.loads(line))
