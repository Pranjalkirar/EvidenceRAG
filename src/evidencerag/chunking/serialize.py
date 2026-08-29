"""Serialization of `Chunk`s and `EvidenceChunkMapping`s to/from JSON
Lines, following the same convention as `evidencerag.ingestion.serialize`
(one JSON object per line, splits kept in separate files, lossless
round-trip).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.chunking.schema import Chunk


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return asdict(chunk)


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=data["chunk_id"],
        paper_id=data["paper_id"],
        split=data["split"],
        section_index=data["section_index"],
        section_title=data["section_title"],
        paragraph_indices=tuple(data["paragraph_indices"]),
        text=data["text"],
        token_count=data["token_count"],
    )


def save_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    """Write chunks as JSON Lines to `path`. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def load_chunks(path: Path) -> Iterator[Chunk]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield chunk_from_dict(json.loads(line))


def evidence_mapping_to_dict(mapping: EvidenceChunkMapping) -> dict[str, Any]:
    return asdict(mapping)


def evidence_mapping_from_dict(data: dict[str, Any]) -> EvidenceChunkMapping:
    return EvidenceChunkMapping(
        paper_id=data["paper_id"],
        split=data["split"],
        question_index=data["question_index"],
        question_id=data.get("question_id"),
        answer_index=data["answer_index"],
        evidence_index=data["evidence_index"],
        is_float_selected=data["is_float_selected"],
        resolved=data["resolved"],
        section_index=data.get("section_index"),
        paragraph_index=data.get("paragraph_index"),
        chunk_ids=tuple(data["chunk_ids"]),
    )


def save_evidence_mappings(mappings: Iterable[EvidenceChunkMapping], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for mapping in mappings:
            f.write(json.dumps(evidence_mapping_to_dict(mapping), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def load_evidence_mappings(path: Path) -> Iterator[EvidenceChunkMapping]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield evidence_mapping_from_dict(json.loads(line))
