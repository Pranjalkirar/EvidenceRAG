"""One canonical corpus, shared by BM25, Dense, and Hybrid retrieval.

BM25 and Dense must index the *exact same* M3 chunks in the *exact
same* order, so that (a) results from each are directly comparable by
chunk_id, and (b) an index built once can later be checked against the
chunk collection it's loaded with, to catch an index silently being
reused against a different corpus.

This module owns both concerns:
  - `build_corpus`: a single deterministic ordering (chunks sorted by
    their already-unique, already-deterministic M3 `chunk_id`) that
    every retriever's `build()` must go through.
  - a corpus fingerprint, computed from that same ordering, that every
    retriever persists alongside its index and can be asked to verify
    against a chunk collection at load time.

This does not chunk, re-parse, or modify any text -- it only orders
and fingerprints chunks that `evidencerag.chunking.chunker` already
produced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from evidencerag.chunking.schema import Chunk


@dataclass(frozen=True)
class Corpus:
    """The one canonical, deterministically ordered corpus. Position i
    in `chunk_ids` and `texts` always refers to the same chunk."""

    chunk_ids: tuple[str, ...]
    texts: tuple[str, ...]
    fingerprint: str


def build_corpus(chunks: Iterable[Chunk]) -> Corpus:
    """Deterministically order `chunks` by `chunk_id` and fingerprint
    them. Raises ValueError on duplicate chunk_ids -- that would mean
    two chunks claiming the same document identity, which every
    retriever assumes cannot happen.
    """
    ordered = sorted(chunks, key=lambda c: c.chunk_id)

    seen: set[str] = set()
    for chunk in ordered:
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id in corpus: {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)

    chunk_ids = tuple(c.chunk_id for c in ordered)
    texts = tuple(c.text for c in ordered)
    fingerprint = compute_fingerprint(chunk_ids, texts)
    return Corpus(chunk_ids=chunk_ids, texts=texts, fingerprint=fingerprint)


def compute_fingerprint(chunk_ids: Sequence[str], texts: Sequence[str]) -> str:
    """A short, deterministic fingerprint of (ordered chunk_id, text)
    pairs. Two corpora with the same fingerprint are, for retrieval
    purposes, the same corpus; a different fingerprint means an index
    built from one must not silently be used against the other.

    Not a sophisticated versioning system -- just enough to catch
    "index built from corpus A, loaded against corpus B" mismatches.
    """
    digest = hashlib.sha256()
    digest.update(str(len(chunk_ids)).encode("utf-8"))
    for chunk_id, text in zip(chunk_ids, texts):
        digest.update(b"\x00")
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def assert_matches_corpus(fingerprint: str, chunks: Iterable[Chunk]) -> None:
    """Raise ValueError if `chunks` doesn't fingerprint to `fingerprint`
    (e.g. a persisted index's recorded fingerprint vs. the chunk
    collection it's about to be used with)."""
    actual = build_corpus(chunks).fingerprint
    if actual != fingerprint:
        raise ValueError(
            "Corpus mismatch: this index was built from a different chunk "
            "collection than the one it's now being checked against "
            f"(expected fingerprint {fingerprint!r}, got {actual!r})."
        )
