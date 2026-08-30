"""BM25 sparse retrieval over M3 chunks.

Uses `rank_bm25.BM25Okapi` rather than a hand-rolled BM25 formula --
we want to study retrieval behavior, not carry implementation risk for
a well-known, already-solved piece of math (see M4 spec).
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from evidencerag.chunking.schema import Chunk
from evidencerag.retrieval.corpus import assert_matches_corpus, build_corpus
from evidencerag.retrieval.schema import RetrievalResult, validate_top_k
from evidencerag.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class BM25Config:
    """BM25 hyperparameters. Defaults are rank_bm25's own standard
    defaults -- a sensible baseline, not a tuned result (M4 establishes
    the baseline; hyperparameter tuning is explicitly out of scope)."""

    k1: float = 1.5
    b: float = 0.75


class BM25Retriever:
    """BM25 over a fixed, deterministically ordered corpus of M3
    chunks. Construct via `build()` or `load()`, not directly."""

    def __init__(
        self, *, bm25: BM25Okapi | None, chunk_ids: tuple[str, ...], fingerprint: str, config: BM25Config
    ) -> None:
        self._bm25 = bm25
        self._chunk_ids = chunk_ids
        self._fingerprint = fingerprint
        self._config = config

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def config(self) -> BM25Config:
        return self._config

    @classmethod
    def build(cls, chunks: Iterable[Chunk], config: BM25Config | None = None) -> "BM25Retriever":
        config = config or BM25Config()
        corpus = build_corpus(chunks)
        if not corpus.chunk_ids:
            # rank_bm25.BM25Okapi divides by zero when constructed with
            # an empty corpus -- handle it explicitly rather than
            # letting that exception leak out of build().
            return cls(bm25=None, chunk_ids=(), fingerprint=corpus.fingerprint, config=config)
        tokenized_corpus = [tokenize(text) for text in corpus.texts]
        bm25 = BM25Okapi(tokenized_corpus, k1=config.k1, b=config.b)
        return cls(bm25=bm25, chunk_ids=corpus.chunk_ids, fingerprint=corpus.fingerprint, config=config)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        validate_top_k(top_k)
        if not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # Deterministic ordering: rank by score descending, then by
        # chunk_id ascending to break ties reproducibly.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._chunk_ids[i]))
        top = order[:top_k]
        return [
            RetrievalResult(chunk_id=self._chunk_ids[i], score=float(scores[i]), rank=rank, retriever="bm25")
            for rank, i in enumerate(top, start=1)
        ]

    def verify_corpus(self, chunks: Iterable[Chunk]) -> None:
        """Raise ValueError if `chunks` isn't the corpus this index was
        built from (see evidencerag.retrieval.corpus)."""
        assert_matches_corpus(self._fingerprint, chunks)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
            "fingerprint": self._fingerprint,
            "config": asdict(self._config),
            "num_chunks": len(self._chunk_ids),
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            bm25=payload["bm25"],
            chunk_ids=tuple(payload["chunk_ids"]),
            fingerprint=payload["fingerprint"],
            config=BM25Config(**payload["config"]),
        )
