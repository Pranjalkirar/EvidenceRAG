"""Minimal common retriever interface.

Deliberately just a `Protocol` (structural typing), mirroring the
style already used for `Tokenizer` in evidencerag.chunking.tokenizer:
anything with a matching `retrieve` method satisfies it, with no
inheritance required. BM25Retriever, DenseRetriever, and
HybridRetriever (rrf.py) all satisfy this.
"""

from __future__ import annotations

from typing import Protocol

from evidencerag.retrieval.schema import RetrievalResult


class Retriever(Protocol):
    """Anything that can rank the corpus it was built from against a
    query and return the top `top_k` results, most relevant first."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...
