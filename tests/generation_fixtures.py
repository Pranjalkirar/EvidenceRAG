"""Shared fixtures for M6 generation tests, following the same
test-only-helper convention as tests/reranking_fixtures.py and
tests/retrieval_fixtures.py.

`FakeGenerator` is a lightweight, fully deterministic stand-in for
`HFGenerator`: no model download, no torch, no network. It records
every prompt it receives (and the `max_new_tokens` it was called
with), so tests can assert exactly what a caller built and passed to
it, without needing a real LLM.
"""

from __future__ import annotations


class FakeGenerator:
    """Deterministic generator for tests.

    By default, returns a short deterministic string derived from the
    prompt's length -- enough for tests to distinguish "different
    prompt -> different answer" without any real language modeling.
    A fixed `canned_answer` can be supplied instead when a test wants
    to dictate the exact answer text (e.g. to check it round-trips
    into `GenerationResult.answer` unchanged).
    """

    def __init__(self, model_name: str = "fake-generator", canned_answer: str | None = None) -> None:
        self._model_name = model_name
        self._canned_answer = canned_answer
        self.calls: list[tuple[str, int]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls.append((prompt, max_new_tokens))
        if self._canned_answer is not None:
            return self._canned_answer
        return f"[fake answer, prompt length={len(prompt)}]"


class FailingGenerator:
    """A `Generator` that always raises -- used to verify that genuine
    runtime errors (e.g. CUDA OOM) from a real generator are never
    swallowed anywhere in the M6 pipeline, the same convention used by
    `test_reranking_cross_encoder.py`'s `_FailingModel`.
    """

    model_name = "failing-generator"

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        raise RuntimeError("CUDA out of memory")


class FakeRetriever:
    """Deterministic stand-in for a `Retriever` (e.g. `HybridRetriever`
    or `RerankingRetriever`): returns a fixed, caller-supplied
    `RetrievalResult` list regardless of the query, and records every
    `(query, top_k)` it was called with so `GenerationPipeline` tests
    can assert it was wired correctly.
    """

    def __init__(self, results) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return self._results[:top_k]
