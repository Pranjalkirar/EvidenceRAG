"""Real-model integration test for generation with the actual
`Qwen/Qwen3-4B-Instruct-2507` model.

Deliberately separate from tests/test_generation_generator.py (which
uses fake tokenizer/model doubles and must always run, fast, with no
network/model/GPU). This test is automatically SKIPPED -- not failed
-- whenever `transformers`/`torch` aren't installed, no GPU/enough
memory is available, or the model can't actually be loaded (no
network access, no cached weights, etc.), so ordinary unit-test runs
never require downloading an ~8 GB model. On a machine that does have
it available (e.g. a Kaggle GPU notebook), this test genuinely loads
the model and checks the M6 pipeline end to end; once the model IS
available, genuine runtime errors are not swallowed.
"""

from __future__ import annotations

import pytest

from evidencerag.generation.generate import generate_answer
from evidencerag.generation.generator import HFGenerator
from evidencerag.retrieval.schema import RetrievalResult

CONTEXT_TEXT_BY_ID = {
    "evidence-1": (
        "The experiments in this paper are conducted on the QASPER dataset, "
        "which consists of 1585 papers with question-answer pairs annotated "
        "by NLP practitioners."
    ),
}


@pytest.fixture(scope="module")
def hf_generator():
    try:
        return HFGenerator()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qwen/Qwen3-4B-Instruct-2507 unavailable in this environment: {exc}")


def test_real_generator_produces_a_nonempty_answer_grounded_in_context(hf_generator):
    candidates = [RetrievalResult(chunk_id="evidence-1", score=0.0, rank=1, retriever="reranker")]
    result = generate_answer(
        "What dataset is used in the experiments?",
        candidates,
        CONTEXT_TEXT_BY_ID,
        hf_generator,
        max_new_tokens=128,
    )
    assert isinstance(result.answer, str)
    assert len(result.answer.strip()) > 0
    assert result.evidence_chunk_ids == ("evidence-1",)
    assert result.model_name == hf_generator.model_name


def test_real_generator_states_insufficiency_when_context_is_empty(hf_generator):
    result = generate_answer(
        "What dataset is used in the experiments?",
        [],
        {},
        hf_generator,
        max_new_tokens=64,
    )
    assert result.evidence_chunk_ids == ()
    assert len(result.answer.strip()) > 0
