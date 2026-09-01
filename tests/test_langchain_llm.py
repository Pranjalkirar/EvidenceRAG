"""Tests for evidencerag.langchain_impl.llm.EvidenceRAGLLM."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from evidencerag.langchain_impl.llm import EvidenceRAGLLM  # noqa: E402
from tests.generation_fixtures import FailingGenerator, FakeGenerator  # noqa: E402


def test_invoke_delegates_to_wrapped_generator():
    generator = FakeGenerator(canned_answer="the answer")
    llm = EvidenceRAGLLM(generator, max_new_tokens=128)

    result = llm.invoke("a prompt")

    assert result == "the answer"
    assert generator.calls == [("a prompt", 128)]


def test_llm_type_and_model_name():
    generator = FakeGenerator(model_name="my-model")
    llm = EvidenceRAGLLM(generator)
    assert llm._llm_type == "evidencerag-generator"
    assert llm.model_name == "my-model"


def test_default_max_new_tokens_matches_generator_default():
    from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS

    generator = FakeGenerator()
    llm = EvidenceRAGLLM(generator)
    llm.invoke("prompt")
    assert generator.calls == [("prompt", DEFAULT_MAX_NEW_TOKENS)]


def test_genuine_generator_errors_are_not_swallowed():
    llm = EvidenceRAGLLM(FailingGenerator())
    with pytest.raises(RuntimeError):
        llm.invoke("prompt")
