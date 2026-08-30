import dataclasses

import pytest

from evidencerag.generation.schema import GenerationResult


def test_construction_with_all_fields():
    result = GenerationResult(
        question="What dataset was used?",
        answer="QASPER.",
        evidence_chunk_ids=("c1", "c2"),
        model_name="fake-generator",
    )
    assert result.question == "What dataset was used?"
    assert result.answer == "QASPER."
    assert result.evidence_chunk_ids == ("c1", "c2")
    assert result.model_name == "fake-generator"


def test_is_frozen():
    result = GenerationResult(question="q", answer="a", evidence_chunk_ids=(), model_name="m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "changed"  # type: ignore[misc]


def test_equality_is_value_based():
    a = GenerationResult(question="q", answer="a", evidence_chunk_ids=("c1",), model_name="m")
    b = GenerationResult(question="q", answer="a", evidence_chunk_ids=("c1",), model_name="m")
    assert a == b


def test_empty_evidence_chunk_ids_is_valid_and_distinct_from_missing_context():
    result = GenerationResult(question="q", answer="a", evidence_chunk_ids=(), model_name="m")
    assert result.evidence_chunk_ids == ()
    assert isinstance(result.evidence_chunk_ids, tuple)


def test_evidence_chunk_id_order_is_preserved_in_the_field():
    result = GenerationResult(question="q", answer="a", evidence_chunk_ids=("z", "a", "m"), model_name="m")
    assert result.evidence_chunk_ids == ("z", "a", "m")
