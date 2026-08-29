import dataclasses

import pytest

from evidencerag.chunking.schema import Chunk


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="train:1000.00001:s000:c0000",
        paper_id="1000.00001",
        split="train",
        section_index=0,
        section_title="Introduction",
        paragraph_indices=(0, 1),
        text="Some chunk text.",
        token_count=4,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_chunk_constructs_with_all_fields():
    chunk = make_chunk()
    assert chunk.chunk_id == "train:1000.00001:s000:c0000"
    assert chunk.paper_id == "1000.00001"
    assert chunk.split == "train"
    assert chunk.section_index == 0
    assert chunk.section_title == "Introduction"
    assert chunk.paragraph_indices == (0, 1)
    assert chunk.text == "Some chunk text."
    assert chunk.token_count == 4


def test_chunk_is_frozen_immutable():
    chunk = make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_chunk_paragraph_indices_is_a_tuple():
    chunk = make_chunk(paragraph_indices=(2, 3, 4))
    assert isinstance(chunk.paragraph_indices, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.paragraph_indices = (0,)  # type: ignore[misc]


def test_chunk_equality_is_value_based():
    a = make_chunk()
    b = make_chunk()
    assert a == b
    assert a is not b


def test_chunk_supports_single_paragraph():
    chunk = make_chunk(paragraph_indices=(0,))
    assert chunk.paragraph_indices == (0,)


def test_chunk_supports_none_section_index_for_abstract_chunks():
    chunk = make_chunk(
        chunk_id="train:1000.00001:abs:c0000",
        section_index=None,
        section_title="Abstract",
        paragraph_indices=(),
    )
    assert chunk.section_index is None
    assert chunk.section_title == "Abstract"
    assert chunk.paragraph_indices == ()
