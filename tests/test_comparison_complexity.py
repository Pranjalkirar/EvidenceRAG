"""Tests for evidencerag.comparison.complexity.measure_complexity.

Deliberately does NOT require the `langchain` extra to be installed --
`measure_complexity` only reads files and imports
`evidencerag.langchain_impl.REQUIRED_PACKAGES` (a plain tuple of
strings), it never calls `require_langchain()` or imports any
`langchain*` package.
"""

from __future__ import annotations

from evidencerag.comparison.complexity import measure_complexity
from evidencerag.comparison.schema import EngineeringComplexity


def test_measure_complexity_returns_custom_and_langchain_snapshots():
    custom, langchain = measure_complexity()
    assert isinstance(custom, EngineeringComplexity)
    assert isinstance(langchain, EngineeringComplexity)
    assert custom.implementation == "custom"
    assert langchain.implementation == "langchain"


def test_custom_side_adds_no_new_dependencies():
    custom, _ = measure_complexity()
    assert custom.dependency_additions == ()


def test_langchain_side_lists_its_dependency_additions():
    _, langchain = measure_complexity()
    assert "langchain-core" in langchain.dependency_additions
    assert "langchain" in langchain.dependency_additions
    assert "langchain-community" in langchain.dependency_additions


def test_both_sides_report_positive_loc_for_this_repository():
    custom, langchain = measure_complexity()
    # This repository's M4/M5/M6 orchestration modules and the
    # langchain_impl package both exist and are non-empty -- a zero
    # here would mean the file lists in complexity.py have drifted
    # from the actual file layout.
    assert custom.relevant_loc > 0
    assert custom.file_count > 0
    assert langchain.relevant_loc > 0
    assert langchain.file_count > 0


def test_measure_complexity_is_deterministic_across_calls():
    first = measure_complexity()
    second = measure_complexity()
    assert first == second
