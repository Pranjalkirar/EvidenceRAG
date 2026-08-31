"""Tests for evidencerag.evaluation.retrieval_metrics."""

from __future__ import annotations

import pytest

from evidencerag.evaluation import retrieval_metrics
from tests.evaluation_fixtures import make_result


# ---- recall_at_k / reciprocal_rank (single reference) ----


def test_recall_at_k_no_relevant_result():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    assert retrieval_metrics.recall_at_k(results, {"z"}, k=3) == 0.0


def test_recall_at_k_relevant_at_rank_1():
    results = [make_result("a", 1), make_result("b", 2)]
    assert retrieval_metrics.recall_at_k(results, {"a"}, k=2) == 1.0


def test_recall_at_k_relevant_at_later_rank():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    assert retrieval_metrics.recall_at_k(results, {"c"}, k=3) == 1.0
    # Not within the truncated top-1 window.
    assert retrieval_metrics.recall_at_k(results, {"c"}, k=1) == 0.0


def test_recall_at_k_multiple_relevant_chunks_partial_overlap():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    # Two of three gold chunks retrieved -> 2/3 recall.
    assert retrieval_metrics.recall_at_k(results, {"a", "c", "z"}, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_requires_non_empty_relevant_set():
    with pytest.raises(ValueError):
        retrieval_metrics.recall_at_k([make_result("a", 1)], set(), k=1)


def test_reciprocal_rank_no_relevant_result():
    results = [make_result("a", 1), make_result("b", 2)]
    assert retrieval_metrics.reciprocal_rank(results, {"z"}) == 0.0


def test_reciprocal_rank_relevant_at_rank_1():
    results = [make_result("a", 1), make_result("b", 2)]
    assert retrieval_metrics.reciprocal_rank(results, {"a"}) == 1.0


def test_reciprocal_rank_relevant_at_later_rank():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    assert retrieval_metrics.reciprocal_rank(results, {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_multiple_relevant_chunks_uses_earliest():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    # "b" (rank 2) and "c" (rank 3) are both relevant -- earliest wins.
    assert retrieval_metrics.reciprocal_rank(results, {"b", "c"}) == pytest.approx(1 / 2)


def test_reciprocal_rank_requires_non_empty_relevant_set():
    with pytest.raises(ValueError):
        retrieval_metrics.reciprocal_rank([make_result("a", 1)], set())


# ---- max_recall_at_k / max_reciprocal_rank (max across references) ----


def test_max_recall_at_k_no_references_returns_none():
    results = [make_result("a", 1)]
    assert retrieval_metrics.max_recall_at_k(results, [], k=1) is None


def test_max_reciprocal_rank_no_references_returns_none():
    results = [make_result("a", 1)]
    assert retrieval_metrics.max_reciprocal_rank(results, []) is None


def test_max_recall_at_k_takes_max_across_references():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    # Reference 1 ({"z"}) scores 0; reference 2 ({"a"}) scores 1 -- max wins.
    references = [frozenset({"z"}), frozenset({"a"})]
    assert retrieval_metrics.max_recall_at_k(results, references, k=3) == 1.0


def test_max_reciprocal_rank_takes_max_across_references():
    results = [make_result("a", 1), make_result("b", 2), make_result("c", 3)]
    # Reference 1 ({"c"}) scores 1/3; reference 2 ({"a"}) scores 1.0 -- max wins.
    references = [frozenset({"c"}), frozenset({"a"})]
    assert retrieval_metrics.max_reciprocal_rank(results, references) == 1.0


def test_max_recall_at_k_does_not_union_references():
    # If references were unioned into {"a", "b"}, recall@1 against just
    # ["a"] would be 1/2. Kept separate, reference {"a"} alone scores 1.0
    # and reference {"b"} alone scores 0.0 -- max is 1.0, not 0.5.
    results = [make_result("a", 1)]
    references = [frozenset({"a"}), frozenset({"b"})]
    assert retrieval_metrics.max_recall_at_k(results, references, k=1) == 1.0
