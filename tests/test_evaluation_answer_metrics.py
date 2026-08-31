"""Tests for evidencerag.evaluation.answer_metrics."""

from __future__ import annotations

import pytest

from evidencerag.evaluation import answer_metrics


# ---- normalize_answer ----


def test_normalize_answer_lowercases():
    assert answer_metrics.normalize_answer("HELLO World") == "hello world"


def test_normalize_answer_strips_punctuation():
    assert answer_metrics.normalize_answer("Hello, world!") == "hello world"


def test_normalize_answer_removes_articles():
    assert answer_metrics.normalize_answer("a the answer") == "answer"


def test_normalize_answer_collapses_whitespace():
    assert answer_metrics.normalize_answer("hello   world") == "hello world"


# ---- token_f1_score ----


def test_token_f1_score_exact_match():
    assert answer_metrics.token_f1_score("the cat sat", "The Cat Sat.") == 1.0


def test_token_f1_score_partial_overlap():
    # normalized prediction: "cat dog" ; reference: "cat bird" -> 1 shared token
    # precision = 1/2, recall = 1/2 -> F1 = 0.5
    assert answer_metrics.token_f1_score("cat dog", "cat bird") == pytest.approx(0.5)


def test_token_f1_score_no_overlap():
    assert answer_metrics.token_f1_score("apple", "orange") == 0.0


def test_token_f1_score_case_and_punctuation_normalization():
    assert answer_metrics.token_f1_score("Yes!", "yes") == 1.0


def test_token_f1_score_article_normalization():
    assert answer_metrics.token_f1_score("a cat", "the cat") == 1.0


# ---- answer_f1_and_type (max across references) ----


def test_answer_f1_and_type_exact_match():
    f1, answer_type = answer_metrics.answer_f1_and_type("the cat sat", [("The Cat Sat.", "abstractive")])
    assert f1 == 1.0
    assert answer_type == "abstractive"


def test_answer_f1_and_type_multiple_references_takes_max_and_matching_type():
    references = [("completely different", "abstractive"), ("the cat sat", "extractive")]
    f1, answer_type = answer_metrics.answer_f1_and_type("the cat sat", references)
    assert f1 == 1.0
    assert answer_type == "extractive"


def test_answer_f1_and_type_boolean():
    f1, answer_type = answer_metrics.answer_f1_and_type("Yes", [("Yes", "boolean")])
    assert f1 == 1.0
    assert answer_type == "boolean"


def test_answer_f1_and_type_unanswerable():
    f1, answer_type = answer_metrics.answer_f1_and_type("Unanswerable", [("Unanswerable", "none")])
    assert f1 == 1.0
    assert answer_type == "none"


def test_answer_f1_and_type_requires_at_least_one_reference():
    with pytest.raises(ValueError):
        answer_metrics.answer_f1_and_type("anything", [])
