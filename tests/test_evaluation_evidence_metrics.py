"""Tests for evidencerag.evaluation.evidence_metrics."""

from __future__ import annotations

import pytest

from evidencerag.evaluation import evidence_metrics
from tests.chunking_fixtures import make_paper
from tests.retrieval_fixtures import make_chunk
from evidencerag.ingestion.schema import Section


# ---- paragraph_set_f1 ----


def test_paragraph_set_f1_exact_match():
    assert evidence_metrics.paragraph_set_f1({"para one text"}, {"para one text"}) == 1.0


def test_paragraph_set_f1_empty_predicted_and_empty_reference():
    assert evidence_metrics.paragraph_set_f1(frozenset(), frozenset()) == 1.0


def test_paragraph_set_f1_empty_predicted_non_empty_reference():
    assert evidence_metrics.paragraph_set_f1(frozenset(), {"needed evidence"}) == 0.0


def test_paragraph_set_f1_no_overlap():
    assert evidence_metrics.paragraph_set_f1({"a"}, {"b"}) == 0.0


def test_paragraph_set_f1_partial_overlap():
    predicted = {"shared", "extra"}
    reference = {"shared", "missing"}
    # precision = 1/2, recall = 1/2 -> F1 = 0.5
    assert evidence_metrics.paragraph_set_f1(predicted, reference) == pytest.approx(0.5)


# ---- evidence_f1 (max across references) ----


def test_evidence_f1_multiple_references_takes_max():
    predicted = {"para A"}
    references = [frozenset({"para B"}), frozenset({"para A"})]
    assert evidence_metrics.evidence_f1(predicted, references) == 1.0


def test_evidence_f1_requires_at_least_one_reference():
    with pytest.raises(ValueError):
        evidence_metrics.evidence_f1(frozenset({"x"}), [])


# ---- retrieved_paragraph_texts ----


def _paper_with_two_paragraphs():
    section = Section(section_index=0, title="Intro", paragraphs=("Paragraph zero text.", "Paragraph one text."))
    return make_paper(sections=(section,))


def test_retrieved_paragraph_texts_recovers_original_text():
    paper = _paper_with_two_paragraphs()
    chunks = [make_chunk(chunk_id="c1", text="irrelevant", section_index=0, paragraph_indices=(0,))]
    chunk_by_id = {c.chunk_id: c for c in chunks}
    result = evidence_metrics.retrieved_paragraph_texts(["c1"], chunk_by_id, paper)
    assert result == frozenset({"Paragraph zero text."})


def test_retrieved_paragraph_texts_deduplicates_shared_paragraph():
    paper = _paper_with_two_paragraphs()
    # Two chunks both covering paragraph 0 (e.g. an oversized paragraph
    # split across chunks) must contribute its text only once.
    chunks = [
        make_chunk(chunk_id="c1", text="part 1", section_index=0, paragraph_indices=(0,)),
        make_chunk(chunk_id="c2", text="part 2", section_index=0, paragraph_indices=(0,)),
    ]
    chunk_by_id = {c.chunk_id: c for c in chunks}
    result = evidence_metrics.retrieved_paragraph_texts(["c1", "c2"], chunk_by_id, paper)
    assert result == frozenset({"Paragraph zero text."})


def test_retrieved_paragraph_texts_abstract_chunk_contributes_nothing():
    paper = _paper_with_two_paragraphs()
    chunks = [make_chunk(chunk_id="abs", text="abstract text", section_index=None, paragraph_indices=())]
    chunk_by_id = {c.chunk_id: c for c in chunks}
    result = evidence_metrics.retrieved_paragraph_texts(["abs"], chunk_by_id, paper)
    assert result == frozenset()


def test_retrieved_paragraph_texts_raises_on_paper_mismatch():
    paper = make_paper(paper_id="paper-A", sections=(Section(section_index=0, title="S", paragraphs=("text",)),))
    chunks = [make_chunk(chunk_id="c1", text="x", paper_id="paper-B", section_index=0, paragraph_indices=(0,))]
    chunk_by_id = {c.chunk_id: c for c in chunks}
    with pytest.raises(ValueError):
        evidence_metrics.retrieved_paragraph_texts(["c1"], chunk_by_id, paper)
