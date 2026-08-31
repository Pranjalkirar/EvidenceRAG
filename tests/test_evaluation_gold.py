"""Tests for evidencerag.evaluation.gold."""

from __future__ import annotations

import pytest

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.evaluation.gold import build_gold, reference_answer_and_type
from tests.chunking_fixtures import (
    make_answer,
    make_float_selected_evidence,
    make_paper,
    make_question,
    make_resolved_evidence,
    make_unresolved_evidence,
)
from tests.evaluation_fixtures import (
    make_boolean_answer,
    make_extractive_answer,
    make_unanswerable_answer,
)


def _mapping(
    paper_id="9999.00001",
    split="train",
    question_index=0,
    question_id="Q1",
    answer_index=0,
    evidence_index=0,
    is_float_selected=False,
    resolved=True,
    section_index=0,
    paragraph_index=0,
    chunk_ids=(),
) -> EvidenceChunkMapping:
    return EvidenceChunkMapping(
        paper_id=paper_id,
        split=split,
        question_index=question_index,
        question_id=question_id,
        answer_index=answer_index,
        evidence_index=evidence_index,
        is_float_selected=is_float_selected,
        resolved=resolved,
        section_index=section_index,
        paragraph_index=paragraph_index,
        chunk_ids=chunk_ids,
    )


# ---- reference_answer_and_type precedence ----


def test_reference_answer_and_type_unanswerable():
    answer = make_unanswerable_answer()
    assert reference_answer_and_type(answer) == ("Unanswerable", "none")


def test_reference_answer_and_type_extractive():
    answer = make_extractive_answer((), extractive_spans=("span one", "span two"))
    assert reference_answer_and_type(answer) == ("span one, span two", "extractive")


def test_reference_answer_and_type_abstractive():
    answer = make_answer(())
    assert reference_answer_and_type(answer) == ("An answer.", "abstractive")


def test_reference_answer_and_type_boolean_yes():
    answer = make_boolean_answer((), yes_no=True)
    assert reference_answer_and_type(answer) == ("Yes", "boolean")


def test_reference_answer_and_type_boolean_no():
    answer = make_boolean_answer((), yes_no=False)
    assert reference_answer_and_type(answer) == ("No", "boolean")


def test_reference_answer_and_type_raises_when_nothing_set():
    from evidencerag.ingestion.schema import Answer

    empty_answer = Answer(
        annotation_id="ann1",
        worker_id="worker1",
        unanswerable=False,
        yes_no=None,
        free_form_answer="",
        extractive_spans=(),
        evidence=(),
    )
    with pytest.raises(ValueError):
        reference_answer_and_type(empty_answer)


# ---- build_gold: chunk_references (union within an answer, never across answers) ----


def test_build_gold_unions_chunk_ids_within_one_answer():
    paper_id = "9999.00001"
    evidence = (
        make_resolved_evidence("ev one", section_index=0, paragraph_index=0),
        make_resolved_evidence("ev two", section_index=0, paragraph_index=1),
    )
    answer = make_answer(evidence)
    question = make_question(paper_id=paper_id, answers=(answer,))
    paper = make_paper(paper_id=paper_id, questions=(question,))

    mappings = (
        _mapping(paper_id=paper_id, question_index=0, answer_index=0, evidence_index=0, chunk_ids=("c1",)),
        _mapping(paper_id=paper_id, question_index=0, answer_index=0, evidence_index=1, chunk_ids=("c2",)),
    )

    gold = build_gold(paper, mappings)
    assert gold[0].chunk_references == (frozenset({"c1", "c2"}),)


def test_build_gold_keeps_separate_answers_as_separate_references_not_unioned():
    paper_id = "9999.00001"
    answer_a = make_answer((make_resolved_evidence("ev A", section_index=0, paragraph_index=0),))
    answer_b = make_answer((make_resolved_evidence("ev B", section_index=0, paragraph_index=1),))
    question = make_question(paper_id=paper_id, answers=(answer_a, answer_b))
    paper = make_paper(paper_id=paper_id, questions=(question,))

    mappings = (
        _mapping(paper_id=paper_id, question_index=0, answer_index=0, evidence_index=0, chunk_ids=("c1",)),
        _mapping(paper_id=paper_id, question_index=0, answer_index=1, evidence_index=0, chunk_ids=("c2",)),
    )

    gold = build_gold(paper, mappings)
    # Two SEPARATE references, not one unioned {"c1", "c2"} set.
    assert gold[0].chunk_references == (frozenset({"c1"}), frozenset({"c2"}))


def test_build_gold_excludes_answer_with_zero_mappable_chunks():
    paper_id = "9999.00001"
    answer_resolved = make_answer((make_resolved_evidence("ev A", section_index=0, paragraph_index=0),))
    answer_unresolved = make_answer((make_unresolved_evidence("ev B"),))
    question = make_question(paper_id=paper_id, answers=(answer_resolved, answer_unresolved))
    paper = make_paper(paper_id=paper_id, questions=(question,))

    mappings = (
        _mapping(paper_id=paper_id, question_index=0, answer_index=0, evidence_index=0, chunk_ids=("c1",)),
        _mapping(
            paper_id=paper_id,
            question_index=0,
            answer_index=1,
            evidence_index=0,
            resolved=False,
            section_index=None,
            paragraph_index=None,
            chunk_ids=(),
        ),
    )

    gold = build_gold(paper, mappings)
    # Only the resolved answer contributes a reference.
    assert gold[0].chunk_references == (frozenset({"c1"}),)


def test_build_gold_question_with_all_answers_unmappable_has_no_chunk_references():
    paper_id = "9999.00001"
    answer = make_answer((make_unresolved_evidence("ev"),))
    question = make_question(paper_id=paper_id, answers=(answer,))
    paper = make_paper(paper_id=paper_id, questions=(question,))

    mappings = (
        _mapping(
            paper_id=paper_id,
            question_index=0,
            answer_index=0,
            evidence_index=0,
            resolved=False,
            section_index=None,
            paragraph_index=None,
            chunk_ids=(),
        ),
    )

    gold = build_gold(paper, mappings)
    assert gold[0].chunk_references == ()


# ---- build_gold: evidence_text_references (text only, resolution-independent) ----


def test_build_gold_evidence_text_references_excludes_float_selected_but_keeps_unresolved():
    paper_id = "9999.00001"
    evidence = (
        make_unresolved_evidence("unresolved text evidence"),
        make_float_selected_evidence("figure caption", figure_index=0),
    )
    answer = make_answer(evidence)
    question = make_question(paper_id=paper_id, answers=(answer,))
    paper = make_paper(paper_id=paper_id, questions=(question,))

    gold = build_gold(paper, mappings=())
    assert gold[0].evidence_text_references == (frozenset({"unresolved text evidence"}),)


def test_build_gold_evidence_text_references_present_per_answer_even_when_empty():
    paper_id = "9999.00001"
    answer = make_unanswerable_answer(evidence=())
    question = make_question(paper_id=paper_id, answers=(answer,))
    paper = make_paper(paper_id=paper_id, questions=(question,))

    gold = build_gold(paper, mappings=())
    assert gold[0].evidence_text_references == (frozenset(),)


# ---- build_gold: mismatched paper/mapping raises ----


def test_build_gold_raises_on_paper_id_mismatch():
    paper = make_paper(paper_id="paper-A", questions=(make_question(paper_id="paper-A"),))
    mappings = (_mapping(paper_id="paper-B"),)
    with pytest.raises(ValueError):
        build_gold(paper, mappings)
