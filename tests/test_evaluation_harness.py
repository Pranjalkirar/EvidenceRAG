"""Tests for evidencerag.evaluation.harness -- orchestration only, using
fakes for every retriever/generator so this file never needs a real
model. Metric correctness itself is covered by
test_evaluation_retrieval_metrics.py / test_evaluation_evidence_metrics.py /
test_evaluation_answer_metrics.py / test_evaluation_gold.py.
"""

from __future__ import annotations

import pytest

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.evaluation.harness import EvaluationConfig, run_evaluation
from evidencerag.evaluation.systems import EvaluationSystems
from evidencerag.ingestion.schema import Section
from tests.chunking_fixtures import make_answer, make_paper, make_question, make_resolved_evidence
from tests.evaluation_fixtures import QueryKeyedFakeRetriever, make_result
from tests.generation_fixtures import FakeGenerator
from tests.retrieval_fixtures import make_chunk


def _single_question_paper():
    paper_id = "9999.00001"
    question = make_question(
        paper_id=paper_id,
        question_text="What is the method?",
        answers=(make_answer((make_resolved_evidence("ev", section_index=0, paragraph_index=0),)),),
    )
    section = Section(section_index=0, title="Intro", paragraphs=("Paragraph zero text.", "Paragraph one text."))
    paper = make_paper(paper_id=paper_id, sections=(section,), questions=(question,))
    mapping = EvidenceChunkMapping(
        paper_id=paper_id,
        split="train",
        question_index=0,
        question_id=question.question_id,
        answer_index=0,
        evidence_index=0,
        is_float_selected=False,
        resolved=True,
        section_index=0,
        paragraph_index=0,
        chunk_ids=("c1",),
    )
    return paper, {paper_id: (mapping,)}


def _fake_systems_all_returning(chunk_id: str, query: str):
    """One `EvaluationSystems` where every one of the 4 systems
    retrieves the same single chunk for the given `query`, at both
    top_k and candidate_depth."""
    retrievers = {}
    for name in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        retrievers[name] = QueryKeyedFakeRetriever({query: [make_result(chunk_id, 1)]})
    candidate_sources = {name: retrievers[name] for name in retrievers}
    candidate_sources["hybrid_rerank"] = retrievers["hybrid"]
    return EvaluationSystems(retrievers=retrievers, candidate_sources=candidate_sources)


def test_retrieval_mode_never_invokes_generator():
    paper, mappings_by_paper = _single_question_paper()
    chunks = [make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,))]
    systems = _fake_systems_all_returning("c1", "What is the method?")
    generator = FakeGenerator()

    config = EvaluationConfig(mode="retrieval", split="train", top_k=5, candidate_depth=20)
    records, summary = run_evaluation(
        papers=[paper], chunks=chunks, evidence_mappings_by_paper=mappings_by_paper,
        systems=systems, config=config, generator=generator,
    )

    assert generator.calls == []
    assert all(record.answer is None for record in records)
    assert all(record.answer_f1 is None for record in records)
    assert summary.systems[0].answer_f1 is None


def test_end_to_end_mode_calls_generator_and_populates_answer_f1():
    paper, mappings_by_paper = _single_question_paper()
    chunks = [make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,))]
    systems = _fake_systems_all_returning("c1", "What is the method?")
    generator = FakeGenerator(canned_answer="An answer.")

    config = EvaluationConfig(mode="end_to_end", split="train", top_k=5, candidate_depth=20)
    records, summary = run_evaluation(
        papers=[paper], chunks=chunks, evidence_mappings_by_paper=mappings_by_paper,
        systems=systems, config=config, generator=generator,
    )

    assert len(generator.calls) == 4  # once per system
    for record in records:
        assert record.answer == "An answer."
        assert record.answer_f1 == 1.0  # matches the single ("An answer.", "abstractive") reference
        assert record.answer_type == "abstractive"
        assert record.generator_model == "fake-generator"
    assert summary.systems[0].answer_f1 == 1.0


def test_end_to_end_mode_requires_a_generator():
    paper, mappings_by_paper = _single_question_paper()
    chunks = [make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,))]
    systems = _fake_systems_all_returning("c1", "What is the method?")

    config = EvaluationConfig(mode="end_to_end", split="train", top_k=5, candidate_depth=20)
    with pytest.raises(ValueError):
        run_evaluation(
            papers=[paper], chunks=chunks, evidence_mappings_by_paper=mappings_by_paper,
            systems=systems, config=config, generator=None,
        )


def test_split_mismatch_raises():
    paper, mappings_by_paper = _single_question_paper()  # paper.split == "train"
    chunks = [make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,))]
    systems = _fake_systems_all_returning("c1", "What is the method?")

    config = EvaluationConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)
    with pytest.raises(ValueError):
        run_evaluation(
            papers=[paper], chunks=chunks, evidence_mappings_by_paper=mappings_by_paper,
            systems=systems, config=config,
        )


def test_evidence_mappings_referencing_unknown_paper_raises():
    paper, _ = _single_question_paper()
    chunks = [make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,))]
    systems = _fake_systems_all_returning("c1", "What is the method?")

    bogus_mapping = EvidenceChunkMapping(
        paper_id="does-not-exist", split="train", question_index=0, question_id=None,
        answer_index=0, evidence_index=0, is_float_selected=False, resolved=True,
        section_index=0, paragraph_index=0, chunk_ids=("c1",),
    )
    config = EvaluationConfig(mode="retrieval", split="train", top_k=5, candidate_depth=20)
    with pytest.raises(ValueError):
        run_evaluation(
            papers=[paper], chunks=chunks,
            evidence_mappings_by_paper={"does-not-exist": (bogus_mapping,)},
            systems=systems, config=config,
        )


def test_recall_at_20_identical_for_hybrid_and_hybrid_rerank_by_construction():
    paper, mappings_by_paper = _single_question_paper()
    chunks = [
        make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,)),
        make_chunk(chunk_id="c2", text="other text", section_index=0, paragraph_indices=(1,)),
    ]
    query = "What is the method?"

    hybrid_candidates = [make_result("c2", 1), make_result("c1", 2)]  # gold chunk "c1" at rank 2 of 20-candidates
    hybrid = QueryKeyedFakeRetriever({query: hybrid_candidates})
    hybrid_rerank_final = QueryKeyedFakeRetriever({query: [make_result("c1", 1)]})  # reranked to top-5

    systems = EvaluationSystems(
        retrievers={
            "bm25": QueryKeyedFakeRetriever({query: [make_result("c1", 1)]}),
            "dense": QueryKeyedFakeRetriever({query: [make_result("c1", 1)]}),
            "hybrid": hybrid,
            "hybrid_rerank": hybrid_rerank_final,
        },
        candidate_sources={
            "bm25": QueryKeyedFakeRetriever({query: [make_result("c1", 1)]}),
            "dense": QueryKeyedFakeRetriever({query: [make_result("c1", 1)]}),
            "hybrid": hybrid,
            "hybrid_rerank": hybrid,  # same object as "hybrid"'s own candidate source
        },
    )

    config = EvaluationConfig(mode="retrieval", split="train", top_k=5, candidate_depth=20)
    records, _ = run_evaluation(
        papers=[paper], chunks=chunks, evidence_mappings_by_paper=mappings_by_paper,
        systems=systems, config=config,
    )

    recall_20_by_system = {record.system: record.recall_at_20 for record in records}
    assert recall_20_by_system["hybrid"] == recall_20_by_system["hybrid_rerank"] == 1.0
