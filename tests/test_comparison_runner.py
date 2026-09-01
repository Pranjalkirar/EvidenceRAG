"""Tests for evidencerag.comparison.runner.run_comparison --
orchestration only, using fakes for every pipeline so this file never
needs a real model or LangChain. Metric correctness itself is covered
by the existing M7 test_evaluation_retrieval_metrics.py /
test_evaluation_evidence_metrics.py / test_evaluation_answer_metrics.py
/ test_evaluation_gold.py (this module reuses those functions
unchanged -- see runner.py's docstring).
"""

from __future__ import annotations

import pytest

from evidencerag.chunking.evidence_map import EvidenceChunkMapping
from evidencerag.comparison.runner import ComparisonConfig, run_comparison
from evidencerag.comparison.schema import StageResult
from evidencerag.ingestion.schema import Section
from tests.chunking_fixtures import make_answer, make_paper, make_question, make_resolved_evidence
from tests.retrieval_fixtures import make_chunk


class FakeQuestionPipeline:
    """Deterministic stand-in for CustomPipeline/LangChainPipeline:
    returns a caller-supplied `StageResult` per distinct question text.
    """

    def __init__(self, results_by_question: dict[str, StageResult]) -> None:
        self._results_by_question = results_by_question
        self.calls: list[str] = []

    def run_question(self, question_text: str) -> StageResult:
        self.calls.append(question_text)
        return self._results_by_question[question_text]


def _stage(
    top5=("c1",), candidates=("c1",), answer=None, generation_latency=None,
    retrieval_latency=0.01, rerank_latency=0.005, total_latency=0.02,
) -> StageResult:
    return StageResult(
        top5_chunk_ids=tuple(top5),
        candidate20_chunk_ids=tuple(candidates),
        retrieval_latency_s=retrieval_latency,
        rerank_latency_s=rerank_latency,
        generation_latency_s=generation_latency,
        total_latency_s=total_latency,
        answer=answer,
    )


def _single_question_paper():
    paper_id = "9999.00001"
    question = make_question(
        paper_id=paper_id,
        question_text="What is the method?",
        answers=(make_answer((make_resolved_evidence("ev", section_index=0, paragraph_index=0),)),),
    )
    section = Section(section_index=0, title="Intro", paragraphs=("Paragraph zero text.", "Paragraph one text."))
    paper = make_paper(paper_id=paper_id, split="validation", sections=(section,), questions=(question,))
    mapping = EvidenceChunkMapping(
        paper_id=paper_id, split="validation", question_index=0, question_id=question.question_id,
        answer_index=0, evidence_index=0, is_float_selected=False, resolved=True,
        section_index=0, paragraph_index=0, chunk_ids=("c1",),
    )
    return paper, {paper_id: (mapping,)}


def _chunks():
    return [make_chunk(chunk_id="c1", text="chunk text", section_index=0, paragraph_indices=(0,))]


def test_records_one_row_per_implementation_per_question():
    paper, mappings_by_paper = _single_question_paper()
    pipelines = {
        "custom": FakeQuestionPipeline({"What is the method?": _stage(top5=("c1",), candidates=("c1",))}),
        "langchain": FakeQuestionPipeline({"What is the method?": _stage(top5=("c1",), candidates=("c1",))}),
    }
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)

    records, summary = run_comparison(
        papers=[paper], chunks=_chunks(), evidence_mappings_by_paper=mappings_by_paper,
        pipelines=pipelines, config=config,
    )

    assert len(records) == 2
    assert {r.implementation for r in records} == {"custom", "langchain"}
    assert all(r.recall_at_5 == 1.0 for r in records)
    assert {s.implementation for s in summary.implementations} == {"custom", "langchain"}


def test_retrieval_mode_leaves_answer_and_answer_f1_none():
    paper, mappings_by_paper = _single_question_paper()
    pipelines = {"custom": FakeQuestionPipeline({"What is the method?": _stage(answer=None)})}
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)

    records, summary = run_comparison(
        papers=[paper], chunks=_chunks(), evidence_mappings_by_paper=mappings_by_paper,
        pipelines=pipelines, config=config,
    )

    assert records[0].answer is None
    assert records[0].answer_f1 is None
    assert summary.implementations[0].answer_f1 is None


def test_end_to_end_mode_scores_answer_f1():
    paper, mappings_by_paper = _single_question_paper()
    pipelines = {
        "custom": FakeQuestionPipeline(
            {"What is the method?": _stage(answer="An answer.", generation_latency=0.1)}
        )
    }
    config = ComparisonConfig(mode="end_to_end", split="validation", top_k=5, candidate_depth=20)

    records, summary = run_comparison(
        papers=[paper], chunks=_chunks(), evidence_mappings_by_paper=mappings_by_paper,
        pipelines=pipelines, config=config,
    )

    assert records[0].answer == "An answer."
    assert records[0].answer_f1 == 1.0  # matches make_answer's "An answer." reference exactly
    assert records[0].answer_type == "abstractive"
    assert summary.implementations[0].answer_f1 == 1.0
    assert summary.implementations[0].mean_generation_latency_s == 0.1


def test_no_pipelines_raises():
    paper, mappings_by_paper = _single_question_paper()
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)
    with pytest.raises(ValueError):
        run_comparison(
            papers=[paper], chunks=_chunks(), evidence_mappings_by_paper=mappings_by_paper,
            pipelines={}, config=config,
        )


def test_split_mismatch_raises():
    paper, mappings_by_paper = _single_question_paper()  # paper.split == "validation"
    pipelines = {"custom": FakeQuestionPipeline({"What is the method?": _stage()})}
    config = ComparisonConfig(mode="retrieval", split="test", top_k=5, candidate_depth=20)
    with pytest.raises(ValueError):
        run_comparison(
            papers=[paper], chunks=_chunks(), evidence_mappings_by_paper=mappings_by_paper,
            pipelines=pipelines, config=config,
        )


def test_evidence_mappings_referencing_unknown_paper_raises():
    paper, _ = _single_question_paper()
    pipelines = {"custom": FakeQuestionPipeline({"What is the method?": _stage()})}
    bogus_mapping = EvidenceChunkMapping(
        paper_id="does-not-exist", split="validation", question_index=0, question_id=None,
        answer_index=0, evidence_index=0, is_float_selected=False, resolved=True,
        section_index=0, paragraph_index=0, chunk_ids=("c1",),
    )
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)
    with pytest.raises(ValueError):
        run_comparison(
            papers=[paper], chunks=_chunks(),
            evidence_mappings_by_paper={"does-not-exist": (bogus_mapping,)},
            pipelines=pipelines, config=config,
        )


def test_max_questions_truncates_across_papers():
    paper1, mappings1 = _single_question_paper()
    question2 = make_question(paper_id="9999.00002", question_text="A second question?")
    section = Section(section_index=0, title="Intro", paragraphs=("Some text.",))
    paper2 = make_paper(paper_id="9999.00002", split="validation", sections=(section,), questions=(question2,))

    pipelines = {
        "custom": FakeQuestionPipeline(
            {"What is the method?": _stage(top5=("c1",), candidates=("c1",)), "A second question?": _stage()}
        )
    }
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20, max_questions=1)

    records, _ = run_comparison(
        papers=[paper1, paper2], chunks=_chunks(),
        evidence_mappings_by_paper={**mappings1, "9999.00002": ()},
        pipelines=pipelines, config=config,
    )

    assert len(records) == 1
    assert records[0].question_text == "What is the method?"


def test_recall_at_20_uses_candidate_ids_not_top5():
    paper, mappings_by_paper = _single_question_paper()
    # gold chunk "c1" only appears in the wider candidate set, not top5
    pipelines = {
        "custom": FakeQuestionPipeline(
            {"What is the method?": _stage(top5=("c9",), candidates=("c9", "c8", "c1"))}
        )
    }
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)
    all_chunks = _chunks() + [
        make_chunk(chunk_id="c8", text="other chunk 8", section_index=0, paragraph_indices=(1,)),
        make_chunk(chunk_id="c9", text="other chunk 9", section_index=0, paragraph_indices=(1,)),
    ]

    records, _ = run_comparison(
        papers=[paper], chunks=all_chunks, evidence_mappings_by_paper=mappings_by_paper,
        pipelines=pipelines, config=config,
    )

    assert records[0].recall_at_5 == 0.0
    assert records[0].recall_at_20 == 1.0


def test_pipelines_are_called_with_question_text():
    paper, mappings_by_paper = _single_question_paper()
    pipeline = FakeQuestionPipeline({"What is the method?": _stage()})
    config = ComparisonConfig(mode="retrieval", split="validation", top_k=5, candidate_depth=20)

    run_comparison(
        papers=[paper], chunks=_chunks(), evidence_mappings_by_paper=mappings_by_paper,
        pipelines={"custom": pipeline}, config=config,
    )

    assert pipeline.calls == ["What is the method?"]
