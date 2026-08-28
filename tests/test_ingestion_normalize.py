from evidencerag.ingestion.normalize import normalize_paper, normalize_split
from tests.qasper_fixtures import PAPER_1_TRAIN, PAPER_2_VALIDATION, raw_dataset_dict


def test_normalize_preserves_paper_question_relationship():
    paper = normalize_paper(PAPER_1_TRAIN, split="train")
    assert paper.paper_id == "1000.00001"
    assert paper.split == "train"
    assert len(paper.questions) == 2
    assert all(q.paper_id == paper.paper_id for q in paper.questions)


def test_normalize_preserves_multiple_answers_per_question():
    paper = normalize_paper(PAPER_1_TRAIN, split="train")
    q2 = paper.questions[1]
    assert len(q2.answers) == 2
    assert q2.answers[0].unanswerable is True
    assert q2.answers[1].yes_no is True


def test_normalize_does_not_flatten_multiple_evidence_pieces():
    # Add a second evidence paragraph to an answer to confirm both are kept
    # as separate EvidenceSpans, not concatenated into one string.
    import copy

    row = copy.deepcopy(PAPER_1_TRAIN)
    row["qas"]["answers"][0]["answer"][0]["evidence"].append(
        "This paper studies event polarity propagation."
    )
    paper = normalize_paper(row, split="train")
    evidence = paper.questions[0].answers[0].evidence
    assert len(evidence) == 2
    assert evidence[0].text != evidence[1].text


def test_evidence_resolves_to_correct_paragraph():
    paper = normalize_paper(PAPER_1_TRAIN, split="train")
    evidence = paper.questions[0].answers[0].evidence[0]
    assert evidence.resolved is True
    assert evidence.is_float_selected is False
    assert evidence.section_index == 1  # "Method" section
    assert evidence.paragraph_index == 0


def test_float_selected_evidence_resolves_to_figure():
    paper = normalize_paper(PAPER_1_TRAIN, split="train")
    evidence = paper.questions[1].answers[1].evidence[0]
    assert evidence.is_float_selected is True
    assert evidence.resolved is True
    assert evidence.figure_or_table_index == 0


def test_unresolvable_evidence_is_marked_unresolved_not_guessed():
    paper = normalize_paper(PAPER_2_VALIDATION, split="validation")
    evidence = paper.questions[0].answers[0].evidence[0]
    assert evidence.resolved is False
    assert evidence.section_index is None
    assert evidence.paragraph_index is None


def test_highlighted_evidence_kept_separate_from_evidence():
    paper = normalize_paper(PAPER_1_TRAIN, split="train")
    answer = paper.questions[0].answers[0]
    assert answer.highlighted_evidence == ("propagates seed polarity scores",)
    assert len(answer.evidence) == 1  # not merged with highlighted_evidence


def test_missing_optional_qas_fields_default_to_none():
    paper = normalize_paper(PAPER_2_VALIDATION, split="validation")
    question = paper.questions[0]
    assert question.question_writer is None
    assert question.search_query is None


def test_empty_section_has_zero_paragraphs():
    paper = normalize_paper(PAPER_2_VALIDATION, split="validation")
    assert paper.sections[0].title == ""
    assert paper.sections[0].paragraphs == ()


def test_missing_figures_and_tables_normalizes_to_empty_tuple():
    paper = normalize_paper(PAPER_2_VALIDATION, split="validation")
    assert paper.figures_and_tables == ()


def test_normalize_split_preserves_split_label_and_does_not_merge():
    raw = raw_dataset_dict()
    train_papers = normalize_split(raw["train"], split="train")
    val_papers = normalize_split(raw["validation"], split="validation")

    assert all(p.split == "train" for p in train_papers)
    assert all(p.split == "validation" for p in val_papers)
    assert {p.paper_id for p in train_papers}.isdisjoint({p.paper_id for p in val_papers})
