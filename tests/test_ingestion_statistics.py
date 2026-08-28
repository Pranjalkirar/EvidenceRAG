from evidencerag.ingestion.normalize import normalize_paper
from evidencerag.ingestion.statistics import compute_statistics
from tests.qasper_fixtures import PAPER_1_TRAIN, PAPER_2_VALIDATION


def test_statistics_are_computed_from_actual_data_not_fabricated():
    papers = [
        normalize_paper(PAPER_1_TRAIN, split="train"),
        normalize_paper(PAPER_2_VALIDATION, split="validation"),
    ]
    stats = compute_statistics(papers)

    train_stats = stats.by_split["train"]
    assert train_stats.num_papers == 1
    assert train_stats.num_questions == 2
    # Q1 has 1 answer, Q2 has 2 answers => 3 total.
    assert train_stats.num_answers == 3
    assert train_stats.num_unanswerable_answers == 1

    val_stats = stats.by_split["validation"]
    assert val_stats.num_papers == 1
    assert val_stats.num_questions == 1
    assert val_stats.num_answers == 1

    assert stats.total_papers == 2
    assert stats.total_questions == 3
    assert stats.total_answers == 4


def test_statistics_split_by_split_not_merged():
    papers = [
        normalize_paper(PAPER_1_TRAIN, split="train"),
        normalize_paper(PAPER_2_VALIDATION, split="validation"),
    ]
    stats = compute_statistics(papers)
    assert set(stats.by_split.keys()) == {"train", "validation"}


def test_statistics_track_evidence_resolution():
    papers = [
        normalize_paper(PAPER_1_TRAIN, split="train"),
        normalize_paper(PAPER_2_VALIDATION, split="validation"),
    ]
    stats = compute_statistics(papers)

    # PAPER_1_TRAIN: one resolved paragraph-evidence answer, one
    # unanswerable (no evidence), one resolved figure-evidence answer.
    train_stats = stats.by_split["train"]
    assert train_stats.num_evidence_resolved == 2
    assert train_stats.num_evidence_unresolved == 0

    # PAPER_2_VALIDATION: one deliberately-unresolvable evidence string.
    val_stats = stats.by_split["validation"]
    assert val_stats.num_evidence_resolved == 0
    assert val_stats.num_evidence_unresolved == 1


def test_format_report_does_not_crash_on_empty_input():
    stats = compute_statistics([])
    report = stats.format_report()
    assert "QASPER dataset statistics" in report
