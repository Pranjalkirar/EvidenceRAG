from evidencerag.ingestion.validate import validate_paper_row, validate_split
from tests.qasper_fixtures import PAPER_1_TRAIN, PAPER_2_VALIDATION


def test_valid_paper_rows_have_no_issues():
    assert validate_paper_row(PAPER_1_TRAIN) == []
    assert validate_paper_row(PAPER_2_VALIDATION) == []


def test_missing_required_key_is_reported():
    broken = {k: v for k, v in PAPER_1_TRAIN.items() if k != "abstract"}
    issues = validate_paper_row(broken)
    assert any("abstract" in issue for issue in issues)


def test_mismatched_parallel_arrays_in_full_text_reported():
    broken = dict(PAPER_1_TRAIN)
    broken["full_text"] = {
        "section_name": ["Introduction", "Method"],
        "paragraphs": [["only one section's worth of paragraphs"]],
    }
    issues = validate_paper_row(broken)
    assert any("parallel arrays" in issue for issue in issues)


def test_answer_missing_key_is_reported():
    import copy

    broken = copy.deepcopy(PAPER_1_TRAIN)
    del broken["qas"]["answers"][0]["answer"][0]["evidence"]
    issues = validate_paper_row(broken)
    assert any("evidence" in issue for issue in issues)


def test_validate_split_prefixes_split_name_and_detects_duplicates():
    rows = [PAPER_1_TRAIN, PAPER_1_TRAIN]  # duplicate id on purpose
    issues = validate_split(rows, split_name="train")
    assert any(issue.startswith("[train]") for issue in issues)
    assert any("duplicate paper id" in issue for issue in issues)


def test_optional_figures_and_tables_absence_is_not_an_issue():
    # PAPER_2_VALIDATION has no 'figures_and_tables' key at all.
    assert "figures_and_tables" not in PAPER_2_VALIDATION
    assert validate_paper_row(PAPER_2_VALIDATION) == []
