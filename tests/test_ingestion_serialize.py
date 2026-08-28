from evidencerag.ingestion.normalize import normalize_paper
from evidencerag.ingestion.serialize import load_papers, paper_from_dict, paper_to_dict, save_papers
from tests.qasper_fixtures import PAPER_1_TRAIN, PAPER_2_VALIDATION


def test_paper_dict_round_trip_is_lossless():
    paper = normalize_paper(PAPER_1_TRAIN, split="train")
    round_tripped = paper_from_dict(paper_to_dict(paper))
    assert round_tripped == paper


def test_save_and_load_papers_jsonl(tmp_path):
    papers = [
        normalize_paper(PAPER_1_TRAIN, split="train"),
        normalize_paper(PAPER_2_VALIDATION, split="validation"),
    ]
    output_path = tmp_path / "papers.jsonl"

    n_written = save_papers(papers, output_path)
    assert n_written == 2
    assert output_path.exists()

    loaded = list(load_papers(output_path))
    assert loaded == papers


def test_save_papers_creates_parent_directories(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "papers.jsonl"
    papers = [normalize_paper(PAPER_1_TRAIN, split="train")]

    save_papers(papers, nested_path)
    assert nested_path.exists()


def test_jsonl_file_has_one_paper_per_line(tmp_path):
    papers = [
        normalize_paper(PAPER_1_TRAIN, split="train"),
        normalize_paper(PAPER_2_VALIDATION, split="validation"),
    ]
    output_path = tmp_path / "papers.jsonl"
    save_papers(papers, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
