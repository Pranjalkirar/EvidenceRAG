from evidencerag.chunking.chunker import chunk_paper
from evidencerag.chunking.evidence_map import map_evidence_to_chunks
from evidencerag.chunking.serialize import (
    chunk_from_dict,
    chunk_to_dict,
    load_chunks,
    load_evidence_mappings,
    save_chunks,
    save_evidence_mappings,
)
from evidencerag.ingestion.schema import Section
from tests.chunking_fixtures import (
    WordCountTokenizer,
    make_answer,
    make_paper,
    make_question,
    make_resolved_evidence,
    make_words_paragraph,
)

TOKENIZER = WordCountTokenizer()


def _build_sample_paper_and_chunks():
    section = Section(
        section_index=0,
        title="Method",
        paragraphs=(make_words_paragraph(30, "a"), make_words_paragraph(30, "b")),
    )
    evidence = make_resolved_evidence(make_words_paragraph(30, "a"), section_index=0, paragraph_index=0)
    answer = make_answer(evidence=(evidence,))
    question = make_question(paper_id="1000.00001", answers=(answer,))
    paper = make_paper(paper_id="1000.00001", sections=(section,), questions=(question,))
    chunks = chunk_paper(paper, tokenizer=TOKENIZER)
    return paper, chunks


def test_chunk_dict_round_trip_is_lossless():
    _, chunks = _build_sample_paper_and_chunks()
    for chunk in chunks:
        assert chunk_from_dict(chunk_to_dict(chunk)) == chunk


def test_chunk_dict_round_trip_handles_none_section_index_for_abstract_chunks():
    section = Section(section_index=0, title="Method", paragraphs=(make_words_paragraph(30),))
    paper = make_paper(paper_id="1000.00001", abstract="A short abstract.", sections=(section,))
    chunks = chunk_paper(paper, tokenizer=TOKENIZER)

    abstract_chunks = [c for c in chunks if c.section_title == "Abstract"]
    assert abstract_chunks
    for chunk in abstract_chunks:
        assert chunk.section_index is None
        round_tripped = chunk_from_dict(chunk_to_dict(chunk))
        assert round_tripped == chunk
        assert round_tripped.section_index is None


def test_save_and_load_chunks_jsonl(tmp_path):
    _, chunks = _build_sample_paper_and_chunks()
    path = tmp_path / "chunks.jsonl"

    n_written = save_chunks(chunks, path)
    assert n_written == len(chunks)
    assert path.exists()

    loaded = list(load_chunks(path))
    assert loaded == list(chunks)


def test_chunks_jsonl_has_one_chunk_per_line(tmp_path):
    _, chunks = _build_sample_paper_and_chunks()
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(chunks)


def test_save_evidence_mappings_and_load_round_trip(tmp_path):
    paper, chunks = _build_sample_paper_and_chunks()
    mappings = map_evidence_to_chunks(paper, chunks)
    path = tmp_path / "evidence_map.jsonl"

    n_written = save_evidence_mappings(mappings, path)
    assert n_written == len(mappings)

    loaded = list(load_evidence_mappings(path))
    assert loaded == list(mappings)


def test_save_chunks_creates_parent_directories(tmp_path):
    _, chunks = _build_sample_paper_and_chunks()
    nested_path = tmp_path / "nested" / "chunks" / "train.jsonl"
    save_chunks(chunks, nested_path)
    assert nested_path.exists()
