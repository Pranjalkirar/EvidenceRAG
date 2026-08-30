import pytest

from evidencerag.chunking.schema import Chunk
from evidencerag.generation.generate import GenerationConfig, GenerationPipeline, generate_answer
from evidencerag.retrieval.schema import RetrievalResult
from tests.generation_fixtures import FailingGenerator, FakeGenerator, FakeRetriever


def _result(chunk_id: str, rank: int, retriever: str = "reranker", score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, rank=rank, retriever=retriever)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        split="test",
        section_index=0,
        section_title="Results",
        paragraph_indices=(0,),
        text=text,
        token_count=len(text.split()),
    )


RANKED = [_result("A", 1), _result("B", 2), _result("C", 3)]
TEXT_BY_ID = {"A": "alpha text", "B": "beta text", "C": "gamma text"}


# ---- generate_answer() ----------------------------------------------------


def test_generate_answer_preserves_question():
    generator = FakeGenerator()
    result = generate_answer("what is alpha?", RANKED, TEXT_BY_ID, generator)
    assert result.question == "what is alpha?"


def test_generate_answer_preserves_chunk_ids_in_rank_order():
    generator = FakeGenerator()
    result = generate_answer("q", RANKED, TEXT_BY_ID, generator)
    assert result.evidence_chunk_ids == ("A", "B", "C")


def test_generate_answer_records_the_generators_model_name():
    generator = FakeGenerator(model_name="my-fake-model")
    result = generate_answer("q", RANKED, TEXT_BY_ID, generator)
    assert result.model_name == "my-fake-model"


def test_generate_answer_returns_the_generators_answer_text_unchanged():
    generator = FakeGenerator(canned_answer="QASPER is the dataset used.")
    result = generate_answer("q", RANKED, TEXT_BY_ID, generator)
    assert result.answer == "QASPER is the dataset used."


def test_generate_answer_builds_a_prompt_containing_every_candidate_chunk_text():
    generator = FakeGenerator()
    generate_answer("q", RANKED, TEXT_BY_ID, generator)
    prompt, _ = generator.calls[0]
    for text in TEXT_BY_ID.values():
        assert text in prompt


def test_generate_answer_forwards_max_new_tokens():
    generator = FakeGenerator()
    generate_answer("q", RANKED, TEXT_BY_ID, generator, max_new_tokens=128)
    _, max_new_tokens = generator.calls[0]
    assert max_new_tokens == 128


def test_generate_answer_with_empty_context_still_produces_a_result():
    generator = FakeGenerator()
    result = generate_answer("q", [], TEXT_BY_ID, generator)
    assert result.evidence_chunk_ids == ()
    assert result.question == "q"
    assert len(generator.calls) == 1


def test_generate_answer_with_missing_chunk_text_raises_keyerror():
    generator = FakeGenerator()
    incomplete = {"A": "alpha text", "B": "beta text"}  # missing "C"
    with pytest.raises(KeyError):
        generate_answer("q", RANKED, incomplete, generator)


def test_generate_answer_never_widens_evidence_beyond_the_supplied_candidates():
    generator = FakeGenerator()
    text_by_id_with_extra = dict(TEXT_BY_ID, D="delta text, not a candidate")
    result = generate_answer("q", RANKED, text_by_id_with_extra, generator)
    assert set(result.evidence_chunk_ids) == {"A", "B", "C"}
    prompt, _ = generator.calls[0]
    assert "delta text, not a candidate" not in prompt


def test_generate_answer_is_deterministic_across_repeated_calls():
    generator = FakeGenerator()
    first = generate_answer("q", RANKED, TEXT_BY_ID, generator)
    second = generate_answer("q", RANKED, TEXT_BY_ID, generator)
    assert first == second


def test_generate_answer_propagates_genuine_generator_errors():
    with pytest.raises(RuntimeError):
        generate_answer("q", RANKED, TEXT_BY_ID, FailingGenerator())


# ---- GenerationPipeline ----------------------------------------------------


def test_pipeline_calls_the_retriever_with_the_question_and_configured_top_k():
    retriever = FakeRetriever(RANKED)
    generator = FakeGenerator()
    chunks = [_chunk(cid, text) for cid, text in TEXT_BY_ID.items()]
    pipeline = GenerationPipeline(retriever, generator, chunks, config=GenerationConfig(top_k=2))

    pipeline.answer("what is alpha?")

    assert retriever.calls == [("what is alpha?", 2)]


def test_pipeline_result_evidence_chunk_ids_match_what_the_retriever_returned():
    retriever = FakeRetriever(RANKED)
    generator = FakeGenerator()
    chunks = [_chunk(cid, text) for cid, text in TEXT_BY_ID.items()]
    pipeline = GenerationPipeline(retriever, generator, chunks)

    result = pipeline.answer("q")

    assert result.evidence_chunk_ids == ("A", "B", "C")


def test_pipeline_looks_up_chunk_text_only_for_returned_chunk_ids():
    retriever = FakeRetriever(RANKED[:1])  # only "A"
    generator = FakeGenerator()
    chunks = [_chunk(cid, text) for cid, text in TEXT_BY_ID.items()]  # A, B, C all available
    pipeline = GenerationPipeline(retriever, generator, chunks)

    pipeline.answer("q")

    prompt, _ = generator.calls[0]
    assert "alpha text" in prompt
    assert "beta text" not in prompt
    assert "gamma text" not in prompt


def test_pipeline_with_no_retrieved_results_still_returns_a_result():
    retriever = FakeRetriever([])
    generator = FakeGenerator()
    chunks = [_chunk(cid, text) for cid, text in TEXT_BY_ID.items()]
    pipeline = GenerationPipeline(retriever, generator, chunks)

    result = pipeline.answer("q")

    assert result.evidence_chunk_ids == ()
    assert result.question == "q"


def test_pipeline_default_config_uses_top_k_five():
    retriever = FakeRetriever(RANKED)
    generator = FakeGenerator()
    chunks = [_chunk(cid, text) for cid, text in TEXT_BY_ID.items()]
    pipeline = GenerationPipeline(retriever, generator, chunks)

    pipeline.answer("q")

    assert retriever.calls == [("q", 5)]


def test_pipeline_result_model_name_matches_the_generator():
    retriever = FakeRetriever(RANKED)
    generator = FakeGenerator(model_name="pipeline-fake-model")
    chunks = [_chunk(cid, text) for cid, text in TEXT_BY_ID.items()]
    pipeline = GenerationPipeline(retriever, generator, chunks)

    result = pipeline.answer("q")

    assert result.model_name == "pipeline-fake-model"
