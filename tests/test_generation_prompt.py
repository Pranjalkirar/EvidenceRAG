from evidencerag.generation.prompt import (
    NO_CONTEXT_NOTICE,
    SYSTEM_INSTRUCTIONS,
    ContextChunk,
    build_prompt,
)

CONTEXT = [
    ContextChunk(chunk_id="c1", text="The model was trained on QASPER."),
    ContextChunk(chunk_id="c2", text="QASPER contains 1585 papers."),
]


def test_prompt_contains_the_question():
    prompt = build_prompt("What dataset was used?", CONTEXT)
    assert "What dataset was used?" in prompt


def test_prompt_contains_every_chunk_id_and_its_text():
    prompt = build_prompt("q", CONTEXT)
    for chunk in CONTEXT:
        assert chunk.chunk_id in prompt
        assert chunk.text in prompt


def test_prompt_instructs_context_only_answering():
    prompt = build_prompt("q", CONTEXT)
    assert "ONLY" in prompt
    assert "outside knowledge" in prompt


def test_prompt_instructs_against_inventing_information():
    prompt = build_prompt("q", CONTEXT)
    assert "do not invent" in prompt.lower()


def test_prompt_instructs_stating_insufficient_evidence():
    prompt = build_prompt("q", CONTEXT)
    assert "insufficient" in prompt.lower()


def test_prompt_instructs_concise_qasper_style_answers():
    prompt = build_prompt("q", CONTEXT)
    assert "concise" in prompt.lower()
    assert "research paper" in prompt.lower()


def test_prompt_instructs_using_the_retrieved_evidence():
    prompt = build_prompt("q", CONTEXT)
    assert "evidence" in prompt.lower()


def test_system_instructions_constant_is_embedded_verbatim():
    prompt = build_prompt("q", CONTEXT)
    assert SYSTEM_INSTRUCTIONS in prompt


def test_context_order_is_preserved_as_given():
    prompt = build_prompt("q", CONTEXT)
    assert prompt.index("c1") < prompt.index("c2")


def test_reversed_context_order_changes_the_rendered_order():
    forward = build_prompt("q", CONTEXT)
    backward = build_prompt("q", list(reversed(CONTEXT)))
    assert forward != backward
    assert backward.index("c2") < backward.index("c1")


def test_empty_context_produces_explicit_no_context_notice_not_a_blank_section():
    prompt = build_prompt("q", [])
    assert NO_CONTEXT_NOTICE in prompt


def test_empty_context_does_not_crash_and_still_contains_question():
    prompt = build_prompt("What is the accuracy?", [])
    assert "What is the accuracy?" in prompt


def test_deterministic_same_inputs_produce_identical_prompt_string():
    first = build_prompt("q", CONTEXT)
    second = build_prompt("q", CONTEXT)
    assert first == second


def test_deterministic_across_freshly_constructed_equal_context_objects():
    context_copy = [ContextChunk(chunk_id=c.chunk_id, text=c.text) for c in CONTEXT]
    assert build_prompt("q", CONTEXT) == build_prompt("q", context_copy)


def test_different_question_produces_different_prompt():
    assert build_prompt("q1", CONTEXT) != build_prompt("q2", CONTEXT)


def test_different_context_text_produces_different_prompt():
    other = [ContextChunk(chunk_id="c1", text="different text entirely")]
    assert build_prompt("q", CONTEXT) != build_prompt("q", other)


def test_prompt_ends_with_an_answer_cue():
    prompt = build_prompt("q", CONTEXT)
    assert prompt.rstrip().endswith("Answer:")
