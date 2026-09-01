"""Tests for evidencerag.langchain_impl.prompt.build_langchain_prompt.

Asserts the LangChain-templated prompt is byte-for-byte identical to
`evidencerag.generation.prompt.build_prompt`'s output for the
equivalent input -- the whole point of reusing SYSTEM_INSTRUCTIONS
rather than retyping it (see that module's docstring).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from evidencerag.generation.prompt import ContextChunk, build_prompt  # noqa: E402
from evidencerag.langchain_impl.documents import chunk_to_document  # noqa: E402
from evidencerag.langchain_impl.prompt import build_langchain_prompt  # noqa: E402
from tests.retrieval_fixtures import make_chunk  # noqa: E402


def test_matches_custom_build_prompt_with_context():
    chunks = [
        make_chunk(chunk_id="c1", text="First passage."),
        make_chunk(chunk_id="c2", text="Second passage."),
    ]
    documents = [chunk_to_document(c) for c in chunks]
    context = [ContextChunk(chunk_id=c.chunk_id, text=c.text) for c in chunks]

    langchain_prompt = build_langchain_prompt("What happened?", documents)
    custom_prompt = build_prompt("What happened?", context)

    assert langchain_prompt == custom_prompt


def test_matches_custom_build_prompt_with_empty_context():
    langchain_prompt = build_langchain_prompt("What happened?", [])
    custom_prompt = build_prompt("What happened?", [])
    assert langchain_prompt == custom_prompt


def test_contains_no_context_notice_when_empty():
    from evidencerag.generation.prompt import NO_CONTEXT_NOTICE

    prompt = build_langchain_prompt("Q?", [])
    assert NO_CONTEXT_NOTICE in prompt
