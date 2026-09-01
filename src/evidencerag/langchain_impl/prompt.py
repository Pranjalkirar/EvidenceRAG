"""LangChain `PromptTemplate` for M8 generation, built from the SAME
grounding instructions M6 uses.

`SYSTEM_INSTRUCTIONS` and `NO_CONTEXT_NOTICE` are imported, not
retyped, from `evidencerag.generation.prompt` -- the custom and
LangChain pipelines are given the exact same grounding contract
(answer only from evidence, cite by chunk_id, say so when insufficient,
stay concise), so a difference in answer quality between the two
pipelines can't be attributed to accidentally-different wording. Only
the templating mechanism differs: `build_prompt()` is Python
string-formatting, this module's `build_langchain_prompt()` is
LangChain's `PromptTemplate.format()`.
"""

from __future__ import annotations

from typing import Sequence

from evidencerag.langchain_impl import require_langchain

require_langchain()

from langchain_core.documents import Document  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from evidencerag.generation.prompt import NO_CONTEXT_NOTICE, SYSTEM_INSTRUCTIONS  # noqa: E402
from evidencerag.langchain_impl.documents import document_chunk_id  # noqa: E402

QA_PROMPT_TEMPLATE = PromptTemplate.from_template(
    SYSTEM_INSTRUCTIONS + "\n\n{evidence_block}\n\nQuestion: {question}\n\nAnswer:"
)


def _render_passage(index: int, document: Document) -> str:
    return f"[{index}] chunk_id={document_chunk_id(document)}\n{document.page_content}"


def build_langchain_prompt(question: str, context: Sequence[Document]) -> str:
    """Build the full generation prompt for `question` grounded in the
    ordered `context` documents, via LangChain's `PromptTemplate`.

    Produces the same rendered text as
    `evidencerag.generation.prompt.build_prompt` would for the
    equivalent `ContextChunk` list (same instructions, same
    numbered/chunk_id-labeled passage format, same
    `NO_CONTEXT_NOTICE` for an empty context) -- the two functions are
    intentionally kept in lockstep so `M8` measures orchestration
    overhead, not incidental prompt-wording drift.
    """
    if context:
        passages = "\n\n".join(_render_passage(i, document) for i, document in enumerate(context, start=1))
        evidence_block = f"Evidence passages:\n\n{passages}"
    else:
        evidence_block = f"Evidence passages:\n\n{NO_CONTEXT_NOTICE}"

    return QA_PROMPT_TEMPLATE.format(question=question, evidence_block=evidence_block)
