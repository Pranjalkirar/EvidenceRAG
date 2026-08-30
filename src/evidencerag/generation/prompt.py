"""Deterministic prompt construction for M6 generation.

`build_prompt` is pure string formatting -- no model, no network, no
randomness, no wall-clock/timestamp, no dependence on dict ordering
(context is consumed as an ordered `Sequence`). Same `question` +
same `context` (same objects, same order) always produces the exact
same string, byte for byte, so this function can be fully unit-tested
without downloading or running any LLM (see
tests/test_generation_prompt.py).

This is deliberately the ONLY place the grounding instructions are
worded, so every `Generator` implementation (fake or real) is fed the
exact same contract:

  - answer using ONLY the supplied evidence passages,
  - never invent facts/numbers/claims the evidence doesn't support,
  - explicitly say so when the evidence is insufficient,
  - keep the answer concise, in a style appropriate for QASPER-style
    scientific paper QA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Kept as a module-level constant (rather than inlined in build_prompt)
# so tests can assert on its exact wording/content without having to
# re-parse the full assembled prompt string.
SYSTEM_INSTRUCTIONS = (
    "You are a scientific question-answering assistant. Answer the question "
    "using ONLY the evidence passages supplied below -- do not use any "
    "outside knowledge, and do not invent facts, numbers, or claims that "
    "the supplied evidence does not directly support. Each passage is "
    "numbered and labeled with its chunk_id; you may refer to a passage by "
    "its chunk_id when useful. If the supplied evidence is insufficient to "
    "answer the question, say explicitly that the evidence is insufficient "
    "rather than guessing. Keep the answer concise and precise, in a style "
    "appropriate for answering a question about a scientific research paper."
)

# Shown in place of an evidence block when no chunks were retrieved --
# a real, explicit statement rather than an empty section, so the
# model is never left to guess whether context was omitted by mistake.
NO_CONTEXT_NOTICE = "(No evidence passages were retrieved for this question.)"


@dataclass(frozen=True)
class ContextChunk:
    """One piece of evidence placed into the prompt.

    `chunk_id` is embedded directly into the rendered passage (not
    just carried alongside it) so the model can see and cite it, and
    so a human inspecting a raw prompt string can trace every passage
    back to its `chunk_id` without needing the original
    `RetrievalResult` list.
    """

    chunk_id: str
    text: str


def _render_passage(index: int, chunk: ContextChunk) -> str:
    return f"[{index}] chunk_id={chunk.chunk_id}\n{chunk.text}"


def build_prompt(question: str, context: Sequence[ContextChunk]) -> str:
    """Build the full generation prompt for `question` grounded in the
    ordered `context` chunks.

    `context` is consumed in the order given -- this function performs
    no reordering, deduplication, or filtering of its own; deciding
    which chunks to include, and in what order, is the caller's job
    (typically: an M5 `RerankingRetriever`'s output, in reranked
    order), matching the "caller decides the candidate set" convention
    already used by `rerank()`.
    """
    if context:
        passages = "\n\n".join(_render_passage(i, chunk) for i, chunk in enumerate(context, start=1))
        evidence_block = f"Evidence passages:\n\n{passages}"
    else:
        evidence_block = f"Evidence passages:\n\n{NO_CONTEXT_NOTICE}"

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{evidence_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
