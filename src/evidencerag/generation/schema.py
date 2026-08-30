"""Generation result model, shared by every generator implementation.

Mirrors evidencerag.retrieval.schema.RetrievalResult /
evidencerag.chunking.schema.Chunk: `chunk_id` is the canonical
evidence identity everywhere in this project, and that convention
continues into M6. A `GenerationResult` never invents a new identifier
for the evidence it was grounded on -- `evidence_chunk_ids` is exactly
the (ordered) set of `chunk_id`s that were placed into the model's
prompt context, so M7 can later check grounding against precisely
what the model was given, independent of whatever the model's answer
text happens to say.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    """One generated answer for one question.

    `evidence_chunk_ids` is recorded even when it is empty -- an empty
    tuple means generation genuinely had zero context chunks (e.g. an
    upstream retriever/reranker returned nothing), which is itself
    meaningful provenance for M7, not an error state to hide.

    `model_name` identifies which `Generator` produced `answer` (e.g.
    the real `HFGenerator`'s Hugging Face model id, or a fake
    generator's test name in unit tests) -- the same
    "which model produced this" metadata pattern as
    `Embedder.model_name` / `Reranker.model_name`.
    """

    question: str
    answer: str
    evidence_chunk_ids: tuple[str, ...]
    model_name: str
