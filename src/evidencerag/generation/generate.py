"""M6 generation: M5 reranked top-k chunks -> LLM `Generator` ->
grounded `GenerationResult`.

    M4 HybridRetriever / M5 RerankingRetriever (top `top_k`, ranked)
                    │
                    ▼
        evidencerag.generation.prompt.build_prompt
                    │
                    ▼
        evidencerag.generation.generator.Generator
                    │
                    ▼
              GenerationResult  (answer + evidence chunk_ids)

This module does not duplicate any M4/M5 functionality: it never
retrieves, scores, or reranks candidates itself -- it only accepts an
already-ranked `RetrievalResult` list (or, via `GenerationPipeline`,
any `Retriever`-shaped object -- typically an M5 `RerankingRetriever`
-- and calls its `retrieve()` unchanged) and turns that fixed,
ordered list into prompt context and then an answer. This mirrors
exactly how `reranking/rerank.py`'s `RerankingRetriever` wraps an M4
base retriever without altering its behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evidencerag.chunking.schema import Chunk
from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS, Generator
from evidencerag.generation.prompt import ContextChunk, build_prompt
from evidencerag.generation.schema import GenerationResult
from evidencerag.retrieval.base import Retriever
from evidencerag.retrieval.schema import RetrievalResult


def generate_answer(
    question: str,
    ranked_results: Sequence[RetrievalResult],
    chunk_text_by_id: Mapping[str, str],
    generator: Generator,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> GenerationResult:
    """Generate a grounded answer from an already-ranked result list.

    `ranked_results` are consumed exactly as given, in the order
    given -- this function performs no retrieval, scoring, or
    reranking of its own; it assumes the caller (typically an M5
    `RerankingRetriever`) already produced the best-first ordering.
    Every result becomes one numbered evidence passage in the prompt,
    and its `chunk_id` is recorded in the returned result's
    `evidence_chunk_ids`, in the same order -- so the answer's
    grounding can always be traced back to exactly what the model saw.

    `chunk_text_by_id` supplies the text for each result's chunk_id; a
    missing entry is a caller bug and raises `KeyError`, the same
    "caller's job" contract `rerank()` places on its own caller.

    An empty `ranked_results` is not an error: it produces a prompt
    that explicitly tells the model no evidence was retrieved (see
    `prompt.NO_CONTEXT_NOTICE`), and the returned result's
    `evidence_chunk_ids` is simply an empty tuple.
    """
    context = [ContextChunk(chunk_id=r.chunk_id, text=chunk_text_by_id[r.chunk_id]) for r in ranked_results]
    prompt = build_prompt(question, context)
    answer = generator.generate(prompt, max_new_tokens=max_new_tokens)
    return GenerationResult(
        question=question,
        answer=answer,
        evidence_chunk_ids=tuple(r.chunk_id for r in ranked_results),
        model_name=generator.model_name,
    )


@dataclass(frozen=True)
class GenerationConfig:
    """`top_k` is how many ranked results `GenerationPipeline` asks its
    wrapped retriever for before turning them into prompt context --
    the same `top_k` naming/role as `Retriever.retrieve()`'s own
    parameter and `RerankConfig`/`RRFConfig`'s depth settings.
    """

    top_k: int = 5
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


class GenerationPipeline:
    """Question -> `retriever.retrieve()` -> `generate_answer()` ->
    `GenerationResult`.

    Wraps `retriever` (an M4 `HybridRetriever`, or -- the intended M6
    usage -- an M5 `RerankingRetriever`) unchanged: it does not alter
    BM25, dense, RRF, or cross-encoder reranking behavior at all, and
    never touches the corpus directly. Chunk text is looked up only
    for `chunk_id`s the wrapped retriever itself returned, exactly
    like `RerankingRetriever` looks up text only for candidates its
    base retriever returned.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        chunks: Iterable[Chunk],
        config: GenerationConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
        self._config = config or GenerationConfig()

    def answer(self, question: str) -> GenerationResult:
        ranked = self._retriever.retrieve(question, top_k=self._config.top_k)
        return generate_answer(
            question,
            ranked,
            self._chunk_text_by_id,
            self._generator,
            max_new_tokens=self._config.max_new_tokens,
        )
