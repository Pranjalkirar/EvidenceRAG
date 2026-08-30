"""Answer generation (Milestone 6).

LLM generation applied on top of M4 hybrid retrieval + M5 cross-encoder
reranking results:

    evidencerag.reranking.rerank.RerankingRetriever
                    │  (top `top_k` reranked RetrievalResult, chunk_id-identified)
                    ▼
    evidencerag.generation.prompt.build_prompt
    (deterministic, context-only, cites chunk_id, unit-testable without an LLM)
                    │
                    ▼
    evidencerag.generation.generator.HFGenerator
    (Qwen/Qwen3-4B-Instruct-2507, via transformers, lazily loaded)
                    │
                    ▼
    evidencerag.generation.schema.GenerationResult
    (question + answer + evidence chunk_ids + model_name)

See `generator.py` for the model abstraction (`Generator` protocol +
`HFGenerator`), `prompt.py` for deterministic prompt construction, and
`generate.py` for the generation pipeline (`generate_answer()` and
`GenerationPipeline`).
"""
