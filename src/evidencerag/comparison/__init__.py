"""M8: Custom vs LangChain comparison.

Runs the SAME QASPER questions through two Hybrid(BM25+Dense
RRF)+Cross-Encoder-Reranker(+LLM) pipelines -- the custom M1-M7
implementation (`evidencerag.comparison.custom_pipeline.CustomPipeline`)
and a LangChain-based one
(`evidencerag.langchain_impl.pipeline.LangChainPipeline`) -- built from
the SAME chunks, embedding model, reranker model, and generation model,
and scores both with the UNCHANGED M7 evaluation metrics
(`evidencerag.evaluation.retrieval_metrics`,
`evidencerag.evaluation.evidence_metrics`,
`evidencerag.evaluation.answer_metrics`), plus per-stage latency.

The custom implementation remains the project's primary
implementation; this package (and `evidencerag.langchain_impl`) is an
experimental comparison branch layered on top of it, not a
replacement -- see the M8 section of `README.md`.

    evidencerag.comparison.schema           StageResult / ComparisonRecord / metadata / summary dataclasses
    evidencerag.comparison.custom_pipeline  CustomPipeline: timed custom BM25+Dense-RRF-Rerank(-Generate)
    evidencerag.comparison.complexity       LOC / dependency / component-count snapshot, computed from disk
    evidencerag.comparison.runner           orchestrates pipelines + M7 metrics into ComparisonRecords + a summary
    evidencerag.comparison.io               metadata.json / summary.json / results.jsonl persistence

`evidencerag.langchain_impl.pipeline.LangChainPipeline` is the
LangChain-side counterpart driven by the same `runner.run_comparison`;
it is optional (LangChain is an optional dependency -- see
`evidencerag.langchain_impl.require_langchain`) and this package does
not import it at module load time, so `evidencerag.comparison` can
always be imported (and its custom-only functionality used) even when
LangChain is not installed.

See `scripts/evaluate_m8.py` for the CLI entry point.
"""
