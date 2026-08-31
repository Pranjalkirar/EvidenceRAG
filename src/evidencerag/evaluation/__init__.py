"""M7: evaluation of the M4/M5/M6 pipeline against QASPER gold data.

Compares four `Retriever`-shaped systems -- BM25, Dense, Hybrid, and
Hybrid+Cross-Encoder-Reranker -- on identical QASPER questions, using
the existing M4/M5/M6 interfaces unchanged. No new retrievers,
rerankers, or generators are introduced here; this package only scores
what M2-M6 already produce.

    evidencerag.evaluation.schema             result/metadata/summary dataclasses
    evidencerag.evaluation.gold                M2 Paper + M3 EvidenceChunkMapping -> per-question gold references
    evidencerag.evaluation.retrieval_metrics   Recall@k / reciprocal rank, single-reference and max-over-references
    evidencerag.evaluation.evidence_metrics    QASPER-style Evidence F1 (paragraph text, official semantics)
    evidencerag.evaluation.answer_metrics      QASPER-style Answer F1 (SQuAD-style token F1, official semantics)
    evidencerag.evaluation.systems             builds the four Retriever-shaped systems from Settings, unchanged
    evidencerag.evaluation.harness             orchestrates the above into EvalRecords + a RunSummary
    evidencerag.evaluation.io                  metadata.json / summary.json / results.jsonl persistence

See `scripts/evaluate_m7.py` for the CLI entry point.
"""
