# EvidenceRAG

An evaluation-driven Retrieval-Augmented Generation (RAG) system over ML/AI
research papers, built to rigorously compare retrieval strategies rather
than to ship a chatbot.

## Overview

EvidenceRAG evaluates and compares multiple retrieval approaches on the
same dataset, questions, models, and metrics, so the comparisons are fair:

1. **BM25** sparse retrieval
2. **Dense retrieval** using neural embeddings + FAISS
3. **Hybrid retrieval** — BM25 + dense retrieval combined with Reciprocal
   Rank Fusion (RRF)
4. **Hybrid + cross-encoder reranking**
5. A **custom implementation** compared against an equivalent
   **LangChain-based** implementation

## Motivation

Most public RAG projects are demos: a single retrieval method wired to a
single LLM, with no systematic comparison of *why* one retrieval strategy
outperforms another. EvidenceRAG is built as a portfolio-quality
DS/ML project that instead treats retrieval as the object of study —
sparse vs. dense vs. hybrid vs. reranked, and a hand-written implementation
vs. a framework (LangChain) implementation — under identical experimental
conditions.

The retrieval mechanisms are kept understandable and inspectable by
design. LangChain is used later purely as a comparison point, not to
hide how the custom retrieval methods work.

## Planned Architecture

```
Raw QASPER papers
      │
      ▼
  Ingestion  ──►  Chunking
                       │
                       ▼
        ┌──────────────┴──────────────┐
        ▼                             ▼
   BM25 retrieval              Dense retrieval (FAISS)
        └──────────────┬──────────────┘
                        ▼
              Hybrid retrieval (RRF)
                        │
                        ▼
            Cross-encoder reranking
                        │
                        ▼
                  Answer generation
                        │
                        ▼
                    Evaluation
```

Each stage above corresponds to a subpackage under `src/evidencerag/`
(`ingestion`, `chunking`, `retrieval`, `reranking`, `generation`,
`evaluation`). A LangChain-based pipeline mirroring the same stages will
be added as a separate comparison implementation.

## Planned Experiments

- BM25 vs. dense vs. hybrid vs. hybrid+rerank, on identical questions
- Custom implementation vs. LangChain implementation, same settings
- Metrics tracked and compared via MLflow (retrieval quality, e.g.
  recall@k / nDCG, and end-to-end answer quality against QASPER's
  supporting evidence)

**Dataset:** [QASPER](https://allenai.org/data/qasper) — chosen because it
provides research papers, questions, answers, and supporting evidence,
which suits evaluating retrieval against known evidence spans.

**Planned models:**
- `Qwen3-Embedding-0.6B` — embeddings for dense retrieval
- `Qwen3-Reranker-0.6B` — cross-encoder reranking
- `Qwen3-4B` — answer generation (subject to available GPU memory)

## Planned Technology Stack

Python · PyTorch · Hugging Face Transformers · Sentence Transformers ·
BM25 · FAISS · LangChain · MLflow · FastAPI · Streamlit · Docker ·
Git/GitHub

## Hardware / Compute Setup

- **Local development:** Windows 11, AMD Ryzen 5 5600H, 8 GB RAM, no
  local NVIDIA/CUDA GPU. Local development is kept lightweight —
  no model downloads or GPU-bound work happens here.
- **GPU-heavy experiments** (embedding generation, reranking, generation,
  fine-tuning if any): run on an online GPU environment such as Kaggle.

## Current Status

This project has completed **Milestone 1 (foundation)** and
**Milestone 2 (QASPER ingestion + data model)**. Retrieval, reranking,
and generation are not implemented yet.

**Implemented:**
- Project skeleton and package layout (`src/evidencerag/`)
- `pyproject.toml` / `requirements.txt`
- Centralized path/settings configuration (`evidencerag.config`)
- Basic (non-GPU) Dockerfile
- **QASPER ingestion** (`evidencerag.ingestion`):
  - typed internal data model (`Paper` → `Question` → `Answer` → `EvidenceSpan`)
  - raw dataset acquisition via the Hugging Face `datasets` library (`loader.py`)
  - structural validation of raw rows (`validate.py`)
  - normalization into the internal model, including best-effort
    evidence-to-paragraph/figure provenance resolution (`normalize.py`)
  - JSON Lines serialization with lossless round-tripping (`serialize.py`)
  - dataset statistics computed from actual loaded data (`statistics.py`)
  - CLI: `scripts/ingest_qasper.py`
- Test suite: package import, config, and full ingestion pipeline
  (validation, normalization, evidence provenance, serialization,
  statistics) against a small synthetic fixture — no full-dataset
  download required to run tests

**Explicitly NOT implemented yet** (all subpackages below are empty
placeholders with docstrings only):
- Chunking
- BM25 retrieval
- Dense retrieval / FAISS
- Hybrid retrieval / RRF
- Cross-encoder reranking
- Answer generation
- Evaluation harness / metrics
- LangChain comparison pipeline
- FastAPI service
- Streamlit UI
- MLflow experiment tracking

### QASPER data model

```
Paper (id, title, abstract, split, sections, figures_and_tables)
  └── Question (question_id, question_text, ...)
        └── Answer (unanswerable / yes_no / free_form_answer / extractive_spans)
              └── EvidenceSpan (text, resolved, section_index, paragraph_index, ...)
```

Each `EvidenceSpan` keeps the original evidence text and records
whether it could be traced back to a specific paragraph or
figure/table in the paper — the groundwork for a future chunking step
to derive `question → ground-truth evidence → chunk(s)` mappings.
Evidence pieces are never concatenated; a question's multiple
evidence paragraphs, or multiple workers' answers, are all preserved
individually. See `src/evidencerag/ingestion/schema.py` and
`normalize.py` for details, including documented ambiguities in the
raw QASPER schema and how they were resolved.

### Running ingestion

```bash
python scripts/ingest_qasper.py                       # all splits
python scripts/ingest_qasper.py --splits train         # one split
python scripts/ingest_qasper.py --max-papers-per-split 5   # quick smoke test
```

Downloads/caches the raw dataset under `data/raw/qasper_hf_cache/`
(never modified afterward) and writes normalized `train.jsonl`,
`validation.jsonl`, `test.jsonl` under `data/processed/`, then prints
dataset statistics.

## Project Layout

```
EvidenceRAG/
├── app/                    # (planned) FastAPI / Streamlit app code
├── configs/                # (planned) experiment/config files
├── data/
│   ├── raw/                 # (gitignored) raw QASPER data
│   └── processed/           # (gitignored) processed data
├── experiments/             # (planned) experiment tracking artifacts
├── notebooks/                # exploratory notebooks
├── scripts/                 # (planned) standalone utility scripts
├── src/evidencerag/
│   ├── config.py            # project paths & settings (IMPLEMENTED)
│   ├── chunking/             # (planned)
│   ├── evaluation/           # (planned)
│   ├── generation/           # (planned)
│   ├── ingestion/            # (planned)
│   ├── reranking/            # (planned)
│   └── retrieval/            # (planned)
└── tests/                   # test suite
```

## Local Setup (Milestone 1)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
pytest
```
