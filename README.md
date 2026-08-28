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

This project is at the **foundation stage (Milestone 1)**. Nothing
retrieval-, model-, or dataset-related is implemented yet.

**Implemented:**
- Project skeleton and package layout (`src/evidencerag/`)
- `pyproject.toml` / `requirements.txt` for the foundation stage
- Centralized path/settings configuration (`evidencerag.config`)
- Minimal test suite verifying the package imports correctly
- Basic (non-GPU) Dockerfile
- This README

**Explicitly NOT implemented yet** (all subpackages below are empty
placeholders with docstrings only):
- QASPER download/ingestion
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
