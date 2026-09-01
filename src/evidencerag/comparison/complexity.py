"""Computes `evidencerag.comparison.schema.EngineeringComplexity` for
both implementations from the repository's actual source files -- no
number here is hand-estimated; every field is derived by reading the
files listed below at call time, so it changes automatically if the
code it describes changes.

Methodology (documented, not hidden, since "LOC" is always a somewhat
arbitrary measure -- see the M8 README section for the same caveat):

  - Both implementations SHARE the underlying model-loading code
    (`evidencerag.retrieval.embeddings.QwenEmbedder`,
    `evidencerag.reranking.reranker.CrossEncoderReranker`,
    `evidencerag.generation.generator.HFGenerator`) -- LangChain wraps
    those same objects rather than reimplementing model loading, so
    counting that shared code against BOTH sides would double-count it
    and understate LangChain's relative simplicity there.
    `CUSTOM_FILES` therefore covers only the custom
    indexing/fusion/reranking/generation ORCHESTRATION modules (BM25
    index, dense/FAISS index, RRF fusion, the rerank() function, the
    prompt builder, the generate_answer() function, and
    `comparison.custom_pipeline`), not the three model-wrapper files.
  - `LANGCHAIN_FILES` covers the entire `evidencerag.langchain_impl`
    package -- ALL of it is integration code written to plug the same
    shared models into LangChain's abstractions (`Document`,
    `Embeddings`, `BaseRetriever`, `LLM`, `PromptTemplate`,
    `EnsembleRetriever`), so none of it is excluded as "shared".
  - Line counts are non-blank lines (`str.strip()` truthy), including
    docstrings and comments -- deliberately not "logical lines of
    code"; this project favors extensive module/function docstrings
    (see any file in `src/evidencerag/`), so a stricter LOC-only count
    would understate both sides' true file size similarly, without
    changing which side is relatively larger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evidencerag.comparison.schema import EngineeringComplexity
from evidencerag.config import PATHS
from evidencerag.langchain_impl import REQUIRED_PACKAGES

# Paths relative to PATHS.root / "src" / "evidencerag".
CUSTOM_FILES: tuple[str, ...] = (
    "retrieval/corpus.py",
    "retrieval/tokenize.py",
    "retrieval/bm25.py",
    "retrieval/dense.py",
    "retrieval/rrf.py",
    "retrieval/base.py",
    "retrieval/schema.py",
    "reranking/rerank.py",
    "generation/prompt.py",
    "generation/generate.py",
    "generation/schema.py",
    "comparison/custom_pipeline.py",
)

LANGCHAIN_FILES: tuple[str, ...] = (
    "langchain_impl/__init__.py",
    "langchain_impl/documents.py",
    "langchain_impl/embeddings.py",
    "langchain_impl/reranking.py",
    "langchain_impl/llm.py",
    "langchain_impl/prompt.py",
    "langchain_impl/retrievers.py",
    "langchain_impl/pipeline.py",
)


def _count_nonblank_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _count_class_defs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("class "))


def _measure(implementation: str, relative_files: Sequence[str], dependency_additions: tuple[str, ...], notes: tuple[str, ...]) -> EngineeringComplexity:
    base = PATHS.root / "src" / "evidencerag"
    paths = [base / relative for relative in relative_files]
    existing = [path for path in paths if path.exists()]
    return EngineeringComplexity(
        implementation=implementation,
        relevant_loc=sum(_count_nonblank_lines(path) for path in paths),
        file_count=len(existing),
        custom_component_count=sum(_count_class_defs(path) for path in paths),
        dependency_additions=dependency_additions,
        notes=notes,
    )


def measure_complexity() -> tuple[EngineeringComplexity, EngineeringComplexity]:
    """`(custom, langchain)` `EngineeringComplexity` snapshots, computed
    from the files listed above as they exist on disk right now."""
    custom = _measure(
        "custom",
        CUSTOM_FILES,
        dependency_additions=(),
        notes=(
            "Counts only orchestration modules (BM25 index, FAISS/dense "
            "index, RRF fusion, rerank(), prompt builder, generate_answer(), "
            "CustomPipeline) -- excludes the three model-wrapper files "
            "(QwenEmbedder/CrossEncoderReranker/HFGenerator), which are "
            "shared with the LangChain side, not duplicated by it.",
        ),
    )
    langchain = _measure(
        "langchain",
        LANGCHAIN_FILES,
        dependency_additions=REQUIRED_PACKAGES,
        notes=(
            "Counts the entire evidencerag.langchain_impl package -- all "
            "of it is integration code on top of the SAME shared model "
            "wrappers the custom side uses (no model is loaded twice).",
        ),
    )
    return custom, langchain
