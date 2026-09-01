"""Milestone 8: a LangChain-based comparison pipeline, wired to the
SAME underlying models/algorithms as the custom M1-M7 implementation
wherever LangChain provides a genuine equivalent.

This package is an EXPERIMENTAL COMPARISON BRANCH, not a replacement
for the custom pipeline (`evidencerag.retrieval` / `evidencerag.reranking`
/ `evidencerag.generation` remain the project's primary implementation).
Nothing in M1-M7 imports from this package, and nothing here is
imported by M1-M7 -- it is purely additive.

LangChain is an OPTIONAL dependency (see `pyproject.toml`'s
`[project.optional-dependencies].langchain` extra). Importing this
package's submodules requires `langchain-core`, `langchain`, and
`langchain-community` to be installed; importing
`evidencerag.langchain_impl` itself does not (this module only raises
a clear, actionable error -- via `require_langchain()` -- when a
caller actually tries to use LangChain-backed functionality without
it installed, rather than failing at unrelated import time elsewhere
in the project).

    evidencerag.langchain_impl.documents   Chunk <-> LangChain Document conversion
    evidencerag.langchain_impl.embeddings  Embeddings adapter around our own Embedder
    evidencerag.langchain_impl.reranking   Document reranking adapter around our own Reranker
    evidencerag.langchain_impl.llm         LangChain LLM adapter around our own Generator
    evidencerag.langchain_impl.prompt      LangChain PromptTemplate, reusing M6's exact wording
    evidencerag.langchain_impl.retrievers  BM25 / dense (FAISS) / hybrid (RRF) LangChain retrievers
    evidencerag.langchain_impl.pipeline    LangChainPipeline: ties the above into one timed,
                                            per-question call, mirroring
                                            evidencerag.comparison.custom_pipeline.CustomPipeline

See `evidencerag.comparison` for the M8 orchestration that runs this
pipeline (and the custom one) over the same QASPER questions and
scores both with the unchanged M7 evaluation metrics, and
`scripts/evaluate_m8.py` for the CLI entry point.
"""

from __future__ import annotations

# Package names as declared in `pyproject.toml`'s `langchain` extra --
# kept here as the single source of truth for the error message below,
# so the extra name and the message can never silently drift apart.
REQUIRED_PACKAGES: tuple[str, ...] = ("langchain-core", "langchain", "langchain-community")


class LangChainNotInstalledError(ImportError):
    """Raised when LangChain-backed M8 functionality is used without
    the optional `langchain` extra installed. A plain `ImportError`
    subclass (not a new exception hierarchy) so callers that already
    catch `ImportError` around optional-dependency code keep working
    unchanged.
    """


def require_langchain() -> None:
    """Raise `LangChainNotInstalledError` with an actionable message if
    any required LangChain package is missing. Call this at the start
    of any function that is about to import a `langchain*` module, so
    the failure is a clear, single, project-specific error rather than
    whatever raw `ModuleNotFoundError` the first missing sub-import
    happens to produce.
    """
    missing: list[str] = []
    for module_name, package_name in (
        ("langchain_core", "langchain-core"),
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
    ):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        raise LangChainNotInstalledError(
            "M8's LangChain comparison pipeline requires the optional "
            "'langchain' extra. Missing: " + ", ".join(missing) + ". "
            "Install with `pip install -e .[langchain]` (or "
            "`pip install -r requirements-langchain.txt`) to run "
            "scripts/evaluate_m8.py or use evidencerag.langchain_impl "
            "directly. The custom M1-M7 pipeline does not need this."
        )
