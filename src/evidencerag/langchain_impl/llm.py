"""LangChain `LLM` adapter around our own `Generator`.

Subclasses `langchain_core.language_models.llms.LLM` (the standard,
long-stable way to plug an arbitrary text-completion callable into
LangChain's `Runnable`/LCEL machinery) rather than adding a second,
independent LangChain-native model integration -- `EvidenceRAGLLM`
delegates every call to a wrapped `Generator` instance (typically the
SAME `HFGenerator` object the custom pipeline uses), so the model is
loaded once, and custom vs LangChain generation differ ONLY in how the
prompt is assembled and the call is issued, never in which model
produces the text.
"""

from __future__ import annotations

from typing import Any, List, Optional

from evidencerag.langchain_impl import require_langchain

require_langchain()

from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.llms import LLM  # noqa: E402

from evidencerag.generation.generator import DEFAULT_MAX_NEW_TOKENS, Generator  # noqa: E402


class EvidenceRAGLLM(LLM):
    """Wraps an `evidencerag.generation.generator.Generator` (real or
    fake) as a LangChain `LLM`.

    `generator` and `max_new_tokens` are plain instance attributes set
    via `object.__setattr__` in `__init__` rather than declared as
    pydantic fields, because `Generator` is a structural `Protocol`
    (not a type LangChain's pydantic-based `LLM` base class can
    validate) -- this sidesteps needing `arbitrary_types_allowed`
    configuration that has moved between pydantic v1/v2-style APIs
    across LangChain releases.
    """

    def __init__(self, generator: Generator, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_generator", generator)
        object.__setattr__(self, "_max_new_tokens", max_new_tokens)

    @property
    def _llm_type(self) -> str:
        return "evidencerag-generator"

    @property
    def model_name(self) -> str:
        return self._generator.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # `stop` sequences are accepted for interface compatibility but
        # not applied -- our `Generator` Protocol has no stop-sequence
        # concept (see generation/generator.py), and the custom
        # pipeline's `generate_answer()` doesn't apply one either, so
        # leaving `stop` unused here keeps both pipelines' generation
        # behavior identical rather than giving LangChain generation an
        # extra capability the comparison isn't measuring.
        return self._generator.generate(prompt, max_new_tokens=self._max_new_tokens)
