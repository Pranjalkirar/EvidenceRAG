"""Tests for `QwenEmbedder.embed_queries`'s "query" prompt handling.

These construct a `QwenEmbedder` WITHOUT calling `__init__` (which
lazily imports sentence-transformers and downloads the real model) --
instead a fake stand-in is dropped directly into `_model`, so this
exercises the exact fallback logic in embeddings.py without needing
sentence-transformers, torch, or network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidencerag.retrieval.embeddings import QwenEmbedder


def _make_embedder_with_fake_model(fake_model) -> QwenEmbedder:
    embedder = object.__new__(QwenEmbedder)  # skip __init__ (no sentence-transformers import)
    embedder._model_name = "fake-qwen"
    embedder._model = fake_model
    return embedder


class _ModelWithNoQueryPrompt:
    """Mimics sentence-transformers raising KeyError for an unknown
    prompt name -- the specific, expected "prompt unavailable" case."""

    def encode(self, texts, prompt_name=None, convert_to_numpy=True, show_progress_bar=False):
        if prompt_name is not None:
            raise KeyError(f"Prompt name {prompt_name!r} not found in the configured prompts.")
        return np.zeros((len(texts), 4), dtype=np.float32)


class _ModelWithNoPromptsAttributeAtAll:
    """Mimics an older/minimal model object with no prompt concept at
    all -- AttributeError is the other narrow, expected case."""

    def encode(self, texts, prompt_name=None, convert_to_numpy=True, show_progress_bar=False):
        if prompt_name is not None:
            raise AttributeError("'Model' object has no attribute 'prompts'")
        return np.zeros((len(texts), 4), dtype=np.float32)


class _ModelWithGenuineRuntimeError:
    """A real failure (e.g. CUDA OOM) that must NOT be swallowed."""

    def encode(self, texts, prompt_name=None, convert_to_numpy=True, show_progress_bar=False):
        if prompt_name is not None:
            raise RuntimeError("CUDA out of memory")
        return np.zeros((len(texts), 4), dtype=np.float32)


def test_falls_back_when_query_prompt_name_not_found():
    embedder = _make_embedder_with_fake_model(_ModelWithNoQueryPrompt())
    result = embedder.embed_queries(["hello"])
    assert result.shape == (1, 4)


def test_falls_back_when_model_has_no_prompts_concept():
    embedder = _make_embedder_with_fake_model(_ModelWithNoPromptsAttributeAtAll())
    result = embedder.embed_queries(["hello"])
    assert result.shape == (1, 4)


def test_genuine_runtime_errors_are_not_silently_swallowed():
    embedder = _make_embedder_with_fake_model(_ModelWithGenuineRuntimeError())
    with pytest.raises(RuntimeError):
        embedder.embed_queries(["hello"])
