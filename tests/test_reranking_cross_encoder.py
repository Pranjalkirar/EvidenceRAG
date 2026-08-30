"""Unit tests for `CrossEncoderReranker.score()`.

These construct a `CrossEncoderReranker` WITHOUT calling `__init__`
(which lazily imports sentence-transformers and downloads the real
model) -- instead a fake stand-in is dropped directly into `_model`,
mirroring tests/test_retrieval_embeddings.py's approach for
`QwenEmbedder`. This exercises the exact pair-construction/score-
mapping logic in reranker.py without needing sentence-transformers,
torch, or network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidencerag.reranking.reranker import CROSS_ENCODER_MODEL, CrossEncoderReranker


def _make_reranker_with_fake_model(fake_model) -> CrossEncoderReranker:
    reranker = object.__new__(CrossEncoderReranker)  # skip __init__ (no sentence-transformers import)
    reranker._model_name = "fake-cross-encoder"
    reranker._model = fake_model
    return reranker


class _RecordingModel:
    """Mimics `sentence_transformers.CrossEncoder.predict`: takes a
    list of [query, text] pairs and returns one float per pair.
    Records exactly what it was called with.
    """

    def __init__(self) -> None:
        self.received = None

    def predict(self, sentence_pairs):
        self.received = sentence_pairs
        return np.array([float(i) for i in range(len(sentence_pairs))])


def test_model_name_defaults_to_the_m5_spec_model():
    assert CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_score_forwards_pairs_to_the_underlying_model_as_lists():
    model = _RecordingModel()
    reranker = _make_reranker_with_fake_model(model)
    reranker.score([("q1", "text1"), ("q1", "text2")])
    assert model.received == [["q1", "text1"], ["q1", "text2"]]


def test_score_returns_one_float_per_pair_in_order():
    model = _RecordingModel()
    reranker = _make_reranker_with_fake_model(model)
    scores = reranker.score([("q", "a"), ("q", "b"), ("q", "c")])
    assert scores.shape == (3,)
    assert list(scores) == [0.0, 1.0, 2.0]


def test_score_with_empty_pairs_returns_empty_array_without_calling_the_model():
    model = _RecordingModel()
    reranker = _make_reranker_with_fake_model(model)
    scores = reranker.score([])
    assert scores.shape == (0,)
    assert model.received is None


def test_genuine_runtime_errors_from_the_model_are_not_swallowed():
    class _FailingModel:
        def predict(self, sentence_pairs):
            raise RuntimeError("CUDA out of memory")

    reranker = _make_reranker_with_fake_model(_FailingModel())
    with pytest.raises(RuntimeError):
        reranker.score([("q", "a")])
