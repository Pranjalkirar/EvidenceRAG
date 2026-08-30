"""Unit tests for `HFGenerator.generate()`.

These construct an `HFGenerator` WITHOUT calling `__init__` (which
lazily imports `transformers`/`torch` and downloads the real model) --
instead fake tokenizer/model stand-ins are dropped directly into
`_tokenizer`/`_model`, mirroring
tests/test_reranking_cross_encoder.py's approach for
`CrossEncoderReranker`. This exercises the exact chat-template /
generate / decode wiring in generator.py without needing
`transformers`, `torch`, or network access.
"""

from __future__ import annotations

import pytest

from evidencerag.generation.generator import GENERATION_MODEL, HFGenerator


def _make_generator_with_fakes(tokenizer, model) -> HFGenerator:
    generator = object.__new__(HFGenerator)  # skip __init__ (no transformers import)
    generator._model_name = "fake-hf-generator"
    generator._tokenizer = tokenizer
    generator._model = model
    return generator


class _FakeBatchEncoding(dict):
    """Mimics the small slice of `transformers.BatchEncoding`'s
    interface `HFGenerator.generate` relies on: dict-unpackable via
    `**`, `.to(device)` returns self, and `.input_ids` is readable as
    an attribute.
    """

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class _FakeGeneratedRow(list):
    """Mimics one row of a `torch` tensor: `.tolist()` returns a plain
    list of the same ints."""

    def tolist(self):
        return list(self)


class _RecordingTokenizer:
    def __init__(self, prompt_token_ids, decoded_text="a concise grounded answer"):
        self._prompt_token_ids = prompt_token_ids
        self._decoded_text = decoded_text
        self.chat_template_messages = None
        self.encoded_texts = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.chat_template_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return f"TEMPLATED::{messages[0]['content']}"

    def __call__(self, texts, return_tensors="pt"):
        self.encoded_texts = texts
        assert return_tensors == "pt"
        return _FakeBatchEncoding(input_ids=[self._prompt_token_ids])

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_ids = ids
        assert skip_special_tokens is True
        return self._decoded_text


class _RecordingModel:
    device = "cpu"

    def __init__(self, prompt_token_ids, extra_generated_ids):
        self.received_kwargs = None
        self._full_row = list(prompt_token_ids) + list(extra_generated_ids)

    def generate(self, **kwargs):
        self.received_kwargs = kwargs
        return [_FakeGeneratedRow(self._full_row)]


def test_model_name_defaults_to_the_m6_spec_model():
    assert GENERATION_MODEL == "Qwen/Qwen3-4B-Instruct-2507"


def test_generate_builds_a_single_user_turn_chat_template_from_the_prompt():
    tokenizer = _RecordingTokenizer(prompt_token_ids=[1, 2, 3])
    model = _RecordingModel(prompt_token_ids=[1, 2, 3], extra_generated_ids=[4, 5])
    generator = _make_generator_with_fakes(tokenizer, model)

    generator.generate("MY GROUNDED PROMPT")

    assert tokenizer.chat_template_messages == [{"role": "user", "content": "MY GROUNDED PROMPT"}]
    assert tokenizer.encoded_texts == ["TEMPLATED::MY GROUNDED PROMPT"]


def test_generate_only_decodes_newly_generated_tokens_not_the_prompt():
    tokenizer = _RecordingTokenizer(prompt_token_ids=[1, 2, 3])
    model = _RecordingModel(prompt_token_ids=[1, 2, 3], extra_generated_ids=[4, 5])
    generator = _make_generator_with_fakes(tokenizer, model)

    generator.generate("prompt")

    assert tokenizer.decoded_ids == [4, 5]


def test_generate_returns_the_decoded_and_stripped_answer():
    tokenizer = _RecordingTokenizer(prompt_token_ids=[1], decoded_text="  padded answer  ")
    model = _RecordingModel(prompt_token_ids=[1], extra_generated_ids=[2])
    generator = _make_generator_with_fakes(tokenizer, model)

    answer = generator.generate("prompt")

    assert answer == "padded answer"


def test_max_new_tokens_is_forwarded_to_the_underlying_model():
    tokenizer = _RecordingTokenizer(prompt_token_ids=[1])
    model = _RecordingModel(prompt_token_ids=[1], extra_generated_ids=[2])
    generator = _make_generator_with_fakes(tokenizer, model)

    generator.generate("prompt", max_new_tokens=64)

    assert model.received_kwargs["max_new_tokens"] == 64


def test_genuine_runtime_errors_from_the_model_are_not_swallowed():
    class _FailingModel:
        device = "cpu"

        def generate(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    tokenizer = _RecordingTokenizer(prompt_token_ids=[1])
    generator = _make_generator_with_fakes(tokenizer, _FailingModel())

    with pytest.raises(RuntimeError):
        generator.generate("prompt")
