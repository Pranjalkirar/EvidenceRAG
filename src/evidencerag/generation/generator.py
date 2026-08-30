"""LLM generation model abstraction.

Mirrors the `Embedder` / `Reranker` Protocol pattern in
evidencerag.retrieval.embeddings / evidencerag.reranking.reranker: a
minimal structural interface, implemented by the real model
(`HFGenerator`) and, in tests, by a lightweight deterministic fake
(see tests/generation_fixtures.py) -- ordinary unit tests must never
need to download or load the real model.

Import of `transformers` (and therefore `torch`) is deferred to
`HFGenerator.__init__`, exactly like `QwenEmbedder` defers importing
`sentence_transformers` -- importing this module never pulls in ML
dependencies unless the real generator is actually constructed.
"""

from __future__ import annotations

from typing import Protocol

# Fixed per M6 spec (matches the README's "Planned models" entry,
# `Qwen3-4B`, resolved to the specific Hugging Face model id) -- must
# not be silently substituted. Qwen3-4B-Instruct-2507 is the
# non-thinking-mode instruction-tuned checkpoint: ~4B parameters (fits
# comfortably in bf16/fp16 on a single Kaggle T4/P100 GPU), a standard
# chat template, and no <think> preamble to strip out of the answer.
GENERATION_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Deliberately modest: QASPER-style answers are short (a sentence or a
# few sentences), and long_answer generation is expensive on a single
# consumer/Kaggle GPU. Callers needing longer answers can override it.
DEFAULT_MAX_NEW_TOKENS = 512


class Generator(Protocol):
    """Minimal interface generation depends on: turn one fully-built
    prompt string into one answer string. Prompt construction (what
    goes into that string) is deliberately NOT part of this interface
    -- see prompt.py -- so a `Generator` implementation never needs to
    know anything about chunks, chunk_ids, or grounding instructions;
    it only ever sees the final text to complete.
    """

    @property
    def model_name(self) -> str: ...

    def generate(self, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str: ...


class HFGenerator:
    """`Qwen/Qwen3-4B-Instruct-2507` (or any compatible causal LM) via
    Hugging Face `transformers`, using the model's own chat template
    with the fully-built grounded prompt as a single user turn -- the
    integration path documented on the model's own model card.

    Requires the `transformers` and `torch` dependencies and network
    access to the Hugging Face Hub (or a local/cached copy of the
    model) -- neither is needed just to import this module.
    """

    def __init__(
        self,
        model_name: str = GENERATION_MODEL,
        device: str | None = None,
        torch_dtype: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device or "auto"
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # transformers' `.generate()` is itself already no-grad; wrapping
        # it again here would be redundant, not more correct.
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        # Convert the full generated sequence to a plain list before
        # slicing off the prompt, rather than slicing the (framework)
        # tensor first -- keeps this method's contract with its inputs
        # to exactly two operations (`[0]` then `.tolist()`), which is
        # all a fake/test double needs to implement.
        full_ids = generated_ids[0].tolist()
        output_ids = full_ids[len(model_inputs.input_ids[0]) :]
        return self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
