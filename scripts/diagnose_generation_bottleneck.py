#!/usr/bin/env python
"""One-call diagnostic: run BEFORE choosing between batching, quantization,
or a different model.

Answers the question raised by "Qwen3-0.6B took roughly the same time as
Qwen3-4B": is the bottleneck (a) the model silently running partly/fully on
CPU because `device_map="auto"` misjudged available GPU memory, or (b) fixed
per-call overhead (tokenization, KV-cache setup, the transformers generation
loop) that doesn't scale down with model size?

Usage (on the Kaggle GPU environment):
    python scripts/diagnose_generation_bottleneck.py
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

PROMPT = (
    "You are a scientific question-answering assistant. Answer using only "
    "the evidence below.\n\nEvidence passages:\n\n[1] chunk_id=demo\n"
    "This is a placeholder evidence passage of representative length, "
    "similar to a real chunk from a research paper section, so timing "
    "reflects realistic prompt sizes rather than a trivially short string. "
    "Repeat this sentence a few times to reach a plausible chunk length. "
    "Repeat this sentence a few times to reach a plausible chunk length.\n\n"
    "Question: What does this evidence describe?\n\nAnswer:"
)


def main() -> None:
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t1 = time.time()
    print(f"Tokenizer load: {t1 - t0:.1f}s")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="bfloat16", device_map="auto"
    )
    t2 = time.time()
    print(f"Model load: {t2 - t1:.1f}s")

    # THE key check: where did the model's layers actually land?
    if hasattr(model, "hf_device_map"):
        print(f"hf_device_map = {model.hf_device_map}")
    print(f"First parameter device = {next(model.parameters()).device}")

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t3 = time.time()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t4 = time.time()
    print(f"Tokenize + move to device: {t4 - t3:.2f}s, prompt_tokens={inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    t5 = time.time()
    print(f"generate() for 64 new tokens: {t5 - t4:.1f}s  <-- compare this to your ~112s/call pilot number")

    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    t6 = time.time()
    print(f"Decode: {t6 - t5:.2f}s")
    print(f"\nAnswer: {answer[:200]}")

    print("\n--- Interpretation ---")
    print("If any parameter device above is 'cpu', or hf_device_map shows")
    print("'cpu' entries: that's the bottleneck (CPU offload). Fix by passing")
    print("device_map={'': 0} explicitly instead of 'auto'. Batching and")
    print("quantization will NOT fix this on their own.")
    print("If everything is on cuda:0 and generate() is still ~100s+ for")
    print("only 64 tokens: it's per-call overhead, not FLOPs. Batching")
    print("(generate_batch below) is the correct fix.")


if __name__ == "__main__":
    main()
