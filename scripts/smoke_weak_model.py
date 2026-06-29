"""Smoke test for HFWeakModel — run on a GPU node.

Asserts the weak model produces a COHERENT, non-degenerate continuation (this
would have caught the chat-template whitespace-degeneration bug, where the loop
collapsed into endless newlines).

Usage:
    HF_HOME=/scratch2/ml23/smur0075/hf_cache HF_TOKEN=... \
        python scripts/smoke_weak_model.py
"""
from w2s_research.core.weak_model import HFWeakModel
from w2s_research.core.benchmarks import build_instruction

weak = HFWeakModel("meta-llama/Llama-3.2-1B-Instruct")
instr = build_instruction(
    "gsm8k",
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did she sell altogether in April and May?",
)

weak.begin(instr)
text = ""
for _ in range(220):
    step = weak.peek()
    if step.is_eos:
        break
    weak.commit(step.top_token_id)
    text += step.text_piece

print("GENERATED:\n", text[:700])

# Coherence guards (regression test for the whitespace-degeneration bug):
assert len(text) > 0, "weak model produced no text"
assert any(c.isdigit() for c in text), "output has no digits — degenerate"
ws = sum(c.isspace() for c in text)
assert ws / len(text) < 0.5, f"output is {ws}/{len(text)} whitespace — degenerate"
print("\nOK: HFWeakModel smoke passed (coherent, non-degenerate)")
