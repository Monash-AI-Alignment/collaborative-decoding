"""Smoke test for VLLMStrongModel — run on a GPU node.

Usage:
    HF_HOME=/scratch2/ml23/smur0075/hf_cache python scripts/smoke_strong_model.py
"""
from w2s_research.core.strong_model import VLLMStrongModel
from w2s_research.core.benchmarks import build_instruction

strong = VLLMStrongModel("Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.6)
instr = build_instruction("gsm8k", "What is 2 + 2?")

# span mode: stop at newline, should NOT be finished
span = strong.generate(instr, "", stop=["\n"], max_tokens=64, temperature=0.0)
print("SPAN:", repr(span.text), "finished=", span.finished)

# full mode: no stop, should finish on EOS
full = strong.generate(instr, "", stop=None, max_tokens=256, temperature=0.0)
print("FULL:", repr(full.text[:200]), "finished=", full.finished)
assert full.text.strip(), "strong model produced no text"
print("OK: VLLMStrongModel smoke passed")
