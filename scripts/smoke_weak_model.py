"""Smoke test for HFWeakModel — run on a GPU node.

Usage:
    HF_HOME=/scratch2/ml23/smur0075/hf_cache HF_TOKEN=... \
        python scripts/smoke_weak_model.py
"""
from w2s_research.core.weak_model import HFWeakModel
from w2s_research.core.benchmarks import build_instruction

weak = HFWeakModel("meta-llama/Llama-3.2-1B-Instruct")
instr = build_instruction("gsm8k", "What is 2 + 2?")
text = ""
for _ in range(40):
    step = weak.next_step(instr, text)
    if step.is_eos:
        break
    print(f"piece={step.text_piece!r}  entropy={step.entropy:.3f}  margin={step.margin:.3f}")
    text += step.text_piece
print("\nGENERATED:", text)
assert len(text) > 0, "weak model produced no text"
print("OK: HFWeakModel smoke passed")
