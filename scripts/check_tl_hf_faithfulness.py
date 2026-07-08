"""Back the faithfulness claim: does TLWhiteBoxWeakModel's greedy decode match
HFWeakModel's (AutoModelForCausalLM) token-for-token?

TransformerLens re-implements the forward pass, so logits are not bit-identical to
HF and greedy argmax can differ near ties (more likely in bf16). This script runs a
multi-prompt, token-by-token comparison so the TL-vs-HF parity is measured, not
assumed, before TL-produced entries are compared to HF-measured leaderboard
baselines. Runs bf16 on GPU (the pipeline dtype) if available, else float32 on CPU.

Run:  PYTHONPATH=<repo> python scripts/check_tl_hf_faithfulness.py --k 24
"""
import argparse
import os

import torch

from w2s_research.core.white_box import TLWhiteBoxWeakModel

NAME = os.getenv("WEAK_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

PROMPTS = [
    "Explain why the sky is blue.",
    "Compute 17 * 23 and show your working.",
    "Give three tips for writing clear code.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "List the first five prime numbers and explain what prime means.",
    "Explain the difference between weather and climate.",
    "How does a refrigerator keep food cold?",
    "What is 48 divided by 6, and how would you check the answer?",
    "Describe why the ocean has tides.",
]


def hf_greedy(hf, tok, instruction, k, device):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(device)
    out = []
    for _ in range(k):
        with torch.no_grad():
            logits = hf(ids).logits[0, -1, :].float()
        nxt = int(logits.argmax().item())
        out.append(nxt)
        if nxt == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(f"=== TL vs HF greedy faithfulness  device={device} dtype={dtype}  K={args.k} ===", flush=True)

    weak = TLWhiteBoxWeakModel(NAME, device=device, dtype=dtype)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=weak.dtype).to(device).eval()
    tok = weak.tokenizer

    tot_agree = tot = 0
    diverged = 0
    for p in PROMPTS:
        tl = [r["top_token_id"] for r in weak.harvest(p, max_new_tokens=args.k)]
        hg = hf_greedy(hf, tok, p, args.k, device)
        n = min(len(tl), len(hg))
        agree = sum(int(tl[i] == hg[i]) for i in range(n))
        fd = next((i for i in range(n) if tl[i] != hg[i]), None)
        tot_agree += agree
        tot += n
        diverged += int(fd is not None)
        flag = "OK " if fd is None else f"DIV@{fd}"
        print(f"  {flag}  {agree}/{n}  :: {p[:52]}", flush=True)

    rate = tot_agree / tot if tot else float("nan")
    print(f"\nOVERALL token agreement {tot_agree}/{tot} = {rate:.4f}  "
          f"({len(PROMPTS)-diverged}/{len(PROMPTS)} prompts fully identical)  "
          f"device={device} dtype={dtype}", flush=True)
    if rate < 1.0:
        print("NOTE: <100% agreement — TL and HF greedy diverge (expected near ties, "
              "esp. bf16). Quantify before treating TL entries as HF-comparable.", flush=True)


if __name__ == "__main__":
    main()
