# Phase 1 Gate 1a/1b Results (collaborative decoding)

**Date:** 2026-06-29 · **Hardware:** 1× A100-80GB (Monash M3) · **Code:** branch `collaborative-decoding` @ `bddeca8`+

## Setup
- **Weak (white-box):** `meta-llama/Llama-3.2-1B-Instruct` (HF Transformers, stateful greedy decode)
- **Strong (black-box):** `Qwen/Qwen2.5-7B-Instruct` (vLLM, text-in/out only; **different tokenizer**)
- **Benchmark:** GSM8K, `n = 30` test problems · **Policies:** baselines + entropy-threshold sweep
- Deferral handoff at the text level (`assistant_text`); `f_weak` = char-weighted fraction from the weak model.

## Gate 1a — baselines (PASS)
| | utility | f_weak |
|---|---|---|
| `weak_only` | **0.333** | 1.000 |
| `strong_only` (free-running) | **0.933** | 0.000 |

`gap = U_strong − U_weak = 0.60` — a real, healthy recovery gap.

## Gate 1b — entropy frontier (PASS)
| τ | utility | f_weak | utility_recovery | avg defers |
|---|---|---|---|---|
| 0.3 | 0.900 | 0.113 | 0.945 | 13.1 |
| 0.6 | 0.867 | 0.174 | 0.889 | 12.2 |
| 1.0 | 0.733 | 0.261 | 0.667 | 9.7 |
| 1.5 | 0.767 | 0.443 | 0.722 | 7.8 |
| 2.0 | 0.500 | 0.655 | 0.278 | 4.6 |

Raising τ trades weak-fraction up and utility/recovery down, with all points lying **between** the two
baselines. Collaborative output is coherent across the tokenizer boundary (e.g. a τ=0.6 example produced a
clean weak+strong step-by-step solution ending `#### $18`, with 18 handoffs). Minor non-monotonicity
(τ=1.0 vs 1.5) is noise at n=30.

## Headline finding (motivates the research)
Target metric = **max `f_weak` subject to `utility_recovery ≥ 0.98`**. The naive entropy baseline **never
reaches 0.98** (best ≈ 0.945 at τ=0.3, where the weak model carries only ~11% of tokens). So the core
problem — *high weak-token-fraction while preserving ≥98% utility* — is **real and unsolved by the obvious
baseline**. This is the gap the autonomous agents (Phase 2/3) will attack.

## Notes
- These numbers VALIDATE the sandbox end-to-end (engine, cross-tokenizer handoff, scoring). They are not
  final baselines — scale up via `scripts/run_gate.py --eval-size <full> --benchmark {gsm8k,math}` and add
  seeds / AlpacaEval for publication-grade numbers.
- A prior gate run reported `U_weak = 0.0` due to a weak-decode degeneration bug (chat-template whitespace
  stripping); fixed in `bddeca8` (stateful weak model). Regression guard added to `scripts/smoke_weak_model.py`.
- Gate ran in ~12 min via `scripts/run_gate.py` (single model load) vs ~49 min for per-policy `decode_cli`.
