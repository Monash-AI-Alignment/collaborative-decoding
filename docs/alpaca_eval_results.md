# AlpacaEval Open-Ended Results (collaborative decoding) — exploratory

**Date:** 2026-06-30 · **Hardware:** 1× A100-80GB (Monash M3) · **Code:** branch `collaborative-decoding` @ `c69da94`+ · **Status:** exploratory (n=20), pipeline + metric validated.

First open-ended run of the sandbox. Unlike GSM8K/MATH (CPU exact-match), utility here is an
**LLM-judge winrate**, measured by a local Gemma-4-31B judge.

## Setup
- **Weak (white-box):** `meta-llama/Llama-3.2-1B-Instruct` · **Strong (black-box):** `Qwen/Qwen2.5-7B-Instruct`
- **Judge:** `google/gemma-4-31B-it` (local vLLM, `http://m3u006:8001/v1`), **continuous logprob-weighted** preference (AlpacaEval-2.0 style), position-swapped.
- **Benchmark:** AlpacaEval prompts, `n = 20` · **Utility metric:** length-controlled (LC) winrate.
- **Reference = the STRONG model's own outputs** (not the GPT-4-turbo AlpacaEval baseline — see "Why" below).
  `recovery = 1.0` ⇔ **parity with the strong model**; `recovery = 0` ⇔ weak-only.

## Why the strong model is the reference (not GPT-4-turbo)
Against the AlpacaEval GPT-4-turbo baseline, **both** our models score ≈0 winrate (Gemma decisively prefers
GPT-4-turbo over a 1B *and* a 7B), so `U_strong − U_weak` is degenerate (gap ≈ 0.005) — no signal to optimize,
even with continuous preferences. Using the strong model's own outputs as the reference gives a healthy,
discriminative gap and directly expresses the research goal ("preserve *the strong model's* utility").

## Baselines (n=20)
| | LC winrate vs strong | note |
|---|---|---|
| `U_weak` (Llama-1B vs Qwen-7B) | **0.166** | weak genuinely *beats* strong on 3/20 prompts, ties 1 |
| `U_strong` (parity, by definition) | **0.500** | strong vs itself; never judged |
| **gap** | **0.334** | usable |

## Headline result — frontier at parity
**6 of 29 policies hold `recovery ≥ 0.98` (parity with the strong model).** Best by weak-fraction:

| f_weak | recovery | LC winrate | policy |
|---|---|---|---|
| **0.438** | 1.01 | 0.503 | **`margin_threshold` (τ=0.05), span 64** |
| 0.383 | 1.16 | 0.553 | `budget_entropy` (τ=0.3, budget=10), span 64 |
| 0.273 | 1.06 | 0.521 | `entropy_streak` (τ=0.5, k=2), span 64 |
| 0.151 | 1.05 | 0.518 | `entropy_threshold` (τ=1.0), span 256 |
| 0.144 | 1.10 | 0.534 | `or_gate` (τ=1.0, margin 0.05), span 64 |
| 0.141 | 1.03 | 0.510 | `entropy_threshold` (τ=0.7), span 32 |

## Findings
1. **Open-ended is the favorable domain.** The weak model carries **~44% of characters at parity** here vs
   **~27%** on GSM8K — confirming the hypothesis that open-ended generation (mostly fluent prose) lets the
   weak model do far more while the strong model covers the few hard spans.
2. **Different policies win.** Uncertainty-based gating (`margin_threshold`, `budget_entropy`,
   `entropy_streak`) leads — *not* the math-specific `context_gate`/`answer_protect` (correctly excluded for
   open-ended). Low-margin deferral (defer only when the weak model's top-2 are nearly tied) is the standout.
3. **Several configs reach recovery > 1.0** — the collaborative output occasionally *beats* strong-only per
   the judge (plausible: the weak+strong mix phrases some answers better). Not statistically established at n=20.

## Caveats (this is exploratory)
- **n = 20 → noisy.** Winrate std-error ≈ 0.11, so `recovery ≈ 1.0–1.16` is "at/near parity" — the ordering
  among the top configs is within noise, and `recovery > 1.0` is not established. The *robust* takeaway is the
  domain comparison (open-ended f_weak ≈ 0.4 ≫ GSM8K ≈ 0.27), not exact per-policy numbers.
- **Judge = Gemma-31B, reference = strong-only.** These are internal-comparison numbers, not
  AlpacaEval-leaderboard figures. LC here is a per-method fit (recovery/ranking valid; absolute LC not
  directly AlpacaEval-2.0-comparable).
- **25% of the best config's generations hit the length cap** (`finished_frac = 0.75`); open-ended answers run
  long. Consider a higher `max_chars`/`max_steps` or report unfinished-rate.
- **Refinement got 0 rounds** — the 29-config curated sweep consumed the 85-min budget (the run was
  time-boxed to fit the GPU window). No hill-climbing on the winners yet.

## Next
- Re-run at larger `n` (≥ 100) with seeds + the refinement phase, to tighten the winrate estimates and push
  `margin_threshold`/`budget_entropy` further. **Requires the Gemma judge server kept alive for the run.**
- The winning open-ended policies (`margin_threshold` τ≈0.05, `budget_entropy`) are the seeds for the
  autonomous agents on open-ended tasks.

Raw outputs: `/scratch2/ml23/smur0075/w2s_decode_runs/alpaca_57965548/`
(`baselines.json`, `results.jsonl` [29 rows], `frontier.json`, `best.json`, `meeting_bar/*.json`).
