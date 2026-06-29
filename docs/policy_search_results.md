# Unattended Policy-Search Results (collaborative decoding)

**Date:** 2026-06-29 · **Hardware:** 1× A100-80GB (Monash M3, job 57965548) · **Code:** branch
`collaborative-decoding` @ `306230f`+ · **Runtime:** 7h (25,238s), 143 configs, fully autonomous.

This run was launched unattended (`scripts/policy_search.py`) while the researcher was away. It loads the
weak + strong models once, measures free-running baselines, then searches deferral policies to **maximize the
weak-token-fraction `f_weak` subject to `utility_recovery ≥ 0.98`** — the strict bar that Phase 1's naive
entropy baseline could not reach. It is a pure search loop (no `claude` CLI), so it consumes no Claude usage.

## Setup
- **Weak (white-box):** `meta-llama/Llama-3.2-1B-Instruct` (HF, stateful greedy decode)
- **Strong (black-box):** `Qwen/Qwen2.5-7B-Instruct` (vLLM, text in/out; different tokenizer)
- **Benchmark:** GSM8K, `n = 50` test problems · **Baselines (free-running):** `U_weak = 0.40`,
  `U_strong = 0.94`, `gap = 0.54`
- `utility_recovery = (U − U_weak) / gap`; headline = max `f_weak` s.t. `recovery ≥ 0.98`.

## Headline result
**`context_gate` is the only policy that cleared the 0.98 bar — and it did so robustly.**

| best @ `recovery ≥ 0.98` | utility | recovery | **f_weak** | params |
|---|---|---|---|---|
| **`context_gate`** | 0.94 | **1.00** | **0.272** | `defer_threshold=0.313`, `span_max=64` |

- The weak Llama-1B carried **~27% of all characters with zero utility loss** (matched the strong model's
  exact-match count, 47/50).
- This is **~2× the weak-fraction of naive entropy** at comparable recovery. Naive `entropy_threshold` peaked
  at `recovery ≈ 0.963` (never reaching 0.98), consistent with the Phase-1 finding.
- 6 of 143 configs met the bar — **all of them `context_gate`** with `span_max=64`, `defer_threshold` in
  `[0.20, 0.313]` (`f_weak` 0.234 → 0.272). The mechanism, not a single lucky setting, is what works.

**Winning mechanism:** defer to the strong model only when the weak model is uncertain *and* the text is at a
computation-critical position (trailing `= + - * / :` etc.), letting the weak model write the cheap reasoning
prose. This is exactly the "weak does prose, strong does the arithmetic" hypothesis — and it preserves utility
because only the few utility-critical tokens are deferred.

## Utility-vs-f_weak frontier (best f_weak at each recovery level)

| recovery | f_weak | utility | policy |
|---|---|---|---|
| 1.00 | 0.272 | 0.94 | context_gate (τ=0.313, sm=64) |
| 0.96 | 0.287 | 0.92 | context_gate (τ=0.369, sm=64) |
| 0.92 | 0.290 | 0.90 | context_gate (τ=0.382, sm=64) |
| 0.88 | 0.296 | 0.88 | context_gate (τ=0.402, sm=96) |
| 0.86 | 0.404 | 0.86 | margin_threshold (0.2, sm=64) |
| 0.74 | 0.457 | 0.80 | entropy_cooldown (τ=0.3, m=8) |
| 0.56 | 0.633 | 0.70 | entropy_streak (τ=0.5, k=3) |
| 0.38 | 0.737 | 0.60 | margin_threshold (0.05, sm=64) |
| 0.30 | 0.692 | 0.56 | answer_protect (τ=2.0, sm=256) |

`context_gate` dominates the **high-recovery band (≥0.88)** and gives a *flat plateau* there — recovery rises
0.88 → 1.00 while f_weak only dips 0.296 → 0.272. If you relax the utility bar, other policies (margin, streak,
cooldown) trade far more weak-fraction for utility.

## Caveats (important — these are exploratory numbers)
1. **n = 50.** `recovery = 1.00` means "matched the strong model's correct count," not a tight estimate.
   Validate the winner at full GSM8K + multiple seeds before any claim.
2. **Search allocation is greedy.** The hill-climb seeds from the best bar-meeting config, so it spent
   111/143 configs refining `context_gate`; the other 6 novel policies got only their 2–4 curated configs and
   were never tuned. "Only context_gate meets the bar" is partly that other policies were under-explored at
   high recovery — a fairer/longer search (or the autonomous agents) may lift them.
3. **f_weak ≈ 0.27 is modest.** Big headroom remains; this is the gap the autonomous research agents should
   attack (smarter computation-position detection, hybrids of context_gate + cooldown/streak, span control).

## Next steps
- Re-validate `context_gate` (τ≈0.3, span_max=64) at full GSM8K + seeds; add MATH.
- Use `context_gate` as the **seed policy** the autonomous Claude agents start from in the supervised
  Phase-2/3 build (server scoring + MCP tools + agent prompt + SLURM loop).

Raw outputs: `/scratch2/ml23/smur0075/w2s_decode_runs/search_57965548/`
(`baselines.json`, `results.jsonl` [143 rows], `frontier.json`, `best.json`, `meeting_bar/*.json`).
