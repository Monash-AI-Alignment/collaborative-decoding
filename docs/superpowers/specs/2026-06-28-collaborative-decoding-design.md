# Collaborative Decoding Research Sandbox — Design

**Date:** 2026-06-28
**Status:** Draft for review
**Branch:** `collaborative-decoding`

## 1. Problem statement

We are repurposing the *automated weak-to-strong research* sandbox into an *automated
collaborative-decoding research* sandbox. The autonomous Claude agent infrastructure
(agent loop, MCP tools, dashboard, leaderboard, findings forum, idea-plugin pattern)
is kept; the scientific core is replaced.

**Research question.** Given a cheap **weak** model and an expensive **strong** model,
how large a fraction of generated tokens can the weak model produce while *preserving*
the strong model's task utility? The method must decide, during generation, **when to
defer** the next part of the output to the strong model (e.g., when the weak model is
uncertain — high logit entropy).

### Setting / assumptions (hard constraints)

- **No fine-tuning** of either model. Everything is inference-time orchestration.
- **Weak model = white-box**: full access to per-step logits / probability distribution.
- **Strong model = black-box**: text in, text out only. No logits, no verification signal.
- **Tokenizers may differ** between weak and strong. Handoff therefore happens at the
  **text level**, not the token level.

### Concrete instantiation (v1)

| Role | Model | Access | Tokenizer |
|------|-------|--------|-----------|
| Weak | `meta-llama/Llama-3.2-1B-Instruct` | white-box (HF Transformers, per-step logits) | Llama |
| Strong | `Qwen/Qwen2.5-7B-Instruct` | black-box (vLLM, text in/out only) | Qwen (different) |

The **weak** model (Llama-3.2-1B) is gated on HuggingFace: requires license acceptance and
`HF_TOKEN`. The strong model (Qwen2.5) is ungated. The two use different tokenizers, preserving
the cross-tokenizer constraint. Strong-model size is a knob (`Qwen2.5-7B` default; `-14B` for a
higher ceiling) — both fit alongside the 1B weak on one A100-80GB / H100.

## 2. Goals and non-goals

**Goals**
- A trusted, shared **collaborative-decoding engine** that runs the weak/strong loop,
  measures the weak-token-fraction, and is the *only* component that measures it (so an
  idea cannot fake its own efficiency number).
- A pluggable **deferral-policy contract** — an idea implements only `decide()`.
- Benchmarks GSM8K, MATH (Hendrycks), AlpacaEval with server-side utility scoring.
- A leaderboard ranking on **`f_weak` subject to a strict utility-preservation bar**.
- Autonomous Claude agent running on Monash M3 via SLURM (one agent, architected to
  scale to N parallel agents sharing a findings forum).

**Non-goals (v1)**
- Any training / fine-tuning / LoRA / RL.
- Cloud deployment (RunPod / S3 / Docker) — left dormant, not the primary path.
- Token-level interleaving that assumes a shared vocabulary (ruled out by the
  different-tokenizer constraint).
- Speculative-decoding-style verify/accept-reject (ruled out by the black-box constraint —
  there is no target distribution to verify against).

## 3. Architecture

### 3.1 Component verdict map (grounded in current files)

| File / area | Verdict | Notes |
|---|---|---|
| `research_loop/agent.py` | **KEEP** | Bump researcher model to latest Opus; update `allowed_tools` and prompt. |
| `research_loop/tools/server_api_tools.py` | **ADAPT** | `evaluate_predictions` → `evaluate_generations`; `share_finding` metrics → utility/`f_weak`. |
| `research_loop/tools/findings_sync.py`, `prior_work_tools.py`, `http_utils.py`, `telemetry/*`, `hooks/log_*` | **KEEP** | Filesystem/HTTP based, cloud-agnostic. |
| `research_loop/hooks/sync_to_s3.py` | **KEEP (dormant)** | Already no-ops when `LOCAL_MODE` truthy. |
| `research_loop/prompt.jinja2` | **REWRITE** | New problem description, workflow, tools. |
| `web_ui/backend/evaluation.py` | **ADAPT** | Replace `compute_metrics_from_predictions`/PGR with utility + `f_weak` + `utility_recovery`. |
| `web_ui/backend/models.py` | **ADAPT** | Add `utility`, `weak_token_fraction`, `utility_recovery` columns to `Experiment` + `Finding`. |
| `web_ui/backend/app.py` | **ADAPT** | `/api/evaluate-predictions` → `/api/evaluate-generations`; leaderboard ranks by `f_weak`@bar; `/api/config` defaults. |
| `web_ui/backend/worker.py` | **ADAPT** | Add `_deploy_to_slurm` (sbatch) launch path; keep local subprocess. |
| `web_ui/frontend/*` | **ADAPT (low priority)** | Relabel PGR→utility/`f_weak`; can lag backend. |
| `core/data.py` | **ADAPT** | Keep helpers; add generative-benchmark loaders. |
| `core/eval.py` | **ADAPT** | Reuse answer-checking; utility computation moves to `benchmarks.py`. |
| `core/vllm_inference.py` | **ADAPT** | Reuse for fast strong-model + baseline generation. |
| `core/seed_utils.py`, `core/config.py` (`RunConfig`) | **KEEP/ADAPT** | Drop training fields; add decoding fields. |
| `core/train.py`, `ideas/vanilla_w2s/loss.py` | **DROP** | Training-only. |
| `ideas/{vanilla_w2s,critic,ue_*,train_only_on_confident_labels}` | **ARCHIVE** | Move out of active `ideas/`; not deleted (reference). |
| `ideas/ue_zeroshot/math_eval_tools.py`, `math_normalize.py` | **REUSE** | Existing MATH/GSM8K answer normalization + checking. |
| `infrastructure/runpod.py`, `execute_autonomous.py`, most of `s3_utils.py` | **DORMANT** | Not used by SLURM path. |
| `Dockerfile`, `entrypoint.sh` | **DORMANT** | SLURM is the primary execution path. |

### 3.2 New core modules

```
w2s_research/core/
  weak_model.py      # WeakModel: HF Transformers, white-box per-step logits + detokenize
  strong_model.py    # StrongModel interface (black-box text in/out); VLLMStrongModel impl
  collab_decode.py   # CollaborativeDecoder engine + WeakStepState + DecodeResult
  policy.py          # DeferralPolicy base class (the idea contract)
  benchmarks.py      # GSM8K / MATH / AlpacaEval: load, render prompt, extract answer, score utility
```

### 3.3 The collaborative-decoding engine (`collab_decode.py`)

Per example with rendered prompt `P`:

```
output_text = ""
weak_chars = 0, strong_chars = 0
while not done and len(generated_tokens) < max_new_tokens:
    logits = weak.next_token_logits(P + output_text)     # WHITE-BOX full distribution
    state  = WeakStepState(logits, probs, entropy, margin, output_text, step_idx, ...)
    if policy.decide(state) == CONTINUE:
        tok   = sample(probs, gen_params)                # argmax or temperature sample
        piece = weak.detokenize([tok])
        output_text += piece;  weak_chars += len(piece)
        done = is_eos(tok)
    else:  # DEFER a span to the strong model
        span = strong.generate(P + output_text, stop=span_boundary, max_tokens=span_budget)
        output_text += span;   strong_chars += len(span)
        done = span_terminates(span)                      # e.g., strong emitted EOS/answer
return DecodeResult(text=output_text, weak_chars, strong_chars, trace=[...])
```

- **Cross-tokenizer is free**: weak emits *text* (detokenized), strong consumes/produces *text*.
- **Measurement unit = characters**, char-weighted across the eval set:
  `f_weak = Σ weak_chars / Σ (weak_chars + strong_chars)`. Tokenizer-robust. The engine
  owns this number; idea code never reports it.
- **Span boundary (default)**: strong generates until end-of-line / sentence terminator
  (or `span_budget` tokens), then control returns to the weak model. Configurable via
  `RunConfig`.
- **Weak runtime**: HF Transformers manual decode loop (needed for true per-step
  distributions). **Strong runtime**: vLLM (fast; prefix caching mitigates repeated
  prefill of the growing context). Baselines (`weak_only`, `strong_only`) may use vLLM
  end-to-end for speed.

### 3.4 Strong-model interface (`strong_model.py`)

```python
class StrongModel(Protocol):
    def generate(self, prompt: str, stop: list[str] | None, max_tokens: int,
                 temperature: float) -> str: ...   # text in, text out ONLY
```

`VLLMStrongModel` wraps a local vLLM-served Qwen2.5-7B-Instruct but **exposes no logits** —
enforcing the black-box constraint in code so ideas cannot cheat. An `APIStrongModel`
(Anthropic/OpenAI) adapter is a later drop-in.

### 3.5 The idea / deferral-policy contract (`policy.py`)

```python
class DeferralPolicy:
    def __init__(self, config: RunConfig): ...
    def decide(self, state: WeakStepState) -> Decision:   # CONTINUE | DEFER
        ...
    # optional: on_strong_span(span) -> handback hook
```

An idea directory provides `run.py` exposing `build_policy(config) -> DeferralPolicy`
(plus optional metadata). `run.py`'s `run_experiment(config)` is a thin wrapper the engine
calls; it returns `{utility, weak_token_fraction, utility_recovery, operating_points, ...}`.

**Baselines shipped:**
- `weak_only` — never defer (lower utility bound, `f_weak=1.0`).
- `strong_only` — always defer (upper utility bound, `f_weak=0.0`).
- `random_defer` — defer each step with prob `p` (frontier reference).
- `entropy_threshold` — defer when token entropy `> τ` (the canonical baseline).
- `margin_threshold` — defer when `top1 − top2` probability margin `< τ`.

### 3.6 Benchmarks & utility (`benchmarks.py`)

| Benchmark | Utility | Scoring | Where computed |
|---|---|---|---|
| GSM8K | exact-match on final number | `math_normalize` + extract | server (CPU) |
| MATH (Hendrycks) | exact-match on `\boxed{}` answer | `math_eval_tools` | server (CPU) |
| AlpacaEval | winrate vs reference outputs | LLM judge (Claude API, configurable) | server (needs API egress + cost) |

Gold answers / references are held **server-side**; the agent only sees questions and
submits generations. GSM8K + MATH are CPU-only and self-contained. AlpacaEval is wired
last because of the judge dependency.

## 4. Metric (precise)

Per benchmark `B`, fixed baselines computed once:
- `U_weak(B)` = utility of `weak_only`
- `U_strong(B)` = utility of `strong_only`

For a method `M`:
- `U_M(B)` = utility on `B`
- `f_weak(M,B)` = char-weighted weak fraction (engine-measured)
- `utility_recovery(M,B) = (U_M − U_weak) / (U_strong − U_weak)` (may exceed 1; reported raw)

**Headline leaderboard rank:** among methods with `utility_recovery ≥ R_bar`
(**default `R_bar = 0.98`** — utility loss is heavily penalized), **maximize `f_weak`**;
tie-break by higher `U_M`. The full set of `(U_M, f_weak)` operating points (e.g., from a
`τ` sweep) is stored so we can plot the utility-vs-fraction Pareto frontier. `R_bar` is a
server config knob.

**Baseline measurement.** `U_weak` and `U_strong` must be measured **free-running** (the strong
model generates its whole answer in one call, `--span-stop none`), not via the line-segmented
span handoff used by the deferral policies — otherwise repeated re-prefill could skew `U_strong`,
the denominator of `utility_recovery`. The deferral policies themselves use span-level handoff.

## 5. End-to-end data flow

1. Agent session starts (SLURM GPU job). Reads notebook + leaderboard + shared findings.
2. Proposes a deferral policy; implements it under `ideas/autonomous_<name>/`.
3. Runs the shared engine over a benchmark (weak HF loop + black-box strong) →
   generations + engine-measured `f_weak`.
4. Submits generations to server `POST /api/evaluate-generations` → server scores
   `U_M`, computes `utility_recovery` vs fixed baselines, returns metrics.
5. Agent records to `notebook.json`, shares via `share_finding` (auto-published to
   leaderboard when `finding_type="result"`).
6. Other agents pull the finding via the forum (HTTP hub, or shared-FS forum).

## 6. Config / env changes

**`RunConfig` (core/config.py)** — drop all training fields (`epochs`, `lr*`, `batch_size`,
`lora_*`, `grpo_*`, `sft_*`, `loss`, `optimizer`, `weight_decay`, `warmup*`). Add:
`benchmark` (gsm8k|math|alpaca_eval), `max_new_tokens`, `defer_threshold` (τ),
`span_boundary` (line|sentence|tokens), `span_budget`, `gen_temperature`/`gen_top_p`
(kept), `r_bar`, `eval_size`.

**Env vars** — keep `WEAK_MODEL`, `STRONG_MODEL`, `DATASET_NAME`/`BENCHMARK`, `DATA_DIR`,
`WORKSPACE_DIR`, `SERVER_URL`/`ORCHESTRATOR_API_URL`, `LOCAL_MODE`. Set on M3:
`HF_HOME=/scratch2/ml23/smur0075/hf_cache`, `HF_TOKEN=...` (Llama gate),
`ANTHROPIC_API_KEY` (researcher + AlpacaEval judge). New defaults:
`WEAK_MODEL=meta-llama/Llama-3.2-1B-Instruct`, `STRONG_MODEL=Qwen/Qwen2.5-7B-Instruct`,
`R_BAR=0.98`. `HF_TOKEN` is required for the (gated) weak Llama model. Optional
`SHARED_FINDINGS_DIR` for the filesystem forum.

## 7. Infrastructure (M3 / SLURM)

- **Server**: GPU-free Flask app (GSM8K/MATH utility is CPU string-matching). Runs as a
  small persistent process reachable by compute nodes via `ORCHESTRATOR_API_URL`.
- **Agent**: `AutonomousAgentLoop` on a GPU SLURM job (one A100-80GB or H100 fits
  Llama-1B + Qwen-7B + KV cache). `slurm/agent.sbatch` loads `cuda/12.2.0`, activates the
  venv, sets `HF_HOME`/`HF_TOKEN`, points at the server. The HF-served weak model and the
  vLLM-served strong model share one GPU, so vLLM `gpu_memory_utilization` is capped
  (~0.6) to leave room for the weak model.
- **Scale-out**: `sbatch` N agent jobs against the same server (HTTP findings hub already
  supports this). Optional shared-FS forum via `SHARED_FINDINGS_DIR` if centralizing is
  undesirable.
- New: `slurm/server.sbatch`, `slurm/agent.sbatch`, `scripts/prepare_benchmarks.py`
  (download GSM8K/MATH/AlpacaEval into `data/` with gold held separately).

## 8. Build phases (verify each gate before proceeding)

**Phase 0 — setup.** `uv sync`; download Llama-3.2-1B + Qwen2.5-7B to scratch2; smoke-test
loading weak (HF) + strong (vLLM) on a GPU node.

**Phase 1 — engine + baselines, CLI only.** Implement `weak_model`, `strong_model`,
`collab_decode`, `policy`, `benchmarks`; ship baseline policies. Run `weak_only` and
`strong_only` on GSM8K.
- **GATE 1a:** `U_strong − U_weak` is a *meaningful* gap (expected large for Llama-3.2-1B
  ~44% vs Qwen2.5-7B ~85% on GSM8K) **and** the weak model is coherent enough that
  `weak_only` produces valid, parseable outputs (the flipped risk now that the weak model
  is small — see Risks). Adjust model sizes if either fails.
- **GATE 1b:** cross-tokenizer span handoff yields coherent text; `entropy_threshold` τ-sweep
  produces a sensible `(U, f_weak)` frontier between the two baselines.

**Phase 2 — server + tools + prompt.** Adapt `evaluation.py`, `models.py`, `app.py`,
`server_api_tools.py`, `prompt.jinja2`. 
- **GATE 2:** an idea runs end-to-end through `/api/evaluate-generations` and appears on the
  leaderboard ranked by `f_weak`@`R_bar`.

**Phase 3 — agent + SLURM.** `slurm/agent.sbatch` + `slurm/server.sbatch`; researcher model
bumped to latest Opus. 
- **GATE 3:** one agent autonomously completes propose → implement → run → evaluate → share.
  Then `sbatch` N agents sharing findings.

## 9. Risks & open questions

1. **Weak too weak vs. gap (flipped risk).** With weak=Llama-3.2-1B (~44% GSM8K) and
   strong=Qwen2.5-7B (~85%), the utility gap is now healthily large — good for the recovery
   story. The risk flips: a 1B model may be *too* incoherent on MATH, producing
   unparseable outputs or deferring nearly everything (`f_weak → 0`), which is
   uninteresting. **Mitigation:** Phase 1 Gate 1a checks `weak_only` coherence/parse-rate
   per benchmark; if a 1B weak is unusable on MATH, bump to Qwen2.5-1.5B or run MATH with a
   slightly larger weak. GSM8K is the safer primary benchmark for the 1B weak.
2. **Weak HF decode-loop throughput.** Token-by-token HF generation is slow vs vLLM.
   **Mitigation:** batch examples, cap `max_new_tokens`, use vLLM for baselines; consider
   vLLM per-step logprobs for the weak model if the loop is too slow.
3. **Repeated strong prefill.** Each defer re-sends the growing context to strong.
   **Mitigation:** vLLM automatic prefix caching; span-level (not token-level) handoff.
4. **AlpacaEval judge.** Needs API egress from compute nodes + cost. Wired last; judge model
   configurable; GSM8K/MATH carry Phases 1–2.
5. **`f_weak` integrity.** Self-reported numbers are cheatable; mitigated by making the
   shared engine the sole measurer. Future: server re-runs a sampled subset to audit.
6. **Llama gating.** The weak model (Llama-3.2-1B) requires HF license acceptance +
   `HF_TOKEN` in the SLURM env. The strong Qwen model is ungated.

## 10. Researcher-agent model & session notes

- The autonomous researcher (the Claude driving experiments) currently defaults to
  `claude-opus-4-6` in `agent.py`; bump to the latest Opus.
- "Ultracode" is a *Claude Code session* mode for our development of this repo — it does not
  run inside the autonomous researcher. The researcher's own reasoning depth is governed by
  its prompt and the bundled `/research-thinking` skill.
