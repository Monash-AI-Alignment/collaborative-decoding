# Autonomous Collaborative-Decoding Research Loop — Design

**Date:** 2026-06-30
**Status:** Draft for review
**Branch:** `collaborative-decoding`
**Builds on:** `docs/superpowers/specs/2026-06-28-collaborative-decoding-design.md` (engine + metric),
`docs/alpaca_eval_results.md`, `docs/policy_search_results.md`.

## 1. Goal

Stand up the **full autonomous-research system** for collaborative decoding: a Claude agent (Opus 4.8)
autonomously proposes a deferral policy, runs the shared engine to produce collaborative generations + an
engine-measured weak-token-fraction, scores them, and **shares findings to a server-backed leaderboard +
forum** that multiple agents read and write. **POC target benchmark: AlpacaEval** (open-ended, judge-scored);
the metric framework is designed so math (GSM8K/MATH) plugs in later for joint optimization.

**Gate (definition of done):** one agent autonomously completes propose → implement → run → evaluate →
`share_finding`, and the result appears on `/api/leaderboard` ranked by `f_weak` @ `recovery ≥ R_bar`. Then
`sbatch` N agents that share findings through the server.

## 2. Key architectural decision: server trusts engine-computed metrics

The collaborative-decoding **engine is shared, trusted code**; an idea supplies only
`DeferralPolicy.decide()`. So `f_weak` (engine-measured, char-weighted) and utility (judge-measured) computed
on the agent's GPU node are as trustworthy as if recomputed on the server. Therefore:

- The **server records + ranks** engine-computed metrics; it does **not** re-run the judge or the engine.
- The server **serves the canonical baselines + reference** (Section 4) so `recovery` is identical across
  agents.
- Server-side audit (re-judging a sampled subset) is a **documented future enhancement**, not POC scope.

The integrity principle from the original design — "an idea cannot fake its own efficiency number" — still
holds: `f_weak` comes from the shared engine, not from idea code.

## 3. Components & build order

| # | Component | Files | Responsibility |
|---|---|---|---|
| 1 | Canonical baseline/reference bootstrap | `scripts/bootstrap_baselines.py` (new) | One-time GPU run: per benchmark, generate the strong-reference outputs + measure `U_weak` (and `U_strong` for math), write a canonical artifact, register it with the server. |
| 2 | Engine shared-reference scoring | `w2s_research/core/eval_idea.py` (new) | Agent-facing single-idea evaluation: load canonical reference/baselines, generate the idea's collaborative outputs, score (judge winrate for alpaca / exact-match for math) → `{utility, f_weak, recovery, operating_points, generations}`. Wraps the existing engine + judge + winrate. |
| 3 | Server scoring + schema | `web_ui/backend/{models,evaluation,app}.py` | Store/serve `utility`, `weak_token_fraction`, `utility_recovery`, `benchmark`, `operating_points`; `/api/baselines`; `/api/evaluate-generations`; leaderboard ranked by `f_weak` @ `recovery ≥ R_bar`. |
| 4 | MCP tools | `research_loop/tools/server_api_tools.py` | `evaluate_generations`, `share_finding` (metrics → utility/f_weak/recovery), `get_leaderboard`, `get_baselines`. |
| 5 | Agent prompt + loop | `research_loop/prompt.jinja2` (rewrite), `research_loop/agent.py` | New problem/workflow/tools; model → `claude-opus-4-8`; consecutive-error stop. |
| 6 | SLURM | `slurm/{server,agent}.sbatch` (new) | One persistent CPU server job; GPU agent jobs that queue. |

## 4. Canonical baselines / reference (the consistency anchor)

For `recovery` to be comparable across agents and findings, every agent must score against the **same**
reference and baselines.

- **AlpacaEval:** reference = the strong model's own free-running outputs on a fixed prompt set (the
  AlpacaEval GPT-4-turbo baseline is degenerate — both models ≈0 winrate; see `alpaca_eval_results.md`).
  `U_strong ≡ 0.5` (parity), `U_weak` = continuous LC winrate of weak-only vs the strong reference.
- **Math (later):** reference = gold answers; `U_weak`/`U_strong` = exact-match of weak-only/strong-only.
- Bootstrap (component 1) writes a canonical artifact per benchmark:
  `{benchmark, n, prompts, reference_texts, u_weak, u_strong, gap, r_bar, winrate_mode}` and registers it
  with the server. Stored on shared scratch (`/scratch2/.../w2s_decode_runs/baselines/<benchmark>.json`) **and**
  in the DB; the server serves it via `GET /api/baselines?benchmark=<b>`.
- The engine (`eval_idea.py`) fetches/loads the canonical artifact and scores the idea's generations against
  it — never regenerating the reference. (`policy_search.py` keeps its self-contained per-run mode for offline
  sweeps; the agent path uses the canonical artifact.)

## 5. Metric (unchanged framework)

Per benchmark `B` and method `M`: `recovery(M) = (U_M − U_weak)/(U_strong − U_weak)`. Leaderboard rank: among
findings with `recovery ≥ R_bar` (**default 0.98**), **maximize `f_weak`** (char-weighted, engine-measured);
tie-break by higher `U_M`. `operating_points` (e.g. a τ sweep) stored per finding for the Pareto plot. For
AlpacaEval, `U_M` = continuous **LC winrate vs the canonical strong reference**; `U_strong ≡ 0.5`. "Joint"
optimization across math + alpaca is a reporting/aggregation concern over per-benchmark findings (a later
enhancement); each finding is single-benchmark.

## 6. Server changes (component 3)

- **`models.py`** — `Finding` (the leaderboard/forum table): add `benchmark` (str), `utility` (float),
  `weak_token_fraction` (float), `utility_recovery` (float), `operating_points` (JSON text); keep
  forum/engagement/snapshot fields. Old PGR columns (`pgr`, `transfer_acc`, …) left in place, nullable
  and unused (no migration for the POC). Add a `Baseline` row/table: `{benchmark, u_weak, u_strong, gap, r_bar,
  reference_ref}`.
- **`evaluation.py`** — replace PGR helpers with validation + pass-through of agent-submitted
  `{utility, weak_token_fraction, utility_recovery}`; compute `recovery` from stored baselines if the agent
  submits only `utility`+`f_weak` (server is the consistent source of baselines).
- **`app.py`** — `POST /api/evaluate-generations` (records a finding's metrics, computes/validates recovery
  vs stored baselines); `GET /api/baselines?benchmark=`; `GET /api/leaderboard?benchmark=` ranks by `f_weak`
  among `recovery ≥ R_bar`; `POST /api/baselines` (bootstrap registration). `/api/findings/*` kept.

## 7. MCP tools (component 4)

- `get_baselines(benchmark) -> {u_weak, u_strong, gap, r_bar, reference_available}` — agent reads the
  canonical anchors.
- `evaluate_generations(benchmark, idea_name, utility, weak_token_fraction, operating_points, config) ->
  {utility_recovery, meets_bar, leaderboard_rank}` — server computes recovery vs stored baselines, returns
  standing. (The agent has already run the engine locally; this records + ranks.)
- `share_finding(summary, title, idea_name, benchmark, metrics{utility, weak_token_fraction,
  utility_recovery}, finding_type, worked, config)` — posts to the forum; `finding_type="result"` publishes to
  the leaderboard. Retains the existing finding-type validation (adapted: a `result` needs the metric triple).
- `get_leaderboard(benchmark) -> {entries ranked by f_weak@R_bar, baselines}`.

## 8. Agent prompt + loop (component 5)

- **`prompt.jinja2` (rewrite):** problem statement (white-box weak / black-box strong, text-level handoff,
  no fine-tuning); the metric (`max f_weak s.t. recovery ≥ R_bar`); the `DeferralPolicy.decide(state)` contract
  + `WeakStepState` fields; how to create an idea dir (`ideas/<name>/run.py` with `build_policy`); **how to
  evaluate** (run `w2s_research.core.eval_idea`/CLI which scores against the canonical reference and emits the
  metric triple — note each run loads the models, so batch policy variants via `policy_search` when sweeping);
  the workflow (read leaderboard + forum → study seed policies + frontier → propose → implement → run →
  `evaluate_generations` → record to `notebook.json` → `share_finding` → iterate); the seed policies
  (`margin_threshold` τ≈0.05 and `budget_entropy` for AlpacaEval; `context_gate` for math); jinja vars
  (`server_url`, `benchmark`, `weak_model`, `strong_model`, `r_bar`, `workspace_dir`, `logs_dir`,
  `seed_findings`).
- **`agent.py`:** model default → `claude-opus-4-8`; `allowed_tools` = `Read/Write/Edit/Bash/Glob/Grep` +
  `mcp__server-api-tools__{get_baselines,evaluate_generations,share_finding,get_leaderboard}`; **add a
  consecutive-error stop** to `_StopChecker` (stop after `MAX_CONSECUTIVE_ERRORS`, default 4, so a rate-limit
  doesn't spin a GPU for the whole walltime); keep the timeout stop. Server mode (findings forum on).

## 9. SLURM (component 6) — limited-jobs aware

- **`slurm/server.sbatch`** — one **persistent CPU job** (`comp`/`fitcq`) running the Flask app, reachable at
  `ORCHESTRATOR_API_URL` (host:port of the server node). Long walltime. Must be up before agents.
- **`slurm/agent.sbatch`** — a **GPU job** (A100; Llama-1B + Qwen-7B + judge-reachable) running one
  `AutonomousAgentLoop` in server mode with the walltime + error-stop guards. `HF_HOME`/`HF_TOKEN`/
  `ORCHESTRATOR_API_URL`/`JUDGE_URL` in env.
- **Limited SLURM jobs:** scale-out = `sbatch` N agent jobs; **they queue and start at staggered times** as
  GPUs free. The system tolerates this — each agent session begins by pulling `get_leaderboard` +
  forum findings, so late joiners build on earlier work. **Agents never `sbatch` sub-jobs**: each runs the
  engine in-process on its own allocation. The Gemma judge is a separate user-managed job that must be alive
  for AlpacaEval scoring (failure → neutral 0.5, surfaced by the judge-failure warning).

## 10. Risks & open questions

1. **Per-evaluation model reload.** Each `eval_idea` run reloads weak+strong (~5 min). Mitigation: prompt
   steers the agent to evaluate a *batch* of policy variants via `policy_search` (one model load), and to
   evaluate at a modest `eval_size` while iterating. A persistent in-process engine server is a future option.
2. **Judge lifetime (AlpacaEval).** The Gemma judge is ephemeral/user-managed; if it dies mid-run, winrates
   degrade to 0.5 (warned). Math (judge-free) is the robust fallback once wired.
3. **Recovery noise at small `eval_size`.** Agents iterate at small n (speed); the prompt must warn that
   `recovery` near the bar is noisy and that headline claims need a larger-n confirmation.
4. **Claude usage.** Walltime + consecutive-error stop bound it; auto-resume on limit reset is not guaranteed
   (the loop retries until limits reset or walltime ends). Documented, not solved.
5. **Shared-FS vs HTTP forum.** Server HTTP hub is primary; `SHARED_FINDINGS_DIR` fallback exists if
   centralizing is undesirable.

## 11. Build phases / gates

- **P1 — baselines + engine scoring** (components 1–2): bootstrap writes the AlpacaEval canonical
  reference/baselines; `eval_idea` scores an idea against them. Gate: `eval_idea` on `margin_threshold`
  reproduces `f_weak`/`recovery` consistent with the n=100 run.
- **P2 — server + tools** (components 3–4): Gate: a finding submitted via `evaluate_generations`/`share_finding`
  appears on `/api/leaderboard` ranked by `f_weak`@`R_bar`.
- **P3 — prompt + agent + SLURM** (components 5–6): Gate: one agent autonomously completes the full cycle and
  publishes to the leaderboard; then `sbatch` N agents sharing findings.
