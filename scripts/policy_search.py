"""Autonomous policy-search 'agent' for collaborative decoding.

Loads the weak (HF) + strong (vLLM) models ONCE, measures the free-running
baselines U_weak / U_strong, then explores a space of deferral policies trying to
maximize the weak-token-fraction (f_weak) subject to utility_recovery >= R_bar.

It runs a curated, promising-first sweep first, then -- for whatever wall-clock
remains -- a random hill-climb that perturbs the best operating points found so
far. Every configuration is checkpointed immediately, so the run is valuable even
if interrupted. No `claude` CLI is involved, so it cannot consume Claude usage.

Guarded with `if __name__ == '__main__'`: loading the HF weak model first
initializes CUDA, which forces vLLM to use the `spawn` start method (it re-imports
this module in the engine-core child), so the executable body MUST stay under the
guard.

Usage (on a GPU node, with HF_HOME + HF_TOKEN set):
    python scripts/policy_search.py --benchmark gsm8k --eval-size 50 \
        --max-seconds 25200 --out /scratch2/ml23/smur0075/w2s_decode_runs/search_<job>
"""
import argparse
import importlib
import json
import os
import random
import time


# ----------------------------- scoring one config -----------------------------

def run_one(weak, strong, instructions, golds, benchmark, spec, judge=None, winrate_mode="lc"):
    """Run one deferral-policy configuration; return a metrics dict (+ generations).

    Math benchmarks score with CPU exact-match. alpaca_eval scores via the judge:
    winrate of the generations vs the reference outputs (golds), reporting both
    plain and length-controlled winrate; `utility` is the chosen one (default LC).
    """
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction

    cfg = DecodeConfig(benchmark=benchmark, eval_size=len(instructions))
    cfg.span_stop = spec.get("span_stop", ["\n"])
    cfg.span_max_tokens = spec.get("span_max", 256)
    for k, v in spec.get("params", {}).items():
        setattr(cfg, k, v)

    mod = importlib.import_module(f"w2s_research.ideas.{spec['idea']}.run")
    policy = mod.build_policy(cfg)
    dec = CollaborativeDecoder(weak, strong, policy, cfg)
    results = dec.run_dataset(instructions)
    gens = [r.text for r in results]

    out = {
        "weak_token_fraction": aggregate_weak_fraction(results),
        "avg_defers": sum(r.num_defers for r in results) / len(results),
        "avg_weak_steps": sum(r.num_weak_steps for r in results) / len(results),
        "finished_frac": sum(1 for r in results if r.finished) / len(results),
        "_generations": gens,
    }
    if benchmark == "alpaca_eval":
        if judge is None:                       # generation-only (e.g. producing the reference)
            out["utility"] = None
        else:
            from w2s_research.core.alpaca_eval import score_generations
            from w2s_research.core.winrate import plain_winrate, lc_winrate
            scored = score_generations(judge, instructions, gens, golds)   # golds = reference texts
            out["winrate_plain"] = plain_winrate(scored["per_example"])
            out["winrate_lc"] = lc_winrate(scored["per_example"])
            out["utility"] = out["winrate_lc"] if winrate_mode == "lc" else out["winrate_plain"]
            out["_judge_per_example"] = scored["per_example"]
    else:
        from w2s_research.core.benchmarks import utility
        out["utility"] = utility(benchmark, gens, golds)
    return out


# ----------------------------- curated search space -----------------------------

def curated_specs():
    """Promising-first list of (idea, params, span_max) configurations.

    Hypothesis driving the ordering: defer *early but briefly* (low entropy
    threshold + small strong span) and *precisely* (context / margin gating) to
    keep the weak model carrying most characters while protecting the few
    utility-critical tokens. Phase-1's naive entropy never reached recovery>=0.98.
    """
    specs = []

    # 1. brief-span entropy: catch uncertainty early, hand back fast.
    for tau in (0.3, 0.5, 0.7):
        for sm in (32, 64):
            specs.append({"idea": "entropy_threshold",
                          "params": {"defer_threshold": tau}, "span_max": sm})
    # 2. computation-context gating: weak writes prose, strong does results.
    for tau in (0.2, 0.3, 0.5):
        specs.append({"idea": "context_gate",
                      "params": {"defer_threshold": tau}, "span_max": 64})
    # 3. precision AND-gate (high entropy AND low margin).
    for te, tm in ((0.3, 0.10), (0.5, 0.10), (0.5, 0.20), (0.7, 0.15)):
        specs.append({"idea": "and_gate",
                      "params": {"defer_threshold": te, "margin_threshold": tm},
                      "span_max": 64})
    # 4. cooldown between defers.
    for tau in (0.3, 0.5):
        for m in (4, 8):
            specs.append({"idea": "entropy_cooldown",
                          "params": {"defer_threshold": tau, "cooldown_m": m},
                          "span_max": 64})
    # 5. hysteresis streak.
    for tau in (0.3, 0.5):
        for k in (2, 3):
            specs.append({"idea": "entropy_streak",
                          "params": {"defer_threshold": tau, "streak_k": k},
                          "span_max": 64})
    # 6. capped defer budget.
    for b in (3, 5, 10):
        specs.append({"idea": "budget_entropy",
                      "params": {"defer_threshold": 0.3, "defer_budget": b},
                      "span_max": 64})
    # 7. answer-line protection.
    for tau in (0.7, 1.0, 2.0):
        specs.append({"idea": "answer_protect",
                      "params": {"defer_threshold": tau}, "span_max": 256})
    # 8. margin threshold.
    for tm in (0.05, 0.10, 0.20):
        specs.append({"idea": "margin_threshold",
                      "params": {"margin_threshold": tm}, "span_max": 64})
    # 9. OR-gate (higher recall).
    for te, tm in ((0.7, 0.10), (1.0, 0.05)):
        specs.append({"idea": "or_gate",
                      "params": {"defer_threshold": te, "margin_threshold": tm},
                      "span_max": 64})
    # 10. Phase-1-style entropy at larger spans (continuity / sanity).
    for tau in (0.3, 0.6, 1.0):
        specs.append({"idea": "entropy_threshold",
                      "params": {"defer_threshold": tau}, "span_max": 256})
    return specs


# numeric param ranges for the random hill-climb phase
_PARAM_BOUNDS = {
    "defer_threshold": (0.05, 2.5),
    "margin_threshold": (0.01, 0.5),
    "entropy_hi": (0.5, 3.5),
}
_INT_PARAM_BOUNDS = {
    "streak_k": (2, 6),
    "cooldown_m": (1, 16),
    "defer_budget": (1, 20),
}
_SPAN_CHOICES = (16, 32, 48, 64, 96, 128, 256)


def perturb(spec, rng):
    """Return a randomly perturbed neighbor of `spec`."""
    new = {"idea": spec["idea"],
           "params": dict(spec.get("params", {})),
           "span_stop": spec.get("span_stop", ["\n"]),
           "span_max": rng.choice(_SPAN_CHOICES)}
    for k, v in list(new["params"].items()):
        if k in _PARAM_BOUNDS:
            lo, hi = _PARAM_BOUNDS[k]
            factor = rng.uniform(0.6, 1.6)
            new["params"][k] = round(min(hi, max(lo, v * factor)), 3)
        elif k in _INT_PARAM_BOUNDS:
            lo, hi = _INT_PARAM_BOUNDS[k]
            new["params"][k] = min(hi, max(lo, v + rng.choice((-2, -1, 1, 2))))
    return new


# ----------------------------- bookkeeping -----------------------------

def recovery_of(u, u_weak, gap):
    return (u - u_weak) / gap if gap > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Collaborative-decoding policy search")
    ap.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math", "alpaca_eval"])
    ap.add_argument("--winrate-mode", default="lc", choices=["plain", "lc"],
                    help="alpaca_eval utility: length-controlled (lc, default) or plain winrate")
    ap.add_argument("--alpaca-reference", default="strong", choices=["strong", "baseline"],
                    help="alpaca_eval recovery reference: 'strong' = the strong model's own "
                         "outputs (recovery=1.0 is parity with strong; default), 'baseline' = "
                         "the dataset GPT-4-turbo reference answers")
    ap.add_argument("--eval-size", type=int, default=50)
    ap.add_argument("--max-seconds", type=int, default=25200)   # ~7h
    ap.add_argument("--r-bar", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-refine", action="store_true",
                    help="stop after the curated sweep instead of hill-climbing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.strong_model import VLLMStrongModel
    from w2s_research.core.benchmarks import load_benchmark, build_instruction

    bench, n = args.benchmark, args.eval_size
    out_dir = args.out or f"/scratch2/ml23/smur0075/w2s_decode_runs/search_{bench}_{n}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meeting_bar"), exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    rng = random.Random(args.seed)
    t_start = time.time()

    def elapsed():
        return time.time() - t_start

    exs = load_benchmark(bench, "test", limit=n)
    instrs = [build_instruction(bench, e.question) for e in exs]
    golds = [e.answer for e in exs]
    print(f"[search] benchmark={bench} n={len(exs)} r_bar={args.r_bar} "
          f"budget={args.max_seconds}s out={out_dir}", flush=True)

    base = DecodeConfig(benchmark=bench, eval_size=n)
    print("[search] loading weak (HF)...", flush=True)
    weak = HFWeakModel(base.weak_model, max_model_len=base.weak_max_model_len)
    print("[search] loading strong (vLLM)...", flush=True)
    strong = VLLMStrongModel(base.strong_model,
                             gpu_memory_utilization=base.strong_gpu_memory_utilization,
                             max_model_len=base.strong_max_model_len)

    judge = None
    if bench == "alpaca_eval":
        from w2s_research.core.judge import VLLMJudge
        judge = VLLMJudge()
        print(f"[search] judge: {judge.model} @ {judge.base_url}  "
              f"winrate_mode={args.winrate_mode}", flush=True)

    # --- reference + baselines ---
    # For alpaca_eval with --alpaca-reference strong (default) the recovery reference is the
    # STRONG model's OWN free-running outputs: recovery=1.0 == parity with the strong model
    # (U_strong := 0.5 by definition; we never judge strong-vs-itself). Otherwise (math, or
    # alpaca 'baseline' mode) the reference is the dataset gold/reference answers.
    print("[search] measuring baselines...", flush=True)
    strong_ref_mode = (bench == "alpaca_eval" and args.alpaca_reference == "strong")
    try:
        if strong_ref_mode:
            print("[search] generating strong-only reference outputs...", flush=True)
            strong_ref = run_one(weak, strong, instrs, golds, bench,
                                 {"idea": "strong_only", "params": {}, "span_max": 1024,
                                  "span_stop": None})              # judge=None: generation only
            ref_texts = strong_ref["_generations"]
            weak_base = run_one(weak, strong, instrs, ref_texts, bench,
                                {"idea": "weak_only", "params": {}, "span_max": 256},
                                judge=judge, winrate_mode=args.winrate_mode)
            strong_base = None
            uw, us = weak_base["utility"], 0.5                     # strong vs itself == parity
        else:
            ref_texts = golds
            weak_base = run_one(weak, strong, instrs, ref_texts, bench,
                                {"idea": "weak_only", "params": {}, "span_max": 256},
                                judge=judge, winrate_mode=args.winrate_mode)
            strong_base = run_one(weak, strong, instrs, ref_texts, bench,
                                  {"idea": "strong_only", "params": {}, "span_max": 1024,
                                   "span_stop": None}, judge=judge, winrate_mode=args.winrate_mode)
            uw, us = weak_base["utility"], strong_base["utility"]
    except Exception as e:               # baselines are essential -> fail loud, don't run on garbage
        print(f"[search] FATAL: baseline measurement failed: {e!r}", flush=True)
        raise
    gap = us - uw
    baselines = {"benchmark": bench, "n": len(exs), "u_weak": uw, "u_strong": us,
                 "gap": gap, "r_bar": args.r_bar,
                 "reference": ("strong_model_outputs" if strong_ref_mode else "dataset_gold")}
    if bench == "alpaca_eval":
        baselines["winrate_mode"] = args.winrate_mode
    # keep both winrate flavors + raw judge records for the baselines too (reproducibility)
    for label, b in (("weak_only", weak_base), ("strong_only", strong_base)):
        if b is None:
            continue
        for k in ("winrate_plain", "winrate_lc"):
            if k in b:
                baselines[f"{label}_{k}"] = b[k]
        if "_judge_per_example" in b:
            baselines.setdefault("judge_per_example", {})[label] = b["_judge_per_example"]
    with open(os.path.join(out_dir, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"[search] U_weak={uw:.3f} U_strong={us:.3f} gap={gap:.3f}", flush=True)
    if judge is not None and judge.n_calls and judge.n_failures / judge.n_calls > 0.1:
        print(f"[search] WARNING: judge failed {judge.n_failures}/{judge.n_calls} calls (>10%) "
              f"during baselines; winrates are pulled toward 0.5 and recovery may be inflated. "
              f"Check the judge server before trusting results.", flush=True)
    if gap <= 0:
        print("[search] FATAL: non-positive gap (U_weak >= U_strong) -> recovery undefined. "
              "Aborting before the search burns the budget (baselines.json is saved).", flush=True)
        raise SystemExit(2)

    all_rows = []
    best = {"weak_token_fraction": -1.0}     # best meeting the bar
    rf = open(results_path, "a")

    def record(spec, metrics, phase):
        rec = recovery_of(metrics["utility"], uw, gap)
        row = {"phase": phase, "idea": spec["idea"], "params": spec.get("params", {}),
               "span_max": spec.get("span_max"), "span_stop": spec.get("span_stop", ["\n"]),
               "utility": round(metrics["utility"], 4),
               "weak_token_fraction": round(metrics["weak_token_fraction"], 4),
               "utility_recovery": round(rec, 4) if rec == rec else None,
               "avg_defers": round(metrics["avg_defers"], 2),
               "avg_weak_steps": round(metrics["avg_weak_steps"], 1),
               "finished_frac": round(metrics["finished_frac"], 3),
               "elapsed_s": round(elapsed(), 1)}
        for k in ("winrate_plain", "winrate_lc"):     # alpaca_eval: keep both flavors
            if k in metrics:
                row[k] = round(metrics[k], 4)
        all_rows.append(row)
        rf.write(json.dumps(row) + "\n")
        rf.flush()
        meets = rec == rec and rec >= args.r_bar
        flag = " *** MEETS BAR" if meets else ""
        print(f"[{row['idea']:>16} {str(row['params'])[:46]:<46} sm={row['span_max']}] "
              f"U={row['utility']:.3f} fw={row['weak_token_fraction']:.3f} "
              f"rec={row['utility_recovery']}{flag} ({elapsed():.0f}s)", flush=True)

        if meets:
            with open(os.path.join(out_dir, "meeting_bar",
                                   f"{row['idea']}_{len(all_rows)}.json"), "w") as g:
                json.dump({"row": row, "generations": metrics["_generations"],
                           "judge_per_example": metrics.get("_judge_per_example")}, g, indent=2)
            if row["weak_token_fraction"] > best["weak_token_fraction"]:
                best.update(row)
                with open(os.path.join(out_dir, "best.json"), "w") as g:
                    json.dump(best, g, indent=2)

        # frontier = best f_weak at each rounded recovery bucket
        frontier = {}
        for r in all_rows:
            if r["utility_recovery"] is None:
                continue
            key = f"{r['utility_recovery']:.2f}"
            if key not in frontier or r["weak_token_fraction"] > frontier[key]["weak_token_fraction"]:
                frontier[key] = r
        with open(os.path.join(out_dir, "frontier.json"), "w") as g:
            json.dump({"baselines": baselines,
                       "frontier": sorted(frontier.values(),
                                          key=lambda r: r["utility_recovery"]),
                       "best_meeting_bar": best if best["weak_token_fraction"] >= 0 else None,
                       "num_configs": len(all_rows)}, g, indent=2)
        return row

    def safe_run(spec, phase):
        try:
            m = run_one(weak, strong, instrs, ref_texts, bench, spec,
                        judge=judge, winrate_mode=args.winrate_mode)
            return record(spec, m, phase)
        except Exception as e:           # never let one config kill the search
            print(f"[search] ERROR on {spec}: {e!r}", flush=True)
            return None

    # --- curated phase ---
    specs = curated_specs()
    if bench == "alpaca_eval":               # math-only policies don't transfer to open-ended
        specs = [s for s in specs if s["idea"] not in ("context_gate", "answer_protect")]
    print(f"[search] curated phase: {len(specs)} configs", flush=True)
    for spec in specs:
        if elapsed() > args.max_seconds:
            print("[search] time budget reached during curated phase.", flush=True)
            break
        safe_run(spec, "curated")

    # --- random hill-climb phase ---
    if not args.no_refine:
        print("[search] refinement phase (hill-climb on best operating points)...", flush=True)
        rounds = 0
        while elapsed() < args.max_seconds:
            rounds += 1
            scored = [r for r in all_rows if r["utility_recovery"] is not None]
            meeting = [r for r in scored if r["utility_recovery"] >= args.r_bar]
            if meeting:                  # push f_weak up while staying above the bar
                seed_row = max(meeting, key=lambda r: r["weak_token_fraction"])
            elif scored:                 # no winner yet -> climb toward the bar
                seed_row = max(scored, key=lambda r: r["utility_recovery"])
            else:
                break
            seed_spec = {"idea": seed_row["idea"], "params": seed_row["params"],
                         "span_max": seed_row["span_max"],
                         "span_stop": seed_row.get("span_stop", ["\n"])}
            safe_run(perturb(seed_spec, rng), "refine")
        print(f"[search] refinement did {rounds} rounds.", flush=True)

    rf.close()
    print("\n=== SEARCH COMPLETE ===", flush=True)
    print(f"configs evaluated: {len(all_rows)}  elapsed: {elapsed():.0f}s", flush=True)
    if best["weak_token_fraction"] >= 0:
        print(f"BEST @ recovery>={args.r_bar}: {best['idea']} {best['params']} "
              f"span_max={best['span_max']} -> f_weak={best['weak_token_fraction']:.3f} "
              f"utility={best['utility']:.3f} recovery={best['utility_recovery']}", flush=True)
    else:
        scored = [r for r in all_rows if r["utility_recovery"] is not None]
        if scored:
            top = max(scored, key=lambda r: r["utility_recovery"])
            print(f"no config reached recovery>={args.r_bar}. "
                  f"closest: {top['idea']} {top['params']} sm={top['span_max']} "
                  f"-> recovery={top['utility_recovery']} f_weak={top['weak_token_fraction']:.3f}",
                  flush=True)
    print(f"results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
