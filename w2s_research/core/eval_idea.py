"""Score ONE deferral-policy idea against the canonical per-benchmark reference.

This is the agent-facing evaluator: it loads the shared canonical artifact (so
recovery is comparable across agents), runs the idea through the engine to get
engine-measured f_weak + generations, and scores utility (judge winrate vs the
strong reference for alpaca_eval; exact-match for math). The server trusts these.
"""
import importlib
import json
import os

_DEFAULT_BASELINES_DIR = os.environ.get(
    "W2S_BASELINES_DIR", "/scratch2/ml23/smur0075/w2s_decode_runs/baselines")


def load_canonical(benchmark, baselines_dir=None):
    d = baselines_dir or _DEFAULT_BASELINES_DIR
    with open(os.path.join(d, f"{benchmark}.json")) as f:
        return json.load(f)


def recovery_of(u, u_weak, gap):
    return (u - u_weak) / gap if gap > 0 else float("nan")


def score_generations(benchmark, generations, canonical, judge=None, winrate_mode="lc"):
    refs = canonical["reference_texts"][:len(generations)]
    uw, gap = canonical["u_weak"], canonical["gap"]
    out = {}
    if benchmark == "alpaca_eval":
        from w2s_research.core.winrate import plain_winrate, lc_winrate
        prompts = canonical["prompts"][:len(generations)]
        scored = judge.winrate(prompts, generations, refs)
        out["winrate_plain"] = plain_winrate(scored["per_example"])
        out["winrate_lc"] = lc_winrate(scored["per_example"])
        out["per_example"] = scored["per_example"]
        out["utility"] = out["winrate_lc"] if winrate_mode == "lc" else out["winrate_plain"]
    else:
        from w2s_research.core.benchmarks import is_correct
        correct = sum(1 for g, gold in zip(generations, refs) if is_correct(benchmark, g, gold))
        out["utility"] = correct / len(generations) if generations else 0.0
    out["utility_recovery"] = recovery_of(out["utility"], uw, gap)
    return out


def evaluate_idea(idea_name, benchmark, eval_size, weak=None, strong=None,
                  judge=None, winrate_mode="lc", baselines_dir=None):
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction

    canonical = load_canonical(benchmark, baselines_dir)
    prompts = canonical["prompts"][:eval_size]

    cfg = DecodeConfig(benchmark=benchmark, eval_size=len(prompts))
    cfg.span_stop = ["\n"]
    mod = importlib.import_module(f"w2s_research.ideas.{idea_name}.run")
    policy = mod.build_policy(cfg)
    if weak is None:
        from w2s_research.core.weak_model import HFWeakModel
        weak = HFWeakModel(cfg.weak_model, max_model_len=cfg.weak_max_model_len)
    if strong is None:
        from w2s_research.core.strong_model import VLLMStrongModel
        strong = VLLMStrongModel(cfg.strong_model,
                                 gpu_memory_utilization=cfg.strong_gpu_memory_utilization,
                                 max_model_len=cfg.strong_max_model_len)
    if judge is None and benchmark == "alpaca_eval":
        from w2s_research.core.judge import VLLMJudge
        judge = VLLMJudge()

    dec = CollaborativeDecoder(weak, strong, policy, cfg)
    results = dec.run_dataset(prompts)
    gens = [r.text for r in results]
    scored = score_generations(benchmark, gens, canonical, judge=judge, winrate_mode=winrate_mode)
    return {"idea": idea_name, "benchmark": benchmark, "n": len(prompts),
            "weak_token_fraction": aggregate_weak_fraction(results),
            "utility": scored["utility"], "utility_recovery": scored["utility_recovery"],
            "operating_points": [], "generations": gens,
            **{k: scored[k] for k in ("winrate_plain", "winrate_lc") if k in scored}}
