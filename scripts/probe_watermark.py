"""Phase 1 of the watermark-removal probe: GENERATE + save (no detector here).

Generic over the watermark method (``--watermark eth_french | inference_kgw | ...``).
Runs strong-only, weak-only, and the SOTA seamgate policy with the method's WATERMARKED
strong model on the method's prompts, and saves each policy's completions + f_weak +
recovery. Detection runs in a SEPARATE process (scripts/detect_watermark.py) so the vLLM
engine's GPU memory is fully freed before the detector's large allocation.

SIDE-REPORTING — the p-value never feeds the policy. Guarded with __main__ (loading the
HF weak model first initializes CUDA -> vLLM spawn re-imports this module).

Run (GPU node, judge up):
  python scripts/probe_watermark.py --watermark inference_kgw --n 100 --tau 0.16 --out gens.json
  python scripts/probe_watermark.py --watermark eth_french --fingerprinted-model <ckpt> \
      --embedding-config <yaml> --n 100 --out gens.json
"""
import argparse
import json
import os
import sys


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watermark", required=True, help="registered method name")
    ap.add_argument("--weak-model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.16, help="fallback CG_TAU if the SOTA has no recorded config")
    ap.add_argument("--span-max", type=int, default=64, help="fallback span cap if the SOTA has no recorded config")
    ap.add_argument("--policies", default="strong_only,weak_only,sota",
                    help="comma list; 'sota' tests the current leaderboard #1 (not hardcoded)")
    ap.add_argument("--sota-from", default="docs/data.json",
                    help="leaderboard source to resolve the current SOTA policy + its config")
    ap.add_argument("--out", default="/scratch2/ml23/smur0075/w2s_decode_runs/wm_gens.json")
    return ap


def main():
    ap = build_arg_parser()
    from w2s_research.core.watermarks import get_watermark_cls
    known, _ = ap.parse_known_args()
    cls = get_watermark_cls(known.watermark)
    cls.add_cli_args(ap)                       # method-specific args (model paths, kgw params)
    args = ap.parse_args()
    method = cls.from_cli_args(args)
    want = [p.strip() for p in args.policies.split(",") if p.strip()]

    os.environ.setdefault("CG_TAU", str(args.tau))
    os.environ.setdefault("SPAN_MAX_TOKENS", str(args.span_max))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/ for policy_search
    import policy_search
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.judge import VLLMJudge

    B = method.benchmark
    prompts = method.get_prompts(args.n)
    print(f"[gen] watermark={args.watermark} {len(prompts)} prompts benchmark={B}", flush=True)

    base = DecodeConfig(benchmark=B, eval_size=len(prompts))
    # Serve "logits" so cache-contract policies (e.g. the SOTA factgate/contentgate,
    # which derive margin/top-token from state.activations["logits"]) can run. The hf
    # backend can only serve "logits"; internal-hook (TL) SOTAs would need the tl backend.
    weak = HFWeakModel(args.weak_model, max_model_len=base.weak_max_model_len,
                       capture_hooks=["logits"])
    strong = method.make_strong(gpu_memory_utilization=base.strong_gpu_memory_utilization,
                                max_model_len=base.strong_max_model_len)
    judge = VLLMJudge()

    # strong-only reference is always generated: judge reference for weak/seam AND the
    # watermark-present anchor.
    print("[gen] strong-only reference...", flush=True)
    strong_ref = policy_search.run_one(weak, strong, prompts, prompts, B,
                                       {"idea": "strong_only", "params": {}, "span_max": 1024,
                                        "span_stop": None})
    ref = strong_ref["_generations"]

    rows = []
    if "strong_only" in want:
        rows.append({"policy": "strong_only", "f_weak": strong_ref["weak_token_fraction"],
                     "utility": 0.5, "recovery": 1.0, "completions": ref})

    u_weak, u_strong = None, 0.5
    if "weak_only" in want:
        print("[gen] weak-only...", flush=True)
        weak_only = policy_search.run_one(weak, strong, prompts, ref, B,
                                          {"idea": "weak_only", "params": {}, "span_max": 256},
                                          judge=judge)
        u_weak = weak_only["utility"]
    gap = (u_strong - u_weak) if u_weak is not None else None
    rec = lambda u: (u - u_weak) / gap if (u is not None and u_weak is not None and gap) else None

    if "weak_only" in want:
        rows.append({"policy": "weak_only", "f_weak": weak_only["weak_token_fraction"],
                     "utility": u_weak, "recovery": rec(u_weak),
                     "completions": weak_only["_generations"]})
    sota_info = None
    if ("sota" in want) or ("seamgate" in want):   # 'seamgate' kept as a back-compat alias
        from sota_policy import resolve_sota
        sota = resolve_sota(args.sota_from)
        idea = sota["idea_name"]
        span_max = sota["span_max"] or args.span_max
        for k, v in sota["env"].items():
            os.environ[k] = str(v)                  # apply the SOTA's RECORDED config (not hardcoded)
        if not sota["config_recorded"]:
            print(f"[gen] WARNING: SOTA {idea} has no recorded config; falling back to "
                  f"--tau={args.tau} --span-max={args.span_max}", flush=True)
            os.environ["CG_TAU"] = str(args.tau)
            span_max = args.span_max
        print(f"[gen] SOTA policy = {idea} (leaderboard f_weak={sota['f_weak']}, "
              f"recovery={sota['recovery']}); env={sota['env']} span_max={span_max}", flush=True)
        seam = policy_search.run_one(weak, strong, prompts, ref, B,
                                     {"idea": idea, "params": {}, "span_max": span_max}, judge=judge)
        rows.append({"policy": idea, "is_sota": True, "f_weak": seam["weak_token_fraction"],
                     "utility": seam["utility"], "recovery": rec(seam["utility"]),
                     "completions": seam["_generations"]})
        sota_info = {"idea_name": idea, "env": sota["env"], "span_max": span_max,
                     "leaderboard_f_weak": sota["f_weak"], "config_recorded": sota["config_recorded"]}

    summary = {"watermark": args.watermark, "n_prompts": len(prompts), "benchmark": B,
               "tau": args.tau, "span_max": args.span_max, "sota_policy": sota_info,
               "u_weak": u_weak, "u_strong": u_strong, "gap": gap,
               "prompts": prompts, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[gen] wrote {args.out}")
    print(f"{'policy':<12} {'f_weak':>7} {'recovery':>9}")
    for r in rows:
        rr = "n/a" if r["recovery"] is None else f"{r['recovery']:.3f}"
        print(f"{r['policy']:<12} {r['f_weak']:>7.3f} {rr:>9}")


if __name__ == "__main__":
    main()
