"""One-time: build the canonical per-benchmark reference + baselines on a GPU node.

alpaca_eval: reference = the strong model's OWN free-running outputs; U_strong:=0.5;
U_weak = continuous LC winrate of weak-only vs that reference. Writes the canonical
artifact (prompts + reference_texts + baselines) all agents score against.

Guarded with `if __name__ == "__main__"` (loading the HF weak model first initializes
CUDA, forcing vLLM's spawn start method, which re-imports this module).

Run (GPU node, HF_HOME+HF_TOKEN, judge reachable):
    python scripts/bootstrap_baselines.py --benchmark alpaca_eval --eval-size 100
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="alpaca_eval", choices=["alpaca_eval"])
    ap.add_argument("--eval-size", type=int, default=100)
    ap.add_argument("--r-bar", type=float, default=0.98)
    ap.add_argument("--winrate-mode", default="lc", choices=["plain", "lc"])
    ap.add_argument("--out-dir", default=os.environ.get(
        "W2S_BASELINES_DIR", "/scratch2/ml23/smur0075/w2s_decode_runs/baselines"))
    ap.add_argument("--server-url", default=os.environ.get("ORCHESTRATOR_API_URL"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/ for policy_search
    import policy_search
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.strong_model import VLLMStrongModel
    from w2s_research.core.benchmarks import load_benchmark, build_instruction
    from w2s_research.core.judge import VLLMJudge

    bench, n = args.benchmark, args.eval_size
    exs = load_benchmark(bench, "eval", limit=n)
    prompts = [build_instruction(bench, e.question) for e in exs]
    base = DecodeConfig(benchmark=bench, eval_size=n)
    print(f"[bootstrap] {bench} n={len(prompts)}: loading weak...", flush=True)
    weak = HFWeakModel(base.weak_model, max_model_len=base.weak_max_model_len)
    print("[bootstrap] loading strong...", flush=True)
    strong = VLLMStrongModel(base.strong_model,
                             gpu_memory_utilization=base.strong_gpu_memory_utilization,
                             max_model_len=base.strong_max_model_len)
    judge = VLLMJudge()

    print("[bootstrap] generating strong reference outputs...", flush=True)
    strong_ref = policy_search.run_one(weak, strong, prompts, prompts, bench,
                                       {"idea": "strong_only", "params": {}, "span_max": 1024,
                                        "span_stop": None})
    ref_texts = strong_ref["_generations"]
    print("[bootstrap] measuring U_weak (weak-only vs strong reference)...", flush=True)
    weak_base = policy_search.run_one(weak, strong, prompts, ref_texts, bench,
                                      {"idea": "weak_only", "params": {}, "span_max": 256},
                                      judge=judge, winrate_mode=args.winrate_mode)
    uw, us = weak_base["utility"], 0.5
    artifact = {"benchmark": bench, "n": len(prompts), "winrate_mode": args.winrate_mode,
                "r_bar": args.r_bar, "u_weak": uw, "u_strong": us, "gap": us - uw,
                "prompts": prompts, "reference_texts": ref_texts}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{bench}.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[bootstrap] wrote {path}  U_weak={uw:.3f} U_strong={us} gap={us-uw:.3f}", flush=True)

    if args.server_url:
        import urllib.request
        body = json.dumps({"benchmark": bench, "u_weak": uw, "u_strong": us,
                           "gap": us - uw, "r_bar": args.r_bar, "reference_path": path}).encode()
        try:
            req = urllib.request.Request(f"{args.server_url}/api/baselines", data=body,
                                         headers={"content-type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
            print(f"[bootstrap] registered baselines with {args.server_url}", flush=True)
        except Exception as e:
            print(f"[bootstrap] WARN: server registration failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
