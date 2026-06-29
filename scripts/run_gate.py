"""Fast in-process Phase-1 gate: load weak+strong ONCE, run baselines + entropy sweep.

Much faster than invoking decode_cli per policy (which reloads both models each
time). Guarded with `if __name__ == "__main__"`: loading the HF weak model first
initializes CUDA, which forces vLLM to use the `spawn` start method for its
engine-core process — and spawn re-imports this module, so the executable body
MUST live under the guard or the child re-runs model loading and crashes.

Usage (on a GPU node, with HF_HOME + HF_TOKEN set):
    python scripts/run_gate.py --benchmark gsm8k --eval-size 30
"""
import argparse
import importlib
import json
import os
import time


def main():
    ap = argparse.ArgumentParser(description="Phase-1 collaborative-decoding gate")
    ap.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math"])
    ap.add_argument("--eval-size", type=int, default=30)
    ap.add_argument("--taus", default="0.3,0.6,1.0,1.5,2.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.strong_model import VLLMStrongModel
    from w2s_research.core.benchmarks import load_benchmark, build_instruction, utility
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction

    bench, ev = args.benchmark, args.eval_size
    out_dir = args.out or f"/scratch2/ml23/smur0075/w2s_decode_runs/gate_{bench}_{ev}"
    os.makedirs(out_dir, exist_ok=True)

    exs = load_benchmark(bench, "test", limit=ev)
    instrs = [build_instruction(bench, e.question) for e in exs]
    golds = [e.answer for e in exs]

    base = DecodeConfig(benchmark=bench, eval_size=ev)
    print("loading weak (HF)...", flush=True)
    weak = HFWeakModel(base.weak_model, max_model_len=base.weak_max_model_len)
    print("loading strong (vLLM)...", flush=True)
    strong = VLLMStrongModel(base.strong_model,
                             gpu_memory_utilization=base.strong_gpu_memory_utilization,
                             max_model_len=base.strong_max_model_len)

    def run(idea, fname, span_stop=("\n",), span_max=256, tau=None):
        cfg = DecodeConfig(benchmark=bench, eval_size=ev)
        cfg.span_stop = list(span_stop) if span_stop is not None else None
        cfg.span_max_tokens = span_max
        if tau is not None:
            cfg.defer_threshold = tau
        mod = importlib.import_module(f"w2s_research.ideas.{idea}.run")
        dec = CollaborativeDecoder(weak, strong, mod.build_policy(cfg), cfg)
        t0 = time.time()
        results = dec.run_dataset(instrs)
        u = utility(bench, [r.text for r in results], golds)
        fw = aggregate_weak_fraction(results)
        json.dump(
            {"idea": mod.IDEA_NAME, "benchmark": bench, "utility": u,
             "weak_token_fraction": fw, "n": len(exs),
             "results": [{"text": r.text, "weak_chars": r.weak_chars,
                          "strong_chars": r.strong_chars, "num_defers": r.num_defers,
                          "finished": r.finished} for r in results]},
            open(os.path.join(out_dir, fname), "w"), indent=2)
        print(f"[{mod.IDEA_NAME} tau={tau}] utility={u:.4f} f_weak={fw:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return u, fw

    uw, _ = run("weak_only", "weak_only.json")
    us, _ = run("strong_only", "strong_only.json", span_stop=None, span_max=1024)
    rows = []
    for tau in [float(x) for x in args.taus.split(",")]:
        u, fw = run("entropy_threshold", f"entropy_tau_{tau}.json", tau=tau)
        rows.append((tau, u, fw))

    gap = us - uw
    print("\n=== GATE 1a/1b SUMMARY ===", flush=True)
    print(f"U_weak={uw:.3f}  U_strong={us:.3f}  gap={gap:.3f}")
    print(f"{'tau':>5}{'utility':>9}{'f_weak':>9}{'recovery':>10}")
    for tau, u, fw in rows:
        rec = (u - uw) / gap if gap > 0 else float("nan")
        print(f"{tau:>5}{u:>9.3f}{fw:>9.3f}{rec:>10.3f}")
    print(f"\nResults -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
