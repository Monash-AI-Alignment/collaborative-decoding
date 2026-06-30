"""CLI: evaluate ONE deferral-policy idea against the canonical reference.

Prints {utility, weak_token_fraction, utility_recovery} scored against the shared
canonical baselines/reference (so the number is comparable across agents).

Guarded with `if __name__ == "__main__"` (loading the HF weak model first initializes
CUDA, forcing vLLM's spawn start method, which re-imports this module).

    python -m w2s_research.core.eval_idea_cli --idea autonomous_myidea --benchmark alpaca_eval --eval-size 60
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description="Evaluate one idea vs the canonical reference")
    ap.add_argument("--idea", required=True, help="idea dir name under w2s_research/ideas/")
    # Only alpaca_eval has a canonical artifact for the POC (bootstrap_baselines builds it).
    ap.add_argument("--benchmark", default="alpaca_eval", choices=["alpaca_eval"])
    ap.add_argument("--eval-size", type=int, default=60)
    ap.add_argument("--winrate-mode", default=None, choices=["plain", "lc"],
                    help="override the canonical artifact's winrate mode (default: use the artifact's)")
    ap.add_argument("--out", default=None, help="optional path to write the full result JSON")
    args = ap.parse_args()

    from w2s_research.core.eval_idea import evaluate_idea
    out = evaluate_idea(args.idea, args.benchmark, args.eval_size,
                        winrate_mode=args.winrate_mode)
    summary = {k: out[k] for k in ("idea", "benchmark", "n", "utility",
                                   "weak_token_fraction", "utility_recovery")}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
