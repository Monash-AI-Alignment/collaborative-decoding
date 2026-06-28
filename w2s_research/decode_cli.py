# w2s_research/decode_cli.py
"""CLI to run a deferral policy over a benchmark and report (utility, f_weak).

Phase 1: local-only. Computes utility locally against gold answers (the server-side
held-out evaluation arrives in Phase 2). Weak/strong adapters are injectable for testing.
"""
import argparse
import importlib
import json
from typing import Optional

from w2s_research.core.benchmarks import build_instruction, load_benchmark, utility
from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction
from w2s_research.core.decode_config import DecodeConfig


def _load_idea(idea: str):
    return importlib.import_module(f"w2s_research.ideas.{idea}.run")


def run_decode(config: DecodeConfig, idea: str, jsonl_path: Optional[str] = None,
               weak=None, strong=None) -> dict:
    mod = _load_idea(idea)
    policy = mod.build_policy(config)

    examples = load_benchmark(config.benchmark, config.split,
                              limit=config.eval_size, jsonl_path=jsonl_path)

    if weak is None:
        from w2s_research.core.weak_model import HFWeakModel
        weak = HFWeakModel(config.weak_model, max_model_len=config.weak_max_model_len)
    if strong is None:
        from w2s_research.core.strong_model import VLLMStrongModel
        strong = VLLMStrongModel(config.strong_model,
                                 gpu_memory_utilization=config.strong_gpu_memory_utilization,
                                 max_model_len=config.strong_max_model_len)

    decoder = CollaborativeDecoder(weak, strong, policy, config)
    instructions = [build_instruction(config.benchmark, ex.question) for ex in examples]
    results = decoder.run_dataset(instructions)

    generations = [r.text for r in results]
    golds = [ex.answer for ex in examples]
    return {
        "idea": mod.IDEA_NAME,
        "benchmark": config.benchmark,
        "utility": utility(config.benchmark, generations, golds),
        "weak_token_fraction": aggregate_weak_fraction(results),
        "n": len(examples),
        "results": [
            {"text": r.text, "weak_chars": r.weak_chars, "strong_chars": r.strong_chars,
             "num_defers": r.num_defers, "finished": r.finished}
            for r in results
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Run a collaborative-decoding policy on a benchmark")
    p.add_argument("--idea", required=True)
    p.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math"])
    p.add_argument("--eval-size", type=int, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--tau", type=float, default=None, help="entropy/margin threshold for the idea")
    p.add_argument("--defer-prob", type=float, default=None, help="random_defer probability")
    p.add_argument("--out", default=None, help="optional path to write results JSON")
    args = p.parse_args()

    cfg = DecodeConfig(benchmark=args.benchmark, eval_size=args.eval_size, split=args.split)
    if args.tau is not None:
        cfg.defer_threshold = args.tau          # consumed by entropy_threshold
        cfg.margin_threshold = args.tau          # consumed by margin_threshold
    if args.defer_prob is not None:
        cfg.defer_prob = args.defer_prob

    out = run_decode(cfg, idea=args.idea)
    summary = {k: out[k] for k in ("idea", "benchmark", "utility", "weak_token_fraction", "n")}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
