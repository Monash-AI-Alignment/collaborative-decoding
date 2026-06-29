"""GPU+judge smoke: 3 AlpacaEval prompts through the engine, scored by the Gemma judge.

Run on a GPU node with HF_HOME + HF_TOKEN set and the judge server reachable:
    python scripts/smoke_alpaca_judge.py

Guarded with `if __name__ == "__main__"`: loading the HF weak model first
initializes CUDA, which forces vLLM to use the `spawn` start method (it re-imports
this module in the engine-core child), so the executable body MUST stay under the guard.
"""

def main():
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.strong_model import VLLMStrongModel
    from w2s_research.core.benchmarks import load_benchmark, build_instruction
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction
    from w2s_research.core.judge import VLLMJudge
    from w2s_research.core.alpaca_eval import score_generations
    from w2s_research.core.winrate import plain_winrate, lc_winrate
    from w2s_research.ideas.entropy_threshold.run import build_policy

    exs = load_benchmark("alpaca_eval", "eval", limit=3)
    instrs = [build_instruction("alpaca_eval", e.question) for e in exs]
    refs = [e.answer for e in exs]

    base = DecodeConfig(benchmark="alpaca_eval", eval_size=3)
    weak = HFWeakModel(base.weak_model, max_model_len=base.weak_max_model_len)
    strong = VLLMStrongModel(base.strong_model,
                             gpu_memory_utilization=base.strong_gpu_memory_utilization,
                             max_model_len=base.strong_max_model_len)
    base.defer_threshold = 0.5
    dec = CollaborativeDecoder(weak, strong, build_policy(base), base)
    results = dec.run_dataset(instrs)
    gens = [r.text for r in results]
    print("f_weak =", round(aggregate_weak_fraction(results), 3), flush=True)

    judge = VLLMJudge()
    print("judge:", judge.model, "@", judge.base_url, flush=True)
    scored = score_generations(judge, instrs, gens, refs)
    print("winrate(plain) =", round(plain_winrate(scored["per_example"]), 3))
    print("winrate(LC)    =", round(lc_winrate(scored["per_example"]), 3))
    for i, p in enumerate(scored["per_example"]):
        print(f"  ex{i}: win={p['win']} cand_len={p['cand_len']} ref_len={p['ref_len']} verdicts={p['verdicts']}")


if __name__ == "__main__":
    main()
