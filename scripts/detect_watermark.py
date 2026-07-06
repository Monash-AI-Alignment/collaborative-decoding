"""Phase 2 of the watermark-removal probe: DETECT on saved completions -> p-values.

Generic over the watermark method (``--watermark`` + the same method args as phase 1).
Runs in a fresh process so the detector gets the full GPU (co-locating it with the vLLM
engine OOMs an 80GB A100). On CUDA OOM the detector subsamples fewer completions and
retries, down to ``--min-frac`` of them.

  python scripts/detect_watermark.py --watermark inference_kgw --gens gens.json --out probe.json
"""
import argparse
import json


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watermark", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frac", type=float, default=0.2,
                    help="smallest fraction of completions to try before giving up on OOM")
    return ap


def score_with_oom_retry(detector, completions, *, min_frac=0.2):
    """Score, halving n_queries on CUDA OOM until it fits (or hits the floor)."""
    import torch
    n = len([c for c in completions if c])
    if n == 0:
        return {"pvalue": 1.0, "is_fingerprinted": False, "n_queries": 0}
    nq = n
    floor = max(1, int(n * min_frac))
    while True:
        try:
            return detector.score(completions, n_queries=nq)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if nq <= floor:
                raise
            nq = max(floor, nq // 2)
            print(f"[detect]   OOM — retrying with n_queries={nq} (of {n})", flush=True)


def main():
    ap = build_arg_parser()
    from w2s_research.core.watermarks import get_watermark_cls
    known, _ = ap.parse_known_args()
    cls = get_watermark_cls(known.watermark)
    cls.add_cli_args(ap)
    args = ap.parse_args()
    method = cls.from_cli_args(args)

    data = json.load(open(args.gens))
    print(f"[detect] watermark={args.watermark} building detector...", flush=True)
    detector = method.build_detector()

    out_rows = []
    for r in data["rows"]:
        res = score_with_oom_retry(detector, r.get("completions", []), min_frac=args.min_frac)
        row = {k: r[k] for k in ("policy", "f_weak", "utility", "recovery") if k in r}
        row.update(watermark_pvalue=res["pvalue"], is_fingerprinted=res["is_fingerprinted"],
                   detect_n_queries=res["n_queries"])
        out_rows.append(row)
        print(f"[detect] {row['policy']:<12} f_weak={row.get('f_weak', float('nan')):.3f} "
              f"recovery={row.get('recovery')} wm_pvalue={res['pvalue']:.2e} "
              f"fingerprinted={res['is_fingerprinted']}", flush=True)

    out = {k: data.get(k) for k in ("watermark", "n_prompts", "benchmark", "tau", "span_max",
                                    "u_weak", "u_strong", "gap")}
    out["rows"] = out_rows
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== WATERMARK-REMOVAL PROBE ({args.watermark}) ===")
    print(f"{'policy':<12} {'f_weak':>7} {'recovery':>9} {'wm_pvalue':>11}  detected")
    for r in out_rows:
        rr = "n/a" if r.get("recovery") is None else f"{r['recovery']:.3f}"
        print(f"{r['policy']:<12} {r.get('f_weak', float('nan')):>7.3f} {rr:>9} "
              f"{r['watermark_pvalue']:>11.2e}  {r['is_fingerprinted']}")


if __name__ == "__main__":
    main()
