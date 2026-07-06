"""Normalize watermark-removal probe result JSONs into docs/watermarks.json for the site.

The probe/detect scripts write one JSON per (watermark, run). This collects them into the
single file the dashboard fetches, tagging each with display metadata. Run after a probe:

  python scripts/build_watermark_site_data.py \
      /scratch2/.../fp_probe_<jid>.json /scratch2/.../wm_probe_inference_kgw_<jid>.json

Method is read from each file's `watermark` field, or inferred as eth_french when a
`fingerprinted_model` is present (the older French probe predates the generic schema).
"""
import argparse
import json
import os

# Display metadata per method (labels/blurbs the dashboard shows).
METHOD_META = {
    "inference_kgw": {
        "label": "Red-green KGW (inference-time)",
        "domain": "English AlpacaEval",
        "strong": "Qwen2.5-7B + green-list logit bias",
        "kind": "inference-time, no fine-tuning",
        "blurb": ("The original Kirchenbauer red-green watermark: at each step a +delta bias is "
                  "added to a green-list (a gamma fraction of the vocab, hashed from the previous "
                  "token). Detected by the green-count z-test. Applied only at generation."),
    },
    "eth_french": {
        "label": "Semantic KGW fingerprint (fine-tuned)",
        "domain": "French",
        "strong": "Qwen2.5-3B fine-tuned",
        "kind": "learned, semantically conditioned",
        "blurb": ("A KGW watermark fine-tuned into the weights so it activates on the French "
                  "domain (ETH-SRI, arXiv 2505.16723). The watermark is learned, not applied at "
                  "inference."),
    },
}
def _policy_label(policy):
    """Anchors keep fixed labels; anything else is whatever policy is currently SOTA."""
    fixed = {"strong_only": "strong-only (anchor)", "weak_only": "weak-only (control)"}
    if policy in fixed:
        return fixed[policy]
    return policy.replace("autonomous_", "") + " (SOTA policy)"


def _method_name(d):
    if d.get("watermark"):
        return d["watermark"]
    if d.get("fingerprinted_model"):
        return "eth_french"
    raise ValueError("cannot determine watermark method from probe JSON")


def build(paths):
    methods = []
    for p in paths:
        d = json.load(open(p))
        name = _method_name(d)
        meta = METHOD_META.get(name, {"label": name, "domain": "?", "strong": "?", "kind": "?", "blurb": ""})
        rows = [{
            "policy": r["policy"],
            "policy_label": _policy_label(r["policy"]),
            "f_weak": r.get("f_weak"),
            "recovery": r.get("recovery"),
            "pvalue": r.get("watermark_pvalue"),
            "detected": r.get("is_fingerprinted"),
            "detect_n": r.get("detect_n_queries"),
        } for r in d["rows"]]
        methods.append({
            "name": name, **meta,
            "n_prompts": d.get("n_prompts"), "alpha": 1e-3,
            "tau": d.get("tau"), "rows": rows,
            "source": os.path.basename(p),
        })
    return {"alpha": 1e-3, "methods": methods}


def _discover_latest(rundir):
    """Newest detect-output probe JSON per watermark method under rundir (for living regen)."""
    import glob
    latest = {}
    for pat in ("wm_probe_*.json", "fp_probe_*.json"):
        for p in glob.glob(os.path.join(rundir, pat)):
            try:
                name = _method_name(json.load(open(p)))
            except Exception:
                continue
            if name not in latest or os.path.getmtime(p) > os.path.getmtime(latest[name]):
                latest[name] = p
    return list(latest.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", nargs="*", help="watermark probe result JSON paths")
    ap.add_argument("--auto", metavar="RUNDIR", help="discover the newest probe JSON per method under RUNDIR")
    ap.add_argument("--out", default="docs/watermarks.json")
    args = ap.parse_args()
    paths = list(args.probes) + (_discover_latest(args.auto) if args.auto else [])
    if not paths:
        ap.error("give probe JSON paths or --auto <rundir>")
    data = build(paths)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.out}: {len(data['methods'])} methods "
          f"({', '.join(m['name'] for m in data['methods'])})")
    for m in data["methods"]:
        print(f"  {m['name']}: " + " | ".join(
            f"{r['policy']} f_weak={r['f_weak']:.2f} p={r['pvalue']:.1e} det={r['detected']}"
            for r in m["rows"]))


if __name__ == "__main__":
    main()
