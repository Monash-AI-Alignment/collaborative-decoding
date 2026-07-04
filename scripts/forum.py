#!/usr/bin/env python3
"""Human CLI view of the collab-decoding forum (stdlib-only; run with any python).

    python scripts/forum.py                       # leaderboard + latest findings
    python scripts/forum.py --full                # include finding summaries
    python scripts/forum.py --watch 120           # refresh every 120s
    ORCHESTRATOR_API_URL=http://<node>:8000 python scripts/forum.py

The server URL defaults to $ORCHESTRATOR_API_URL; find the current one with:
    squeue -u $USER -n collab-server ; grep ORCHESTRATOR slurm-collab-server-<jobid>.out
"""
import argparse
import json
import os
import time
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "  —  "


def show(base_url, bench, full, limit):
    lb = get(f"{base_url}/api/leaderboard?benchmark={bench}")
    fs = get(f"{base_url}/api/findings?benchmark={bench}")["findings"]
    b = lb.get("baseline") or {}
    print(f"=== {bench} @ {base_url}  ({time.strftime('%H:%M:%S')}) ===")
    print(f"baseline: U_weak={fmt(b.get('u_weak'))}  gap={fmt(b.get('gap'))} "
          f" r_bar={fmt(b.get('r_bar'), 2)}   findings={len(fs)}")
    print(f"\n-- LEADERBOARD (recovery >= r_bar, by f_weak) --")
    if not lb["entries"]:
        print("  (empty — nothing meets the bar yet)")
    for i, e in enumerate(lb["entries"], 1):
        print(f"  {i}. {e['idea_name']:<28} f_weak={fmt(e['weak_token_fraction'])} "
              f"rec={fmt(e['utility_recovery'])} u={fmt(e['utility'])}")
    print(f"\n-- FINDINGS (newest first, showing {min(limit, len(fs))}) --")
    for f in fs[:limit]:
        age = (time.time() - f["created_at"]) / 3600
        m = (f"  u={fmt(f.get('utility'))} f_weak={fmt(f.get('weak_token_fraction'))} "
             f"rec={fmt(f.get('utility_recovery'))}" if f.get("utility") is not None else "")
        print(f"  [{age:5.1f}h] {f['finding_type']:>10} | {f['idea_name']:<26} | "
              f"{(f['title'] or '(untitled)')[:76]}{m}")
        if full and f.get("summary"):
            for line in f["summary"].splitlines():
                print(f"           | {line}")
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("ORCHESTRATOR_API_URL", "http://m3e116:8000"))
    ap.add_argument("--benchmark", default="alpaca_eval")
    ap.add_argument("--full", action="store_true", help="print full finding summaries")
    ap.add_argument("--limit", type=int, default=20, help="max findings to show")
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="refresh in a loop")
    args = ap.parse_args()
    while True:
        try:
            show(args.url, args.benchmark, args.full, args.limit)
        except Exception as e:
            print(f"[forum] server unreachable at {args.url}: {e!r}")
        if not args.watch:
            break
        time.sleep(args.watch)
        print("\033c", end="")   # clear terminal between refreshes


if __name__ == "__main__":
    main()
