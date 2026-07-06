"""Resolve the CURRENT state-of-the-art policy (top of the leaderboard) and its config.

The watermark probe uses this instead of hardcoding a policy name/params: whatever is #1
on the leaderboard (highest f_weak with recovery >= bar) is what gets tested, so the
site's watermark graph always reflects the live best policy.

The policy config lives in the finding's `config` field, shape:
    {"env": {"CG_TAU": "0.16", "SPAN_MAX_TOKENS": "64", ...}, "span_max": 64}
`env` is the set of environment overrides the policy read at eval time; `span_max` is the
strong-span cap. If a leaderboard entry has no recorded config, `config_recorded` is False
and the caller must decide whether to fall back or fail loudly.
"""
import json
from typing import Any, Dict


def resolve_sota(data_path: str = "docs/data.json") -> Dict[str, Any]:
    d = json.load(open(data_path))
    lb = d.get("leaderboard") or []
    if not lb:
        raise ValueError(f"no leaderboard entries in {data_path} — nothing is certified yet")
    top = lb[0]                       # server ranks by f_weak among recovery >= bar
    cfg = top.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}
    return {
        "idea_name": top["idea_name"],
        "env": {str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
        "span_max": cfg.get("span_max"),
        "f_weak": top.get("weak_token_fraction"),
        "recovery": top.get("utility_recovery"),
        "config_recorded": bool(cfg),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(resolve_sota(sys.argv[1] if len(sys.argv) > 1 else "docs/data.json"), indent=2))
