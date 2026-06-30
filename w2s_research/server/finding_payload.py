"""Pure (dependency-free) validation + shaping of a share_finding payload.

Lives here (no SDK / no requests) so it is unit-testable on the CPU venv and
reusable by both the MCP tool and the server.
"""
RESULT_METRICS = ("utility", "weak_token_fraction", "utility_recovery")


def build_share_payload(args):
    """Validate + build the /api/findings/share payload.

    Returns (payload, error_or_None). A finding_type='result' must carry the
    metric triple (utility, weak_token_fraction, utility_recovery) since it
    publishes to the leaderboard.
    """
    ft = args.get("finding_type", "result")
    if ft == "result":
        missing = [k for k in RESULT_METRICS if args.get(k) is None]
        if missing:
            return None, f"finding_type='result' requires metrics: {missing} (e.g. utility)"
    payload = {"benchmark": args.get("benchmark"), "idea_name": args.get("idea_name"),
               "summary": args.get("summary", ""), "title": args.get("title", ""),
               "finding_type": ft, "worked": args.get("worked"), "config": args.get("config")}
    for k in RESULT_METRICS + ("operating_points",):
        if args.get(k) is not None:
            payload[k] = args[k]
    return payload, None
