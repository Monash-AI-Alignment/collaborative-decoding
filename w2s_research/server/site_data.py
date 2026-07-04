"""At-source logging of the forum state to the static site's data.json.

The public site is a static page: it can only read files deployed alongside it.
So the server serializes its state to docs/data.json AS PART OF accepting a
finding/baseline (no separate export step); a dumb git-sync loop pushes the file
when it changes and the Pages host redeploys. The DB stays authoritative — this
file is a derived view, regenerated in full on every mutation (it is small).
"""
import json
import os
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SITE_DATA = os.environ.get("W2S_SITE_DATA",
                                   os.path.join(_REPO_ROOT, "docs", "data.json"))


def build_site_data(store, benchmark="alpaca_eval"):
    lb = store.leaderboard(benchmark)
    findings = store.list_findings(benchmark=benchmark, limit=10000)
    r_bar = lb.get("r_bar", 0.98)

    # SOTA trajectory: certified results in time order, keeping each point that
    # raises the best-so-far f_weak. Powers the site's progress chart.
    sota, best = [], 0.0
    for f in sorted((f for f in findings if f["finding_type"] == "result"),
                    key=lambda f: f["created_at"]):
        fw, rec = f.get("weak_token_fraction"), f.get("utility_recovery")
        if fw is not None and rec is not None and rec >= r_bar and fw > best:
            best = fw
            sota.append({"t": f["created_at"], "f_weak": fw, "idea": f["idea_name"]})

    return {
        "benchmark": benchmark,
        "baseline": lb.get("baseline"),
        "r_bar": r_bar,
        "leaderboard": lb.get("entries", []),
        "findings": findings,
        "sota_history": sota,
        "counts": {
            "findings": len(findings),
            "results": sum(1 for f in findings if f["finding_type"] == "result"),
            "ideas": len({f["idea_name"] for f in findings}),
        },
    }


def write_site_data(store, path=None, benchmark="alpaca_eval"):
    """Atomically (re)write the site data file; returns True if it changed.

    No timestamp field on purpose: identical state == identical bytes, so the
    git-sync loop can detect "nothing new" with a plain diff.
    """
    path = path or DEFAULT_SITE_DATA
    body = json.dumps(build_site_data(store, benchmark),
                      indent=1, sort_keys=True, ensure_ascii=False)
    try:
        with open(path) as fh:
            if fh.read() == body:
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True
