"""Minimal Flask server for collaborative-decoding findings + leaderboard.

Trusts engine-computed metrics; serves the scalar baselines. The large W2S
web_ui/backend/app.py is unused (dormant). Run:
    python -m w2s_research.server.app   # honors W2S_SERVER_DB, PORT
"""
import os
import time

from flask import Flask, request, jsonify
from markupsafe import escape

from w2s_research.server.store import Store
from w2s_research.server.finding_payload import build_share_payload


def create_app(db_path=None):
    app = Flask(__name__)
    store = Store(db_path or os.environ.get(
        "W2S_SERVER_DB", "/scratch2/ml23/smur0075/w2s_decode_runs/server.db"))

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/")
    def forum_html():
        """Human-readable forum view (agents use the /api/* endpoints)."""
        bench = request.args.get("benchmark", "alpaca_eval")
        lb = store.leaderboard(bench)
        findings = store.list_findings(benchmark=bench)
        base = lb.get("baseline") or {}

        def fmt(v, nd=3):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

        rows = "".join(
            f"<tr><td>{i+1}</td><td>{escape(e['idea_name'])}</td>"
            f"<td>{fmt(e['weak_token_fraction'])}</td><td>{fmt(e['utility_recovery'])}</td>"
            f"<td>{fmt(e['utility'])}</td></tr>"
            for i, e in enumerate(lb["entries"]))
        cards = "".join(
            f"<div class='f'><div class='meta'>{(time.time()-f['created_at'])/3600:.1f}h ago"
            f" · <b>{escape(f['finding_type'])}</b> · {escape(f['idea_name'])}"
            + (f" · u={fmt(f.get('utility'))} f_weak={fmt(f.get('weak_token_fraction'))}"
               f" rec={fmt(f.get('utility_recovery'))}" if f.get("utility") is not None else "")
            + f"</div><div class='t'>{escape(f['title'] or '(untitled)')}</div>"
            f"<details><summary>summary</summary><pre>{escape(f['summary'] or '')}</pre></details></div>"
            for f in findings)
        return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='60'><title>collab-decoding forum</title><style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
.f{{border:1px solid #ddd;border-radius:6px;padding:8px 12px;margin:8px 0}}
.meta{{color:#666;font-size:.85em}}.t{{font-weight:600;margin:4px 0}}
pre{{white-space:pre-wrap;font-size:.85em}}</style></head><body>
<h1>Collaborative-decoding forum — {escape(bench)}</h1>
<p>baseline: U_weak={fmt(base.get('u_weak'))} · gap={fmt(base.get('gap'))} ·
r_bar={fmt(base.get('r_bar'), 2)} · {len(findings)} findings · auto-refreshes every 60s</p>
<h2>Leaderboard (recovery ≥ r_bar, ranked by f_weak)</h2>
<table><tr><th>#</th><th>idea</th><th>f_weak</th><th>recovery</th><th>utility</th></tr>
{rows or "<tr><td colspan='5'>empty — nothing meets the bar yet</td></tr>"}</table>
<h2>Findings (newest first)</h2>{cards or "<p>none yet</p>"}</body></html>"""

    @app.post("/api/baselines")
    def post_baseline():
        d = request.get_json(force=True)
        store.set_baseline(d["benchmark"], d["u_weak"], d["u_strong"], d["gap"],
                           d.get("r_bar", 0.98), d.get("reference_path", ""))
        return jsonify({"ok": True})

    @app.get("/api/baselines")
    def get_baseline():
        b = store.get_baseline(request.args.get("benchmark", ""))
        return (jsonify(b), 200) if b else (jsonify({"error": "no baseline"}), 404)

    @app.post("/api/evaluate-generations")
    def evaluate_generations():
        d = request.get_json(force=True)
        if not isinstance(d, dict) or not d.get("benchmark") or d.get("utility") is None:
            return jsonify({"error": "benchmark and utility are required"}), 400
        b = store.get_baseline(d["benchmark"])
        if not b:
            return jsonify({"error": "no baseline for benchmark"}), 404
        gap = b["gap"]
        rec = (d["utility"] - b["u_weak"]) / gap if (gap and gap > 0) else None
        return jsonify({"utility_recovery": rec, "gap": gap,
                        "meets_bar": rec is not None and rec >= b["r_bar"]})

    @app.post("/api/findings/share")
    def share_finding():
        d = request.get_json(force=True)
        if not isinstance(d, dict):
            return jsonify({"error": "expected a JSON object"}), 400
        payload, err = build_share_payload(d)
        if err:
            return jsonify({"error": err}), 400
        # Trust engine-measured utility/f_weak, but compute recovery SERVER-SIDE from the
        # canonical baseline so the leaderboard gate can't be set by the submitted value.
        if payload.get("finding_type", "result") == "result":
            b = store.get_baseline(payload.get("benchmark"))
            if not b:
                return jsonify({"error": "no baseline for benchmark; cannot publish a result"}), 400
            gap, u = b["gap"], payload.get("utility")
            payload["utility_recovery"] = (
                (u - b["u_weak"]) / gap if (gap and gap > 0 and u is not None) else None)
        return jsonify(store.add_finding(payload))

    @app.get("/api/findings")
    def get_findings():
        return jsonify({"findings": store.list_findings(
            benchmark=request.args.get("benchmark"),
            finding_type=request.args.get("finding_type"))})

    @app.get("/api/leaderboard")
    def leaderboard():
        return jsonify(store.leaderboard(request.args.get("benchmark", "alpaca_eval")))

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
