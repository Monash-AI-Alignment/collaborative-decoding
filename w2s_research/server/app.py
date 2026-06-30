"""Minimal Flask server for collaborative-decoding findings + leaderboard.

Trusts engine-computed metrics; serves the scalar baselines. The large W2S
web_ui/backend/app.py is unused (dormant). Run:
    python -m w2s_research.server.app   # honors W2S_SERVER_DB, PORT
"""
import os
from flask import Flask, request, jsonify
from w2s_research.server.store import Store


def create_app(db_path=None):
    app = Flask(__name__)
    store = Store(db_path or os.environ.get(
        "W2S_SERVER_DB", "/scratch2/ml23/smur0075/w2s_decode_runs/server.db"))

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

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
        b = store.get_baseline(d["benchmark"])
        if not b:
            return jsonify({"error": "no baseline for benchmark"}), 404
        gap = b["gap"]
        rec = (d["utility"] - b["u_weak"]) / gap if gap > 0 else None
        return jsonify({"utility_recovery": rec, "gap": gap,
                        "meets_bar": rec is not None and rec >= b["r_bar"]})

    @app.post("/api/findings/share")
    def share_finding():
        return jsonify(store.add_finding(request.get_json(force=True)))

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
