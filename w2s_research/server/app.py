"""Minimal Flask server for collaborative-decoding findings + leaderboard.

Trusts engine-computed metrics; serves the scalar baselines. The large W2S
web_ui/backend/app.py is unused (dormant). Run:
    python -m w2s_research.server.app   # honors W2S_SERVER_DB, PORT
"""
import os
import subprocess
import threading
import time

from flask import Flask, request, jsonify
from markupsafe import escape

from w2s_research.server.site_data import write_site_data
from w2s_research.server.store import Store
from w2s_research.server.finding_payload import build_share_payload


def create_app(db_path=None, site_data_path=None):
    app = Flask(__name__)
    store = Store(db_path or os.environ.get(
        "W2S_SERVER_DB", "/scratch2/ml23/smur0075/w2s_decode_runs/server.db"))
    benchmark_default = os.environ.get("BENCHMARK", "alpaca_eval")
    # Only write the public site file when explicitly configured (arg or W2S_SITE_DATA env).
    # Defaulting to the repo's docs/data.json here is a footgun: a test/caller that builds an
    # app on a throwaway DB would clobber the production site data. Opt in, don't default.
    site_data_path = site_data_path if site_data_path is not None else os.environ.get("W2S_SITE_DATA")

    def sync_site(benchmark=None):
        if not site_data_path:
            return                                                # no site output configured
        try:                                                      # never break the research API
            write_site_data(store, site_data_path, benchmark or benchmark_default)
        except Exception as e:                                    # noqa: BLE001
            app.logger.warning("site_data write failed: %r", e)

    sync_site()   # backfill on startup so the file always reflects the DB

    # --- watermark trigger ---------------------------------------------------
    # When a submitted result becomes the new leaderboard #1, launch the watermark
    # probe(s) HERE — the server holds the champion's recorded config at submission
    # time, so the graph point is reproduced exactly (no polling / config-guessing).
    # The server does the sbatch, not the agent (the agents-never-sbatch rule holds).
    _WM_REPO = os.environ.get("W2S_REPO", "/fs04/ax74/smur0075/automated-w2s-research")
    _WM_METHODS = ("inference_kgw", "eth_french")
    _WM_STATE = os.environ.get(
        "W2S_WM_STATE", "/scratch2/ml23/smur0075/w2s_decode_runs/.wm_champion")
    _wm_lock = threading.Lock()

    def _wm_probe_busy():
        try:
            r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""),
                                "-n", "wm-probe", "-t", "R,PD", "-h"],
                               capture_output=True, text=True, timeout=15)
            return bool(r.stdout.strip())
        except Exception:                                             # noqa: BLE001
            return False

    def trigger_watermark_if_new_champion(new_post_id, benchmark):
        if not _wm_lock.acquire(blocking=False):
            return                                                    # a trigger is already running
        try:
            entries = store.leaderboard(benchmark).get("entries") or []
            if not entries or entries[0].get("post_id") != new_post_id:
                return                                                # this result is not the new #1
            champ = entries[0]
            idea = champ.get("idea_name") or ""
            hf = os.environ.get("HF_TOKEN")
            if not hf:
                app.logger.warning("watermark: no HF_TOKEN in server env — cannot probe %s", idea)
                return
            try:
                last = open(_WM_STATE).read().strip()
            except Exception:                                         # noqa: BLE001
                last = ""
            if idea == last or _wm_probe_busy():
                return                                                # already handled / run in flight
            cfg = champ.get("config") if isinstance(champ.get("config"), dict) else {}
            env = (cfg or {}).get("env") or {}
            env_str = "".join(f",{k}={v}" for k, v in env.items())
            if not env:
                app.logger.warning("watermark: champion %s has NO recorded config — probe uses "
                                   "fallback config (graph point may not match its operating point)", idea)
            ok = True
            for m in _WM_METHODS:
                cmd = ["sbatch", f"--export=ALL,HF_TOKEN={hf},WATERMARK={m}{env_str}",
                       os.path.join(_WM_REPO, "slurm", "probe_watermark.sbatch")]
                try:
                    subprocess.run(cmd, timeout=30, check=True, capture_output=True, text=True)
                except Exception as e:                                # noqa: BLE001
                    app.logger.warning("watermark: sbatch %s failed: %r", m, e)
                    ok = False
            if ok:
                try:
                    with open(_WM_STATE, "w") as fh:
                        fh.write(idea)
                except Exception:                                     # noqa: BLE001
                    pass
                app.logger.info("watermark: launched probes for new champion %s (config=%s)",
                                idea, "exact" if env else "fallback")
        except Exception as e:                                        # noqa: BLE001
            app.logger.warning("watermark trigger failed: %r", e)
        finally:
            _wm_lock.release()

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
        sync_site(d["benchmark"])
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
        out = store.add_finding(payload)
        sync_site(payload.get("benchmark"))
        if payload.get("finding_type", "result") == "result":
            # off the request path: check #1 + launch the watermark probe if champion changed
            threading.Thread(target=trigger_watermark_if_new_champion,
                             args=(out.get("post_id"), payload.get("benchmark")),
                             daemon=True).start()
        return jsonify(out)

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
