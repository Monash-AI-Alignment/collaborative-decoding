# Autonomous Collaborative-Decoding Research Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the full autonomous research system — a Claude agent proposes/runs/evaluates/shares deferral-policy findings against a server-backed leaderboard + forum, POC on AlpacaEval.

**Architecture:** A one-time GPU **bootstrap** writes a canonical per-benchmark artifact (strong-reference outputs + baselines) to shared scratch. An agent-facing **`eval_idea`** scores one idea against that canonical reference (engine-measured `f_weak` + judge/exact-match utility + recovery). A **fresh minimal Flask server** (`w2s_research/server/`) stores + ranks engine-computed findings and serves the scalar baselines (the 2280-line W2S `app.py` stays dormant). **MCP tools** let the agent read baselines/leaderboard and share findings. The **agent loop** (`agent.py`, Opus 4.8, consecutive-error stop) drives it; **SLURM** runs one persistent CPU server + queued GPU agents.

**Tech Stack:** Python 3.12, Flask + SQLAlchemy/SQLite (server), `claude_agent_sdk` (`@tool`/`create_sdk_mcp_server`), the existing engine (`collab_decode`, `weak_model`, `strong_model`, `judge`, `winrate`, `alpaca_eval`). CPU tests: `~/venvs/w2s-decode/bin/python -m pytest`. GPU venv: `~/venvs/w2s-decode-gpu`.

## Global Constraints

- **Server TRUSTS engine-computed metrics** — it records + ranks `{utility, weak_token_fraction, utility_recovery}` and serves baselines; it never re-runs the judge or engine. `f_weak` integrity comes from the shared engine (idea supplies only `decide()`).
- **Canonical reference is the consistency anchor**: every agent scores against the SAME reference + baselines so `recovery` is comparable. Reference artifact lives on shared scratch; server holds the scalar baselines.
- **Metric:** `recovery = (U_M − U_weak)/(U_strong − U_weak)`; leaderboard ranks by `f_weak` (char-weighted, engine-measured) among `recovery ≥ R_bar` (default `0.98`), tie-break higher `U_M`. AlpacaEval `U` = continuous LC winrate vs the strong reference, `U_strong ≡ 0.5`.
- **No fine-tuning; weak=white-box, strong=black-box; text-level handoff.**
- **SLURM is limited**: one persistent CPU server; GPU agents queue and start staggered; each agent pulls leaderboard+forum on session start; **agents never `sbatch` sub-jobs**.
- **Usage guard**: walltime + stop after `MAX_CONSECUTIVE_ERRORS` (default 4) consecutive session failures.
- Canonical artifact schema (used by bootstrap, eval_idea, server): `{"benchmark", "n", "winrate_mode", "r_bar", "u_weak", "u_strong", "gap", "prompts": [...], "reference_texts": [...]}`. Path: `$W2S_BASELINES_DIR/<benchmark>.json` (env `W2S_BASELINES_DIR`, default `/scratch2/ml23/smur0075/w2s_decode_runs/baselines`).
- Run CPU tests with `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest`.

---

## Phase P1 — canonical baselines + agent-facing scoring

### Task 1: `eval_idea` — score one idea against the canonical reference

**Files:**
- Create: `w2s_research/core/eval_idea.py`
- Test: `tests/test_eval_idea.py`

**Interfaces:**
- Consumes: `judge.VLLMJudge.winrate`, `winrate.{plain_winrate,lc_winrate}`, `benchmarks.{is_correct}`, `collab_decode.{CollaborativeDecoder,aggregate_weak_fraction}`.
- Produces:
  - `load_canonical(benchmark, baselines_dir=None) -> dict` (reads `$W2S_BASELINES_DIR/<benchmark>.json`)
  - `score_generations(benchmark, generations, canonical, judge=None, winrate_mode="lc") -> dict` → `{"utility", "weak_token_fraction"?, "utility_recovery", "winrate_plain"?, "winrate_lc"?, "per_example"?}` — the pure scoring step (no models), testable with a fake judge.
  - `recovery_of(u, u_weak, gap) -> float` (NaN-safe, gap>0 guard).
  - `evaluate_idea(idea_name, benchmark, eval_size, weak=None, strong=None, judge=None, winrate_mode="lc") -> dict` — the full GPU path: load canonical, build the idea's policy, run the engine over `canonical["prompts"][:eval_size]`, compute `f_weak`, score vs `canonical["reference_texts"]`, return `{utility, weak_token_fraction, utility_recovery, operating_points, generations, n}`.

- [ ] **Step 1: Write the failing test** (pure scoring, fake judge — no GPU)

```python
# tests/test_eval_idea.py
from w2s_research.core.eval_idea import score_generations, recovery_of
from w2s_research.core.judge import VLLMJudge


def test_recovery_of():
    assert recovery_of(0.5, 0.166, 0.334) == (0.5 - 0.166) / 0.334
    assert recovery_of(0.166, 0.166, 0.334) == 0.0
    import math
    assert math.isnan(recovery_of(0.6, 0.6, 0.0))     # gap<=0 -> NaN


def test_score_generations_alpaca_strong_ref():
    canonical = {"benchmark": "alpaca_eval", "winrate_mode": "lc",
                 "u_weak": 0.166, "u_strong": 0.5, "gap": 0.334,
                 "reference_texts": ["ref a", "ref b"], "prompts": ["p1", "p2"]}
    judge = VLLMJudge(pref_fn=lambda p: 1.0)   # always prefers A -> position-swap -> win 0.5 each
    out = score_generations("alpaca_eval", ["g1", "g2"], canonical, judge=judge)
    assert "winrate_lc" in out and "winrate_plain" in out
    assert abs(out["utility"] - out["winrate_lc"]) < 1e-9
    # winrate ~0.5 -> recovery = (0.5-0.166)/0.334 ~ 1.0
    assert abs(out["utility_recovery"] - (out["utility"] - 0.166) / 0.334) < 1e-9


def test_score_generations_math_exact_match():
    canonical = {"benchmark": "gsm8k", "u_weak": 0.4, "u_strong": 0.94, "gap": 0.54,
                 "reference_texts": ["7", "12"], "prompts": ["q1", "q2"]}
    out = score_generations("gsm8k", ["#### 7", "#### 99"], canonical)   # 1/2 correct
    assert out["utility"] == 0.5
    assert abs(out["utility_recovery"] - (0.5 - 0.4) / 0.54) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_eval_idea.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/eval_idea.py
"""Score ONE deferral-policy idea against the canonical per-benchmark reference.

This is the agent-facing evaluator: it loads the shared canonical artifact (so
recovery is comparable across agents), runs the idea through the engine to get
engine-measured f_weak + generations, and scores utility (judge winrate vs the
strong reference for alpaca_eval; exact-match for math). The server trusts these.
"""
import importlib
import json
import os

_DEFAULT_BASELINES_DIR = os.environ.get(
    "W2S_BASELINES_DIR", "/scratch2/ml23/smur0075/w2s_decode_runs/baselines")


def load_canonical(benchmark, baselines_dir=None):
    d = baselines_dir or _DEFAULT_BASELINES_DIR
    with open(os.path.join(d, f"{benchmark}.json")) as f:
        return json.load(f)


def recovery_of(u, u_weak, gap):
    return (u - u_weak) / gap if gap > 0 else float("nan")


def score_generations(benchmark, generations, canonical, judge=None, winrate_mode="lc"):
    refs = canonical["reference_texts"][:len(generations)]
    uw, gap = canonical["u_weak"], canonical["gap"]
    out = {}
    if benchmark == "alpaca_eval":
        from w2s_research.core.winrate import plain_winrate, lc_winrate
        prompts = canonical["prompts"][:len(generations)]
        scored = judge.winrate(prompts, generations, refs)
        out["winrate_plain"] = plain_winrate(scored["per_example"])
        out["winrate_lc"] = lc_winrate(scored["per_example"])
        out["per_example"] = scored["per_example"]
        out["utility"] = out["winrate_lc"] if winrate_mode == "lc" else out["winrate_plain"]
    else:
        from w2s_research.core.benchmarks import is_correct
        correct = sum(1 for g, gold in zip(generations, refs) if is_correct(benchmark, g, gold))
        out["utility"] = correct / len(generations) if generations else 0.0
    out["utility_recovery"] = recovery_of(out["utility"], uw, gap)
    return out


def evaluate_idea(idea_name, benchmark, eval_size, weak=None, strong=None,
                  judge=None, winrate_mode="lc", baselines_dir=None):
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction

    canonical = load_canonical(benchmark, baselines_dir)
    prompts = canonical["prompts"][:eval_size]

    cfg = DecodeConfig(benchmark=benchmark, eval_size=len(prompts))
    cfg.span_stop = ["\n"]
    mod = importlib.import_module(f"w2s_research.ideas.{idea_name}.run")
    policy = mod.build_policy(cfg)
    if weak is None:
        from w2s_research.core.weak_model import HFWeakModel
        weak = HFWeakModel(cfg.weak_model, max_model_len=cfg.weak_max_model_len)
    if strong is None:
        from w2s_research.core.strong_model import VLLMStrongModel
        strong = VLLMStrongModel(cfg.strong_model,
                                 gpu_memory_utilization=cfg.strong_gpu_memory_utilization,
                                 max_model_len=cfg.strong_max_model_len)
    if judge is None and benchmark == "alpaca_eval":
        from w2s_research.core.judge import VLLMJudge
        judge = VLLMJudge()

    dec = CollaborativeDecoder(weak, strong, policy, cfg)
    results = dec.run_dataset(prompts)
    gens = [r.text for r in results]
    scored = score_generations(benchmark, gens, canonical, judge=judge, winrate_mode=winrate_mode)
    return {"idea": idea_name, "benchmark": benchmark, "n": len(prompts),
            "weak_token_fraction": aggregate_weak_fraction(results),
            "utility": scored["utility"], "utility_recovery": scored["utility_recovery"],
            "operating_points": [], "generations": gens,
            **{k: scored[k] for k in ("winrate_plain", "winrate_lc") if k in scored}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_eval_idea.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/eval_idea.py tests/test_eval_idea.py
git commit -m "feat(eval): eval_idea — score one idea vs the canonical reference"
```

---

### Task 2: `bootstrap_baselines.py` — write the canonical AlpacaEval artifact

**Files:**
- Create: `scripts/bootstrap_baselines.py`
- Test: manual GPU+judge smoke (no unit test — needs models + judge)

**Interfaces:**
- Consumes: `policy_search.run_one` (generation + judge scoring), `eval_idea` (none directly), `benchmarks.load_benchmark`, `judge.VLLMJudge`.
- Produces: writes `$W2S_BASELINES_DIR/alpaca_eval.json` with the Global-Constraints schema; optionally POSTs the scalar baselines to the server (`POST /api/baselines`). `if __name__ == "__main__"` guarded (vLLM spawn).

- [ ] **Step 1: Write the script**

```python
# scripts/bootstrap_baselines.py
"""One-time: build the canonical per-benchmark reference + baselines on a GPU node.

alpaca_eval: reference = the strong model's OWN free-running outputs; U_strong:=0.5;
U_weak = continuous LC winrate of weak-only vs that reference. Writes the canonical
artifact (prompts + reference_texts + baselines) all agents score against.

Run (GPU node, HF_HOME+HF_TOKEN, judge reachable):
    python scripts/bootstrap_baselines.py --benchmark alpaca_eval --eval-size 100
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="alpaca_eval", choices=["alpaca_eval"])
    ap.add_argument("--eval-size", type=int, default=100)
    ap.add_argument("--r-bar", type=float, default=0.98)
    ap.add_argument("--winrate-mode", default="lc", choices=["plain", "lc"])
    ap.add_argument("--out-dir", default=os.environ.get(
        "W2S_BASELINES_DIR", "/scratch2/ml23/smur0075/w2s_decode_runs/baselines"))
    ap.add_argument("--server-url", default=os.environ.get("ORCHESTRATOR_API_URL"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import policy_search   # scripts/ is on sys.path when run as a script
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.strong_model import VLLMStrongModel
    from w2s_research.core.benchmarks import load_benchmark, build_instruction
    from w2s_research.core.judge import VLLMJudge
    from w2s_research.core.winrate import lc_winrate, plain_winrate

    bench, n = args.benchmark, args.eval_size
    exs = load_benchmark(bench, "eval", limit=n)
    prompts = [build_instruction(bench, e.question) for e in exs]
    base = DecodeConfig(benchmark=bench, eval_size=n)
    weak = HFWeakModel(base.weak_model, max_model_len=base.weak_max_model_len)
    strong = VLLMStrongModel(base.strong_model,
                             gpu_memory_utilization=base.strong_gpu_memory_utilization,
                             max_model_len=base.strong_max_model_len)
    judge = VLLMJudge()

    # strong reference (generation only)
    strong_ref = policy_search.run_one(weak, strong, prompts, prompts, bench,
                                       {"idea": "strong_only", "params": {}, "span_max": 1024,
                                        "span_stop": None})
    ref_texts = strong_ref["_generations"]
    # U_weak = weak-only vs the strong reference
    weak_base = policy_search.run_one(weak, strong, prompts, ref_texts, bench,
                                      {"idea": "weak_only", "params": {}, "span_max": 256},
                                      judge=judge, winrate_mode=args.winrate_mode)
    uw, us = weak_base["utility"], 0.5
    artifact = {"benchmark": bench, "n": len(prompts), "winrate_mode": args.winrate_mode,
                "r_bar": args.r_bar, "u_weak": uw, "u_strong": us, "gap": us - uw,
                "prompts": prompts, "reference_texts": ref_texts}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{bench}.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[bootstrap] wrote {path}  U_weak={uw:.3f} U_strong={us} gap={us-uw:.3f}", flush=True)

    if args.server_url:
        import urllib.request
        body = json.dumps({"benchmark": bench, "u_weak": uw, "u_strong": us,
                           "gap": us - uw, "r_bar": args.r_bar,
                           "reference_path": path}).encode()
        try:
            req = urllib.request.Request(f"{args.server_url}/api/baselines", data=body,
                                         headers={"content-type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
            print(f"[bootstrap] registered baselines with {args.server_url}", flush=True)
        except Exception as e:
            print(f"[bootstrap] WARN: server registration failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: GPU+judge smoke**

Run (GPU node):
```bash
HF_HOME=/scratch2/ml23/smur0075/hf_cache PYTHONPATH=. \
  ~/venvs/w2s-decode-gpu/bin/python scripts/bootstrap_baselines.py --benchmark alpaca_eval --eval-size 20
```
Expected: writes `$W2S_BASELINES_DIR/alpaca_eval.json`; prints `U_weak≈0.17 U_strong=0.5 gap≈0.33`. (Re-uses the validated n=100 result; n=20 here just for the smoke. Replace with eval-size 100 for the real artifact.)

- [ ] **Step 3: Commit**

```bash
git add scripts/bootstrap_baselines.py
git commit -m "feat(bootstrap): canonical AlpacaEval reference + baselines artifact"
```

---

## Phase P2 — minimal server + MCP tools

### Task 3: server store (SQLite-backed findings + baselines)

**Files:**
- Create: `w2s_research/server/__init__.py` (empty), `w2s_research/server/store.py`
- Test: `tests/test_server_store.py`

**Interfaces:**
- Produces a `Store(db_path)` class:
  - `set_baseline(benchmark, u_weak, u_strong, gap, r_bar, reference_path) -> None`
  - `get_baseline(benchmark) -> dict | None`
  - `add_finding(d: dict) -> dict` (d has `benchmark, idea_name, summary, title, finding_type, utility, weak_token_fraction, utility_recovery, operating_points, config, worked`; returns the stored row incl. generated `post_id`, `created_at`)
  - `list_findings(benchmark=None, finding_type=None, limit=100) -> list[dict]`
  - `leaderboard(benchmark, r_bar=None) -> dict` → `{"entries": [...result findings with recovery>=r_bar sorted by weak_token_fraction desc, tie-break utility desc...], "baseline": {...}}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_store.py
from w2s_research.server.store import Store


def _store(tmp_path):
    return Store(str(tmp_path / "t.db"))


def test_baseline_roundtrip(tmp_path):
    s = _store(tmp_path)
    assert s.get_baseline("alpaca_eval") is None
    s.set_baseline("alpaca_eval", 0.166, 0.5, 0.334, 0.98, "/x/alpaca_eval.json")
    b = s.get_baseline("alpaca_eval")
    assert b["u_weak"] == 0.166 and b["gap"] == 0.334 and b["r_bar"] == 0.98


def test_add_finding_and_leaderboard_ranks_by_fweak_at_bar(tmp_path):
    s = _store(tmp_path)
    s.set_baseline("alpaca_eval", 0.166, 0.5, 0.334, 0.98, "/x")
    # two meet the bar (recovery>=0.98), one below
    s.add_finding({"benchmark": "alpaca_eval", "idea_name": "margin", "finding_type": "result",
                   "utility": 0.503, "weak_token_fraction": 0.44, "utility_recovery": 1.01,
                   "summary": "m", "title": "m"})
    s.add_finding({"benchmark": "alpaca_eval", "idea_name": "budget", "finding_type": "result",
                   "utility": 0.55, "weak_token_fraction": 0.38, "utility_recovery": 1.16,
                   "summary": "b", "title": "b"})
    s.add_finding({"benchmark": "alpaca_eval", "idea_name": "lowrec", "finding_type": "result",
                   "utility": 0.30, "weak_token_fraction": 0.9, "utility_recovery": 0.40,
                   "summary": "l", "title": "l"})
    lb = s.leaderboard("alpaca_eval")
    names = [e["idea_name"] for e in lb["entries"]]
    assert names == ["margin", "budget"]          # below-bar excluded; ranked by f_weak desc
    assert lb["baseline"]["gap"] == 0.334


def test_only_results_on_leaderboard(tmp_path):
    s = _store(tmp_path)
    s.set_baseline("alpaca_eval", 0.166, 0.5, 0.334, 0.98, "/x")
    s.add_finding({"benchmark": "alpaca_eval", "idea_name": "h", "finding_type": "hypothesis",
                   "utility": 0.9, "weak_token_fraction": 0.9, "utility_recovery": 2.0,
                   "summary": "h", "title": "h"})
    assert s.leaderboard("alpaca_eval")["entries"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_server_store.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation** (stdlib `sqlite3`, WAL for concurrent agents)

```python
# w2s_research/server/store.py
"""SQLite-backed store for collaborative-decoding findings + baselines.

Trusts engine-computed metrics (records + ranks them). WAL mode so multiple
queued agents can write concurrently. No GPU, no judge.
"""
import json
import os
import sqlite3
import time
import uuid


class Store:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS baselines(
                benchmark TEXT PRIMARY KEY, u_weak REAL, u_strong REAL, gap REAL,
                r_bar REAL, reference_path TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS findings(
                post_id TEXT PRIMARY KEY, created_at REAL, benchmark TEXT, idea_name TEXT,
                finding_type TEXT, title TEXT, summary TEXT, utility REAL,
                weak_token_fraction REAL, utility_recovery REAL, operating_points TEXT,
                config TEXT, worked INTEGER)""")

    def set_baseline(self, benchmark, u_weak, u_strong, gap, r_bar, reference_path):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO baselines VALUES (?,?,?,?,?,?)",
                      (benchmark, u_weak, u_strong, gap, r_bar, reference_path))

    def get_baseline(self, benchmark):
        with self._conn() as c:
            r = c.execute("SELECT * FROM baselines WHERE benchmark=?", (benchmark,)).fetchone()
        return dict(r) if r else None

    def add_finding(self, d):
        row = {"post_id": uuid.uuid4().hex, "created_at": time.time(),
               "benchmark": d.get("benchmark"), "idea_name": d.get("idea_name"),
               "finding_type": d.get("finding_type", "result"), "title": d.get("title", ""),
               "summary": d.get("summary", ""), "utility": d.get("utility"),
               "weak_token_fraction": d.get("weak_token_fraction"),
               "utility_recovery": d.get("utility_recovery"),
               "operating_points": json.dumps(d.get("operating_points")),
               "config": json.dumps(d.get("config")), "worked": int(bool(d.get("worked", False)))}
        with self._conn() as c:
            c.execute("""INSERT INTO findings VALUES
                (:post_id,:created_at,:benchmark,:idea_name,:finding_type,:title,:summary,
                 :utility,:weak_token_fraction,:utility_recovery,:operating_points,:config,:worked)""", row)
        return row

    def list_findings(self, benchmark=None, finding_type=None, limit=100):
        q, args = "SELECT * FROM findings WHERE 1=1", []
        if benchmark:
            q += " AND benchmark=?"; args.append(benchmark)
        if finding_type:
            q += " AND finding_type=?"; args.append(finding_type)
        q += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def leaderboard(self, benchmark, r_bar=None):
        base = self.get_baseline(benchmark)
        bar = r_bar if r_bar is not None else (base["r_bar"] if base else 0.98)
        rows = [f for f in self.list_findings(benchmark=benchmark, finding_type="result", limit=10000)
                if f["utility_recovery"] is not None and f["utility_recovery"] >= bar]
        rows.sort(key=lambda f: (-(f["weak_token_fraction"] or 0), -(f["utility"] or 0)))
        return {"entries": rows, "baseline": base, "r_bar": bar}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_server_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/server/__init__.py w2s_research/server/store.py tests/test_server_store.py
git commit -m "feat(server): SQLite store — findings + baselines + leaderboard ranking"
```

---

### Task 4: minimal Flask server (routes over the store)

**Files:**
- Create: `w2s_research/server/app.py`
- Test: `tests/test_server_app.py`

**Interfaces:**
- Consumes: `Store` (Task 3).
- Produces `create_app(db_path) -> Flask` with routes:
  - `GET  /api/health` → `{"ok": true}`
  - `POST /api/baselines` `{benchmark,u_weak,u_strong,gap,r_bar,reference_path}` → `{"ok":true}`
  - `GET  /api/baselines?benchmark=` → baseline dict or 404
  - `POST /api/evaluate-generations` `{benchmark,idea_name,utility,weak_token_fraction,operating_points?,config?}` → `{utility_recovery, meets_bar, gap}` (computes recovery from stored baseline)
  - `POST /api/findings/share` `{benchmark,idea_name,summary,title,finding_type,utility,weak_token_fraction,utility_recovery,worked,config}` → stored finding (publishes to leaderboard if `finding_type=="result"`)
  - `GET  /api/findings?benchmark=&finding_type=` → `{findings:[...]}`
  - `GET  /api/leaderboard?benchmark=` → `store.leaderboard(...)`

- [ ] **Step 1: Write the failing test** (Flask test client, temp DB)

```python
# tests/test_server_app.py
import json
from w2s_research.server.app import create_app


def _client(tmp_path):
    app = create_app(str(tmp_path / "t.db"))
    app.config["TESTING"] = True
    return app.test_client()


def test_health(tmp_path):
    assert _client(tmp_path).get("/api/health").get_json()["ok"] is True


def test_baselines_and_recovery_and_leaderboard(tmp_path):
    c = _client(tmp_path)
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.166,
           "u_strong": 0.5, "gap": 0.334, "r_bar": 0.98, "reference_path": "/x"})
    assert c.get("/api/baselines?benchmark=alpaca_eval").get_json()["u_weak"] == 0.166
    # evaluate-generations computes recovery from the stored baseline
    ev = c.post("/api/evaluate-generations", json={"benchmark": "alpaca_eval",
                "idea_name": "margin", "utility": 0.5, "weak_token_fraction": 0.44}).get_json()
    assert abs(ev["utility_recovery"] - (0.5 - 0.166) / 0.334) < 1e-9
    assert ev["meets_bar"] is True
    # share a result -> leaderboard
    c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "margin",
           "finding_type": "result", "utility": 0.5, "weak_token_fraction": 0.44,
           "utility_recovery": ev["utility_recovery"], "summary": "s", "title": "t"})
    lb = c.get("/api/leaderboard?benchmark=alpaca_eval").get_json()
    assert [e["idea_name"] for e in lb["entries"]] == ["margin"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_server_app.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/server/app.py
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
```

- [ ] **Step 4: Run test + full suite**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_server_app.py -q && PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest -q`
Expected: PASS (server tests + full suite green). If Flask is missing from the CPU venv, `~/venvs/w2s-decode/bin/pip install flask` first (note in commit).

- [ ] **Step 5: Commit**

```bash
git add w2s_research/server/app.py tests/test_server_app.py
git commit -m "feat(server): minimal Flask app — baselines, evaluate, findings, leaderboard"
```

---

### Task 5: MCP tools (agent ↔ server)

**Files:**
- Create: `w2s_research/research_loop/tools/collab_api_tools.py`
- Test: `tests/test_collab_api_tools.py`

**Interfaces:**
- Consumes: `claude_agent_sdk.{tool,create_sdk_mcp_server}`, `http_utils.{get_server_url,async_http_get,async_http_post}` (existing).
- Produces module-level async tool fns + `create_collab_api_tools_server()`:
  - `get_baselines(benchmark)`, `get_leaderboard(benchmark)`, `evaluate_generations(benchmark,idea_name,utility,weak_token_fraction,...)`, `share_finding(...)`.
  - Each builds the URL from `get_server_url()` and calls the server. Tools return MCP-formatted dicts. Factor the HTTP-shaping into a private helper `_pure_*` function that is unit-tested without the network (the network calls themselves are exercised by the server tests + the GPU integration).

- [ ] **Step 1: Write the failing test** (test the pure payload builders, no network)

```python
# tests/test_collab_api_tools.py
from w2s_research.research_loop.tools import collab_api_tools as t


def test_share_payload_requires_metrics_for_result():
    payload, err = t._share_payload({"finding_type": "result", "benchmark": "alpaca_eval",
                                     "idea_name": "m", "summary": "s"})
    assert err is not None and "utility" in err            # result needs the metric triple

    payload, err = t._share_payload({"finding_type": "result", "benchmark": "alpaca_eval",
                                     "idea_name": "m", "summary": "s", "utility": 0.5,
                                     "weak_token_fraction": 0.44, "utility_recovery": 1.01})
    assert err is None and payload["weak_token_fraction"] == 0.44


def test_hypothesis_needs_no_metrics():
    payload, err = t._share_payload({"finding_type": "hypothesis", "benchmark": "alpaca_eval",
                                     "idea_name": "m", "summary": "idea"})
    assert err is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_collab_api_tools.py -v`
Expected: FAIL (ModuleNotFoundError / `_share_payload` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/research_loop/tools/collab_api_tools.py
"""MCP tools for the collaborative-decoding server: baselines, evaluate, share, leaderboard."""
import json
from typing import Any, Dict

from claude_agent_sdk import tool, create_sdk_mcp_server
from .http_utils import get_server_url, async_http_get, async_http_post

_RESULT_METRICS = ("utility", "weak_token_fraction", "utility_recovery")


def _share_payload(args):
    """Validate + build the /api/findings/share payload. Returns (payload, error_or_None)."""
    ft = args.get("finding_type", "result")
    if ft == "result":
        missing = [k for k in _RESULT_METRICS if args.get(k) is None]
        if missing:
            return None, f"finding_type='result' requires metrics: {missing} (e.g. utility)"
    payload = {"benchmark": args.get("benchmark"), "idea_name": args.get("idea_name"),
               "summary": args.get("summary", ""), "title": args.get("title", ""),
               "finding_type": ft, "worked": args.get("worked"),
               "config": args.get("config")}
    for k in _RESULT_METRICS + ("operating_points",):
        if args.get(k) is not None:
            payload[k] = args[k]
    return payload, None


def _ok(d):
    return {"content": [{"type": "text", "text": json.dumps(d)}]}


@tool("get_baselines", "Get U_weak/U_strong/gap/r_bar for a benchmark (the recovery anchor).",
      {"type": "object", "properties": {"benchmark": {"type": "string"}}, "required": ["benchmark"]})
async def get_baselines(args: Dict[str, Any]):
    return _ok(await async_http_get(f"{get_server_url()}/api/baselines?benchmark={args['benchmark']}"))


@tool("get_leaderboard", "Leaderboard for a benchmark: result findings with recovery>=r_bar, ranked by f_weak.",
      {"type": "object", "properties": {"benchmark": {"type": "string"}}, "required": ["benchmark"]})
async def get_leaderboard(args: Dict[str, Any]):
    return _ok(await async_http_get(f"{get_server_url()}/api/leaderboard?benchmark={args['benchmark']}"))


@tool("evaluate_generations",
      "Submit engine-computed metrics; returns recovery vs the canonical baselines and whether it meets the bar.",
      {"type": "object", "properties": {
          "benchmark": {"type": "string"}, "idea_name": {"type": "string"},
          "utility": {"type": "number"}, "weak_token_fraction": {"type": "number"}},
       "required": ["benchmark", "idea_name", "utility", "weak_token_fraction"]})
async def evaluate_generations(args: Dict[str, Any]):
    return _ok(await async_http_post(f"{get_server_url()}/api/evaluate-generations", dict(args)))


@tool("share_finding",
      "Share a finding to the forum/leaderboard. finding_type='result' requires utility, "
      "weak_token_fraction, utility_recovery and publishes to the leaderboard.",
      {"type": "object", "properties": {
          "benchmark": {"type": "string"}, "idea_name": {"type": "string"},
          "summary": {"type": "string"}, "title": {"type": "string"},
          "finding_type": {"type": "string"}, "utility": {"type": "number"},
          "weak_token_fraction": {"type": "number"}, "utility_recovery": {"type": "number"},
          "worked": {"type": "boolean"}},
       "required": ["benchmark", "idea_name", "summary"]})
async def share_finding(args: Dict[str, Any]):
    payload, err = _share_payload(args)
    if err:
        return _ok({"error": err})
    return _ok(await async_http_post(f"{get_server_url()}/api/findings/share", payload))


def create_collab_api_tools_server():
    return create_sdk_mcp_server(
        name="collab-api-tools",
        tools=[get_baselines, get_leaderboard, evaluate_generations, share_finding])
```

- [ ] **Step 4: Run test + full suite**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_collab_api_tools.py -q && PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add w2s_research/research_loop/tools/collab_api_tools.py tests/test_collab_api_tools.py
git commit -m "feat(tools): collab-api MCP tools (baselines/evaluate/share/leaderboard)"
```

---

## Phase P3 — agent prompt, loop, SLURM

### Task 6: agent loop changes (model, error-stop, collab tools)

**Files:**
- Modify: `w2s_research/research_loop/agent.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `collab_api_tools.create_collab_api_tools_server` (Task 5).
- Produces (changes):
  - `_StopChecker` gains a `max_consecutive_errors` (default from env `MAX_CONSECUTIVE_ERRORS`, fallback 4) and `check()` returns a new `StopReason.MAX_ERRORS` when `consecutive_errors >= max_consecutive_errors`.
  - `BaseAgent` default `model="claude-opus-4-8"`; `AutonomousAgentLoop` default `model="claude-opus-4-8"`.
  - MCP server wiring uses `create_collab_api_tools_server()` under key `"collab-api-tools"`; `_create_agent` allowed_tools = `Read/Write/Edit/Bash/Glob/Grep`, `WebSearch/WebFetch`, `mcp__collab-api-tools__{get_baselines,get_leaderboard,evaluate_generations,share_finding}`.

- [ ] **Step 1: Write the failing test** (StopChecker error-stop; no SDK/network)

```python
# tests/test_agent_loop.py
from w2s_research.research_loop.agent import _StopChecker, StopReason


def test_consecutive_error_stop():
    sc = _StopChecker(max_runtime=10_000, max_consecutive_errors=3)
    assert sc.check() is None
    sc.record_error(); sc.record_error()
    assert sc.check() is None                 # 2 < 3
    sc.record_error()
    assert sc.check() is StopReason.MAX_ERRORS
    sc2 = _StopChecker(max_runtime=10_000, max_consecutive_errors=3)
    sc2.record_error(); sc2.record_success()  # success resets
    sc2.record_error()
    assert sc2.check() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_agent_loop.py -v`
Expected: FAIL (`_StopChecker` has no `max_consecutive_errors`; no `StopReason.MAX_ERRORS`)

- [ ] **Step 3: Implement** in `w2s_research/research_loop/agent.py`:

Add to `StopReason`:
```python
    MAX_ERRORS = "max_consecutive_errors"
```
Replace `_StopChecker.__init__`/`check`:
```python
    def __init__(self, max_runtime: float, max_consecutive_errors: int = None):
        self.max_runtime = max_runtime
        self.start_time = time.time()
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors or int(
            os.getenv("MAX_CONSECUTIVE_ERRORS", "4"))

    def check(self) -> Optional[StopReason]:
        if self.elapsed_time >= self.max_runtime:
            return StopReason.TIMEOUT
        if self.consecutive_errors >= self.max_consecutive_errors:
            return StopReason.MAX_ERRORS
        return None
```
Change both `model: str = "claude-opus-4-6"` defaults (BaseAgent + AutonomousAgentLoop) to `"claude-opus-4-8"`.
Replace the MCP wiring in `AutonomousAgentLoop.__init__` (the `server-api-tools` block) with:
```python
        from .tools.collab_api_tools import create_collab_api_tools_server
        self.mcp_servers = {}
        try:
            self.mcp_servers["collab-api-tools"] = create_collab_api_tools_server()
        except Exception as e:
            print(f"[Init] Warning: collab API tools unavailable: {e}")
```
Replace `_create_agent` `allowed_tools`:
```python
        allowed_tools = [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch",
            "mcp__collab-api-tools__get_baselines",
            "mcp__collab-api-tools__get_leaderboard",
            "mcp__collab-api-tools__evaluate_generations",
            "mcp__collab-api-tools__share_finding",
        ]
```
(Remove the `prior-work-tools` and old `server-api-tools` references in `__init__`/`_create_agent`.)

- [ ] **Step 4: Run test + full suite**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_agent_loop.py -q && PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest -q`
Expected: PASS. (If `import` of agent.py pulls heavy SDK deps unavailable in the CPU venv, the test imports only `_StopChecker`/`StopReason`; keep those import-light — they already are.)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/research_loop/agent.py tests/test_agent_loop.py
git commit -m "feat(agent): Opus 4.8, consecutive-error stop, collab-api MCP tools"
```

---

### Task 7: prompt rewrite + jinja vars

**Files:**
- Rewrite: `w2s_research/research_loop/prompt.jinja2`
- Modify: `w2s_research/research_loop/agent.py` (`resolve_prompt` — add `benchmark`, `r_bar` vars)
- Test: `tests/test_prompt_render.py`

**Interfaces:**
- Consumes: existing `resolve_prompt`.
- Produces: `prompt.jinja2` renders with vars `{workspace_dir, weak_model, strong_model, server_url, benchmark, r_bar, logs_dir, local_mode}`; `resolve_prompt` passes `benchmark=os.getenv("BENCHMARK","alpaca_eval")` and `r_bar=os.getenv("R_BAR","0.98")`.

- [ ] **Step 1: Write the failing test** (render contains the collab-decoding contract, not PGR)

```python
# tests/test_prompt_render.py
from jinja2 import Template
from pathlib import Path


def test_prompt_is_collaborative_decoding():
    tmpl = Path("w2s_research/research_loop/prompt.jinja2").read_text()
    out = Template(tmpl).render(workspace_dir="/ws", weak_model="W", strong_model="S",
                                server_url="http://s:8000", benchmark="alpaca_eval",
                                r_bar="0.98", logs_dir="/ws/logs", local_mode="false")
    assert "weak_token_fraction" in out and "utility_recovery" in out
    assert "DeferralPolicy" in out and "build_policy" in out
    assert "share_finding" in out and "get_leaderboard" in out
    assert "PGR" not in out and "weak labels" not in out    # old W2S content gone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_prompt_render.py -v`
Expected: FAIL (current prompt is the W2S/PGR one).

- [ ] **Step 3: Rewrite `prompt.jinja2`** with these sections (full prose, collaborative-decoding):
  - **Title + role**: autonomous researcher discovering deferral policies for collaborative decoding; multi-worker note (forum) vs local.
  - **Problem**: a white-box weak model (`{{ weak_model }}`) and a black-box strong model (`{{ strong_model }}`); maximize the weak model's share of generated tokens while preserving the strong model's task utility; decide *during generation* when to defer the next span to the strong model. Constraints: NO fine-tuning; weak=logits, strong=text-only; tokenizers differ → text-level handoff.
  - **Metric**: `recovery = (U_M − U_weak)/(U_strong − U_weak)`; **maximize `f_weak` (char-weighted, engine-measured) subject to `recovery ≥ {{ r_bar }}`**. For `{{ benchmark }}` utility is an LLM-judge winrate vs the strong model's own outputs (parity = recovery 1.0).
  - **The idea contract**: an idea is a dir `w2s_research/ideas/autonomous_<name>/` with `__init__.py` + `run.py` exposing `IDEA_NAME` and `build_policy(config) -> DeferralPolicy`. `DeferralPolicy.decide(state) -> Decision.{CONTINUE,DEFER}`; `WeakStepState(step_index, entropy, top1_prob, margin, top_token_id, text_so_far)` — scalars + text only (no logits leak). Stateful policies self-reset on `text_so_far == ""`.
  - **Seed policies to study + build on**: `margin_threshold` (τ≈0.05) and `budget_entropy` are the open-ended winners (f_weak≈0.44 @ parity); `context_gate` wins on math. Read `w2s_research/ideas/*/run.py`.
  - **How to evaluate**: run the engine on the canonical eval set, e.g.
    ```bash
    python -m w2s_research.core.eval_idea_cli --idea autonomous_<name> --benchmark {{ benchmark }} --eval-size 60
    ```
    which prints `{utility, weak_token_fraction, utility_recovery}` scored against the canonical reference. NOTE each run loads the models (~5 min) — to sweep variants, batch them via `scripts/policy_search.py`. Iterate at a modest eval-size; `recovery` near the bar is noisy at small n.
  - **Tools**: `get_baselines`, `get_leaderboard`, `evaluate_generations`, `share_finding` (`finding_type='result'` needs the metric triple → publishes to the leaderboard).
  - **Memory**: read/update `{{ workspace_dir }}/w2s_research/research_loop/notebook.json` each session; session logs in `{{ logs_dir }}`.
  - **Workflow**: Review (notebook + `get_leaderboard` + read seed ideas) → Propose (update `current_idea`) → Implement under `ideas/autonomous_<name>/` → Run `eval_idea_cli` → `evaluate_generations` → Record to notebook → `share_finding` → Decide/iterate → Clean up.
  - **Do science, don't cheat**: `f_weak` is engine-measured; don't modify the shared engine/judge; write idea-local code only.

  Also add a tiny CLI the prompt references: **Create `w2s_research/core/eval_idea_cli.py`** — argparse over `evaluate_idea(...)` printing the metric JSON (mirror `decode_cli` structure; `if __name__=="__main__"` guard).

- [ ] **Step 4: Update `resolve_prompt`** in `agent.py` — add to the `template.render(...)` call:
```python
        benchmark=os.getenv("BENCHMARK", "alpaca_eval"),
        r_bar=os.getenv("R_BAR", "0.98"),
```

- [ ] **Step 5: Run test + full suite**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_prompt_render.py -q && PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add w2s_research/research_loop/prompt.jinja2 w2s_research/research_loop/agent.py w2s_research/core/eval_idea_cli.py tests/test_prompt_render.py
git commit -m "feat(agent): collaborative-decoding prompt + eval_idea_cli + benchmark/r_bar vars"
```

---

### Task 8: SLURM launch scripts

**Files:**
- Create: `slurm/server.sbatch`, `slurm/agent.sbatch`
- Test: shell lint + a dry note (no unit test)

**Interfaces:** none (ops scripts). Produces a persistent CPU server job and a GPU agent job.

- [ ] **Step 1: Write `slurm/server.sbatch`** — CPU partition, long walltime, runs the Flask server, prints its host:port so agents can set `ORCHESTRATOR_API_URL`:
```bash
#!/bin/bash
#SBATCH --job-name=collab-server
#SBATCH --partition=comp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
export W2S_SERVER_DB="${W2S_SERVER_DB:-/scratch2/ml23/smur0075/w2s_decode_runs/server.db}"
export PORT="${PORT:-8000}"
cd /fs04/ml23/smur0075/automated-w2s-research
echo "[server] node $(hostname) port $PORT  -> set ORCHESTRATOR_API_URL=http://$(hostname):$PORT"
PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m w2s_research.server.app
```

- [ ] **Step 2: Write `slurm/agent.sbatch`** — GPU, judge-reachable, server-mode loop with usage guards:
```bash
#!/bin/bash
#SBATCH --job-name=collab-agent
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=ml23
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${ORCHESTRATOR_API_URL:?set ORCHESTRATOR_API_URL=http://<server-node>:8000}"
: "${HF_TOKEN:?set HF_TOKEN (Llama gated)}"
export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export JUDGE_URL="${JUDGE_URL:-http://m3u006:8001/v1}"
export BENCHMARK="${BENCHMARK:-alpaca_eval}"
export R_BAR="${R_BAR:-0.98}"
export WEAK_MODEL="${WEAK_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export STRONG_MODEL="${STRONG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export FULL_AUTO_MAX_RUNTIME_SECONDS="${FULL_AUTO_MAX_RUNTIME_SECONDS:-25200}"
export MAX_CONSECUTIVE_ERRORS="${MAX_CONSECUTIVE_ERRORS:-4}"
export IDEA_UID="${IDEA_UID:-collab-$(date +%s)}"
export IDEA_NAME="${IDEA_NAME:-collaborative_decoding}"
cd /fs04/ml23/smur0075/automated-w2s-research
PYTHONPATH=. ~/venvs/w2s-decode-gpu/bin/python -m w2s_research.research_loop.agent
```
(Scale-out: `sbatch` N copies of `agent.sbatch`; they queue and share findings through the server.)

- [ ] **Step 3: Commit**

```bash
git add slurm/server.sbatch slurm/agent.sbatch
git commit -m "feat(slurm): persistent CPU server + queued GPU agent launch scripts"
```

---

## Integration gates (run after the tasks)

- **Gate P1:** `bootstrap_baselines.py --eval-size 100` writes `alpaca_eval.json`; `eval_idea_cli --idea margin_threshold --benchmark alpaca_eval --eval-size 60` reports `f_weak`/`recovery` consistent with `docs/alpaca_eval_results.md`. (GPU.)
- **Gate P2:** start the server (`python -m w2s_research.server.app`), POST a baseline + a `result` finding, confirm it ranks on `/api/leaderboard`. (CPU; can use the Flask test client or a live localhost server.)
- **Gate P3 (the headline gate):** `sbatch slurm/server.sbatch`; set `ORCHESTRATOR_API_URL`; `sbatch slurm/agent.sbatch`; confirm one agent autonomously completes propose → implement → run → `evaluate_generations` → `share_finding` and the finding appears on `/api/leaderboard`. Then `sbatch` a 2nd agent and confirm it reads the 1st's findings.
