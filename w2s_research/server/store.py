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
            q += " AND benchmark=?"
            args.append(benchmark)
        if finding_type:
            q += " AND finding_type=?"
            args.append(finding_type)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def leaderboard(self, benchmark, r_bar=None):
        base = self.get_baseline(benchmark)
        bar = r_bar if r_bar is not None else (base["r_bar"] if base else 0.98)
        rows = [f for f in self.list_findings(benchmark=benchmark, finding_type="result", limit=10000)
                if f["utility_recovery"] is not None and f["utility_recovery"] >= bar]
        rows.sort(key=lambda f: (-(f["weak_token_fraction"] or 0), -(f["utility"] or 0)))
        return {"entries": rows, "baseline": base, "r_bar": bar}
