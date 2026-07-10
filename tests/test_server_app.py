from w2s_research.server.app import create_app


def _client(tmp_path):
    # site_data_path MUST be a tmp file — without it create_app defaults to the real
    # docs/data.json and the test store would clobber the production site data.
    app = create_app(str(tmp_path / "t.db"), site_data_path=str(tmp_path / "data.json"))
    app.config["TESTING"] = True
    return app.test_client()


def test_health(tmp_path):
    assert _client(tmp_path).get("/api/health").get_json()["ok"] is True


def test_baselines_and_recovery_and_leaderboard(tmp_path):
    c = _client(tmp_path)
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.166,
           "u_strong": 0.5, "gap": 0.334, "r_bar": 0.98, "reference_path": "/x"})
    assert c.get("/api/baselines?benchmark=alpaca_eval").get_json()["u_weak"] == 0.166
    ev = c.post("/api/evaluate-generations", json={"benchmark": "alpaca_eval",
                "idea_name": "margin", "utility": 0.5, "weak_token_fraction": 0.44}).get_json()
    assert abs(ev["utility_recovery"] - (0.5 - 0.166) / 0.334) < 1e-9
    assert ev["meets_bar"] is True
    c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "margin",
           "finding_type": "result", "utility": 0.5, "weak_token_fraction": 0.44,
           "utility_recovery": ev["utility_recovery"], "summary": "s", "title": "t"})
    lb = c.get("/api/leaderboard?benchmark=alpaca_eval").get_json()
    assert [e["idea_name"] for e in lb["entries"]] == ["margin"]


def test_missing_baseline_404(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/baselines?benchmark=nope").status_code == 404
    assert c.post("/api/evaluate-generations", json={"benchmark": "nope",
                  "idea_name": "x", "utility": 0.5, "weak_token_fraction": 0.1}).status_code == 404


def test_malformed_posts_return_400_not_500(tmp_path):
    c = _client(tmp_path)
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.166,
           "u_strong": 0.5, "gap": 0.334, "r_bar": 0.98})
    # evaluate-generations missing 'utility' -> 400 (was KeyError 500)
    assert c.post("/api/evaluate-generations",
                  json={"benchmark": "alpaca_eval", "idea_name": "x"}).status_code == 400
    # share a 'result' without weak_token_fraction -> 400 (validation)
    assert c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "m",
                  "finding_type": "result", "utility": 0.5, "summary": "s"}).status_code == 400
    # null JSON body -> 400 (was AttributeError 500)
    assert c.post("/api/findings/share", data="null",
                  content_type="application/json").status_code == 400


def test_share_recomputes_recovery_server_side(tmp_path):
    c = _client(tmp_path)
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.166,
           "u_strong": 0.5, "gap": 0.334, "r_bar": 0.98})
    # agent submits a BOGUS utility_recovery (99.0); server must recompute from utility+baseline
    c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "m",
           "finding_type": "result", "utility": 0.5, "weak_token_fraction": 0.44,
           "utility_recovery": 99.0, "summary": "s", "title": "t"})
    lb = c.get("/api/leaderboard?benchmark=alpaca_eval").get_json()
    assert len(lb["entries"]) == 1
    assert abs(lb["entries"][0]["utility_recovery"] - (0.5 - 0.166) / 0.334) < 1e-9   # not 99.0


def test_new_champion_triggers_watermark_with_config(tmp_path, monkeypatch):
    """A result that becomes leaderboard #1 sbatch's BOTH watermark methods with the
    champion's recorded config; a repeat or a non-#1 result does not."""
    import types
    import w2s_research.server.app as appmod
    monkeypatch.setenv("HF_TOKEN", "hf_TEST")
    monkeypatch.setenv("W2S_WM_STATE", str(tmp_path / "champ"))
    calls = []

    def fake_run(cmd, *a, **k):
        if cmd and cmd[0] != "squeue":
            calls.append(cmd[1])                       # the --export arg
        return types.SimpleNamespace(returncode=0, stdout="")   # squeue stdout="" -> not busy

    class _Inline:                                     # run the trigger thread synchronously
        def __init__(self, target=None, args=(), daemon=None):
            self._t, self._a = target, args
        def start(self):
            self._t(*self._a)

    monkeypatch.setattr(appmod.subprocess, "run", fake_run)
    monkeypatch.setattr(appmod.threading, "Thread", _Inline)

    c = _client(tmp_path)
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.166,
           "u_strong": 0.5, "gap": 0.334, "r_bar": 0.98})
    r = c.post("/api/findings/share", json={"benchmark": "alpaca_eval",
               "idea_name": "autonomous_factgate", "finding_type": "result",
               "utility": 0.51, "weak_token_fraction": 0.474, "utility_recovery": 1.03,
               "summary": "s", "title": "t",
               "config": {"env": {"CG_TAU": "0.16", "FG_ENTITY_TAU": "0.25"}}})
    assert r.status_code == 200
    assert len(calls) == 2                             # both methods
    assert any("WATERMARK=inference_kgw" in e for e in calls)
    assert any("WATERMARK=eth_french" in e for e in calls)
    assert all("CG_TAU=0.16" in e and "FG_ENTITY_TAU=0.25" in e for e in calls)

    calls.clear()                                      # a certified but NON-#1 result -> no trigger
    c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "weaker",
           "finding_type": "result", "utility": 0.50, "weak_token_fraction": 0.20,
           "utility_recovery": 1.0, "summary": "s", "title": "t"})
    assert calls == []


def test_forum_html_view(tmp_path):
    c = _client(tmp_path)
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.166,
           "u_strong": 0.5, "gap": 0.334, "r_bar": 0.98, "reference_path": "/x"})
    c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "margin",
           "finding_type": "result", "utility": 0.5, "weak_token_fraction": 0.44,
           "utility_recovery": 1.0, "summary": "x <script>alert(1)</script> y",
           "title": "the title"})
    r = c.get("/?benchmark=alpaca_eval")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Leaderboard" in html and "margin" in html and "the title" in html
    assert "<script>alert" not in html      # user content must be escaped
    assert "&lt;script&gt;alert" in html
