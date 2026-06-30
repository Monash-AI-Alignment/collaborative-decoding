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
