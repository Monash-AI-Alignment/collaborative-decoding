"""At-source site logging: sharing a finding must update docs data.json itself."""
import json

from w2s_research.server.app import create_app
from w2s_research.server.site_data import build_site_data, write_site_data
from w2s_research.server.store import Store


def _seed(store):
    store.set_baseline("alpaca_eval", 0.2, 0.5, 0.3, 0.98, "/ref")
    store.add_finding({"benchmark": "alpaca_eval", "idea_name": "sub", "finding_type": "result",
                       "utility": 0.3, "weak_token_fraction": 0.6, "utility_recovery": 0.33})
    store.add_finding({"benchmark": "alpaca_eval", "idea_name": "good", "finding_type": "result",
                       "utility": 0.5, "weak_token_fraction": 0.4, "utility_recovery": 1.0})


def test_build_site_data_shape(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    _seed(store)
    d = build_site_data(store)
    assert d["counts"] == {"findings": 2, "results": 2, "ideas": 2}
    assert [e["idea_name"] for e in d["leaderboard"]] == ["good"]
    assert [p["idea"] for p in d["sota_history"]] == ["good"]   # sub-bar point excluded


def test_write_site_data_change_detection(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    _seed(store)
    path = str(tmp_path / "data.json")
    assert write_site_data(store, path) is True
    assert write_site_data(store, path) is False        # identical state -> no rewrite
    store.add_finding({"benchmark": "alpaca_eval", "idea_name": "x",
                       "finding_type": "insight", "summary": "s"})
    assert write_site_data(store, path) is True


def test_share_finding_updates_site_file(tmp_path):
    site = tmp_path / "data.json"
    app = create_app(str(tmp_path / "t.db"), site_data_path=str(site))
    app.config["TESTING"] = True
    c = app.test_client()
    assert site.exists()                                # startup backfill
    c.post("/api/baselines", json={"benchmark": "alpaca_eval", "u_weak": 0.2,
           "u_strong": 0.5, "gap": 0.3, "r_bar": 0.98, "reference_path": "/x"})
    c.post("/api/findings/share", json={"benchmark": "alpaca_eval", "idea_name": "m",
           "finding_type": "result", "utility": 0.5, "weak_token_fraction": 0.44,
           "utility_recovery": 1.0, "summary": "s", "title": "t"})
    d = json.loads(site.read_text())
    assert d["counts"]["findings"] == 1
    assert d["leaderboard"][0]["idea_name"] == "m"
