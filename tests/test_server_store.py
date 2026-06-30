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
