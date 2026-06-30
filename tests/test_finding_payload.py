from w2s_research.server.finding_payload import build_share_payload


def test_result_requires_metrics():
    p, e = build_share_payload({"finding_type": "result", "benchmark": "alpaca_eval",
                                "idea_name": "m", "summary": "s"})
    assert p is None and e is not None and "utility" in e

    p, e = build_share_payload({"finding_type": "result", "benchmark": "alpaca_eval",
                                "idea_name": "m", "summary": "s", "utility": 0.5,
                                "weak_token_fraction": 0.44, "utility_recovery": 1.01})
    assert e is None and p["weak_token_fraction"] == 0.44 and p["finding_type"] == "result"


def test_hypothesis_needs_no_metrics():
    p, e = build_share_payload({"finding_type": "hypothesis", "benchmark": "alpaca_eval",
                                "idea_name": "m", "summary": "idea"})
    assert e is None and p["finding_type"] == "hypothesis" and "utility" not in p
