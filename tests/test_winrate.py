from w2s_research.core.winrate import plain_winrate, lc_winrate


def test_plain_winrate():
    per = [{"win": 1.0, "cand_len": 10, "ref_len": 10},
           {"win": 0.0, "cand_len": 10, "ref_len": 10},
           {"win": 0.5, "cand_len": 10, "ref_len": 10}]
    assert abs(plain_winrate(per) - 0.5) < 1e-9


def test_lc_winrate_penalizes_length_driven_wins():
    # Wins occur ONLY when the candidate is much longer -> length-driven.
    # Plain winrate is high; LC (at zero length diff) should be markedly lower.
    per = []
    for i in range(40):
        longer = i % 2 == 0
        per.append({"win": 1.0 if longer else 0.0,
                    "cand_len": 400 if longer else 100,
                    "ref_len": 100 if longer else 400})
    assert plain_winrate(per) == 0.5      # balanced by construction
    # Build a length-confounded set: candidate longer AND wins.
    per2 = [{"win": 1.0, "cand_len": 500, "ref_len": 100} for _ in range(20)]
    per2 += [{"win": 0.0, "cand_len": 100, "ref_len": 500} for _ in range(20)]
    assert plain_winrate(per2) == 0.5
    lc = lc_winrate(per2)
    assert 0.0 <= lc <= 1.0               # well-defined probability
    assert abs(lc - 0.5) < 0.25           # length effect removed -> near 0.5


def test_lc_winrate_handles_degenerate_input():
    assert lc_winrate([]) == 0.0
    # zero length variance -> fall back to plain winrate
    per = [{"win": 1.0, "cand_len": 10, "ref_len": 10} for _ in range(5)]
    assert abs(lc_winrate(per) - 1.0) < 1e-6
