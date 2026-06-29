from w2s_research.core.judge import VLLMJudge, _parse_verdict, _pref_from_logprobs


def test_parse_verdict():
    assert _parse_verdict("A") == "A"
    assert _parse_verdict(" The better answer is B.") == "B"
    assert _parse_verdict("tie") == "tie"
    assert _parse_verdict("TIE - both equal") == "tie"
    assert _parse_verdict("garbage") == "tie"   # default to tie when unclear
    assert _parse_verdict("A is tied to the prompt") == "A"   # "tied" must NOT read as tie
    assert _parse_verdict("B (tier 1)") == "B"               # "tier" must NOT read as tie


def test_pref_from_logprobs():
    import math
    # A is near-certain
    entries = [{"token": "A", "logprob": math.log(0.9)}, {"token": "B", "logprob": math.log(0.1)}]
    assert abs(_pref_from_logprobs(entries) - 0.9) < 1e-9
    # whitespace / case variants accumulate to the right letter
    entries = [{"token": " a", "logprob": math.log(0.4)}, {"token": "B", "logprob": math.log(0.6)}]
    assert abs(_pref_from_logprobs(entries) - 0.4) < 1e-9
    # neither letter present -> None (caller falls back)
    assert _pref_from_logprobs([{"token": "hello", "logprob": -1.0}]) is None
    assert _pref_from_logprobs([]) is None


def test_winrate_one_position_swapped_continuous():
    # Pure position bias: judge always assigns P(A)=1.0 regardless of content.
    j = VLLMJudge(pref_fn=lambda prompt: 1.0)
    r = j.winrate_one("inst", "cand", "reference_text")
    # p1 (A=cand)=1.0, p2 (A=ref)=1.0 -> win=(1.0 + (1-1.0))/2 = 0.5 (bias cancels)
    assert r["win"] == 0.5
    assert r["cand_len"] == len("cand") and r["ref_len"] == len("reference_text")
    assert r["prefs"] == [1.0, 1.0]


def test_winrate_one_graded_preference():
    # Continuous: prefer the side containing "good" with prob 0.8, else 0.2.
    def pref(prompt):
        a_block = prompt.split("Response A:")[1].split("Response B:")[0]
        return 0.8 if "good" in a_block else 0.2
    j = VLLMJudge(pref_fn=pref)
    r = j.winrate_one("inst", "good answer", "bad answer")
    # p1 (A=cand "good")=0.8, p2 (A=ref "bad")=0.2 -> win=(0.8 + (1-0.2))/2 = 0.8
    assert abs(r["win"] - 0.8) < 1e-9


def test_winrate_aggregates():
    j = VLLMJudge(pref_fn=lambda p: 1.0, max_workers=2)
    out = j.winrate(["i1", "i2"], ["c1", "c2"], ["r1", "r2"])
    assert out["winrate"] == 0.5
    assert len(out["per_example"]) == 2
