from w2s_research.core.judge import VLLMJudge, _parse_verdict


def test_parse_verdict():
    assert _parse_verdict("A") == "A"
    assert _parse_verdict(" The better answer is B.") == "B"
    assert _parse_verdict("tie") == "tie"
    assert _parse_verdict("TIE - both equal") == "tie"
    assert _parse_verdict("garbage") == "tie"   # default to tie when unclear


def test_winrate_one_position_swapped():
    # Fake judge that ALWAYS says the first-listed answer (A) is better -> pure position bias.
    j = VLLMJudge(chat_fn=lambda prompt: "A")
    r = j.winrate_one("inst", "cand", "reference_text")
    # call1 (A=cand) -> cand wins; call2 (A=ref) -> ref wins. Swapping cancels bias => 0.5
    assert r["win"] == 0.5
    assert r["cand_len"] == len("cand")
    assert r["ref_len"] == len("reference_text")
    assert r["verdicts"] == ["A", "A"]


def test_winrate_one_genuine_preference():
    # Judge prefers whichever side contains "good" regardless of position.
    def chat(prompt):
        a_block = prompt.split("Response A:")[1].split("Response B:")[0]
        return "A" if "good" in a_block else "B"
    j = VLLMJudge(chat_fn=chat)
    r = j.winrate_one("inst", "good answer", "bad answer")
    assert r["win"] == 1.0          # candidate preferred in both orderings


def test_winrate_aggregates():
    j = VLLMJudge(chat_fn=lambda p: "A", max_workers=2)
    out = j.winrate(["i1", "i2"], ["c1", "c2"], ["r1", "r2"])
    assert out["winrate"] == 0.5
    assert len(out["per_example"]) == 2
