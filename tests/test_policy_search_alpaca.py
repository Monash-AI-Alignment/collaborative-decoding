import sys
sys.path.insert(0, "scripts")
import policy_search
from w2s_research.core.judge import VLLMJudge
from tests.fakes import FakeWeakModel, FakeStrongModel
from w2s_research.core.interfaces import WeakStep, StrongOutput


def _weak():
    return FakeWeakModel([
        WeakStep(top_token_id=5, text_piece="hello ", is_eos=False),
        WeakStep(top_token_id=6, text_piece="world", is_eos=True),
    ])


def test_run_one_alpaca_uses_lc_winrate():
    judge = VLLMJudge(pref_fn=lambda p: 1.0)   # always prefers Response A (position-swapped -> 0.5)
    weak, strong = _weak(), FakeStrongModel([])
    spec = {"idea": "weak_only", "params": {}, "span_max": 64}
    m = policy_search.run_one(weak, strong, ["say hi"], ["reference answer"],
                              "alpaca_eval", spec, judge=judge)   # default winrate_mode="lc"
    assert "winrate_plain" in m and "winrate_lc" in m
    assert m["utility"] == m["winrate_lc"]           # LC is the primary metric
    assert abs(m["utility"] - 0.5) < 1e-9            # position-swap cancels -> 0.5
    assert len(m["_judge_per_example"]) == 1


def test_run_one_alpaca_judge_none_is_generation_only():
    # judge=None -> produce generations WITHOUT scoring (used to make the strong reference)
    weak, strong = _weak(), FakeStrongModel([])
    spec = {"idea": "weak_only", "params": {}, "span_max": 64}
    m = policy_search.run_one(weak, strong, ["say hi"], ["ref"], "alpaca_eval", spec, judge=None)
    assert m["utility"] is None
    assert m["_generations"] == ["hello "]   # 2nd weak step is EOS -> not appended
    assert "winrate_lc" not in m and "_judge_per_example" not in m


def test_run_one_math_unchanged():
    weak = FakeWeakModel([
        WeakStep(top_token_id=5, text_piece="#### 7", is_eos=False),
        WeakStep(top_token_id=6, text_piece="", is_eos=True),
    ])
    spec = {"idea": "weak_only", "params": {}, "span_max": 64}
    m = policy_search.run_one(weak, FakeStrongModel([]), ["q"], ["7"], "gsm8k", spec)
    assert m["utility"] == 1.0
