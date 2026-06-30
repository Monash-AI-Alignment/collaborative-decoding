from w2s_research.core.eval_idea import score_generations, recovery_of
from w2s_research.core.judge import VLLMJudge


def test_recovery_of():
    assert recovery_of(0.5, 0.166, 0.334) == (0.5 - 0.166) / 0.334
    assert recovery_of(0.166, 0.166, 0.334) == 0.0
    import math
    assert math.isnan(recovery_of(0.6, 0.6, 0.0))     # gap<=0 -> NaN


def test_score_generations_alpaca_strong_ref():
    canonical = {"benchmark": "alpaca_eval", "winrate_mode": "lc",
                 "u_weak": 0.166, "u_strong": 0.5, "gap": 0.334,
                 "reference_texts": ["ref a", "ref b"], "prompts": ["p1", "p2"]}
    judge = VLLMJudge(pref_fn=lambda p: 1.0)   # always prefers A -> position-swap -> win 0.5 each
    out = score_generations("alpaca_eval", ["g1", "g2"], canonical, judge=judge)
    assert "winrate_lc" in out and "winrate_plain" in out
    assert abs(out["utility"] - out["winrate_lc"]) < 1e-9
    # winrate ~0.5 -> recovery = (0.5-0.166)/0.334 ~ 1.0
    assert abs(out["utility_recovery"] - (out["utility"] - 0.166) / 0.334) < 1e-9


def test_score_generations_math_exact_match():
    canonical = {"benchmark": "gsm8k", "u_weak": 0.4, "u_strong": 0.94, "gap": 0.54,
                 "reference_texts": ["7", "12"], "prompts": ["q1", "q2"]}
    out = score_generations("gsm8k", ["#### 7", "#### 99"], canonical)   # 1/2 correct
    assert out["utility"] == 0.5
    assert abs(out["utility_recovery"] - (0.5 - 0.4) / 0.54) < 1e-9
