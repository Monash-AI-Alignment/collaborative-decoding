from w2s_research.core.alpaca_eval import AlpacaExample, score_generations
from w2s_research.core.judge import VLLMJudge


def test_score_generations_uses_judge():
    # judge prefers (P(A)=1.0) the response containing "STRONG", else P(A)=0.0
    def pref(prompt):
        a = prompt.split("Response A:")[1].split("Response B:")[0]
        return 1.0 if "STRONG" in a else 0.0
    judge = VLLMJudge(pref_fn=pref)
    instr = ["q1", "q2"]
    gens = ["STRONG ans", "weak ans"]
    refs = ["ref one", "STRONG ref"]
    out = score_generations(judge, instr, gens, refs)
    # gen1 beats ref1 (gen has STRONG) -> win 1.0 ; gen2 loses to ref2 -> 0.0
    assert out["per_example"][0]["win"] == 1.0
    assert out["per_example"][1]["win"] == 0.0
    assert abs(out["winrate"] - 0.5) < 1e-9


def test_alpaca_example_shape():
    ex = AlpacaExample(instruction="write a poem", reference="roses are red")
    assert ex.instruction and ex.reference
