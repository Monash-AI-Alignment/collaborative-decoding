"""AlpacaEval open-ended benchmark: load prompts + reference outputs, score by judge winrate.

Utility = winrate of a method's generations vs the AlpacaEval baseline reference
outputs (GPT-4-turbo, the AlpacaEval-2.0 baseline), judged by the local Gemma judge.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AlpacaExample:
    instruction: str
    reference: str


def load_alpaca_eval(limit: Optional[int] = None,
                     config: str = "alpaca_eval_gpt4_baseline") -> List[AlpacaExample]:
    from datasets import load_dataset            # lazy (heavy)
    ds = load_dataset("tatsu-lab/alpaca_eval", config, split="eval",
                      trust_remote_code=True)
    exs = [AlpacaExample(instruction=r["instruction"], reference=r["output"]) for r in ds]
    return exs[:limit] if limit is not None else exs


def score_generations(judge, instructions, generations, references) -> dict:
    return judge.winrate(instructions, generations, references)
