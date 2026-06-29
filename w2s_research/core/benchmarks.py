"""Generative benchmarks: load examples, build prompts, extract + score answers.

GSM8K and MATH are scored with the repo's existing sympy-backed grader
(w2s_research.ideas.ue_zeroshot.math_eval_tools.grade_answer), so equivalent
forms (0.5 == 1/2, etc.) count as correct.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from w2s_research.ideas.ue_zeroshot import math_normalize
from w2s_research.ideas.ue_zeroshot.math_eval_tools import grade_answer

SUPPORTED = ("gsm8k", "math", "alpaca_eval")

_GSM8K_INSTRUCTION = (
    "Solve the following grade-school math problem. Show your reasoning, then give the "
    "final answer on its own line in the form '#### <number>'.\n\nProblem: {q}"
)
_MATH_INSTRUCTION = (
    "Solve the following competition math problem. Show your reasoning, then put the final "
    "answer in \\boxed{{}}.\n\nProblem: {q}"
)


@dataclass
class BenchmarkExample:
    question: str
    answer: str


def build_instruction(name: str, question: str) -> str:
    if name == "gsm8k":
        return _GSM8K_INSTRUCTION.format(q=question)
    if name == "math":
        return _MATH_INSTRUCTION.format(q=question)
    if name == "alpaca_eval":
        return question                       # open-ended: raw instruction, no wrapper
    raise ValueError(f"Unknown benchmark: {name}")


def _last_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def _last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_answer(name: str, text: str) -> Optional[str]:
    if name == "gsm8k":
        after = text.split("####")[-1] if "####" in text else text
        return _last_number(after)
    if name == "math":
        return _last_boxed(text)
    raise ValueError(f"Unknown benchmark: {name}")


def is_correct(name: str, generated_text: str, gold: str) -> bool:
    pred = extract_answer(name, generated_text)
    if pred is None:
        return False
    if name == "gsm8k":
        gold_num = _last_number(gold) or gold
        return grade_answer(pred, gold_num)
    if name == "math":
        gold_clean = math_normalize.remove_boxed(gold) or gold
        return grade_answer(pred, gold_clean)
    raise ValueError(f"Unknown benchmark: {name}")


def utility(name: str, generations: List[str], golds: List[str]) -> float:
    if name == "alpaca_eval":
        raise ValueError("alpaca_eval utility is judge-scored; "
                         "use w2s_research.core.alpaca_eval.score_generations")
    assert len(generations) == len(golds)
    if not generations:
        return 0.0
    correct = sum(1 for g, gold in zip(generations, golds) if is_correct(name, g, gold))
    return correct / len(generations)


def load_benchmark(name: str, split: str, limit: Optional[int] = None,
                   jsonl_path: Optional[str] = None) -> List[BenchmarkExample]:
    if name not in SUPPORTED:
        raise ValueError(f"Unknown benchmark: {name}")
    if jsonl_path is not None:
        rows = [json.loads(line) for line in open(jsonl_path) if line.strip()]
        exs = [BenchmarkExample(question=r["question"], answer=str(r["answer"])) for r in rows]
        return exs[:limit] if limit is not None else exs
    if name == "alpaca_eval":
        from w2s_research.core.alpaca_eval import load_alpaca_eval
        exs = load_alpaca_eval(limit=limit)
        return [BenchmarkExample(question=e.instruction, answer=e.reference) for e in exs]
    return _load_from_hf(name, split, limit)


def _load_from_hf(name: str, split: str, limit: Optional[int]) -> List[BenchmarkExample]:
    from datasets import load_dataset  # lazy import (heavy)
    exs: List[BenchmarkExample] = []
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for row in ds:
            gold = row["answer"].split("####")[-1].strip()
            exs.append(BenchmarkExample(question=row["question"], answer=gold))
    elif name == "math":
        ds = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
        for row in ds:
            gold = _last_boxed(row["solution"])
            if gold is None:
                continue
            exs.append(BenchmarkExample(question=row["problem"], answer=gold))
    if limit is not None:
        exs = exs[:limit]
    return exs
