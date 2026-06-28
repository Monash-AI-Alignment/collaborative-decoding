# tests/test_benchmarks.py
from pathlib import Path
from w2s_research.core.benchmarks import (
    BenchmarkExample, build_instruction, extract_answer, is_correct,
    utility, load_benchmark,
)

FIX = Path(__file__).parent / "fixtures"


def test_build_instruction_mentions_question():
    instr = build_instruction("gsm8k", "What is 2+2?")
    assert "What is 2+2?" in instr


def test_extract_gsm8k_last_number():
    assert extract_answer("gsm8k", "First 3, then ... The answer is 18.") == "18"
    assert extract_answer("gsm8k", "So we get #### 42") == "42"
    assert extract_answer("gsm8k", "no number here") is None


def test_extract_math_boxed():
    assert extract_answer("math", r"thus \boxed{\frac{1}{2}} is final") == r"\frac{1}{2}"
    assert extract_answer("math", "no box") is None


def test_is_correct_gsm8k_numeric():
    assert is_correct("gsm8k", "The answer is 18", "18") is True
    assert is_correct("gsm8k", "The answer is 19", "18") is False


def test_is_correct_math_via_grader():
    assert is_correct("math", r"\boxed{0.5}", r"\frac{1}{2}") is True


def test_utility_is_fraction_correct():
    gens = ["answer 18", "answer 7", "answer 100"]
    golds = ["18", "8", "100"]
    assert utility("gsm8k", gens, golds) == 2 / 3


def test_load_benchmark_from_jsonl(tmp_path):
    exs = load_benchmark("gsm8k", "test", limit=None, jsonl_path=str(FIX / "gsm8k_tiny.jsonl"))
    assert len(exs) == 2
    assert isinstance(exs[0], BenchmarkExample)
    assert exs[0].answer == "18"


def test_load_benchmark_respects_limit():
    exs = load_benchmark("gsm8k", "test", limit=1, jsonl_path=str(FIX / "gsm8k_tiny.jsonl"))
    assert len(exs) == 1


def test_load_benchmark_limit_zero_is_empty():
    exs = load_benchmark("gsm8k", "test", limit=0, jsonl_path=str(FIX / "gsm8k_tiny.jsonl"))
    assert exs == []
