import pytest
from w2s_research.core import benchmarks as B


def test_alpaca_in_supported():
    assert "alpaca_eval" in B.SUPPORTED


def test_build_instruction_raw():
    assert B.build_instruction("alpaca_eval", "Write a haiku") == "Write a haiku"


def test_utility_alpaca_rejects():
    with pytest.raises(ValueError):
        B.utility("alpaca_eval", ["a"], ["b"])
