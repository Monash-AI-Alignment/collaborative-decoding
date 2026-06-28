import math
from w2s_research.core.uncertainty import entropy_of, top2_margin


def test_entropy_uniform_two():
    assert math.isclose(entropy_of([0.5, 0.5]), math.log(2), rel_tol=1e-9)


def test_entropy_deterministic_is_zero():
    assert math.isclose(entropy_of([1.0, 0.0, 0.0]), 0.0, abs_tol=1e-12)


def test_entropy_ignores_zero_probs():
    # zeros must not produce NaN from log(0)
    assert math.isclose(entropy_of([1.0, 0.0]), 0.0, abs_tol=1e-12)


def test_margin_basic():
    assert math.isclose(top2_margin([0.7, 0.2, 0.1]), 0.5, rel_tol=1e-9)


def test_margin_single_element():
    assert math.isclose(top2_margin([1.0]), 1.0, rel_tol=1e-9)
