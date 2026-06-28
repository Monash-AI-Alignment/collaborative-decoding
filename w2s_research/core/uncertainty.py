"""Pure-Python uncertainty summaries over a probability distribution.

These are used by model adapters (and tests) to summarise a weak-model
next-token distribution into scalars a DeferralPolicy can act on.
"""
import math
from typing import Sequence


def entropy_of(probs: Sequence[float]) -> float:
    """Shannon entropy in nats. Zero-probability entries are skipped."""
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def top2_margin(probs: Sequence[float]) -> float:
    """Difference between the largest and second-largest probabilities.

    For a single-element distribution the margin is the sole probability.
    """
    if len(probs) == 0:
        return 0.0
    ordered = sorted(probs, reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]
