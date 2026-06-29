"""Winrate aggregations over per-example judge records.

plain_winrate: mean win.
lc_winrate (length-controlled, AlpacaEval-2.0 spirit): fit a logistic model
win ~ sigmoid(b0 + b1 * z), where z is the standardized (cand_len - ref_len),
then report the predicted win probability at length-difference = 0. This
removes the judge's systematic preference for longer answers.

Pure-Python (no numpy) so the core module carries no heavy dependency; the
eval sets are small (<=805) and the gradient descent is cheap.
"""
import math


def plain_winrate(per_example):
    if not per_example:
        return 0.0
    return sum(p["win"] for p in per_example) / len(per_example)


def _sigmoid(x):
    if x >= 0:                       # overflow-safe logistic
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def lc_winrate(per_example, steps=2000, lr=0.1):
    if not per_example:
        return 0.0
    diff = [p["cand_len"] - p["ref_len"] for p in per_example]
    y = [p["win"] for p in per_example]
    n = len(y)
    mu = sum(diff) / n
    sd = (sum((d - mu) ** 2 for d in diff) / n) ** 0.5
    if sd < 1e-9:                    # no length variation -> LC == plain
        return sum(y) / n
    z = [(d - mu) / sd for d in diff]
    b0, b1 = 0.0, 0.0
    for _ in range(steps):           # gradient descent on logistic NLL
        g0 = g1 = 0.0
        for zi, yi in zip(z, y):
            err = _sigmoid(b0 + b1 * zi) - yi
            g0 += err
            g1 += err * zi
        b0 -= lr * g0 / n
        b1 -= lr * g1 / n
    z0 = (0.0 - mu) / sd             # standardized value of "equal length"
    return _sigmoid(b0 + b1 * z0)
