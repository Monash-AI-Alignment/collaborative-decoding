"""entropy_streak: defer only after k consecutive uncertain steps (hysteresis).

Isolated single-step entropy spikes are often harmless (the weak model recovers on
the next token). Requiring a *streak* of uncertainty before deferring avoids those
one-off handoffs, so the weak model keeps more of the sequence (higher f_weak).

Stateful across the engine's per-example loop; resets when a new example begins
(detected by an empty `text_so_far`).
"""
from w2s_research.core.policy import Decision, DeferralPolicy
from w2s_research.core import signals

IDEA_NAME = "entropy_streak"


class EntropyStreak(DeferralPolicy):
    name = "entropy_streak"
    required_hooks = ["logits"]

    def __init__(self, tau, k):
        self.tau = tau
        self.k = max(1, int(k))
        self._streak = 0

    def decide(self, state):
        if state.text_so_far == "":          # new example -> reset hysteresis
            self._streak = 0
        if signals.entropy(state.activations["logits"]) > self.tau:
            self._streak += 1
            if self._streak >= self.k:
                self._streak = 0
                return Decision.DEFER
            return Decision.CONTINUE
        self._streak = 0
        return Decision.CONTINUE


def build_policy(config):
    return EntropyStreak(
        tau=getattr(config, "defer_threshold", 0.5),
        k=getattr(config, "streak_k", 3),
    )
