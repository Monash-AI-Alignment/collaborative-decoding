"""budget_entropy: entropy-threshold deferral, capped at B defers per example.

Spends a fixed strong-call budget on the *first* B uncertain steps, then lets the
weak model finish on its own. Tests whether a small, early dose of strong help is
enough to preserve utility while keeping f_weak high.

Stateful; the per-example defer counter resets on an empty `text_so_far`.
"""
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "budget_entropy"


class BudgetEntropy(DeferralPolicy):
    name = "budget_entropy"

    def __init__(self, tau, budget):
        self.tau = tau
        self.budget = max(0, int(budget))
        self._defers = 0

    def decide(self, state):
        if state.text_so_far == "":          # new example
            self._defers = 0
        if state.entropy > self.tau and self._defers < self.budget:
            self._defers += 1
            return Decision.DEFER
        return Decision.CONTINUE


def build_policy(config):
    return BudgetEntropy(
        tau=getattr(config, "defer_threshold", 0.3),
        budget=getattr(config, "defer_budget", 5),
    )
