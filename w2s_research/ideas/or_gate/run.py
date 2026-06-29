"""or_gate: defer when EITHER uncertainty signal fires (high entropy OR low margin).

Higher-recall deferral: catches more uncertain steps than entropy alone, trading
weak-fraction down for safer utility. Frontier reference against `and_gate`.
"""
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "or_gate"


class OrGate(DeferralPolicy):
    name = "or_gate"

    def __init__(self, tau_e, tau_m):
        self.tau_e = tau_e
        self.tau_m = tau_m

    def decide(self, state):
        uncertain = state.entropy > self.tau_e or state.margin < self.tau_m
        return Decision.DEFER if uncertain else Decision.CONTINUE


def build_policy(config):
    return OrGate(
        tau_e=getattr(config, "defer_threshold", 0.7),
        tau_m=getattr(config, "margin_threshold", 0.10),
    )
