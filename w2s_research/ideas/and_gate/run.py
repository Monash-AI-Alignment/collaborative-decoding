"""and_gate: defer only when the weak model is uncertain on BOTH signals.

Higher-precision deferral than naive entropy: requires high entropy AND a small
top1-top2 margin before handing off. Fewer, more-justified defers -> the weak
model carries more characters (higher f_weak) at, hopefully, similar utility.
"""
from w2s_research.core.policy import Decision, DeferralPolicy
from w2s_research.core import signals

IDEA_NAME = "and_gate"


class AndGate(DeferralPolicy):
    name = "and_gate"
    required_hooks = ["logits"]

    def __init__(self, tau_e, tau_m):
        self.tau_e = tau_e
        self.tau_m = tau_m

    def decide(self, state):
        logits = state.activations["logits"]
        uncertain = signals.entropy(logits) > self.tau_e and signals.margin(logits) < self.tau_m
        return Decision.DEFER if uncertain else Decision.CONTINUE


def build_policy(config):
    return AndGate(
        tau_e=getattr(config, "defer_threshold", 0.5),
        tau_m=getattr(config, "margin_threshold", 0.10),
    )
