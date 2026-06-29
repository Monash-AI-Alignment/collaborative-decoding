"""context_gate: defer at computation-critical positions, let the weak model write prose.

Hypothesis for high f_weak AT high utility: most characters in a math solution are
reasoning prose (cheap, weak-model territory); the utility-critical tokens are the
arithmetic *results*. So only defer when the weak model is uncertain AND the trailing
context looks like a result is about to be produced -- the text ends in an operator /
'=' / ':' -- or when entropy is extreme (a safety override). Everything else the weak
model carries.

Uses only `text_so_far` + the scalar uncertainty signals (no token-text peeking), so
it stays within the white-box-weak / black-box-strong contract.
"""
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "context_gate"

_CRITICAL_TRAILING = set("=+-*/:xX×÷")   # = + - * / : x × ÷


class ContextGate(DeferralPolicy):
    name = "context_gate"

    def __init__(self, tau, tau_hi):
        self.tau = tau
        self.tau_hi = tau_hi

    def decide(self, state):
        if state.entropy <= self.tau:
            return Decision.CONTINUE
        ctx = state.text_so_far.rstrip()
        critical = bool(ctx) and ctx[-1] in _CRITICAL_TRAILING
        if critical or state.entropy > self.tau_hi:
            return Decision.DEFER
        return Decision.CONTINUE


def build_policy(config):
    tau = getattr(config, "defer_threshold", 0.3)
    return ContextGate(tau=tau, tau_hi=getattr(config, "entropy_hi", tau + 1.0))
