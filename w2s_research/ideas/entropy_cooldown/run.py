"""entropy_cooldown: defer on high entropy, but force >=m weak steps between defers.

After the strong model fixes a hard span, the next few tokens are usually easy, so
deferring again immediately wastes strong budget. A cooldown forces the weak model
to carry at least `m` tokens after each defer, raising f_weak.

Stateful; resets per example (empty `text_so_far`). `step_index` only advances on
accepted weak tokens, so the cooldown is measured in weak tokens since the last defer.
"""
from w2s_research.core import signals
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "entropy_cooldown"

_NEG = -10 ** 9


class EntropyCooldown(DeferralPolicy):
    name = "entropy_cooldown"
    required_hooks = ["logits"]

    def __init__(self, tau, cooldown):
        self.tau = tau
        self.cooldown = max(0, int(cooldown))
        self._last_defer = _NEG

    def decide(self, state):
        if state.text_so_far == "":          # new example
            self._last_defer = _NEG
        ready = (state.step_index - self._last_defer) >= self.cooldown
        if signals.entropy(state.activations["logits"]) > self.tau and ready:
            self._last_defer = state.step_index
            return Decision.DEFER
        return Decision.CONTINUE


def build_policy(config):
    return EntropyCooldown(
        tau=getattr(config, "defer_threshold", 0.3),
        cooldown=getattr(config, "cooldown_m", 4),
    )
