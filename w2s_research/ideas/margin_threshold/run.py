import os

from w2s_research.core.policy import Decision, DeferralPolicy
from w2s_research.core import signals

IDEA_NAME = "margin_threshold"


class MarginThreshold(DeferralPolicy):
    name = "margin_threshold"
    required_hooks = ["logits"]
    def __init__(self, tau):
        self.tau = tau
    def decide(self, state):
        margin = signals.margin(state.activations["logits"])
        return Decision.DEFER if margin < self.tau else Decision.CONTINUE


def build_policy(config):
    # Env overrides let eval_idea reproduce the policy_search winner (tau=0.05, span 64)
    # without a code edit; defaults match the original curated-sweep behavior.
    config.span_max_tokens = int(os.getenv("SPAN_MAX_TOKENS", str(config.span_max_tokens)))
    tau = float(os.getenv("MARGIN_TAU", str(getattr(config, "margin_threshold", 0.10))))
    return MarginThreshold(tau=tau)
