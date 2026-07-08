from w2s_research.core.policy import Decision, DeferralPolicy
from w2s_research.core import signals

IDEA_NAME = "entropy_threshold"


class EntropyThreshold(DeferralPolicy):
    name = "entropy_threshold"
    required_hooks = ["logits"]
    def __init__(self, tau):
        self.tau = tau
    def decide(self, state):
        entropy = signals.entropy(state.activations["logits"])
        return Decision.DEFER if entropy > self.tau else Decision.CONTINUE


def build_policy(config):
    return EntropyThreshold(tau=getattr(config, "defer_threshold", 1.0))
