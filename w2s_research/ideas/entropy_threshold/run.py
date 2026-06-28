from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "entropy_threshold"


class EntropyThreshold(DeferralPolicy):
    name = "entropy_threshold"
    def __init__(self, tau):
        self.tau = tau
    def decide(self, state):
        return Decision.DEFER if state.entropy > self.tau else Decision.CONTINUE


def build_policy(config):
    return EntropyThreshold(tau=getattr(config, "defer_threshold", 1.0))
