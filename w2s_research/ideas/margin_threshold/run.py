from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "margin_threshold"


class MarginThreshold(DeferralPolicy):
    name = "margin_threshold"
    def __init__(self, tau):
        self.tau = tau
    def decide(self, state):
        return Decision.DEFER if state.margin < self.tau else Decision.CONTINUE


def build_policy(config):
    return MarginThreshold(tau=getattr(config, "margin_threshold", 0.10))
