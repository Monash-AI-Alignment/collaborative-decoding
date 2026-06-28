from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "strong_only"


class StrongOnly(DeferralPolicy):
    name = "strong_only"
    def decide(self, state):
        return Decision.DEFER


def build_policy(config):
    return StrongOnly()
