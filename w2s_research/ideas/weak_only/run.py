from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "weak_only"


class WeakOnly(DeferralPolicy):
    name = "weak_only"
    def decide(self, state):
        return Decision.CONTINUE


def build_policy(config):
    return WeakOnly()
