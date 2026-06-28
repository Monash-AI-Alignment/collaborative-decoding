import random
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "random_defer"


class RandomDefer(DeferralPolicy):
    name = "random_defer"
    def __init__(self, defer_prob, seed):
        self.defer_prob = defer_prob
        self._rng = random.Random(seed)
    def decide(self, state):
        return Decision.DEFER if self._rng.random() < self.defer_prob else Decision.CONTINUE


def build_policy(config):
    return RandomDefer(defer_prob=getattr(config, "defer_prob", 0.5),
                       seed=getattr(config, "seed", 42))
