"""answer_protect: weak model does the reasoning, strong model writes the final answer.

For exact-match benchmarks only the final answer token(s) are graded. This policy lets
the weak model carry the (long, cheap) reasoning under an entropy threshold, but once
the answer marker appears in the text ('####' for GSM8K) it always defers, so the
strong model produces the graded answer conditioned on the weak model's reasoning.

The marker is configurable; default targets GSM8K's '#### <number>' format.
"""
from w2s_research.core.policy import Decision, DeferralPolicy
from w2s_research.core import signals

IDEA_NAME = "answer_protect"


class AnswerProtect(DeferralPolicy):
    name = "answer_protect"
    required_hooks = ["logits"]

    def __init__(self, tau, marker):
        self.tau = tau
        self.marker = marker

    def decide(self, state):
        if self.marker and self.marker in state.text_so_far:
            return Decision.DEFER
        entropy = signals.entropy(state.activations["logits"])
        return Decision.DEFER if entropy > self.tau else Decision.CONTINUE


def build_policy(config):
    return AnswerProtect(
        tau=getattr(config, "defer_threshold", 1.0),
        marker=getattr(config, "answer_marker", "####"),
    )
