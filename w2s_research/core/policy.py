"""Deferral-policy contract: the single method an 'idea' implements."""
from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    CONTINUE = "continue"   # accept the weak model's next token
    DEFER = "defer"         # hand the next span to the strong (black-box) model


@dataclass
class WeakStepState:
    """Everything a policy may see about the weak model's current step.

    Deliberately scalar + text only: NO logits/token distributions and NO
    strong-model internals, so a policy cannot peek past the white/black-box line.
    """
    step_index: int      # number of weak tokens already accepted
    entropy: float       # entropy (nats) of the weak next-token distribution
    top1_prob: float     # probability of the greedy token
    margin: float        # top1 - top2 probability
    top_token_id: int    # weak-vocab id of the greedy token
    text_so_far: str     # assistant text generated so far (weak + strong)


class DeferralPolicy:
    """Base class. An idea subclasses this and implements `decide`."""
    name: str = "base"

    def decide(self, state: WeakStepState) -> Decision:
        raise NotImplementedError
