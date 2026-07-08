"""Deferral-policy contract: the single method an 'idea' implements."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class Decision(Enum):
    CONTINUE = "continue"   # accept the weak model's next token
    DEFER = "defer"         # hand the next span to the strong (black-box) model


@dataclass
class WeakStepState:
    """What a policy sees each step: exactly the weak-model activations it asked for,
    plus the running text and step index.

    The weak model is WHITE-BOX. Declare the hooks you want as `required_hooks` on
    your policy (a list); the engine captures them and delivers them here in
    `activations` ({hook_name -> tensor}). Nothing is precomputed for you: request
    "logits" and derive any distributional signal (entropy, margin, top-k, ...) via
    `w2s_research.core.signals`; request internal hooks (e.g.
    "blocks.8.hook_resid_post") for probes / mechinterp / any read-only readout. The
    STRONG model stays black-box — none of its internals ever appear here.

    `step_index` / `text_so_far` cannot be derived from a single step's activations,
    so they are provided. Stateful policies self-reset when `text_so_far == ""`.
    """
    activations: Dict[str, Any] = field(default_factory=dict)
    text_so_far: str = ""    # assistant text generated so far (weak + strong)
    step_index: int = 0      # number of weak tokens already accepted


class DeferralPolicy:
    """Base class. An idea subclasses this and implements `decide`."""
    name: str = "base"

    def decide(self, state: WeakStepState) -> Decision:
        raise NotImplementedError
