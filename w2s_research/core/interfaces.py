"""Model adapter interfaces for collaborative decoding.

The engine depends ONLY on these protocols, so real HF/vLLM adapters and test
fakes are interchangeable. The StrongModel surface is text-in/text-out by
construction (the black-box constraint lives in the type, not in convention).
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class WeakStep:
    """Greedy next-token summary from the white-box weak model."""
    top_token_id: int
    text_piece: str      # marginal text contributed by accepting top_token_id
    entropy: float
    top1_prob: float
    margin: float
    is_eos: bool


@runtime_checkable
class WeakModel(Protocol):
    """Stateful greedy decoder.

    The engine drives it as: begin(instruction) -> [peek(); commit(id)]* with
    resync(...) after any strong-model span. The weak model maintains its OWN
    token sequence across CONTINUE steps (so generation is faithful, like
    model.generate); only resync re-encodes from text (for the cross-tokenizer
    handback). It never reconstructs the context from text per step.
    """
    def begin(self, instruction: str) -> None:
        """Start a fresh generation; set the context to the prompt for `instruction`."""
        ...

    def peek(self) -> WeakStep:
        """Greedy next-token summary for the current context (does NOT advance)."""
        ...

    def commit(self, token_id: int) -> None:
        """Advance the context by accepting `token_id` (the token peek() proposed)."""
        ...

    def resync(self, instruction: str, assistant_text: str) -> None:
        """Re-encode the context from text after a strong-model span (defer handback)."""
        ...


@dataclass
class StrongOutput:
    """Text emitted by the black-box strong model for one span."""
    text: str
    finished: bool       # True iff generation ended on EOS (not a stop string / length cap)


@runtime_checkable
class StrongModel(Protocol):
    def generate(self, instruction: str, assistant_text: str, *,
                 stop: Optional[List[str]], max_tokens: int, temperature: float) -> StrongOutput:
        """Continue the assistant turn as text. No logits/token ids are returned."""
        ...
