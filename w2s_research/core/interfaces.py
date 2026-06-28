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
    def next_step(self, instruction: str, assistant_text: str) -> WeakStep:
        """Return the greedy next-token summary given (instruction, assistant_text)."""
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
