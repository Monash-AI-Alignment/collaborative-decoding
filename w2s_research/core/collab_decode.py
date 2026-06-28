# w2s_research/core/collab_decode.py
"""The collaborative-decoding engine.

Drives a loop over a logical prompt (instruction, assistant_text). At each step
the weak model proposes a greedy next token; the policy decides whether to accept
it (CONTINUE) or hand the next span to the strong black-box model (DEFER). All
handoff is through the assistant_text string, so different tokenizers compose.
The engine is the sole measurer of weak_token_fraction (char-weighted).
"""
from dataclasses import dataclass
from typing import List

from .decode_config import DecodeConfig
from .interfaces import StrongModel, WeakModel
from .policy import Decision, DeferralPolicy, WeakStepState


@dataclass
class DecodeResult:
    text: str
    weak_chars: int
    strong_chars: int
    num_weak_steps: int
    num_defers: int
    finished: bool

    @property
    def total_chars(self) -> int:
        return self.weak_chars + self.strong_chars

    @property
    def weak_fraction(self) -> float:
        return self.weak_chars / self.total_chars if self.total_chars else 0.0


class CollaborativeDecoder:
    def __init__(self, weak: WeakModel, strong: StrongModel,
                 policy: DeferralPolicy, config: DecodeConfig):
        self.weak = weak
        self.strong = strong
        self.policy = policy
        self.config = config

    def run_example(self, instruction: str) -> DecodeResult:
        cfg = self.config
        assistant = ""
        weak_chars = strong_chars = num_weak_steps = num_defers = 0
        finished = False

        for _ in range(cfg.max_steps):
            step = self.weak.next_step(instruction, assistant)
            state = WeakStepState(
                step_index=num_weak_steps,
                entropy=step.entropy,
                top1_prob=step.top1_prob,
                margin=step.margin,
                top_token_id=step.top_token_id,
                text_so_far=assistant,
            )
            if self.policy.decide(state) is Decision.CONTINUE:
                if step.is_eos:
                    finished = True
                    break
                assistant += step.text_piece
                weak_chars += len(step.text_piece)
                num_weak_steps += 1
            else:
                out = self.strong.generate(
                    instruction, assistant,
                    stop=cfg.span_stop, max_tokens=cfg.span_max_tokens,
                    temperature=cfg.strong_temperature,
                )
                assistant += out.text
                strong_chars += len(out.text)
                num_defers += 1
                if out.finished:
                    finished = True
                    break
                if out.text == "":          # no progress (stall) -> stop; NOT a successful finish
                    finished = False
                    break

            if len(assistant) >= cfg.max_chars:
                break

        return DecodeResult(
            text=assistant, weak_chars=weak_chars, strong_chars=strong_chars,
            num_weak_steps=num_weak_steps, num_defers=num_defers, finished=finished,
        )

    def run_dataset(self, instructions: List[str]) -> List[DecodeResult]:
        return [self.run_example(instr) for instr in instructions]


def aggregate_weak_fraction(results: List[DecodeResult]) -> float:
    weak = sum(r.weak_chars for r in results)
    strong = sum(r.strong_chars for r in results)
    total = weak + strong
    return weak / total if total else 0.0
