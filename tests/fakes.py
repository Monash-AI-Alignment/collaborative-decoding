"""CPU-only fake adapters for engine tests (no models, no GPU)."""
import torch

from w2s_research.core import signals
from w2s_research.core.interfaces import WeakStep, StrongOutput


def logits_with_margin(m, top_id=0, vocab=8):
    """A logits tensor whose signals.margin ≈ m and argmax == top_id."""
    p = torch.full((vocab,), 1e-9, dtype=torch.float32)
    p[top_id] = (1.0 + m) / 2.0
    p[1 if top_id != 1 else 0] = max((1.0 - m) / 2.0, 1e-9)
    return torch.log(p / p.sum())


def logits_with_entropy(h, top_id=0, vocab=64):
    """A logits tensor whose signals.entropy ≈ h (nats) and argmax == top_id."""
    lo, hi = 0.0, 80.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        lg = torch.zeros(vocab, dtype=torch.float32); lg[top_id] = t
        if signals.entropy(lg) > h:
            lo = t          # too much entropy -> sharpen the peak
        else:
            hi = t
    lg = torch.zeros(vocab, dtype=torch.float32); lg[top_id] = 0.5 * (lo + hi)
    return lg


def synth_activations(entropy=None, margin=None, top_id=0):
    """Build {"logits": ...} matching a target margin OR entropy (margin wins if
    both given — one distribution can't hit arbitrary values of both)."""
    if margin is not None:
        return {"logits": logits_with_margin(margin, top_id)}
    return {"logits": logits_with_entropy(0.1 if entropy is None else entropy, top_id)}


class FakeWeakModel:
    """Stateful fake: replays a scripted list of WeakStep.

    peek() returns the current step without advancing; commit() advances to the
    next; resync() also advances (the step that triggered a defer is consumed).
    Once exhausted, peek() returns an EOS step.
    """
    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    def begin(self, instruction):
        self._i = 0

    def peek(self):
        if self._i >= len(self._steps):
            return WeakStep(top_token_id=-1, text_piece="", is_eos=True)
        return self._steps[self._i]

    def commit(self, token_id):
        self._i += 1

    def resync(self, instruction, assistant_text):
        self._i += 1


class FakeStrongModel:
    """Replays a scripted list of StrongOutput; once exhausted, returns finished empty."""
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0
        self.calls = []

    def generate(self, instruction, assistant_text, *, stop, max_tokens, temperature):
        self.calls.append(assistant_text)
        if self._i >= len(self._outputs):
            return StrongOutput(text="", finished=True)
        out = self._outputs[self._i]
        self._i += 1
        return out
