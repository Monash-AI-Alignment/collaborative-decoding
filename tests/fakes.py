"""CPU-only fake adapters for engine tests (no models, no GPU)."""
from w2s_research.core.interfaces import WeakStep, StrongOutput


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
            return WeakStep(top_token_id=-1, text_piece="", entropy=0.0,
                            top1_prob=1.0, margin=1.0, is_eos=True)
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
