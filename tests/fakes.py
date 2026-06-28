"""CPU-only fake adapters for engine tests (no models, no GPU)."""
from w2s_research.core.interfaces import WeakStep, StrongOutput


class FakeWeakModel:
    """Replays a scripted list of WeakStep; once exhausted, returns EOS."""
    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    def next_step(self, instruction, assistant_text):
        if self._i >= len(self._steps):
            return WeakStep(top_token_id=-1, text_piece="", entropy=0.0,
                            top1_prob=1.0, margin=1.0, is_eos=True)
        step = self._steps[self._i]
        self._i += 1
        return step


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
