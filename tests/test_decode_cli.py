# tests/test_decode_cli.py
from pathlib import Path
from w2s_research.core.interfaces import WeakStep, StrongOutput
from w2s_research.core.decode_config import DecodeConfig
from w2s_research.decode_cli import run_decode
from tests.fakes import FakeWeakModel, FakeStrongModel

FIX = Path(__file__).parent / "fixtures"


class CycleWeak:
    """Stateful weak fake: emits '#### <answer>' once per example, then EOS."""
    def __init__(self, answer):
        self.answer = answer
        self._emitted = False
    def begin(self, instruction):
        self._emitted = False
    def peek(self):
        if not self._emitted:
            return WeakStep(top_token_id=1, text_piece=f"#### {self.answer}",
                            entropy=0.0, top1_prob=1.0, margin=1.0, is_eos=False)
        return WeakStep(top_token_id=-1, text_piece="", entropy=0.0,
                        top1_prob=1.0, margin=1.0, is_eos=True)
    def commit(self, token_id):
        self._emitted = True
    def resync(self, instruction, assistant_text):
        self._emitted = True


def test_run_decode_weak_only_perfect(monkeypatch):
    # Build a weak model that always answers "18" regardless of question.
    cfg = DecodeConfig(benchmark="gsm8k", eval_size=2)
    # both fixture answers differ (18, 56); a constant "18" weak is right once -> utility 0.5
    weak = CycleWeak("18")
    strong = FakeStrongModel(outputs=[])
    out = run_decode(cfg, idea="weak_only",
                     jsonl_path=str(FIX / "gsm8k_tiny.jsonl"),
                     weak=weak, strong=strong)
    assert out["idea"] == "weak_only"
    assert out["benchmark"] == "gsm8k"
    assert out["n"] == 2
    assert out["weak_token_fraction"] == 1.0
    assert out["utility"] == 0.5
