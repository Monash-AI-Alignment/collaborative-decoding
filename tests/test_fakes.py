from w2s_research.core.interfaces import WeakStep, StrongOutput
from tests.fakes import FakeWeakModel, FakeStrongModel


def test_fake_weak_emits_scripted_steps():
    weak = FakeWeakModel(steps=[
        WeakStep(top_token_id=1, text_piece="2", entropy=0.1, top1_prob=0.9, margin=0.8, is_eos=False),
        WeakStep(top_token_id=2, text_piece=" + 2", entropy=2.0, top1_prob=0.3, margin=0.05, is_eos=False),
    ])
    s0 = weak.next_step("inst", "")
    assert s0.text_piece == "2" and s0.entropy == 0.1
    s1 = weak.next_step("inst", "2")
    assert s1.text_piece == " + 2" and s1.entropy == 2.0


def test_fake_weak_runs_out_returns_eos():
    weak = FakeWeakModel(steps=[])
    step = weak.next_step("inst", "")
    assert step.is_eos is True
    assert step.text_piece == ""


def test_fake_strong_returns_scripted_output():
    strong = FakeStrongModel(outputs=[StrongOutput(text="= 4\n", finished=False),
                                      StrongOutput(text="#### 4", finished=True)])
    o0 = strong.generate("inst", "2 + 2", stop=["\n"], max_tokens=16, temperature=0.0)
    assert o0.text == "= 4\n" and o0.finished is False
    o1 = strong.generate("inst", "2 + 2= 4\n", stop=["\n"], max_tokens=16, temperature=0.0)
    assert o1.finished is True
    assert strong.calls == ["2 + 2", "2 + 2= 4\n"]
