import pytest
from w2s_research.core.policy import Decision, WeakStepState, DeferralPolicy


def make_state(**kw):
    base = dict(step_index=0, entropy=0.1, top1_prob=0.9, margin=0.8,
                top_token_id=5, text_so_far="")
    base.update(kw)
    return WeakStepState(**base)


def test_decision_members():
    assert {d.name for d in Decision} == {"CONTINUE", "DEFER"}


def test_base_policy_is_abstract():
    with pytest.raises(NotImplementedError):
        DeferralPolicy().decide(make_state())


def test_subclass_can_decide():
    class AlwaysDefer(DeferralPolicy):
        name = "always_defer"
        def decide(self, state):
            return Decision.DEFER
    assert AlwaysDefer().decide(make_state()) is Decision.DEFER
