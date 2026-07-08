import pytest
from w2s_research.core.policy import Decision, WeakStepState, DeferralPolicy


def make_state(**kw):
    return WeakStepState(activations=kw.get("activations", {}),
                         text_so_far=kw.get("text_so_far", ""),
                         step_index=kw.get("step_index", 0))


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
