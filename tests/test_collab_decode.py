# tests/test_collab_decode.py
from w2s_research.core.interfaces import WeakStep, StrongOutput
from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction
from w2s_research.core.decode_config import DecodeConfig
from w2s_research.core.policy import Decision, DeferralPolicy
from tests.fakes import FakeWeakModel, FakeStrongModel


class AlwaysContinue(DeferralPolicy):
    def decide(self, state): return Decision.CONTINUE

class AlwaysDefer(DeferralPolicy):
    def decide(self, state): return Decision.DEFER

class HighEntropyDefers(DeferralPolicy):
    def decide(self, state):
        return Decision.DEFER if state.entropy > 1.0 else Decision.CONTINUE


def W(piece, entropy=0.1, eos=False):
    return WeakStep(top_token_id=1, text_piece=piece, entropy=entropy,
                    top1_prob=0.9, margin=0.8, is_eos=eos)


def test_weak_only_path_counts_all_chars_weak():
    weak = FakeWeakModel(steps=[W("Hello"), W(" world"), W("", eos=True)])
    strong = FakeStrongModel(outputs=[])
    dec = CollaborativeDecoder(weak, strong, AlwaysContinue(), DecodeConfig())
    r = dec.run_example("inst")
    assert r.text == "Hello world"
    assert r.weak_chars == len("Hello world")
    assert r.strong_chars == 0
    assert r.weak_fraction == 1.0
    assert r.finished is True
    assert r.num_defers == 0


def test_strong_only_path_counts_all_chars_strong():
    weak = FakeWeakModel(steps=[W("x"), W("y")])    # never consumed (always defers)
    strong = FakeStrongModel(outputs=[StrongOutput(text="42", finished=True)])
    dec = CollaborativeDecoder(weak, strong, AlwaysDefer(), DecodeConfig())
    r = dec.run_example("inst")
    assert r.text == "42"
    assert r.weak_chars == 0
    assert r.strong_chars == 2
    assert r.weak_fraction == 0.0
    assert r.finished is True


def test_span_handoff_and_handback():
    # weak emits "2+2", then a high-entropy step -> defer one line, hand back, weak ends.
    weak = FakeWeakModel(steps=[
        W("2+2"),
        W(" ", entropy=2.0),                 # this step is high-entropy -> DEFER instead
        W("", eos=True),                     # after handback, weak finishes
    ])
    strong = FakeStrongModel(outputs=[StrongOutput(text=" = 4\n", finished=False)])
    dec = CollaborativeDecoder(weak, strong, HighEntropyDefers(), DecodeConfig())
    r = dec.run_example("inst")
    assert r.text == "2+2 = 4\n"
    assert r.weak_chars == len("2+2")
    assert r.strong_chars == len(" = 4\n")
    assert r.num_defers == 1
    # the strong model was asked to continue from the weak prefix:
    assert strong.calls == ["2+2"]


def test_max_chars_stops_runaway():
    weak = FakeWeakModel(steps=[W("a")] * 100000)
    strong = FakeStrongModel(outputs=[])
    cfg = DecodeConfig(max_chars=10)
    dec = CollaborativeDecoder(weak, strong, AlwaysContinue(), cfg)
    r = dec.run_example("inst")
    assert len(r.text) >= 10 and len(r.text) <= 11
    assert r.finished is False


def test_aggregate_weak_fraction_is_char_weighted():
    weak = FakeWeakModel(steps=[W("aaaa"), W("", eos=True)])     # 4 weak chars
    strong = FakeStrongModel(outputs=[])
    r1 = CollaborativeDecoder(weak, strong, AlwaysContinue(), DecodeConfig()).run_example("i")
    weak2 = FakeWeakModel(steps=[W("x"), W("x"), W("", eos=True)])
    strong2 = FakeStrongModel(outputs=[])
    r2 = CollaborativeDecoder(weak2, strong2, AlwaysContinue(), DecodeConfig()).run_example("i")
    # all-weak in both -> aggregate fraction 1.0
    assert aggregate_weak_fraction([r1, r2]) == 1.0


def test_aggregate_distinguishes_char_weighting_from_averaging():
    # r1: 6 weak chars, 0 strong chars -> per-result fraction 1.0
    r1 = CollaborativeDecoder(
        FakeWeakModel(steps=[W("aaaaaa"), W("", eos=True)]),
        FakeStrongModel(outputs=[]),
        AlwaysContinue(),
        DecodeConfig(),
    ).run_example("i")
    # r2: 0 weak chars, 2 strong chars -> per-result fraction 0.0
    r2 = CollaborativeDecoder(
        FakeWeakModel(steps=[W("zzz")]),
        FakeStrongModel(outputs=[StrongOutput(text="xx", finished=True)]),
        AlwaysDefer(),
        DecodeConfig(),
    ).run_example("i")
    # char-weighted aggregate = 6/(6+2) = 0.75, NOT the naive average (1.0+0.0)/2 = 0.5
    assert aggregate_weak_fraction([r1, r2]) == 0.75


def test_empty_strong_span_marks_unfinished():
    weak = FakeWeakModel(steps=[W("x", entropy=2.0)])           # high entropy -> policy defers
    strong = FakeStrongModel(outputs=[StrongOutput(text="", finished=False)])  # empty span -> stall
    r = CollaborativeDecoder(weak, strong, HighEntropyDefers(), DecodeConfig()).run_example("i")
    assert r.finished is False
    assert r.strong_chars == 0
