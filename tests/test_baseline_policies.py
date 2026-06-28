import importlib
from w2s_research.core.policy import Decision, WeakStepState
from w2s_research.core.decode_config import DecodeConfig


def state(entropy=0.1, margin=0.8, top1=0.9, step=0):
    return WeakStepState(step_index=step, entropy=entropy, top1_prob=top1,
                         margin=margin, top_token_id=0, text_so_far="")


def build(idea, **overrides):
    cfg = DecodeConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    mod = importlib.import_module(f"w2s_research.ideas.{idea}.run")
    return mod.IDEA_NAME, mod.build_policy(cfg)


def test_weak_only_never_defers():
    name, p = build("weak_only")
    assert name == "weak_only"
    assert p.decide(state(entropy=99.0)) is Decision.CONTINUE


def test_strong_only_always_defers():
    _, p = build("strong_only")
    assert p.decide(state(entropy=0.0)) is Decision.DEFER


def test_entropy_threshold_defers_above_tau():
    _, p = build("entropy_threshold", defer_threshold=1.0)
    assert p.decide(state(entropy=1.5)) is Decision.DEFER
    assert p.decide(state(entropy=0.5)) is Decision.CONTINUE


def test_margin_threshold_defers_below_tau():
    _, p = build("margin_threshold", margin_threshold=0.1)
    assert p.decide(state(margin=0.05)) is Decision.DEFER
    assert p.decide(state(margin=0.5)) is Decision.CONTINUE


def test_random_defer_is_seed_deterministic():
    _, p1 = build("random_defer", defer_prob=0.5, seed=123)
    _, p2 = build("random_defer", defer_prob=0.5, seed=123)
    seq1 = [p1.decide(state()) for _ in range(20)]
    seq2 = [p2.decide(state()) for _ in range(20)]
    assert seq1 == seq2                     # same seed -> identical decisions
    assert Decision.DEFER in seq1 and Decision.CONTINUE in seq1
