# Collaborative Decoding — Phase 1 Implementation Plan (engine + baselines, CLI-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-unit-testable collaborative-decoding engine where a white-box weak model defers spans of generation to a black-box strong model, plus baseline deferral policies and a CLI that reports `(utility, weak_token_fraction)` on GSM8K/MATH — with no autonomous agent or server yet.

**Architecture:** A `CollaborativeDecoder` engine drives a loop over a *logical prompt* `(instruction, assistant_text)`. Each step it asks a `WeakModel` for the greedy next-token summary (entropy/margin/top-prob/text-piece); a `DeferralPolicy` returns `CONTINUE` (append the weak token) or `DEFER` (ask a `StrongModel` to generate the next span as text). Models apply their own chat templates internally, so different tokenizers/templates compose at the text level. The engine is the sole measurer of `weak_token_fraction` (char-weighted). Model adapters are injected, so the engine + policies + benchmarks are fully unit-tested on CPU with fakes; the real HF/vLLM adapters are smoke-tested on a GPU node.

**Tech Stack:** Python 3.12, PyTorch 2.8, HuggingFace Transformers (white-box weak model), vLLM 0.11 (black-box strong model), `datasets` (benchmarks), pytest (TDD), sympy/pylatexenc via the repo's existing `math_eval_tools`.

## Global Constraints

- Python `>=3.12`; all new modules live under `w2s_research/core/` and `w2s_research/ideas/`.
- **No training / fine-tuning / LoRA anywhere in this phase.** Inference only.
- Weak model = white-box (HF Transformers, per-step logits). Strong model = **black-box**: its adapter exposes only text in / text out — no logits, no token ids leak to policy or engine.
- Weak/strong tokenizers differ; **all weak↔strong handoff happens through the `assistant_text` string**, never through token ids.
- `weak_token_fraction` is **char-weighted** and computed ONLY inside the engine: `Σ weak_chars / Σ(weak_chars + strong_chars)`.
- Default models: weak `meta-llama/Llama-3.2-1B-Instruct` (gated → `HF_TOKEN`), strong `Qwen/Qwen2.5-7B-Instruct` (ungated).
- Metric target (computed in CLI here, server later): maximize `weak_token_fraction` subject to `utility_recovery = (U_M − U_weak)/(U_strong − U_weak) ≥ 0.98`.
- M3 caching: `HF_HOME=/scratch2/ml23/smur0075/hf_cache`. Never download models under `/home` or `/projects`.
- Unit tests must run on CPU (no GPU, no model download). GPU adapters are exercised only by explicit smoke commands.
- Idea contract: each idea dir exposes `IDEA_NAME: str` and `build_policy(config: DecodeConfig) -> DeferralPolicy`.

---

### Task 1: Phase-1 scaffolding + `DecodeConfig`

**Files:**
- Create: `w2s_research/core/decode_config.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Test: `tests/test_decode_config.py`

**Interfaces:**
- Produces: `DecodeConfig` dataclass with fields used by every later task:
  `benchmark:str`, `split:str`, `eval_size:Optional[int]`, `weak_model:str`, `strong_model:str`,
  `max_steps:int`, `max_chars:int`, `span_stop:Optional[list[str]]`, `span_max_tokens:int`,
  `strong_temperature:float`, `weak_max_model_len:int`, `strong_gpu_memory_utilization:float`,
  `strong_max_model_len:int`, `seed:int`, `r_bar:float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decode_config.py
from w2s_research.core.decode_config import DecodeConfig


def test_defaults_match_locked_models():
    cfg = DecodeConfig()
    assert cfg.weak_model == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg.strong_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.benchmark == "gsm8k"
    assert cfg.r_bar == 0.98
    assert cfg.span_stop == ["\n"]


def test_eval_size_override():
    cfg = DecodeConfig(eval_size=32, benchmark="math")
    assert cfg.eval_size == 32
    assert cfg.benchmark == "math"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decode_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.core.decode_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/decode_config.py
"""Configuration for collaborative-decoding experiments (inference only)."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DecodeConfig:
    # Benchmark
    benchmark: str = "gsm8k"          # "gsm8k" | "math"
    split: str = "test"
    eval_size: Optional[int] = None   # None = full split

    # Models
    weak_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    strong_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Engine limits
    max_steps: int = 512              # max weak-token steps per example
    max_chars: int = 4000             # hard cap on assistant_text length per example

    # Strong-model span generation
    span_stop: Optional[List[str]] = field(default_factory=lambda: ["\n"])
    span_max_tokens: int = 256
    strong_temperature: float = 0.0

    # Runtime / memory
    weak_max_model_len: int = 4096
    strong_max_model_len: int = 4096
    strong_gpu_memory_utilization: float = 0.6   # leave room for HF weak model on same GPU

    # Reproducibility / metric
    seed: int = 42
    r_bar: float = 0.98               # utility_recovery bar for the headline metric
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Ensure the repo root is importable when running pytest from anywhere."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decode_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/decode_config.py tests/__init__.py tests/conftest.py tests/test_decode_config.py
git commit -m "feat(decode): add DecodeConfig and pytest scaffolding"
```

---

### Task 2: Uncertainty helpers

**Files:**
- Create: `w2s_research/core/uncertainty.py`
- Test: `tests/test_uncertainty.py`

**Interfaces:**
- Produces: `entropy_of(probs: Sequence[float]) -> float` (Shannon entropy in nats),
  `top2_margin(probs: Sequence[float]) -> float` (top1 − top2 probability).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_uncertainty.py
import math
from w2s_research.core.uncertainty import entropy_of, top2_margin


def test_entropy_uniform_two():
    assert math.isclose(entropy_of([0.5, 0.5]), math.log(2), rel_tol=1e-9)


def test_entropy_deterministic_is_zero():
    assert math.isclose(entropy_of([1.0, 0.0, 0.0]), 0.0, abs_tol=1e-12)


def test_entropy_ignores_zero_probs():
    # zeros must not produce NaN from log(0)
    assert math.isclose(entropy_of([1.0, 0.0]), 0.0, abs_tol=1e-12)


def test_margin_basic():
    assert math.isclose(top2_margin([0.7, 0.2, 0.1]), 0.5, rel_tol=1e-9)


def test_margin_single_element():
    assert math.isclose(top2_margin([1.0]), 1.0, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uncertainty.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.core.uncertainty'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/uncertainty.py
"""Pure-Python uncertainty summaries over a probability distribution.

These are used by model adapters (and tests) to summarise a weak-model
next-token distribution into scalars a DeferralPolicy can act on.
"""
import math
from typing import Sequence


def entropy_of(probs: Sequence[float]) -> float:
    """Shannon entropy in nats. Zero-probability entries are skipped."""
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def top2_margin(probs: Sequence[float]) -> float:
    """Difference between the largest and second-largest probabilities.

    For a single-element distribution the margin is the sole probability.
    """
    if len(probs) == 0:
        return 0.0
    ordered = sorted(probs, reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uncertainty.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/uncertainty.py tests/test_uncertainty.py
git commit -m "feat(decode): add entropy/margin uncertainty helpers"
```

---

### Task 3: Policy core types

**Files:**
- Create: `w2s_research/core/policy.py`
- Test: `tests/test_policy_core.py`

**Interfaces:**
- Produces:
  - `class Decision(Enum)` with members `CONTINUE`, `DEFER`.
  - `@dataclass WeakStepState`: `step_index:int`, `entropy:float`, `top1_prob:float`,
    `margin:float`, `top_token_id:int`, `text_so_far:str`.
  - `class DeferralPolicy` with class attr `name:str = "base"` and method
    `decide(self, state: WeakStepState) -> Decision` (raises `NotImplementedError`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_policy_core.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policy_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.core.policy'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/policy.py
"""Deferral-policy contract: the single method an 'idea' implements."""
from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    CONTINUE = "continue"   # accept the weak model's next token
    DEFER = "defer"         # hand the next span to the strong (black-box) model


@dataclass
class WeakStepState:
    """Everything a policy may see about the weak model's current step.

    Deliberately scalar + text only: NO logits/token distributions and NO
    strong-model internals, so a policy cannot peek past the white/black-box line.
    """
    step_index: int      # number of weak tokens already accepted
    entropy: float       # entropy (nats) of the weak next-token distribution
    top1_prob: float     # probability of the greedy token
    margin: float        # top1 - top2 probability
    top_token_id: int    # weak-vocab id of the greedy token
    text_so_far: str     # assistant text generated so far (weak + strong)


class DeferralPolicy:
    """Base class. An idea subclasses this and implements `decide`."""
    name: str = "base"

    def decide(self, state: WeakStepState) -> Decision:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_policy_core.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/policy.py tests/test_policy_core.py
git commit -m "feat(decode): add Decision/WeakStepState/DeferralPolicy contract"
```

---

### Task 4: Model interfaces + test fakes

**Files:**
- Create: `w2s_research/core/interfaces.py`
- Create: `tests/fakes.py`
- Test: `tests/test_fakes.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass WeakStep`: `top_token_id:int`, `text_piece:str`, `entropy:float`,
    `top1_prob:float`, `margin:float`, `is_eos:bool`.
  - `class WeakModel(Protocol)`: `next_step(self, instruction:str, assistant_text:str) -> WeakStep`.
  - `@dataclass StrongOutput`: `text:str`, `finished:bool`.
  - `class StrongModel(Protocol)`:
    `generate(self, instruction:str, assistant_text:str, *, stop:Optional[list[str]], max_tokens:int, temperature:float) -> StrongOutput`.
  - `tests/fakes.py`: `FakeWeakModel` (scripted steps) and `FakeStrongModel` (scripted outputs),
    used by the engine tests in Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fakes.py
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
    assert weak.next_step("inst", "").is_eos is True


def test_fake_strong_returns_scripted_output():
    strong = FakeStrongModel(outputs=[StrongOutput(text="= 4\n", finished=False),
                                      StrongOutput(text="#### 4", finished=True)])
    o0 = strong.generate("inst", "2 + 2", stop=["\n"], max_tokens=16, temperature=0.0)
    assert o0.text == "= 4\n" and o0.finished is False
    o1 = strong.generate("inst", "2 + 2= 4\n", stop=["\n"], max_tokens=16, temperature=0.0)
    assert o1.finished is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fakes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.core.interfaces'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/interfaces.py
"""Model adapter interfaces for collaborative decoding.

The engine depends ONLY on these protocols, so real HF/vLLM adapters and test
fakes are interchangeable. The StrongModel surface is text-in/text-out by
construction (the black-box constraint lives in the type, not in convention).
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class WeakStep:
    """Greedy next-token summary from the white-box weak model."""
    top_token_id: int
    text_piece: str      # marginal text contributed by accepting top_token_id
    entropy: float
    top1_prob: float
    margin: float
    is_eos: bool


@runtime_checkable
class WeakModel(Protocol):
    def next_step(self, instruction: str, assistant_text: str) -> WeakStep:
        """Return the greedy next-token summary given (instruction, assistant_text)."""
        ...


@dataclass
class StrongOutput:
    """Text emitted by the black-box strong model for one span."""
    text: str
    finished: bool       # True iff generation ended on EOS (not a stop string / length cap)


@runtime_checkable
class StrongModel(Protocol):
    def generate(self, instruction: str, assistant_text: str, *,
                 stop: Optional[List[str]], max_tokens: int, temperature: float) -> StrongOutput:
        """Continue the assistant turn as text. No logits/token ids are returned."""
        ...
```

```python
# tests/fakes.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fakes.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/interfaces.py tests/fakes.py tests/test_fakes.py
git commit -m "feat(decode): add WeakModel/StrongModel interfaces and test fakes"
```

---

### Task 5: Baseline deferral-policy ideas

**Files:**
- Create: `w2s_research/ideas/weak_only/__init__.py` (empty)
- Create: `w2s_research/ideas/weak_only/run.py`
- Create: `w2s_research/ideas/strong_only/__init__.py` (empty)
- Create: `w2s_research/ideas/strong_only/run.py`
- Create: `w2s_research/ideas/random_defer/__init__.py` (empty)
- Create: `w2s_research/ideas/random_defer/run.py`
- Create: `w2s_research/ideas/entropy_threshold/__init__.py` (empty)
- Create: `w2s_research/ideas/entropy_threshold/run.py`
- Create: `w2s_research/ideas/margin_threshold/__init__.py` (empty)
- Create: `w2s_research/ideas/margin_threshold/run.py`
- Test: `tests/test_baseline_policies.py`

**Interfaces:**
- Consumes: `Decision`, `WeakStepState`, `DeferralPolicy` (Task 3); `DecodeConfig` (Task 1).
- Produces (per idea module): module attr `IDEA_NAME: str` and
  `build_policy(config: DecodeConfig) -> DeferralPolicy`.
- Policy params read off `config` via `getattr` with defaults:
  `defer_threshold` (entropy τ, default `1.0`), `margin_threshold` (default `0.10`),
  `defer_prob` (random, default `0.5`). (These are extra optional attrs the CLI may set on the
  config object; they are not core `DecodeConfig` fields.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline_policies.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline_policies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.ideas.weak_only.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/ideas/weak_only/run.py
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "weak_only"


class WeakOnly(DeferralPolicy):
    name = "weak_only"
    def decide(self, state):
        return Decision.CONTINUE


def build_policy(config):
    return WeakOnly()
```

```python
# w2s_research/ideas/strong_only/run.py
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "strong_only"


class StrongOnly(DeferralPolicy):
    name = "strong_only"
    def decide(self, state):
        return Decision.DEFER


def build_policy(config):
    return StrongOnly()
```

```python
# w2s_research/ideas/entropy_threshold/run.py
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "entropy_threshold"


class EntropyThreshold(DeferralPolicy):
    name = "entropy_threshold"
    def __init__(self, tau):
        self.tau = tau
    def decide(self, state):
        return Decision.DEFER if state.entropy > self.tau else Decision.CONTINUE


def build_policy(config):
    return EntropyThreshold(tau=getattr(config, "defer_threshold", 1.0))
```

```python
# w2s_research/ideas/margin_threshold/run.py
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "margin_threshold"


class MarginThreshold(DeferralPolicy):
    name = "margin_threshold"
    def __init__(self, tau):
        self.tau = tau
    def decide(self, state):
        return Decision.DEFER if state.margin < self.tau else Decision.CONTINUE


def build_policy(config):
    return MarginThreshold(tau=getattr(config, "margin_threshold", 0.10))
```

```python
# w2s_research/ideas/random_defer/run.py
import random
from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "random_defer"


class RandomDefer(DeferralPolicy):
    name = "random_defer"
    def __init__(self, defer_prob, seed):
        self.defer_prob = defer_prob
        self._rng = random.Random(seed)
    def decide(self, state):
        return Decision.DEFER if self._rng.random() < self.defer_prob else Decision.CONTINUE


def build_policy(config):
    return RandomDefer(defer_prob=getattr(config, "defer_prob", 0.5),
                       seed=getattr(config, "seed", 42))
```

Also create the five empty `__init__.py` files:

```bash
: > w2s_research/ideas/weak_only/__init__.py
: > w2s_research/ideas/strong_only/__init__.py
: > w2s_research/ideas/random_defer/__init__.py
: > w2s_research/ideas/entropy_threshold/__init__.py
: > w2s_research/ideas/margin_threshold/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline_policies.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/ideas/weak_only w2s_research/ideas/strong_only w2s_research/ideas/random_defer w2s_research/ideas/entropy_threshold w2s_research/ideas/margin_threshold tests/test_baseline_policies.py
git commit -m "feat(decode): add five baseline deferral-policy ideas"
```

---

### Task 6: `CollaborativeDecoder` engine

**Files:**
- Create: `w2s_research/core/collab_decode.py`
- Test: `tests/test_collab_decode.py`

**Interfaces:**
- Consumes: `WeakModel`, `StrongModel`, `WeakStep`, `StrongOutput` (Task 4);
  `Decision`, `WeakStepState`, `DeferralPolicy` (Task 3); `DecodeConfig` (Task 1);
  `FakeWeakModel`, `FakeStrongModel` (Task 4 tests).
- Produces:
  - `@dataclass DecodeResult`: `text:str`, `weak_chars:int`, `strong_chars:int`,
    `num_weak_steps:int`, `num_defers:int`, `finished:bool`, plus property
    `weak_fraction -> float`.
  - `class CollaborativeDecoder(weak, strong, policy, config)` with
    `run_example(instruction:str) -> DecodeResult` and
    `run_dataset(instructions:list[str]) -> list[DecodeResult]`.
  - `aggregate_weak_fraction(results:list[DecodeResult]) -> float` (char-weighted).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collab_decode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.core.collab_decode'`

- [ ] **Step 3: Write minimal implementation**

```python
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
                if out.text == "":          # no progress -> stop to avoid an infinite loop
                    finished = True
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collab_decode.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/collab_decode.py tests/test_collab_decode.py
git commit -m "feat(decode): add CollaborativeDecoder engine with char-weighted f_weak"
```

---

### Task 7: Benchmarks (load, prompt, extract, score)

**Files:**
- Create: `w2s_research/core/benchmarks.py`
- Create: `tests/fixtures/gsm8k_tiny.jsonl`
- Create: `tests/fixtures/math_tiny.jsonl`
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- Consumes: `w2s_research.ideas.ue_zeroshot.math_eval_tools.grade_answer` (existing),
  `w2s_research.ideas.ue_zeroshot.math_normalize` (existing).
- Produces:
  - `@dataclass BenchmarkExample`: `question:str`, `answer:str`.
  - `build_instruction(name:str, question:str) -> str`.
  - `extract_answer(name:str, text:str) -> Optional[str]`.
  - `is_correct(name:str, generated_text:str, gold:str) -> bool`.
  - `utility(name:str, generations:list[str], golds:list[str]) -> float`.
  - `load_benchmark(name:str, split:str, limit:Optional[int], jsonl_path:Optional[str]) -> list[BenchmarkExample]`.
    (Loads from a jsonl file if `jsonl_path` is given — used by tests and the prepared-data path;
    otherwise from the `datasets` library.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py
from pathlib import Path
from w2s_research.core.benchmarks import (
    BenchmarkExample, build_instruction, extract_answer, is_correct,
    utility, load_benchmark,
)

FIX = Path(__file__).parent / "fixtures"


def test_build_instruction_mentions_question():
    instr = build_instruction("gsm8k", "What is 2+2?")
    assert "What is 2+2?" in instr


def test_extract_gsm8k_last_number():
    assert extract_answer("gsm8k", "First 3, then ... The answer is 18.") == "18"
    assert extract_answer("gsm8k", "So we get #### 42") == "42"
    assert extract_answer("gsm8k", "no number here") is None


def test_extract_math_boxed():
    assert extract_answer("math", r"thus \boxed{\frac{1}{2}} is final") == r"\frac{1}{2}"
    assert extract_answer("math", "no box") is None


def test_is_correct_gsm8k_numeric():
    assert is_correct("gsm8k", "The answer is 18", "18") is True
    assert is_correct("gsm8k", "The answer is 19", "18") is False


def test_is_correct_math_via_grader():
    assert is_correct("math", r"\boxed{0.5}", r"\frac{1}{2}") is True


def test_utility_is_fraction_correct():
    gens = ["answer 18", "answer 7", "answer 100"]
    golds = ["18", "8", "100"]
    assert utility("gsm8k", gens, golds) == 2 / 3


def test_load_benchmark_from_jsonl(tmp_path):
    exs = load_benchmark("gsm8k", "test", limit=None, jsonl_path=str(FIX / "gsm8k_tiny.jsonl"))
    assert len(exs) == 2
    assert isinstance(exs[0], BenchmarkExample)
    assert exs[0].answer == "18"


def test_load_benchmark_respects_limit():
    exs = load_benchmark("gsm8k", "test", limit=1, jsonl_path=str(FIX / "gsm8k_tiny.jsonl"))
    assert len(exs) == 1
```

- [ ] **Step 2: Create fixtures and run test to verify it fails**

```jsonl
# tests/fixtures/gsm8k_tiny.jsonl
{"question": "Natalia sold clips to 48 friends then half as many. How many total?", "answer": "18"}
{"question": "What is 7 times 8?", "answer": "56"}
```

```jsonl
# tests/fixtures/math_tiny.jsonl
{"question": "Compute 1/2 + 0.", "answer": "\\frac{1}{2}"}
{"question": "What is 2 squared?", "answer": "4"}
```

Run: `python -m pytest tests/test_benchmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.core.benchmarks'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/benchmarks.py
"""Generative benchmarks: load examples, build prompts, extract + score answers.

GSM8K and MATH are scored with the repo's existing sympy-backed grader
(w2s_research.ideas.ue_zeroshot.math_eval_tools.grade_answer), so equivalent
forms (0.5 == 1/2, etc.) count as correct.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from w2s_research.ideas.ue_zeroshot import math_normalize
from w2s_research.ideas.ue_zeroshot.math_eval_tools import grade_answer

SUPPORTED = ("gsm8k", "math")

_GSM8K_INSTRUCTION = (
    "Solve the following grade-school math problem. Show your reasoning, then give the "
    "final answer on its own line in the form '#### <number>'.\n\nProblem: {q}"
)
_MATH_INSTRUCTION = (
    "Solve the following competition math problem. Show your reasoning, then put the final "
    "answer in \\boxed{{}}.\n\nProblem: {q}"
)


@dataclass
class BenchmarkExample:
    question: str
    answer: str


def build_instruction(name: str, question: str) -> str:
    if name == "gsm8k":
        return _GSM8K_INSTRUCTION.format(q=question)
    if name == "math":
        return _MATH_INSTRUCTION.format(q=question)
    raise ValueError(f"Unknown benchmark: {name}")


def _last_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def _last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_answer(name: str, text: str) -> Optional[str]:
    if name == "gsm8k":
        after = text.split("####")[-1] if "####" in text else text
        return _last_number(after)
    if name == "math":
        return _last_boxed(text)
    raise ValueError(f"Unknown benchmark: {name}")


def is_correct(name: str, generated_text: str, gold: str) -> bool:
    pred = extract_answer(name, generated_text)
    if pred is None:
        return False
    if name == "gsm8k":
        gold_num = _last_number(gold) or gold
        return grade_answer(pred, gold_num)
    if name == "math":
        gold_clean = math_normalize.remove_boxed(gold) or gold
        return grade_answer(pred, gold_clean)
    raise ValueError(f"Unknown benchmark: {name}")


def utility(name: str, generations: List[str], golds: List[str]) -> float:
    assert len(generations) == len(golds)
    if not generations:
        return 0.0
    correct = sum(1 for g, gold in zip(generations, golds) if is_correct(name, g, gold))
    return correct / len(generations)


def load_benchmark(name: str, split: str, limit: Optional[int] = None,
                   jsonl_path: Optional[str] = None) -> List[BenchmarkExample]:
    if name not in SUPPORTED:
        raise ValueError(f"Unknown benchmark: {name}")
    if jsonl_path is not None:
        rows = [json.loads(line) for line in open(jsonl_path) if line.strip()]
        exs = [BenchmarkExample(question=r["question"], answer=str(r["answer"])) for r in rows]
        return exs[:limit] if limit else exs
    return _load_from_hf(name, split, limit)


def _load_from_hf(name: str, split: str, limit: Optional[int]) -> List[BenchmarkExample]:
    from datasets import load_dataset  # lazy import (heavy)
    exs: List[BenchmarkExample] = []
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for row in ds:
            gold = row["answer"].split("####")[-1].strip()
            exs.append(BenchmarkExample(question=row["question"], answer=gold))
    elif name == "math":
        ds = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
        for row in ds:
            gold = _last_boxed(row["solution"])
            if gold is None:
                continue
            exs.append(BenchmarkExample(question=row["problem"], answer=gold))
    if limit:
        exs = exs[:limit]
    return exs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmarks.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/benchmarks.py tests/test_benchmarks.py tests/fixtures
git commit -m "feat(decode): add GSM8K/MATH benchmark loading, extraction and scoring"
```

---

### Task 8: Real `HFWeakModel` (white-box, GPU)

**Files:**
- Create: `w2s_research/core/weak_model.py`
- Create: `scripts/smoke_weak_model.py`

**Interfaces:**
- Consumes: `WeakStep` (Task 4), `entropy_of`/`top2_margin` (Task 2).
- Produces: `class HFWeakModel(model_name:str, max_model_len:int=4096, device:str="cuda", dtype="bfloat16")`
  implementing `next_step(instruction:str, assistant_text:str) -> WeakStep`.

This task's deliverable is GPU-only and is verified by a **smoke command on a GPU node**, not by
pytest (unit CI never loads models). The engine/policy tests (Tasks 5–6) already cover behaviour
through the `WeakModel` protocol.

- [ ] **Step 1: Write the implementation**

```python
# w2s_research/core/weak_model.py
"""White-box weak model adapter (HuggingFace Transformers).

Exposes per-step next-token uncertainty by running a forward pass and reading the
final-position logits. Chat-templates the (instruction, assistant_text) using the
weak model's OWN tokenizer with assistant-continuation, so it composes with a
different-tokenizer strong model at the text level.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interfaces import WeakStep
from .uncertainty import entropy_of, top2_margin

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFWeakModel:
    def __init__(self, model_name: str, max_model_len: int = 4096,
                 device: str = "cuda", dtype: str = "bfloat16"):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_DTYPES[dtype],
        ).to(device).eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def _build_ids(self, instruction: str, assistant_text: str):
        messages = [{"role": "user", "content": instruction}]
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, continue_final_message=True,
                add_generation_prompt=False, return_tensors="pt",
            )
        else:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
        return ids[:, -self.max_model_len:].to(self.device)

    @torch.no_grad()
    def next_step(self, instruction: str, assistant_text: str) -> WeakStep:
        ids = self._build_ids(instruction, assistant_text)
        logits = self.model(ids).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top_id = int(probs.argmax().item())

        # entropy in nats and top1-top2 margin, computed in torch then summarised
        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        top2 = torch.topk(probs, k=min(2, probs.numel()))
        top1_prob = float(top2.values[0].item())
        margin = float((top2.values[0] - top2.values[1]).item()) if top2.values.numel() > 1 else top1_prob

        is_eos = top_id == self.eos_token_id
        if is_eos:
            text_piece = ""
        else:
            # marginal text contributed by this token, robust to BPE leading spaces:
            prev = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            after = self.tokenizer.decode(
                torch.cat([ids[0], torch.tensor([top_id], device=self.device)]),
                skip_special_tokens=True,
            )
            text_piece = after[len(prev):]

        return WeakStep(top_token_id=top_id, text_piece=text_piece, entropy=ent,
                        top1_prob=top1_prob, margin=margin, is_eos=is_eos)
```

```python
# scripts/smoke_weak_model.py
"""Smoke test for HFWeakModel — run on a GPU node.

Usage:
    HF_HOME=/scratch2/ml23/smur0075/hf_cache HF_TOKEN=... \
        python scripts/smoke_weak_model.py
"""
from w2s_research.core.weak_model import HFWeakModel
from w2s_research.core.benchmarks import build_instruction

weak = HFWeakModel("meta-llama/Llama-3.2-1B-Instruct")
instr = build_instruction("gsm8k", "What is 2 + 2?")
text = ""
for _ in range(40):
    step = weak.next_step(instr, text)
    if step.is_eos:
        break
    print(f"piece={step.text_piece!r}  entropy={step.entropy:.3f}  margin={step.margin:.3f}")
    text += step.text_piece
print("\nGENERATED:", text)
assert len(text) > 0, "weak model produced no text"
print("OK: HFWeakModel smoke passed")
```

- [ ] **Step 2: Run the GPU smoke test**

Run (on a GPU node — `srun --partition=gpu --gres=gpu:1 --mem=48G --time=0:30:00 --pty bash`, then):
```bash
export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_TOKEN=...   # must have accepted the Llama-3.2 license
.venv/bin/python scripts/smoke_weak_model.py
```
Expected: prints decoded pieces with entropy/margin values, a coherent partial answer, and
`OK: HFWeakModel smoke passed`.

- [ ] **Step 3: Commit**

```bash
git add w2s_research/core/weak_model.py scripts/smoke_weak_model.py
git commit -m "feat(decode): add HFWeakModel white-box adapter + GPU smoke test"
```

---

### Task 9: Real `VLLMStrongModel` (black-box, GPU)

**Files:**
- Create: `w2s_research/core/strong_model.py`
- Create: `scripts/smoke_strong_model.py`

**Interfaces:**
- Consumes: `StrongOutput` (Task 4).
- Produces: `class VLLMStrongModel(model_name:str, gpu_memory_utilization:float=0.6, max_model_len:int=4096)`
  implementing `generate(instruction, assistant_text, *, stop, max_tokens, temperature) -> StrongOutput`.
  Returns text only; `finished=True` iff vLLM stopped on EOS (not a stop string or length).

GPU-only deliverable, verified by a smoke command (not pytest).

- [ ] **Step 1: Write the implementation**

```python
# w2s_research/core/strong_model.py
"""Black-box strong model adapter (vLLM).

Wraps a local vLLM engine but the public surface is text-in / text-out only — no
logits or token ids are returned, enforcing the black-box constraint. Applies the
strong model's OWN chat template with assistant-continuation so it composes with a
different-tokenizer weak model at the text level.
"""
from typing import List, Optional

from vllm import LLM, SamplingParams

from .interfaces import StrongOutput


class VLLMStrongModel:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.6,
                 max_model_len: int = 4096):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _build_prompt(self, instruction: str, assistant_text: str) -> str:
        messages = [{"role": "user", "content": instruction}]
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def generate(self, instruction: str, assistant_text: str, *,
                 stop: Optional[List[str]], max_tokens: int, temperature: float) -> StrongOutput:
        prompt = self._build_prompt(instruction, assistant_text)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            include_stop_str_in_output=True,   # keep the "\n" so assistant_text stays well-formed
        )
        out = self.llm.generate([prompt], params)[0].outputs[0]
        # finished on EOS only when vLLM stopped without matching a stop string and not on length
        finished = (out.finish_reason == "stop") and (out.stop_reason is None)
        return StrongOutput(text=out.text, finished=finished)
```

```python
# scripts/smoke_strong_model.py
"""Smoke test for VLLMStrongModel — run on a GPU node.

Usage:
    HF_HOME=/scratch2/ml23/smur0075/hf_cache python scripts/smoke_strong_model.py
"""
from w2s_research.core.strong_model import VLLMStrongModel
from w2s_research.core.benchmarks import build_instruction

strong = VLLMStrongModel("Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.6)
instr = build_instruction("gsm8k", "What is 2 + 2?")

# span mode: stop at newline, should NOT be finished
span = strong.generate(instr, "", stop=["\n"], max_tokens=64, temperature=0.0)
print("SPAN:", repr(span.text), "finished=", span.finished)

# full mode: no stop, should finish on EOS
full = strong.generate(instr, "", stop=None, max_tokens=256, temperature=0.0)
print("FULL:", repr(full.text[:200]), "finished=", full.finished)
assert full.text.strip(), "strong model produced no text"
print("OK: VLLMStrongModel smoke passed")
```

- [ ] **Step 2: Run the GPU smoke test**

Run (on a GPU node):
```bash
export HF_HOME=/scratch2/ml23/smur0075/hf_cache
.venv/bin/python scripts/smoke_strong_model.py
```
Expected: prints a one-line span (`finished=False`) and a full answer ending naturally
(`finished=True`), then `OK: VLLMStrongModel smoke passed`.

- [ ] **Step 3: Commit**

```bash
git add w2s_research/core/strong_model.py scripts/smoke_strong_model.py
git commit -m "feat(decode): add VLLMStrongModel black-box adapter + GPU smoke test"
```

---

### Task 10: CLI + end-to-end (Gate 1a/1b)

**Files:**
- Create: `w2s_research/decode_cli.py`
- Test: `tests/test_decode_cli.py` (CPU, uses fakes via monkeypatch)

**Interfaces:**
- Consumes: everything above — `DecodeConfig`, `load_benchmark`, `build_instruction`, `utility`,
  `CollaborativeDecoder`, `aggregate_weak_fraction`, idea `build_policy`, `HFWeakModel`, `VLLMStrongModel`.
- Produces: `run_decode(config, idea, jsonl_path=None, weak=None, strong=None) -> dict` returning
  `{"idea", "benchmark", "utility", "weak_token_fraction", "n", "results"}`, and a `main()` argparse CLI
  invoked as `python -m w2s_research.decode_cli --idea <name> --benchmark gsm8k --eval-size N [--tau T]`.
  `weak`/`strong` are injectable so the CLI is testable on CPU with fakes; if `None`, real adapters are built.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decode_cli.py
from pathlib import Path
from w2s_research.core.interfaces import WeakStep, StrongOutput
from w2s_research.core.decode_config import DecodeConfig
from w2s_research.decode_cli import run_decode
from tests.fakes import FakeWeakModel, FakeStrongModel

FIX = Path(__file__).parent / "fixtures"


class CycleWeak:
    """A weak model that always emits the correct gsm8k answer then EOS, per example."""
    def __init__(self, answer):
        self.answer = answer
        self._n = 0
    def next_step(self, instruction, assistant_text):
        # emit "#### <answer>" in one piece, then EOS
        if assistant_text == "":
            return WeakStep(top_token_id=1, text_piece=f"#### {self.answer}",
                            entropy=0.0, top1_prob=1.0, margin=1.0, is_eos=False)
        return WeakStep(top_token_id=-1, text_piece="", entropy=0.0,
                        top1_prob=1.0, margin=1.0, is_eos=True)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decode_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'w2s_research.decode_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/decode_cli.py
"""CLI to run a deferral policy over a benchmark and report (utility, f_weak).

Phase 1: local-only. Computes utility locally against gold answers (the server-side
held-out evaluation arrives in Phase 2). Weak/strong adapters are injectable for testing.
"""
import argparse
import importlib
import json
from typing import Optional

from w2s_research.core.benchmarks import build_instruction, load_benchmark, utility
from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction
from w2s_research.core.decode_config import DecodeConfig


def _load_idea(idea: str):
    return importlib.import_module(f"w2s_research.ideas.{idea}.run")


def run_decode(config: DecodeConfig, idea: str, jsonl_path: Optional[str] = None,
               weak=None, strong=None) -> dict:
    mod = _load_idea(idea)
    policy = mod.build_policy(config)

    examples = load_benchmark(config.benchmark, config.split,
                              limit=config.eval_size, jsonl_path=jsonl_path)

    if weak is None:
        from w2s_research.core.weak_model import HFWeakModel
        weak = HFWeakModel(config.weak_model, max_model_len=config.weak_max_model_len)
    if strong is None:
        from w2s_research.core.strong_model import VLLMStrongModel
        strong = VLLMStrongModel(config.strong_model,
                                 gpu_memory_utilization=config.strong_gpu_memory_utilization,
                                 max_model_len=config.strong_max_model_len)

    decoder = CollaborativeDecoder(weak, strong, policy, config)
    instructions = [build_instruction(config.benchmark, ex.question) for ex in examples]
    results = decoder.run_dataset(instructions)

    generations = [r.text for r in results]
    golds = [ex.answer for ex in examples]
    return {
        "idea": mod.IDEA_NAME,
        "benchmark": config.benchmark,
        "utility": utility(config.benchmark, generations, golds),
        "weak_token_fraction": aggregate_weak_fraction(results),
        "n": len(examples),
        "results": [
            {"text": r.text, "weak_chars": r.weak_chars, "strong_chars": r.strong_chars,
             "num_defers": r.num_defers, "finished": r.finished}
            for r in results
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Run a collaborative-decoding policy on a benchmark")
    p.add_argument("--idea", required=True)
    p.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math"])
    p.add_argument("--eval-size", type=int, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--tau", type=float, default=None, help="entropy/margin threshold for the idea")
    p.add_argument("--defer-prob", type=float, default=None, help="random_defer probability")
    p.add_argument("--out", default=None, help="optional path to write results JSON")
    args = p.parse_args()

    cfg = DecodeConfig(benchmark=args.benchmark, eval_size=args.eval_size, split=args.split)
    if args.tau is not None:
        cfg.defer_threshold = args.tau          # consumed by entropy_threshold
        cfg.margin_threshold = args.tau          # consumed by margin_threshold
    if args.defer_prob is not None:
        cfg.defer_prob = args.defer_prob

    out = run_decode(cfg, idea=args.idea)
    summary = {k: out[k] for k in ("idea", "benchmark", "utility", "weak_token_fraction", "n")}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decode_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (Tasks 1–7, 10).

- [ ] **Step 6: Commit**

```bash
git add w2s_research/decode_cli.py tests/test_decode_cli.py
git commit -m "feat(decode): add decode CLI with injectable adapters (CPU-testable)"
```

- [ ] **Step 7: GPU end-to-end (Gate 1a + 1b) — run on a GPU node**

```bash
export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_TOKEN=...
# Baselines define the utility band:
.venv/bin/python -m w2s_research.decode_cli --idea weak_only   --benchmark gsm8k --eval-size 100 --out /scratch2/ml23/smur0075/decode_weak.json
.venv/bin/python -m w2s_research.decode_cli --idea strong_only --benchmark gsm8k --eval-size 100 --out /scratch2/ml23/smur0075/decode_strong.json
# Frontier sweep with the entropy policy:
for T in 0.3 0.6 1.0 1.5 2.0; do
  .venv/bin/python -m w2s_research.decode_cli --idea entropy_threshold --benchmark gsm8k --eval-size 100 --tau $T
done
```
**GATE 1a:** `weak_only` utility (`U_weak`) and `strong_only` utility (`U_strong`) print, and
`U_strong − U_weak` is a meaningful gap (expected ~0.40+ on GSM8K). `weak_only` outputs are coherent
and parseable (utility clearly > 0). If the 1B weak is incoherent or the gap is small, STOP and
revisit model sizes (see spec Risk #1).
**GATE 1b:** the entropy sweep produces intermediate `(utility, weak_token_fraction)` points that lie
between the two baselines — i.e., raising `τ` trades weak fraction up and utility down — confirming the
cross-tokenizer span handoff produces coherent collaborative text.

---

## Self-Review

**1. Spec coverage:**
- Engine (spec §3.3) → Task 6. Strong-model black-box interface (§3.4) → Tasks 4 + 9.
- Weak white-box (§3.1) → Task 8. Policy contract (§3.5) → Tasks 3 + 5. Benchmarks/utility (§3.6) → Task 7.
- Char-weighted `f_weak` (Global Constraints) → Task 6 (`aggregate_weak_fraction`, `DecodeResult.weak_fraction`).
- Metric `utility_recovery`/`R_bar` (§4) → computed at Gate 1a/1b in Task 10 from the two baselines;
  full server-side metric + leaderboard is Phase 2 (out of scope here) — noted in CLI docstring.
- Cross-tokenizer + cross-chat-template handoff → Tasks 8/9 (`continue_final_message`) + Task 6 (text-level).
- Build-order Gate 1a/1b (spec §8) → Task 10 Step 7.
- Models, HF_HOME, HF_TOKEN gating → Global Constraints + Tasks 8/10 smoke commands.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to" — every code step is complete.
The only `...` are inside `Protocol` method bodies, which is correct Python for protocols.

**3. Type consistency:** `WeakStep` fields (`top_token_id`, `text_piece`, `entropy`, `top1_prob`,
`margin`, `is_eos`) are identical across Tasks 4, 5, 6, 8. `StrongOutput(text, finished)` identical in
Tasks 4, 6, 9. `WeakStepState` fields identical in Tasks 3, 5, 6. `DecodeConfig` fields referenced in
Tasks 6/10 all exist in Task 1 (extra policy attrs `defer_threshold`/`margin_threshold`/`defer_prob`
are set dynamically via `setattr` and read via `getattr` with defaults — intentional, documented in
Tasks 5 + 10). `build_policy(config)` / `IDEA_NAME` contract identical across Task 5 ideas and Task 10.

**Deviation from spec noted:** the spec §6 suggested adapting `RunConfig`; this plan instead adds a
separate `DecodeConfig` (cleaner; leaves the training `RunConfig` and archived training ideas
untouched). Functionally equivalent; flagged for the spec to be updated if desired.
