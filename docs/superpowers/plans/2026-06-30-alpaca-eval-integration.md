# AlpacaEval Open-Ended Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an AlpacaEval-2.0-style open-ended benchmark to the collaborative-decoding sandbox, scored by winrate-vs-reference using the local Gemma-4-31B judge, integrated into the existing engine + policy search.

**Architecture:** A `VLLMJudge` HTTP client talks to the local OpenAI-compatible Gemma server. An `alpaca_eval` module loads the 805 AlpacaEval prompts + GPT-4-turbo baseline references. Utility on open-ended prompts = position-swapped pairwise winrate of a method's generations vs the reference, judged by Gemma. We log every raw pairwise judgment + both answers' lengths, so plain winrate (live metric) and length-controlled winrate (follow-up) are both computable from one set of judge calls. The existing `recovery = (U_M − U_weak)/(U_strong − U_weak)` framework is unchanged; only the utility function differs by benchmark.

**Tech Stack:** Python 3.12, HF `datasets`, stdlib `urllib`/`concurrent.futures` (no new heavy deps), `numpy` (already present) for LC winrate. CPU test venv: `~/venvs/w2s-decode/bin/python`. GPU venv: `~/venvs/w2s-decode-gpu`.

## Global Constraints

- **No fine-tuning; weak=white-box, strong=black-box; text-level handoff** (project invariant — unchanged here).
- **The judge is external and must never see logits** — it only compares output *text* vs reference *text*.
- **Judge endpoint is configurable**: default `JUDGE_URL=http://m3u006:8001/v1`, `JUDGE_MODEL=google/gemma-4-31B-it` (env-overridable). It is a separate SLURM job and may move — never hard-code only one literal without an env override.
- **The engine remains the sole measurer of `f_weak`** (char-weighted). The judge measures *only* utility.
- **CPU-testable**: every unit is tested with an injected fake judge (`chat_fn`) — no GPU, no network in unit tests.
- **`utility()` in `benchmarks.py` stays CPU-pure** (math exact-match only). Judge-based scoring lives in `alpaca_eval.py`; runners dispatch by benchmark name.
- Run all unit tests with `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest`.

---

### Task 1: `VLLMJudge` client (pairwise, position-swapped winrate)

**Files:**
- Create: `w2s_research/core/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `VLLMJudge(base_url=DEFAULT_JUDGE_URL, model=DEFAULT_JUDGE_MODEL, max_workers=8, timeout=60, chat_fn=None)`
  - `.compare(instruction: str, output_a: str, output_b: str) -> str` returns `"A" | "B" | "tie"`
  - `.winrate_one(instruction, candidate, reference) -> dict` → `{"win": float, "cand_len": int, "ref_len": int, "verdicts": [str, str]}`
  - `.winrate(instructions: list[str], candidates: list[str], references: list[str]) -> dict` → `{"winrate": float, "per_example": list[dict]}`
  - module consts `DEFAULT_JUDGE_URL`, `DEFAULT_JUDGE_MODEL`, function `_parse_verdict(reply: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge.py
from w2s_research.core.judge import VLLMJudge, _parse_verdict


def test_parse_verdict():
    assert _parse_verdict("A") == "A"
    assert _parse_verdict(" The better answer is B.") == "B"
    assert _parse_verdict("tie") == "tie"
    assert _parse_verdict("TIE - both equal") == "tie"
    assert _parse_verdict("garbage") == "tie"   # default to tie when unclear


def test_winrate_one_position_swapped():
    # Fake judge that ALWAYS says the first-listed answer (A) is better -> pure position bias.
    j = VLLMJudge(chat_fn=lambda prompt: "A")
    r = j.winrate_one("inst", "cand", "reference_text")
    # call1 (A=cand) -> cand wins; call2 (A=ref) -> ref wins. Swapping cancels bias => 0.5
    assert r["win"] == 0.5
    assert r["cand_len"] == len("cand")
    assert r["ref_len"] == len("reference_text")
    assert r["verdicts"] == ["A", "A"]


def test_winrate_one_genuine_preference():
    # Judge prefers whichever side contains "good" regardless of position.
    def chat(prompt):
        # crude: 'A' if Response A block mentions good else 'B'
        a_block = prompt.split("Response A:")[1].split("Response B:")[0]
        return "A" if "good" in a_block else "B"
    j = VLLMJudge(chat_fn=chat)
    r = j.winrate_one("inst", "good answer", "bad answer")
    assert r["win"] == 1.0          # candidate preferred in both orderings


def test_winrate_aggregates():
    j = VLLMJudge(chat_fn=lambda p: "A", max_workers=2)
    out = j.winrate(["i1", "i2"], ["c1", "c2"], ["r1", "r2"])
    assert out["winrate"] == 0.5
    assert len(out["per_example"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_judge.py -v`
Expected: FAIL (ModuleNotFoundError: w2s_research.core.judge)

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/judge.py
"""Black-box LLM judge over an OpenAI-compatible vLLM server (local Gemma).

Measures ONLY utility: it compares a method's output TEXT against a reference
TEXT. It never sees logits or model internals. Pairwise verdicts are
position-swapped (judge twice, A/B reversed) so per-position bias cancels.
"""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_JUDGE_URL = os.environ.get("JUDGE_URL", "http://m3u006:8001/v1")
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")

_PAIRWISE_PROMPT = """You are comparing two AI assistant responses to an instruction.

Instruction:
{instruction}

Response A:
{a}

Response B:
{b}

Which response is better overall (helpfulness, accuracy, relevance)? \
Answer with ONLY a single letter: A or B. If they are genuinely equal, answer: tie."""


def _parse_verdict(reply: str) -> str:
    r = (reply or "").strip().upper()
    if "TIE" in r:
        return "tie"
    for ch in r:
        if ch == "A":
            return "A"
        if ch == "B":
            return "B"
    return "tie"


class VLLMJudge:
    def __init__(self, base_url=DEFAULT_JUDGE_URL, model=DEFAULT_JUDGE_MODEL,
                 max_workers=8, timeout=60, chat_fn=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        self._chat_fn = chat_fn          # injectable for tests

    def _http_chat(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"content-type": "application/json"})
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.loads(resp.read())
                return d["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 1:
                    return "tie"        # judge failure -> neutral, never crash the run

    def _chat(self, prompt: str) -> str:
        return self._chat_fn(prompt) if self._chat_fn else self._http_chat(prompt)

    def compare(self, instruction: str, output_a: str, output_b: str) -> str:
        prompt = _PAIRWISE_PROMPT.format(instruction=instruction, a=output_a, b=output_b)
        return _parse_verdict(self._chat(prompt))

    def winrate_one(self, instruction: str, candidate: str, reference: str) -> dict:
        v1 = self.compare(instruction, candidate, reference)   # A=candidate
        v2 = self.compare(instruction, reference, candidate)   # A=reference
        s1 = {"A": 1.0, "B": 0.0, "tie": 0.5}[v1]
        s2 = {"A": 0.0, "B": 1.0, "tie": 0.5}[v2]
        return {"win": (s1 + s2) / 2, "cand_len": len(candidate),
                "ref_len": len(reference), "verdicts": [v1, v2]}

    def winrate(self, instructions, candidates, references) -> dict:
        triples = list(zip(instructions, candidates, references))
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            per = list(ex.map(lambda t: self.winrate_one(*t), triples))
        wr = sum(p["win"] for p in per) / len(per) if per else 0.0
        return {"winrate": wr, "per_example": per}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_judge.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/judge.py tests/test_judge.py
git commit -m "feat(judge): VLLMJudge pairwise position-swapped winrate client"
```

---

### Task 2: Winrate flavors (plain + length-controlled)

**Files:**
- Create: `w2s_research/core/winrate.py`
- Test: `tests/test_winrate.py`

**Interfaces:**
- Consumes: `per_example` records from Task 1 (`{"win","cand_len","ref_len",...}`).
- Produces:
  - `plain_winrate(per_example: list[dict]) -> float`
  - `lc_winrate(per_example: list[dict]) -> float` (length-controlled: logistic fit of win on length-difference, evaluated at zero length-difference)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_winrate.py
from w2s_research.core.winrate import plain_winrate, lc_winrate


def test_plain_winrate():
    per = [{"win": 1.0, "cand_len": 10, "ref_len": 10},
           {"win": 0.0, "cand_len": 10, "ref_len": 10},
           {"win": 0.5, "cand_len": 10, "ref_len": 10}]
    assert abs(plain_winrate(per) - 0.5) < 1e-9


def test_lc_winrate_penalizes_length_driven_wins():
    # Wins occur ONLY when the candidate is much longer -> length-driven.
    # Plain winrate is high; LC (at zero length diff) should be markedly lower.
    per = []
    for i in range(40):
        longer = i % 2 == 0
        per.append({"win": 1.0 if longer else 0.0,
                    "cand_len": 400 if longer else 100,
                    "ref_len": 100 if longer else 400})
    assert plain_winrate(per) == 0.5      # balanced by construction
    # Build a length-confounded set: candidate longer AND wins.
    per2 = [{"win": 1.0, "cand_len": 500, "ref_len": 100} for _ in range(20)]
    per2 += [{"win": 0.0, "cand_len": 100, "ref_len": 500} for _ in range(20)]
    assert plain_winrate(per2) == 0.5
    lc = lc_winrate(per2)
    assert 0.0 <= lc <= 1.0               # well-defined probability
    assert abs(lc - 0.5) < 0.25           # length effect removed -> near 0.5


def test_lc_winrate_handles_degenerate_input():
    assert lc_winrate([]) == 0.0
    # zero length variance -> fall back to plain winrate
    per = [{"win": 1.0, "cand_len": 10, "ref_len": 10} for _ in range(5)]
    assert abs(lc_winrate(per) - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_winrate.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/winrate.py
"""Winrate aggregations over per-example judge records.

plain_winrate: mean win.
lc_winrate (length-controlled, AlpacaEval-2.0 spirit): fit a logistic model
win ~ sigmoid(b0 + b1 * z), where z is the standardized (cand_len - ref_len),
then report the predicted win probability at length-difference = 0. This
removes the judge's systematic preference for longer answers.
"""
import numpy as np


def plain_winrate(per_example):
    if not per_example:
        return 0.0
    return sum(p["win"] for p in per_example) / len(per_example)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lc_winrate(per_example, steps=2000, lr=0.1):
    if not per_example:
        return 0.0
    diff = np.array([p["cand_len"] - p["ref_len"] for p in per_example], dtype=float)
    y = np.array([p["win"] for p in per_example], dtype=float)
    mu, sd = diff.mean(), diff.std()
    if sd < 1e-9:                         # no length variation -> LC == plain
        return float(y.mean())
    z = (diff - mu) / sd
    b0, b1 = 0.0, 0.0
    n = len(y)
    for _ in range(steps):                # gradient descent on logistic NLL
        p = _sigmoid(b0 + b1 * z)
        g0 = np.mean(p - y)
        g1 = np.mean((p - y) * z)
        b0 -= lr * g0
        b1 -= lr * g1
    z0 = (0.0 - mu) / sd                  # standardized value of "equal length"
    return float(_sigmoid(b0 + b1 * z0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_winrate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add w2s_research/core/winrate.py tests/test_winrate.py
git commit -m "feat(winrate): plain + length-controlled winrate aggregations"
```

---

### Task 3: AlpacaEval loader + reference scoring

**Files:**
- Create: `w2s_research/core/alpaca_eval.py`
- Test: `tests/test_alpaca_eval.py`

**Interfaces:**
- Consumes: `VLLMJudge.winrate` (Task 1).
- Produces:
  - `@dataclass AlpacaExample(instruction: str, reference: str)`
  - `load_alpaca_eval(limit=None, config="alpaca_eval_gpt4_baseline") -> list[AlpacaExample]`
  - `score_generations(judge, instructions: list[str], generations: list[str], references: list[str]) -> dict` → `{"winrate": float, "per_example": [...]}` (pass-through to `judge.winrate`, kept as the benchmark-level seam)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_alpaca_eval.py
from w2s_research.core.alpaca_eval import AlpacaExample, score_generations
from w2s_research.core.judge import VLLMJudge


def test_score_generations_uses_judge():
    # judge prefers the response containing "STRONG"
    def chat(prompt):
        a = prompt.split("Response A:")[1].split("Response B:")[0]
        return "A" if "STRONG" in a else "B"
    judge = VLLMJudge(chat_fn=chat)
    instr = ["q1", "q2"]
    gens = ["STRONG ans", "weak ans"]
    refs = ["ref one", "STRONG ref"]
    out = score_generations(judge, instr, gens, refs)
    # gen1 beats ref1 (gen has STRONG) -> win 1.0 ; gen2 loses to ref2 -> 0.0
    assert out["per_example"][0]["win"] == 1.0
    assert out["per_example"][1]["win"] == 0.0
    assert abs(out["winrate"] - 0.5) < 1e-9


def test_alpaca_example_shape():
    ex = AlpacaExample(instruction="write a poem", reference="roses are red")
    assert ex.instruction and ex.reference
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_alpaca_eval.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# w2s_research/core/alpaca_eval.py
"""AlpacaEval open-ended benchmark: load prompts + reference outputs, score by judge winrate.

Utility = winrate of a method's generations vs the AlpacaEval baseline reference
outputs (GPT-4-turbo, the AlpacaEval-2.0 baseline), judged by the local Gemma judge.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AlpacaExample:
    instruction: str
    reference: str


def load_alpaca_eval(limit: Optional[int] = None,
                     config: str = "alpaca_eval_gpt4_baseline") -> List[AlpacaExample]:
    from datasets import load_dataset            # lazy (heavy)
    ds = load_dataset("tatsu-lab/alpaca_eval", config, split="eval",
                      trust_remote_code=True)
    exs = [AlpacaExample(instruction=r["instruction"], reference=r["output"]) for r in ds]
    return exs[:limit] if limit is not None else exs


def score_generations(judge, instructions, generations, references) -> dict:
    return judge.winrate(instructions, generations, references)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_alpaca_eval.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Verify the dataset loads (integration check, network needed)**

Run:
```bash
HF_HOME=/scratch2/ml23/smur0075/hf_cache PYTHONPATH=. ~/venvs/w2s-decode-gpu/bin/python -c "
from w2s_research.core.alpaca_eval import load_alpaca_eval
exs = load_alpaca_eval(limit=3)
assert len(exs) == 3
for e in exs:
    assert e.instruction and e.reference
print('loaded', len(exs), 'OK; example instr:', exs[0].instruction[:60])
"
```
Expected: prints "loaded 3 OK; ...". If the config name errors, list configs with
`python -c "from datasets import get_dataset_config_names as g; print(g('tatsu-lab/alpaca_eval'))"`
and use the GPT-4-baseline config name it reports (update the `config` default).

- [ ] **Step 6: Commit**

```bash
git add w2s_research/core/alpaca_eval.py tests/test_alpaca_eval.py
git commit -m "feat(alpaca_eval): loader + judge-winrate scoring"
```

---

### Task 4: Register `alpaca_eval` in `benchmarks.py`

**Files:**
- Modify: `w2s_research/core/benchmarks.py`
- Test: `tests/test_benchmarks_alpaca.py`

**Interfaces:**
- Consumes: `load_alpaca_eval` (Task 3).
- Produces (changes to existing functions):
  - `SUPPORTED` now includes `"alpaca_eval"`.
  - `build_instruction("alpaca_eval", q)` returns `q` unchanged (raw instruction).
  - `load_benchmark("alpaca_eval", split, limit)` returns `list[BenchmarkExample]` with `question=instruction`, `answer=reference`.
  - `utility("alpaca_eval", ...)` raises `ValueError` (judge-scored, not CPU) — callers must route alpaca_eval through `alpaca_eval.score_generations`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks_alpaca.py
import pytest
from w2s_research.core import benchmarks as B


def test_alpaca_in_supported():
    assert "alpaca_eval" in B.SUPPORTED


def test_build_instruction_raw():
    assert B.build_instruction("alpaca_eval", "Write a haiku") == "Write a haiku"


def test_utility_alpaca_rejects():
    with pytest.raises(ValueError):
        B.utility("alpaca_eval", ["a"], ["b"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_benchmarks_alpaca.py -v`
Expected: FAIL (alpaca_eval not in SUPPORTED; build_instruction raises ValueError)

- [ ] **Step 3: Write minimal implementation**

In `w2s_research/core/benchmarks.py`:

Change the constant:
```python
SUPPORTED = ("gsm8k", "math", "alpaca_eval")
```

In `build_instruction`, add before the final `raise`:
```python
    if name == "alpaca_eval":
        return question
```

In `utility`, add at the top of the function body (before the math logic):
```python
    if name == "alpaca_eval":
        raise ValueError("alpaca_eval utility is judge-scored; "
                         "use w2s_research.core.alpaca_eval.score_generations")
```

In `load_benchmark`, after the `jsonl_path` block and before `_load_from_hf`, add:
```python
    if name == "alpaca_eval":
        from w2s_research.core.alpaca_eval import load_alpaca_eval
        exs = load_alpaca_eval(limit=limit)
        return [BenchmarkExample(question=e.instruction, answer=e.reference) for e in exs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_benchmarks_alpaca.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full CPU suite (no regressions)**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest -q`
Expected: all prior tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add w2s_research/core/benchmarks.py tests/test_benchmarks_alpaca.py
git commit -m "feat(benchmarks): register alpaca_eval (raw prompts, judge-scored utility)"
```

---

### Task 5: Benchmark-aware scoring in `policy_search.py`

**Files:**
- Modify: `scripts/policy_search.py`
- Test: `tests/test_policy_search_alpaca.py`

**Interfaces:**
- Consumes: `VLLMJudge` (Task 1), `score_generations` (Task 3), `plain_winrate`/`lc_winrate` (Task 2).
- Produces (refactor): `run_one(weak, strong, instructions, golds, benchmark, spec, judge=None, winrate_mode="lc") -> dict`. When `benchmark == "alpaca_eval"`, it computes BOTH `winrate_plain` and `winrate_lc` from the judge records and sets `utility` to the chosen one (**default `lc` — length-controlled is the primary metric**); the dict also carries `"_judge_per_example"`. Otherwise unchanged (math exact-match). `main()` builds a `VLLMJudge` when `--benchmark alpaca_eval`, adds `--winrate-mode {plain,lc}` (default `lc`), loads alpaca examples, threads `judge=`/`winrate_mode=` through baselines + every config; `record()` persists both `winrate_plain` and `winrate_lc` and the per-example judge records. **Recovery is therefore computed on LC winrate by default.**

- [ ] **Step 1: Write the failing test** (run_one routes alpaca_eval through the judge)

```python
# tests/test_policy_search_alpaca.py
import sys
sys.path.insert(0, "scripts")
import policy_search
from w2s_research.core.judge import VLLMJudge
from tests.fakes import FakeWeakModel, FakeStrongModel
from w2s_research.core.interfaces import WeakStep, StrongOutput


def _weak():
    return FakeWeakModel([
        WeakStep(top_token_id=5, text_piece="hello ", entropy=0.1, top1_prob=0.9, margin=0.8, is_eos=False),
        WeakStep(top_token_id=6, text_piece="world", entropy=0.1, top1_prob=0.9, margin=0.8, is_eos=True),
    ])


def test_run_one_alpaca_uses_lc_winrate():
    judge = VLLMJudge(chat_fn=lambda p: "A")   # always prefers Response A (position-swapped -> 0.5)
    weak, strong = _weak(), FakeStrongModel([])
    spec = {"idea": "weak_only", "params": {}, "span_max": 64}
    m = policy_search.run_one(weak, strong, ["say hi"], ["reference answer"],
                              "alpaca_eval", spec, judge=judge)   # default winrate_mode="lc"
    assert "winrate_plain" in m and "winrate_lc" in m
    assert m["utility"] == m["winrate_lc"]           # LC is the primary metric
    assert abs(m["utility"] - 0.5) < 1e-9            # position-swap cancels -> 0.5
    assert len(m["_judge_per_example"]) == 1


def test_run_one_math_unchanged():
    weak = FakeWeakModel([
        WeakStep(top_token_id=5, text_piece="#### 7", entropy=0.1, top1_prob=0.9, margin=0.8, is_eos=False),
        WeakStep(top_token_id=6, text_piece="", entropy=0.0, top1_prob=1.0, margin=1.0, is_eos=True),
    ])
    spec = {"idea": "weak_only", "params": {}, "span_max": 64}
    m = policy_search.run_one(weak, FakeStrongModel([]), ["q"], ["7"], "gsm8k", spec)
    assert m["utility"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_policy_search_alpaca.py -v`
Expected: FAIL (run_one has no `judge` kwarg / doesn't branch on alpaca_eval)

- [ ] **Step 3: Modify `run_one`**

Replace the body of `run_one` in `scripts/policy_search.py` with:

```python
def run_one(weak, strong, instructions, golds, benchmark, spec, judge=None, winrate_mode="lc"):
    """Run one deferral-policy configuration; return a metrics dict (+ generations)."""
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction

    cfg = DecodeConfig(benchmark=benchmark, eval_size=len(instructions))
    cfg.span_stop = spec.get("span_stop", ["\n"])
    cfg.span_max_tokens = spec.get("span_max", 256)
    for k, v in spec.get("params", {}).items():
        setattr(cfg, k, v)

    mod = importlib.import_module(f"w2s_research.ideas.{spec['idea']}.run")
    policy = mod.build_policy(cfg)
    dec = CollaborativeDecoder(weak, strong, policy, cfg)
    results = dec.run_dataset(instructions)
    gens = [r.text for r in results]

    out = {
        "weak_token_fraction": aggregate_weak_fraction(results),
        "avg_defers": sum(r.num_defers for r in results) / len(results),
        "avg_weak_steps": sum(r.num_weak_steps for r in results) / len(results),
        "finished_frac": sum(1 for r in results if r.finished) / len(results),
        "_generations": gens,
    }
    if benchmark == "alpaca_eval":
        from w2s_research.core.alpaca_eval import score_generations
        from w2s_research.core.winrate import plain_winrate, lc_winrate
        scored = score_generations(judge, instructions, gens, golds)
        out["winrate_plain"] = plain_winrate(scored["per_example"])
        out["winrate_lc"] = lc_winrate(scored["per_example"])
        out["utility"] = out["winrate_lc"] if winrate_mode == "lc" else out["winrate_plain"]
        out["_judge_per_example"] = scored["per_example"]
    else:
        from w2s_research.core.benchmarks import utility
        out["utility"] = utility(benchmark, gens, golds)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_policy_search_alpaca.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire `main()` for alpaca_eval**

In `scripts/policy_search.py` `main()`:

(a) add `"alpaca_eval"` to the `--benchmark` choices and add the winrate-mode flag:
```python
    ap.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math", "alpaca_eval"])
    ap.add_argument("--winrate-mode", default="lc", choices=["plain", "lc"],
                    help="alpaca_eval utility: length-controlled (lc, default) or plain winrate")
```

(b) after models are loaded and before "measuring baselines", build the judge:
```python
    judge = None
    if bench == "alpaca_eval":
        from w2s_research.core.judge import VLLMJudge
        judge = VLLMJudge()
        print(f"[search] judge: {judge.model} @ {judge.base_url}  winrate_mode={args.winrate_mode}", flush=True)
```

(c) thread `judge=judge, winrate_mode=args.winrate_mode` into the two baseline `run_one(...)` calls and the `safe_run`/`run_one` used in the loops. Concretely, change `safe_run` to pass them:
```python
    def safe_run(spec, phase):
        try:
            m = run_one(weak, strong, instrs, golds, bench, spec,
                        judge=judge, winrate_mode=args.winrate_mode)
            return record(spec, m, phase)
        except Exception as e:
            print(f"[search] ERROR on {spec}: {e!r}", flush=True)
            return None
```
and the baseline calls:
```python
    uw = run_one(weak, strong, instrs, golds, bench,
                 {"idea": "weak_only", "params": {}, "span_max": 256},
                 judge=judge, winrate_mode=args.winrate_mode)["utility"]
    us = run_one(weak, strong, instrs, golds, bench,
                 {"idea": "strong_only", "params": {}, "span_max": 1024,
                  "span_stop": None}, judge=judge, winrate_mode=args.winrate_mode)["utility"]
```

(d) in `record()`, persist both winrates and judge records when present. After building `row`, add:
```python
        for k in ("winrate_plain", "winrate_lc"):
            if k in metrics:
                row[k] = round(metrics[k], 4)
```
and when writing the `meeting_bar` file, include `metrics.get("_judge_per_example")`:
```python
                json.dump({"row": row, "generations": metrics["_generations"],
                           "judge_per_example": metrics.get("_judge_per_example")}, g, indent=2)
```

(e) when `bench == "alpaca_eval"`, restrict the curated specs to the transferable policies (drop the math-only `context_gate`/`answer_protect`). After `specs = curated_specs()`:
```python
    if bench == "alpaca_eval":
        specs = [s for s in specs if s["idea"] not in ("context_gate", "answer_protect")]
```

- [ ] **Step 6: Re-run the alpaca + full suites**

Run: `PYTHONPATH=. ~/venvs/w2s-decode/bin/python -m pytest tests/test_policy_search_alpaca.py tests/ -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/policy_search.py tests/test_policy_search_alpaca.py
git commit -m "feat(search): benchmark-aware judge scoring for alpaca_eval (plain+LC winrate)"
```

---

### Task 6: GPU + judge end-to-end smoke

**Files:**
- Create: `scripts/smoke_alpaca_judge.py`

**Interfaces:**
- Consumes: everything above + real weak/strong models + the live judge.
- Produces: a runnable smoke that loads models, generates a few collaborative outputs on real AlpacaEval prompts, scores them against the references via the live Gemma judge, and prints winrate + LC winrate. (Manual validation, not a unit test — guarded with `if __name__ == "__main__"` because loading the HF weak model forces vLLM `spawn`.)

- [ ] **Step 1: Write the smoke script**

```python
# scripts/smoke_alpaca_judge.py
"""GPU+judge smoke: 3 AlpacaEval prompts through the engine, scored by the Gemma judge.

Run on a GPU node with HF_HOME + HF_TOKEN set and the judge server reachable:
    python scripts/smoke_alpaca_judge.py
"""

def main():
    from w2s_research.core.decode_config import DecodeConfig
    from w2s_research.core.weak_model import HFWeakModel
    from w2s_research.core.strong_model import VLLMStrongModel
    from w2s_research.core.benchmarks import load_benchmark, build_instruction
    from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction
    from w2s_research.core.judge import VLLMJudge
    from w2s_research.core.alpaca_eval import score_generations
    from w2s_research.core.winrate import plain_winrate, lc_winrate
    from w2s_research.ideas.entropy_threshold.run import build_policy

    exs = load_benchmark("alpaca_eval", "eval", limit=3)
    instrs = [build_instruction("alpaca_eval", e.question) for e in exs]
    refs = [e.answer for e in exs]

    base = DecodeConfig(benchmark="alpaca_eval", eval_size=3)
    weak = HFWeakModel(base.weak_model, max_model_len=base.weak_max_model_len)
    strong = VLLMStrongModel(base.strong_model,
                             gpu_memory_utilization=base.strong_gpu_memory_utilization,
                             max_model_len=base.strong_max_model_len)
    base.defer_threshold = 0.5
    dec = CollaborativeDecoder(weak, strong, build_policy(base), base)
    results = dec.run_dataset(instrs)
    gens = [r.text for r in results]
    print("f_weak =", round(aggregate_weak_fraction(results), 3))

    judge = VLLMJudge()
    print("judge:", judge.model, "@", judge.base_url)
    scored = score_generations(judge, instrs, gens, refs)
    print("winrate(plain) =", round(plain_winrate(scored["per_example"]), 3))
    print("winrate(LC)    =", round(lc_winrate(scored["per_example"]), 3))
    for i, p in enumerate(scored["per_example"]):
        print(f"  ex{i}: win={p['win']} cand_len={p['cand_len']} ref_len={p['ref_len']} verdicts={p['verdicts']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke on the GPU node**

Run:
```bash
HF_HOME=/scratch2/ml23/smur0075/hf_cache PYTHONPATH=. \
  ~/venvs/w2s-decode-gpu/bin/python scripts/smoke_alpaca_judge.py
```
Expected: prints an `f_weak` in (0,1), a plain winrate and an LC winrate in [0,1], and 3 per-example lines with two verdicts each. No crash; judge reachable.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_alpaca_judge.py
git commit -m "test(alpaca): GPU+judge end-to-end smoke"
```

---

## Notes for the launch run (after the plan is implemented)

The full AlpacaEval policy search is launched like the GSM8K one but with the judge:
```bash
HF_HOME=/scratch2/ml23/smur0075/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONPATH=<repo> \
setsid nohup ~/venvs/w2s-decode-gpu/bin/python scripts/policy_search.py \
  --benchmark alpaca_eval --eval-size 80 --max-seconds 25200 \
  --out /scratch2/ml23/smur0075/w2s_decode_runs/alpaca_<jobid> > <out>/search.log 2>&1 < /dev/null &
```
`baselines.json` will hold U_weak/U_strong as winrates-vs-reference; `frontier.json`/`best.json` rank by
`f_weak @ recovery≥0.98` exactly as before. Each `meeting_bar/*.json` carries the raw judge per-example
records, so LC winrate is recomputable without re-judging.
```
```
