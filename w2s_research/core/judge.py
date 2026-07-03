"""Black-box LLM judge over an OpenAI-compatible vLLM server (local Gemma).

Measures ONLY utility: it compares a method's output TEXT against a reference
TEXT. It never sees the weak/strong models' logits or internals.

Uses AlpacaEval-2.0-style CONTINUOUS (logprob-weighted) preferences: the judge's
probability of preferring the A-position response, read from the verdict token's
logprobs -- so a response that usually-but-not-always loses still scores a graded,
non-zero winrate (a hard A/B verdict would collapse these to exactly 0 or 1).
Preferences are position-swapped (judge A/B both orderings) so position bias cancels.
"""
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def _resolve_judge_url():
    """JUDGE_URL env wins; else the always-on endpoint registry (~/bin/vllm-endpoint).

    The judge's SLURM job rolls between nodes, so a hardcoded node:port goes stale.
    The registry pointer is advisory (health is checked on first real request); for a
    blocking, health-checked resolve use `~/bin/vllm-endpoint gemma4 --wait`.
    """
    url = os.environ.get("JUDGE_URL")
    if url:
        return url
    registry = os.path.join(
        os.environ.get("VLLM_REGISTRY_DIR", os.path.expanduser("~/vllm-registry")),
        "gemma4.json",
    )
    try:
        with open(registry) as fh:
            data = json.load(fh)
        if data.get("state") == "serving" and data.get("api_base"):
            return data["api_base"]
    except (OSError, ValueError):
        pass
    return "http://localhost:8001/v1"  # inert last resort: run `~/bin/vllm-endpoint gemma4 --wait`


DEFAULT_JUDGE_URL = _resolve_judge_url()
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")

_PAIRWISE_PROMPT = """You are comparing two AI assistant responses to an instruction.

Instruction:
{instruction}

Response A:
{a}

Response B:
{b}

Which response is better overall (helpfulness, accuracy, relevance)? \
Answer with ONLY a single letter: A or B."""

_VERDICT_RE = re.compile(r"\b([AB])\b")


def _parse_verdict(reply: str) -> str:
    r = (reply or "").strip().upper()
    if re.search(r"\bTIE\b", r):     # word-boundary: don't match "tied"/"tier"/"ties"
        return "tie"
    m = _VERDICT_RE.search(r)        # standalone A/B only (not letters inside words)
    return m.group(1) if m else "tie"


def _pref_from_logprobs(entries):
    """P(A preferred) from a verdict token's top_logprobs list.

    Sums probability mass over tokens that read as 'A' vs 'B' (whitespace/case
    stripped) and normalizes. Returns None if neither letter appears (caller
    then falls back to the hard parsed verdict).
    """
    a = b = 0.0
    for e in entries or []:
        tok = (e.get("token") or "").strip().upper()
        if tok == "A":
            a += math.exp(e["logprob"])
        elif tok == "B":
            b += math.exp(e["logprob"])
    return a / (a + b) if (a + b) > 0 else None


class VLLMJudge:
    def __init__(self, base_url=DEFAULT_JUDGE_URL, model=DEFAULT_JUDGE_MODEL,
                 max_workers=8, timeout=60, pref_fn=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        self._pref_fn = pref_fn          # injectable for tests: prompt -> P(A preferred)
        # diagnostic counters (approximate under threading; used only for a warning heuristic):
        # a high failure rate means winrates are pulled toward the 0.5 neutral fallback.
        self.n_calls = 0
        self.n_failures = 0

    def _http_pref(self, prompt: str) -> float:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": 20,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"content-type": "application/json"})
        self.n_calls += 1
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.loads(resp.read())
                choice = d["choices"][0]
                lp = choice.get("logprobs")
                if lp and lp.get("content"):
                    p = _pref_from_logprobs(lp["content"][0].get("top_logprobs"))
                    if p is not None:
                        return p
                # logprobs unavailable -> fall back to a hard verdict
                return {"A": 1.0, "B": 0.0, "tie": 0.5}[
                    _parse_verdict(choice["message"]["content"])]
            except Exception:
                if attempt == 1:
                    self.n_failures += 1
                    return 0.5          # judge failure -> neutral, never crash the run
                time.sleep(0.5)         # brief backoff before the single retry

    def _pref(self, prompt: str) -> float:
        return self._pref_fn(prompt) if self._pref_fn else self._http_pref(prompt)

    def compare_prob(self, instruction: str, output_a: str, output_b: str) -> float:
        """P(the A-position response is the better one), continuous in [0, 1]."""
        return self._pref(_PAIRWISE_PROMPT.format(instruction=instruction, a=output_a, b=output_b))

    def winrate_one(self, instruction: str, candidate: str, reference: str) -> dict:
        p1 = self.compare_prob(instruction, candidate, reference)   # candidate in A slot
        p2 = self.compare_prob(instruction, reference, candidate)   # reference in A slot
        win = (p1 + (1.0 - p2)) / 2.0                               # position-swapped, continuous
        return {"win": win, "cand_len": len(candidate),
                "ref_len": len(reference), "prefs": [round(p1, 4), round(p2, 4)]}

    def winrate(self, instructions, candidates, references) -> dict:
        triples = list(zip(instructions, candidates, references))
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            per = list(ex.map(lambda t: self.winrate_one(*t), triples))
        wr = sum(p["win"] for p in per) / len(per) if per else 0.0
        return {"winrate": wr, "per_example": per}
