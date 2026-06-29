"""Black-box LLM judge over an OpenAI-compatible vLLM server (local Gemma).

Measures ONLY utility: it compares a method's output TEXT against a reference
TEXT. It never sees logits or model internals. Pairwise verdicts are
position-swapped (judge twice, A/B reversed) so per-position bias cancels.
"""
import json
import os
import re
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


_VERDICT_RE = re.compile(r"\b([AB])\b")


def _parse_verdict(reply: str) -> str:
    r = (reply or "").strip().upper()
    if "TIE" in r:
        return "tie"
    m = _VERDICT_RE.search(r)        # standalone A/B only (not letters inside words)
    return m.group(1) if m else "tie"


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
