"""Classic red-green KGW watermark applied at GENERATION time (no fine-tuning).

At each strong-model decoding step, the logits of a *green-list* — a ``gamma`` fraction
of the vocabulary chosen pseudo-randomly by hashing the previous token — get a ``+delta``
bias. The detector precomputes the same green-lists and runs a z-test: watermarked text
has significantly more green tokens than the ``gamma`` chance rate.

Generator/detector agree BY CONSTRUCTION: both are built from the ETH-SRI KGW code —
``WatermarkBase._get_greenlist_ids`` computes a single seed's green-list on the fly for
generation (the full ``vocab x vocab`` mask would OOM), and ``KGWWatermark`` precomputes
that same mask for detection. Same vocab / gamma / seeding / device => identical lists.

Runs on the ORIGINAL English AlpacaEval + a base Qwen strong model, so deferral policies
are tested at the operating points they were tuned on (unlike the French checkpoint, where
the SOTA policy did not transfer). vLLM v1 rejects per-request logits processors, so
``VLLMStrongModel`` falls back to the v0 engine when a processor is supplied.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import WatermarkMethod

# gamma/seeding match the ETH French config; delta=2.0 is the STANDARD inference-time KGW
# strength (the ETH delta=4 was learned by fine-tuning — applying 4 as an inference bias to a
# base model pushes it off-distribution into degenerate no-EOS output). Override via --kgw-delta.
DEFAULTS = dict(strong_model="Qwen/Qwen2.5-7B-Instruct", gamma=0.25, delta=2.0,
                seeding_scheme="simple_1")


# vLLM expects "module.path:QualName" (colon between module and class).
_V1_PROCESSOR_FQCN = "w2s_research.core.watermarks.kgw_v1_processor:KGWV1LogitsProcessor"


class _KGWDetector:
    """Wraps the ETH ``KGWWatermark`` z-test in the ``.score`` contract the probe expects."""

    def __init__(self, tokenizer, kgw, alpha: float):
        self.tokenizer, self.kgw, self.alpha = tokenizer, kgw, alpha

    def score(self, completions: List[str], *, n_queries: "int | None" = None,
              seed: int = 0) -> Dict[str, Any]:
        from w2s_research.core.fingerprint_detector import score_with
        return score_with(self.tokenizer, self.kgw, completions,
                          n_queries=n_queries, seed=seed, alpha=self.alpha)


class InferenceKGWWatermark(WatermarkMethod):
    name = "inference_kgw"
    benchmark = "alpaca_eval"

    def __init__(self, strong_model: str = DEFAULTS["strong_model"],
                 gamma: float = DEFAULTS["gamma"], delta: float = DEFAULTS["delta"],
                 seeding_scheme: str = DEFAULTS["seeding_scheme"],
                 weak_model: "str | None" = None):
        self.strong_model = strong_model
        self.gamma, self.delta, self.seeding_scheme = gamma, delta, seeding_scheme
        if weak_model:
            self.weak_model = weak_model

    def get_prompts(self, n: int) -> List[str]:
        from w2s_research.core.alpaca_eval import load_alpaca_eval
        return [e.instruction for e in load_alpaca_eval(limit=n)]

    def _vocab(self, tokenizer) -> List[int]:
        # KGW hashes token ids; the green-list is over the tokenizer's id space. Using the
        # same len(tokenizer) in generator and detector keeps the two permutations identical.
        return list(range(len(tokenizer)))

    def make_strong(self, *, gpu_memory_utilization: float, max_model_len: int):
        import os
        from w2s_research.core.strong_model import VLLMStrongModel
        # The v1 processor is instantiated inside the engine (worker inherits these env vars).
        os.environ["KGW_GAMMA"] = str(self.gamma)
        os.environ["KGW_DELTA"] = str(self.delta)
        os.environ["KGW_SEEDING"] = self.seeding_scheme
        return VLLMStrongModel(self.strong_model,
                               gpu_memory_utilization=gpu_memory_utilization,
                               max_model_len=max_model_len,
                               logits_processor_cls=_V1_PROCESSOR_FQCN)

    def build_detector(self):
        import torch
        from transformers import AutoTokenizer
        from robust_fp.watermarks.kgw.kgw_watermark import KGWWatermark
        tok = AutoTokenizer.from_pretrained(self.strong_model, padding_side="left")
        kgw = KGWWatermark(vocab=self._vocab(tok), gamma=self.gamma, delta=self.delta,
                           seeding_scheme=self.seeding_scheme, tokenizer=tok,
                           fast_init=torch.cuda.is_available())
        return _KGWDetector(tok, kgw, self.alpha)

    @staticmethod
    def add_cli_args(parser) -> None:
        parser.add_argument("--strong-model", default=DEFAULTS["strong_model"])
        parser.add_argument("--kgw-gamma", type=float, default=DEFAULTS["gamma"])
        parser.add_argument("--kgw-delta", type=float, default=DEFAULTS["delta"])
        parser.add_argument("--kgw-seeding", default=DEFAULTS["seeding_scheme"])

    @classmethod
    def from_cli_args(cls, args) -> "InferenceKGWWatermark":
        return cls(strong_model=args.strong_model, gamma=args.kgw_gamma, delta=args.kgw_delta,
                   seeding_scheme=args.kgw_seeding, weak_model=getattr(args, "weak_model", None))
