"""ETH-SRI semantically-conditioned KGW fingerprint (the trained French watermark).

The strong model is a Qwen checkpoint FINE-TUNED to emit a KGW watermark on the French
domain (github.com/eth-sri/robust-llm-fingerprints, arXiv 2505.16723). No generation-time
processor is needed — the watermark is baked into the weights. Detection is their semantic
KGW detector, built from the embedding-config YAML (the watermark key).
"""
from __future__ import annotations

from typing import List

from .base import WatermarkMethod


class ETHFrenchWatermark(WatermarkMethod):
    name = "eth_french"
    benchmark = "alpaca_eval"       # French instructions, scored open-ended like AlpacaEval

    def __init__(self, fingerprinted_model: str, embedding_config: str,
                 weak_model: "str | None" = None):
        self.fingerprinted_model = fingerprinted_model
        self.embedding_config = embedding_config
        if weak_model:
            self.weak_model = weak_model

    def get_prompts(self, n: int) -> List[str]:
        from datasets import load_dataset
        ds = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-55K", split="train")
        prompts: List[str] = []
        for i in range(len(ds)):
            row = ds[i]
            inp = (row.get("input") or "").strip()
            instr = (row.get("instruction") or "").strip()
            if instr and not inp:               # pure instructions only
                prompts.append(instr)
            if len(prompts) >= n:
                break
        return prompts

    def make_strong(self, *, gpu_memory_utilization: float, max_model_len: int):
        from w2s_research.core.strong_model import VLLMStrongModel
        return VLLMStrongModel(self.fingerprinted_model,
                               gpu_memory_utilization=gpu_memory_utilization,
                               max_model_len=max_model_len)

    def build_detector(self):
        from w2s_research.core.fingerprint_detector import FingerprintDetector
        return FingerprintDetector(self.embedding_config, self.fingerprinted_model,
                                   alpha=self.alpha)

    @staticmethod
    def add_cli_args(parser) -> None:
        parser.add_argument("--fingerprinted-model", required=True,
                            help="path to the fine-tuned fingerprinted checkpoint")
        parser.add_argument("--embedding-config", required=True,
                            help="ETH embedding-config YAML (the watermark key)")

    @classmethod
    def from_cli_args(cls, args) -> "ETHFrenchWatermark":
        return cls(args.fingerprinted_model, args.embedding_config,
                   getattr(args, "weak_model", None))
