"""Bridge to the ETH-SRI semantically-conditioned KGW watermark detector.

Side-reporting only: measures whether a collaborative-decoding output still carries
the (fingerprinted) strong model's watermark. This is NEVER an optimization target —
it is measured post-hoc on utility-optimal policies.

Mirrors `scripts/compute_decision.py::compute_ours_decision` from
github.com/eth-sri/robust-llm-fingerprints (arXiv 2505.16723): tokenize the
completions, concatenate them into one sequence, run the KGW detector, and
threshold the p-value at ALPHA. The detector is a pure tokenizer+torch statistical
test (no model forward pass).

Requires the `robust_fp` package installed into this env
(`uv pip install --no-deps -e <clone>` + strenum) and the embedding-config YAML
(the watermark key) for the fingerprinted model. The KGW detector needs a GPU
(its config uses a CUDA RNG), so build/run this on a GPU node.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

DEFAULT_ALPHA = 1e-3  # matches compute_ours_decision: is_fingerprinted = pvalue < 1e-3


def decide(pvalue: float, alpha: float = DEFAULT_ALPHA) -> bool:
    """A batch is judged fingerprinted iff its detector p-value is below alpha."""
    return pvalue < alpha


def score_with(tokenizer, detector, completions: List[str], *,
               n_queries: Optional[int] = None, seed: int = 0,
               alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    """Score a list of completion strings with an already-built tokenizer + detector.

    Pure plumbing (no model/config construction) so it is unit-testable with stubs:
    pad-tokenize, deterministically sample ``n_queries`` rows, flatten into one
    sequence, run ``detector.detect``, threshold. Returns
    {pvalue, is_fingerprinted, n_queries}.
    """
    completions = [c for c in completions if c]
    if not completions:
        return {"pvalue": 1.0, "is_fingerprinted": False, "n_queries": 0}
    enc = tokenizer(completions, return_tensors="pt", padding=True)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    n = int(input_ids.shape[0])
    n_queries = n if n_queries is None else min(int(n_queries), n)
    g = torch.Generator().manual_seed(seed)                 # deterministic (theirs is not)
    perm = torch.randperm(n, generator=g)[:n_queries]
    input_ids, attn = input_ids[perm], attn[perm]
    cat_ids = torch.flatten(input_ids).view(1, -1)
    cat_attn = torch.flatten(attn).view(1, -1)
    pvalue = float(detector.detect(cat_ids, cat_attn).item())
    return {"pvalue": pvalue, "is_fingerprinted": decide(pvalue, alpha),
            "n_queries": int(n_queries)}


class FingerprintDetector:
    """Builds the real ETH KGW detector from an embedding config + the model tokenizer."""

    def __init__(self, embedding_config_path: str, model_id_or_path: str,
                 device: Optional[str] = None, alpha: float = DEFAULT_ALPHA):
        from transformers import AutoTokenizer
        from robust_fp.config import MainConfiguration

        self.alpha = alpha
        cfg = MainConfiguration.parse_yaml(embedding_config_path)
        cfg.base_model = model_id_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = cfg.watermark_config.get_detector(device, self.tokenizer)

    def score(self, completions: List[str], *, n_queries: Optional[int] = None,
              seed: int = 0) -> Dict[str, Any]:
        return score_with(self.tokenizer, self.detector, completions,
                          n_queries=n_queries, seed=seed, alpha=self.alpha)
