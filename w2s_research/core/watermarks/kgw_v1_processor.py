"""vLLM v1 model-level LogitsProcessor applying the red-green KGW watermark at generation.

vLLM 0.11 is v1-only in practice (the v0 engine path is dead) and v1 rejects per-request
Python logits processors — it takes model-level LogitsProcessor CLASSES instead (passed to
``LLM(logits_processors=[...])`` by FQCN). This is that class.

Every step, for each request it boosts the green-list — a ``gamma`` fraction of the vocab
seeded by the last ``context_width`` context tokens — by ``+delta``. The green-list is
computed with the SAME ``WatermarkBase._get_greenlist_ids`` the detector's mask is built
from (see ``KGWWatermark.__init__``), so generator and detector agree by construction.

Global watermark config comes from env vars set before the engine is built (the vLLM worker
inherits them): ``KGW_GAMMA``, ``KGW_DELTA``, ``KGW_SEEDING``.
"""
from __future__ import annotations

import os

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class KGWV1LogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        gamma = float(os.environ.get("KGW_GAMMA", "0.25"))
        self.delta = float(os.environ.get("KGW_DELTA", "4.0"))
        seeding = os.environ.get("KGW_SEEDING", "simple_1")
        vocab_size = vllm_config.model_config.get_vocab_size()
        from robust_fp.watermarks.kgw.watermark_processor import WatermarkBase
        self.base = WatermarkBase(vocab=list(range(vocab_size)), gamma=gamma,
                                  delta=self.delta, seeding_scheme=seeding, device=str(device))
        self.cw = self.base.context_width
        self._reqs: dict = {}          # batch index -> (prompt_tok_ids, output_tok_ids-live-ref)

    def is_argmax_invariant(self) -> bool:
        return False                   # the green bias can change the argmax

    def update_state(self, batch_update):
        # track EVERY request (the strong model's whole output is watermarked)
        process_dict_updates(self._reqs, batch_update,
                             lambda params, prompt, out: (prompt or [], out))

    def _seed_ctx(self, prompt, out):
        cw = self.cw
        if len(out) >= cw:
            return out[-cw:]
        return (list(prompt) + list(out))[-cw:]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, (prompt, out) in self._reqs.items():
            seed = self._seed_ctx(prompt, out)
            if len(seed) < self.cw:
                continue               # not enough context yet to seed a green-list
            green = self.base._get_greenlist_ids(
                torch.tensor(list(seed), device=self.device, dtype=torch.long))
            logits[idx, green] += self.delta
        return logits
