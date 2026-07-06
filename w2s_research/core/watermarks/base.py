"""Pluggable watermark / fingerprint methods for the removal probe.

A ``WatermarkMethod`` bundles the three things the probe needs to evaluate one
watermark against collaborative decoding:

  1. ``get_prompts(n)``  — the eval prompts (English AlpacaEval, French, ...).
  2. ``make_strong(...)``— a STRONG model whose text output carries the watermark
                           (a fine-tuned checkpoint, or a base model + a generation-time
                           logits bias).
  3. ``build_detector()``— a detector scoring completions -> p-value (built in a
                           SEPARATE process, after generation frees the GPU).

Add a method by subclassing this and registering it in ``registry.py``; the
``probe_watermark`` / ``detect_watermark`` scripts dispatch on ``--watermark <name>``.
Because those are two processes, a method must also rebuild itself from CLI args
(``add_cli_args`` / ``from_cli_args``) identically in each.

SIDE-REPORTING ONLY: the detector p-value is NEVER an optimization target — it is
measured post-hoc on utility-optimal policies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol


class Detector(Protocol):
    def score(self, completions: List[str], *, n_queries: "int | None" = None,
              seed: int = 0) -> Dict[str, Any]:
        """Return {pvalue, is_fingerprinted, n_queries}."""
        ...


class WatermarkMethod(ABC):
    name: str = "base"
    benchmark: str = "alpaca_eval"      # judge/reference domain the probe scores against
    alpha: float = 1e-3                  # a batch is "detected" iff its p-value < alpha
    weak_model: str = "meta-llama/Llama-3.2-1B-Instruct"

    @abstractmethod
    def get_prompts(self, n: int) -> List[str]:
        """The ``n`` eval prompts for this watermark's domain."""

    @abstractmethod
    def make_strong(self, *, gpu_memory_utilization: float, max_model_len: int):
        """A ``VLLMStrongModel`` whose text output carries the watermark."""

    @abstractmethod
    def build_detector(self) -> Detector:
        """A detector exposing ``.score(completions, n_queries=?, seed=?)``.
        Built in the detect process, after generation has freed the GPU."""

    # --- CLI reconstruction: probe and detect are two separate processes ---
    @staticmethod
    def add_cli_args(parser) -> None:
        """Register this method's CLI args (model paths, watermark params)."""

    @classmethod
    @abstractmethod
    def from_cli_args(cls, args) -> "WatermarkMethod":
        """Rebuild the method from parsed CLI args (same args in both processes)."""
