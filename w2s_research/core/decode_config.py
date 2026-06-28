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
