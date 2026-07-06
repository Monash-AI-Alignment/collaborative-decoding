"""Registry of watermark methods. Add a method here to make it available to the probe."""
from __future__ import annotations

from typing import Dict, Type

from .base import WatermarkMethod
from .eth_french import ETHFrenchWatermark
from .inference_kgw import InferenceKGWWatermark

WATERMARKS: Dict[str, Type[WatermarkMethod]] = {
    ETHFrenchWatermark.name: ETHFrenchWatermark,
    InferenceKGWWatermark.name: InferenceKGWWatermark,
}


def get_watermark_cls(name: str) -> Type[WatermarkMethod]:
    if name not in WATERMARKS:
        raise ValueError(f"unknown watermark '{name}'; known: {sorted(WATERMARKS)}")
    return WATERMARKS[name]
