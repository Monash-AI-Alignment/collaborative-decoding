"""Pluggable watermark/fingerprint methods for the fingerprint-removal probe."""
from .base import WatermarkMethod
from .registry import WATERMARKS, get_watermark_cls

__all__ = ["WatermarkMethod", "WATERMARKS", "get_watermark_cls"]
