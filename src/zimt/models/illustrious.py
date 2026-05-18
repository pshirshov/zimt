"""Illustrious-XL v1.0 — constants only.

OnomaAI ships v0.1 in diffusers format and v1.0+/v2.0 only as single-file
safetensors. transformers 5.x's flattened ``CLIPTextModel`` breaks the
single-file path; we use the community diffusers mirror
``WhiteAiZ/Illustrious-xl-v1.0`` and the shared SDXL loader.
"""

from __future__ import annotations

from .sdxl_factory import make_sdxl_loader

QUALITY_PREFIX: str = "masterpiece, best quality, very aesthetic, absurdres"

load = make_sdxl_loader("WhiteAiZ/Illustrious-xl-v1.0")
