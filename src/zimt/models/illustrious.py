"""Illustrious-XL v1.0 loader (OnomaAI Research).

OnomaAI ships v0.1 in diffusers format and v1.0+/v2.0 only as single-file
safetensors. transformers 5.x's flattened ``CLIPTextModel`` breaks the
single-file path (same bug as Pony), so we use the community diffusers
mirror at ``WhiteAiZ/Illustrious-xl-v1.0``. Same SDXL VAE substitution as
Pony for the bf16 precision/scaling-factor issue.
"""

from __future__ import annotations

from typing import Any

import torch


QUALITY_PREFIX: str = "masterpiece, best quality, very aesthetic, absurdres"


def load(device: str) -> Any:
    from diffusers import (  # type: ignore[attr-defined]
        AutoencoderKL,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLPipeline,
    )

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.bfloat16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "WhiteAiZ/Illustrious-xl-v1.0",
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe
