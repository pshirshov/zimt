"""Pony Diffusion V6 XL loader.

Distribution / wiring caveats baked into this loader:
  * We use the diffusers-format mirror ``kitty7779/ponyDiffusionV6XL`` rather
    than the official single-file safetensors. diffusers ``from_single_file``
    builds a CLIP text encoder by reading ``model.text_model.embeddings…``
    — an attribute path that was removed in transformers 5.x.
    ``from_pretrained`` routes through transformers' own loader and works on
    either layout.
  * We swap in ``madebyollin/sdxl-vae-fp16-fix`` instead of the bundled VAE.
    The bundled VAE has the SD-1.x ``scaling_factor`` (0.18215, not SDXL's
    0.13025) and isn't numerically stable in bf16 — pastel output is the
    classic symptom of both bugs.
"""

from __future__ import annotations

from typing import Any

import torch


SCORE_PREFIX: str = (
    "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
)


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
        "kitty7779/ponyDiffusionV6XL",
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe
