"""Pony Diffusion V6 XL — constants only.

Distribution choice is documented in :mod:`zimt.models.sdxl_factory`:
we use the diffusers-format mirror ``kitty7779/ponyDiffusionV6XL`` and
swap in the fp16-fix VAE.
"""

from __future__ import annotations

from .sdxl_factory import make_sdxl_loader

SCORE_PREFIX: str = (
    "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
)

load = make_sdxl_loader("kitty7779/ponyDiffusionV6XL")
