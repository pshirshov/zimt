"""The canonical model registry.

Keys are the strings users type after ``/model``. Add a model by:
  1. writing a sibling module with ``load(device)`` (+ optional
     ``tokenize_report``) callables,
  2. importing it here, and
  3. inserting a :class:`ModelSpec` into :data:`MODELS`.
"""

from __future__ import annotations

from ..buckets import SDXL_BUCKETS, ZIMAGE_BUCKETS
from . import illustrious, pony, zimage
from .sdxl_common import tokenize_report_sdxl
from .spec import ModelSpec

MODELS: dict[str, ModelSpec] = {
    "z-image-turbo": ModelSpec(
        name="z-image-turbo",
        description="Tongyi-MAI Z-Image-Turbo (6B DiT, Qwen3 text enc, CFG=0)",
        default_steps=9,
        default_cfg=0.0,
        default_negative=(
            "low quality, blurry, jpeg artifacts, watermark, signature, text, "
            "deformed anatomy, extra limbs, bad hands, low resolution"
        ),
        resolutions=list(ZIMAGE_BUCKETS),
        score_tags="",
        load=zimage.load,
        tokenize_report=zimage.tokenize_report,
    ),
    "pony-v6-xl": ModelSpec(
        name="pony-v6-xl",
        description="Pony Diffusion V6 XL (SDXL fine-tune, Euler-a, CFG=7, score-tag prefix)",
        default_steps=25,
        default_cfg=7.0,
        default_negative=(
            "score_6, score_5, score_4, worst quality, low quality, jpeg artifacts, "
            "blurry, watermark, signature, text"
        ),
        resolutions=list(SDXL_BUCKETS),
        score_tags=pony.SCORE_PREFIX,
        load=pony.load,
        tokenize_report=tokenize_report_sdxl,
    ),
    "illustrious-xl-v1": ModelSpec(
        name="illustrious-xl-v1",
        description="Illustrious-XL v1.0 (anime-focused SDXL fine-tune by OnomaAI, Euler-a, CFG=6)",
        default_steps=24,
        default_cfg=6.0,
        default_negative=(
            "worst quality, bad quality, low quality, lowres, jpeg artifacts, "
            "sketch, monochrome, signature, watermark, text, blurry"
        ),
        resolutions=list(SDXL_BUCKETS),
        score_tags=illustrious.QUALITY_PREFIX,
        load=illustrious.load,
        tokenize_report=tokenize_report_sdxl,
    ),
}
