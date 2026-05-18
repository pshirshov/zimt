"""The canonical model registry.

Keys are the strings users type after ``/model``. Add a model by:
  1. writing a sibling module with ``load(device)`` (+ optional
     ``tokenize_report``) callables,
  2. importing it here, and
  3. inserting a :class:`ModelSpec` into :data:`MODELS`.
"""

from __future__ import annotations

from ..buckets import SDXL_BUCKETS, ZIMAGE_BUCKETS
from ..samplers import SDXL_SAMPLERS, ZIMAGE_SAMPLERS
from . import community, illustrious, noobai, pony, zimage
from .sdxl_common import tokenize_report_sdxl
from .spec import ModelSpec

# Shared negative-prompt templates so the new models can reuse rather than
# each defining a near-duplicate string.
_NEG_ILLUSTRIOUS = (
    "worst quality, bad quality, low quality, lowres, jpeg artifacts, "
    "sketch, monochrome, signature, watermark, text, blurry"
)
_NEG_PONY = (
    "score_6, score_5, score_4, worst quality, low quality, jpeg artifacts, "
    "blurry, watermark, signature, text"
)

MODELS: dict[str, ModelSpec] = {
    "z-image-turbo": ModelSpec(
        name="z-image-turbo",
        description="Tongyi-MAI Z-Image-Turbo (6B DiT, Qwen3 text enc, CFG=0)",
        family="zimage",
        default_steps=9,
        default_cfg=0.0,
        default_negative=(
            "low quality, blurry, jpeg artifacts, watermark, signature, text, "
            "deformed anatomy, extra limbs, bad hands, low resolution"
        ),
        resolutions=list(ZIMAGE_BUCKETS),
        samplers=dict(ZIMAGE_SAMPLERS),
        default_sampler="flow-match-euler",
        score_tags="",
        load=zimage.load,
        tokenize_report=zimage.tokenize_report,
    ),
    "pony-v6-xl": ModelSpec(
        name="pony-v6-xl",
        description="Pony Diffusion V6 XL (SDXL fine-tune, Euler-a, CFG=7, score-tag prefix)",
        family="sdxl",
        default_steps=25,
        default_cfg=7.0,
        default_negative=(
            "score_6, score_5, score_4, worst quality, low quality, jpeg artifacts, "
            "blurry, watermark, signature, text"
        ),
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=pony.SCORE_PREFIX,
        load=pony.load,
        tokenize_report=tokenize_report_sdxl,
    ),
    "illustrious-xl-v1": ModelSpec(
        name="illustrious-xl-v1",
        description="Illustrious-XL v1.0 (anime-focused SDXL fine-tune by OnomaAI, Euler-a, CFG=6)",
        family="sdxl",
        default_steps=24,
        default_cfg=6.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=illustrious.QUALITY_PREFIX,
        load=illustrious.load,
        tokenize_report=tokenize_report_sdxl,
    ),

    # ---- Illustrious-family community fine-tunes ----
    "hassaku-xl-illustrious": ModelSpec(
        name="hassaku-xl-illustrious",
        description="Hassaku XL Illustrious v3.1 (anime-realistic Illustrious fine-tune)",
        family="sdxl",
        default_steps=28,
        default_cfg=6.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.ILLUSTRIOUS_FAMILY_PREFIX,
        load=community.load_hassaku,
        tokenize_report=tokenize_report_sdxl,
    ),
    "wai-nsfw-illustrious": ModelSpec(
        name="wai-nsfw-illustrious",
        description="WAI-NSFW-Illustrious v8.0 (Illustrious-based, NSFW-focused)",
        family="sdxl",
        default_steps=28,
        default_cfg=6.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.ILLUSTRIOUS_FAMILY_PREFIX,
        load=community.load_wai_nsfw_illustrious,
        tokenize_report=tokenize_report_sdxl,
    ),
    "noobai-xl-vpred": ModelSpec(
        name="noobai-xl-vpred",
        description=(
            "NoobAI XL Vpred 1.0 (Illustrious-based, v-prediction; "
            "needs prediction_type override since upstream config is wrong)"
        ),
        family="sdxl",
        default_steps=30,
        default_cfg=5.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=noobai.QUALITY_PREFIX,
        scheduler_overrides=dict(noobai.SCHEDULER_OVERRIDES),
        load=noobai.load,
        tokenize_report=tokenize_report_sdxl,
    ),
    "animagine-xl-4": ModelSpec(
        name="animagine-xl-4",
        description="Animagine XL 4.0 (Cagliostro Lab's flagship anime SDXL)",
        family="sdxl",
        default_steps=28,
        default_cfg=5.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.ILLUSTRIOUS_FAMILY_PREFIX,
        load=community.load_animagine_xl_4,
        tokenize_report=tokenize_report_sdxl,
    ),

    # ---- Pony-family community fine-tunes ----
    "cyberrealistic-pony": ModelSpec(
        name="cyberrealistic-pony",
        description="CyberRealistic Pony v8.5 (Pony v6-based, photorealism focus)",
        family="sdxl",
        default_steps=25,
        default_cfg=7.0,
        default_negative=_NEG_PONY,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.PONY_SCORE_PREFIX,
        load=community.load_cyberrealistic_pony,
        tokenize_report=tokenize_report_sdxl,
    ),
}
