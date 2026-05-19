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
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        compatibility_tags=["zimage"],
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
        repo_id="kitty7779/ponyDiffusionV6XL",
        compatibility_tags=["sdxl", "pony"],
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
        repo_id="WhiteAiZ/Illustrious-xl-v1.0",
        compatibility_tags=["sdxl", "illustrious"],
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
        repo_id="John6666/hassaku-xl-illustrious-v31-sdxl",
        compatibility_tags=["sdxl", "illustrious"],
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
        repo_id="John6666/wai-nsfw-illustrious-v80-sdxl",
        compatibility_tags=["sdxl", "illustrious"],
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
    "wai-nsfw-illustrious-v110": ModelSpec(
        name="wai-nsfw-illustrious-v110",
        description="WAI-NSFW-Illustrious v11.0 (Illustrious-based, NSFW-focused)",
        repo_id="John6666/wai-nsfw-illustrious-v110-sdxl",
        compatibility_tags=["sdxl", "illustrious"],
        family="sdxl",
        default_steps=28,
        default_cfg=6.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.ILLUSTRIOUS_FAMILY_PREFIX,
        load=community.load_wai_nsfw_illustrious_v110,
        tokenize_report=tokenize_report_sdxl,
    ),
    "wai-mature-illustrious": ModelSpec(
        name="wai-mature-illustrious",
        description="WAI-Mature-Illustrious v2.0 (Illustrious-based, mature/body/style focus)",
        repo_id="John6666/wai-mature-illustrious-v20-sdxl",
        compatibility_tags=["sdxl", "illustrious"],
        family="sdxl",
        default_steps=28,
        default_cfg=6.0,
        default_negative=_NEG_ILLUSTRIOUS,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.ILLUSTRIOUS_FAMILY_PREFIX,
        load=community.load_wai_mature_illustrious,
        tokenize_report=tokenize_report_sdxl,
    ),
    "noobai-xl-vpred": ModelSpec(
        name="noobai-xl-vpred",
        description=(
            "NoobAI XL Vpred 1.0 (Illustrious-based, v-prediction; "
            "needs prediction_type override since upstream config is wrong)"
        ),
        repo_id="Laxhar/noobai-XL-Vpred-1.0",
        compatibility_tags=["sdxl", "illustrious", "noobai"],
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
        repo_id="cagliostrolab/animagine-xl-4.0",
        compatibility_tags=["sdxl", "illustrious", "animagine"],
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
        repo_id="John6666/cyberrealistic-pony-v85-sdxl",
        compatibility_tags=["sdxl", "pony"],
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
    "pony-realism-v23": ModelSpec(
        name="pony-realism-v23",
        description="Pony Realism v2.3 (Pony-family, photorealism focus)",
        repo_id="John6666/pony-realism-v23-sdxl",
        compatibility_tags=["sdxl", "pony"],
        family="sdxl",
        default_steps=25,
        default_cfg=7.0,
        default_negative=_NEG_PONY,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.PONY_SCORE_PREFIX,
        load=community.load_pony_realism_v23,
        tokenize_report=tokenize_report_sdxl,
    ),
    "spicy-realism-nsfw-mix": ModelSpec(
        name="spicy-realism-nsfw-mix",
        description="Spicy Realism NSFW Mix v3.0 (Pony-family, adult photorealism focus)",
        repo_id="John6666/spicy-realism-nsfw-mix-v30-sdxl",
        compatibility_tags=["sdxl", "pony"],
        family="sdxl",
        default_steps=25,
        default_cfg=7.0,
        default_negative=_NEG_PONY,
        resolutions=list(SDXL_BUCKETS),
        samplers=dict(SDXL_SAMPLERS),
        default_sampler="euler-a",
        score_tags=community.PONY_SCORE_PREFIX,
        load=community.load_spicy_realism_nsfw_mix,
        tokenize_report=tokenize_report_sdxl,
    ),
}
