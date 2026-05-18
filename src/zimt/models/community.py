"""Community SDXL fine-tunes — loaders + tag conventions.

These are vanilla SDXL fine-tunes (no v-prediction or other upstream
wrinkles), so they share the standard :mod:`zimt.models.sdxl_factory`
loader + the fp16-fix VAE + euler-a default.

Tag convention notes:
  * Hassaku-XL, WAI-NSFW-Illustrious, and WAI-Mature-Illustrious are
    Illustrious-family, share its
    ``masterpiece, best quality, very aesthetic, absurdres`` prefix.
  * AnimagineXL 4.0 (Cagliostro's flagship) uses the same prefix
    convention (it's where many fine-tunes inherited it from).
  * CyberRealistic Pony is Pony v6-based, so it uses Pony's score-tag
    prefix instead.
"""

from __future__ import annotations

from .pony import SCORE_PREFIX
from .sdxl_factory import make_sdxl_loader

ILLUSTRIOUS_FAMILY_PREFIX: str = (
    "masterpiece, best quality, very aesthetic, absurdres"
)

# Each export is a load callable bound to a specific HF repo.
load_hassaku                = make_sdxl_loader("John6666/hassaku-xl-illustrious-v31-sdxl")
load_wai_nsfw_illustrious   = make_sdxl_loader("John6666/wai-nsfw-illustrious-v80-sdxl")
load_wai_nsfw_illustrious_v110 = make_sdxl_loader("John6666/wai-nsfw-illustrious-v110-sdxl")
load_wai_mature_illustrious = make_sdxl_loader("John6666/wai-mature-illustrious-v20-sdxl")
load_animagine_xl_4         = make_sdxl_loader("cagliostrolab/animagine-xl-4.0")
load_cyberrealistic_pony    = make_sdxl_loader("John6666/cyberrealistic-pony-v85-sdxl")
load_pony_realism_v23       = make_sdxl_loader("John6666/pony-realism-v23-sdxl")
load_spicy_realism_nsfw_mix = make_sdxl_loader("John6666/spicy-realism-nsfw-mix-v30-sdxl")

# Re-export under a friendlier name for the registry.
PONY_SCORE_PREFIX = SCORE_PREFIX
