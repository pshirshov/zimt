"""NoobAI XL Vpred 1.0 — Illustrious-based, v-prediction trained.

The wrinkle: the upstream ``scheduler/scheduler_config.json`` ships
``prediction_type: "epsilon"`` and ``rescale_betas_zero_snr: false`` even
though the UNet was trained with v-prediction. Loading the bundled
config and rendering with euler-a out of the box produces washed,
desaturated output. The :data:`SCHEDULER_OVERRIDES` here are merged
into every sampler swap by :func:`zimt.samplers.apply_sampler`.

NoobAI's quality tag convention adds ``newest`` to the Illustrious
prefix to bias toward the more recent training corpus.
"""

from __future__ import annotations

from typing import Any

from .sdxl_factory import make_sdxl_loader

QUALITY_PREFIX: str = (
    "masterpiece, best quality, newest, very aesthetic, absurdres"
)

SCHEDULER_OVERRIDES: dict[str, Any] = {
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": True,
}

load = make_sdxl_loader("Laxhar/noobai-XL-Vpred-1.0")
