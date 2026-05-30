"""Scheduler/sampler registry per model family.

Each entry maps a friendly name (used in ``/sampler <name>``) to the
diffusers scheduler class and any constructor kwargs that distinguish
variants like "DPM++ 2M Karras" from plain "DPM++ 2M".

We don't import the diffusers scheduler classes at module load — the
class-name string is resolved lazily in :func:`apply_sampler` so this
file is safe to import in environments without diffusers (eg. pyright).
"""

from __future__ import annotations

from typing import Any

SamplerEntry = tuple[str, dict[str, Any]]
"""(diffusers_class_name, extra_kwargs_for_from_config)"""

# Sampler keys use hyphens-only so they're single-token (tab-completable, no
# whitespace ambiguity with the GREEDY parser).
SDXL_SAMPLERS: dict[str, SamplerEntry] = {
    "euler":             ("EulerDiscreteScheduler", {}),
    "euler-a":           ("EulerAncestralDiscreteScheduler", {}),
    "dpmpp-2m":          ("DPMSolverMultistepScheduler", {}),
    "dpmpp-2m-karras":   ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpmpp-sde":         ("DPMSolverSinglestepScheduler", {}),
    "dpmpp-sde-karras":  ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    "heun":              ("HeunDiscreteScheduler", {}),
    "unipc":             ("UniPCMultistepScheduler", {}),
    "lms":               ("LMSDiscreteScheduler", {}),
    "ddim":              ("DDIMScheduler", {}),
}

# Z-Image is a flow-matching model — only flow-matching schedulers work.
ZIMAGE_SAMPLERS: dict[str, SamplerEntry] = {
    "flow-match-euler": ("FlowMatchEulerDiscreteScheduler", {}),
}

# Flux (1 & 2) ship a FlowMatchEulerDiscreteScheduler; only flow-matching
# schedulers are valid for these transformers.
FLUX_SAMPLERS: dict[str, SamplerEntry] = {
    "flow-match-euler": ("FlowMatchEulerDiscreteScheduler", {}),
}


def apply_sampler(
    pipe: Any,
    samplers: dict[str, SamplerEntry],
    name: str,
    model_overrides: dict[str, Any] | None = None,
) -> None:
    """Swap ``pipe.scheduler`` to match the named sampler.

    ``model_overrides`` is merged into the from_config kwargs and applied
    regardless of which sampler is chosen — used for model-level config
    that the upstream repo got wrong (e.g. NoobAI Vpred is trained with
    ``prediction_type="v_prediction"`` but its scheduler_config.json
    ships ``epsilon``). Sampler-specific kwargs win on conflict.

    Raises :class:`ValueError` if the name isn't in the per-model registry.
    """
    if name not in samplers:
        raise ValueError(
            f"unknown sampler {name!r}; available: {', '.join(samplers)}"
        )
    cls_name, sampler_kwargs = samplers[name]
    import diffusers  # type: ignore[import-not-found]
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise RuntimeError(
            f"diffusers has no scheduler class {cls_name!r} — "
            "either the diffusers version is too old, or the sampler "
            "table needs updating."
        )
    merged: dict[str, Any] = dict(model_overrides or {})
    merged.update(sampler_kwargs)
    pipe.scheduler = cls.from_config(pipe.scheduler.config, **merged)
