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


def apply_sampler(pipe: Any, samplers: dict[str, SamplerEntry], name: str) -> None:
    """Swap ``pipe.scheduler`` to match the named sampler.

    Raises :class:`ValueError` if the name isn't in the per-model registry.
    """
    if name not in samplers:
        raise ValueError(
            f"unknown sampler {name!r}; available: {', '.join(samplers)}"
        )
    cls_name, extra_kwargs = samplers[name]
    import diffusers  # type: ignore[import-not-found]
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise RuntimeError(
            f"diffusers has no scheduler class {cls_name!r} — "
            "either the diffusers version is too old, or the sampler "
            "table needs updating."
        )
    pipe.scheduler = cls.from_config(pipe.scheduler.config, **extra_kwargs)
