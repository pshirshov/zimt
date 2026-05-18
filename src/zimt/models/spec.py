"""``ModelSpec`` — the abstract contract every supported model satisfies.

A spec carries (a) static metadata used by the UI (defaults, presets, score
tags), (b) a ``load(device)`` callable returning a diffusers pipeline, and
(c) a ``tokenize_report(pipe, text)`` callable that prints the per-encoder
token analysis used by the ``/tokenize`` command. Splitting load + tokenize
into callables (rather than subclassing) keeps the registry flat and avoids
attribute-soup classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..buckets import Resolution, SDXL_BUCKETS
from ..samplers import SDXL_SAMPLERS, SamplerEntry

Family = Literal["sdxl", "zimage"]


@dataclass
class ModelSpec:
    name: str
    description: str
    default_steps: int
    default_cfg: float
    default_negative: str
    family: Family = "sdxl"
    """Text-encoder architecture family.

    ``sdxl`` — two CLIP encoders. Supports compel prompt weighting and the
    ``clip_skip`` pipeline kwarg.
    ``zimage`` — single Qwen3 encoder. Compel doesn't have an adapter for
    this family, and clip_skip is a no-op (Z-Image uses penultimate
    hidden states by default).
    """
    resolutions: list[Resolution] = field(default_factory=lambda: list(SDXL_BUCKETS))
    samplers: dict[str, SamplerEntry] = field(
        default_factory=lambda: dict(SDXL_SAMPLERS))
    """Allowed sampler names for this model. Default sampler MUST be a key."""
    default_sampler: str = "euler-a"
    scheduler_overrides: dict[str, Any] = field(default_factory=dict)
    """Per-model scheduler-config overrides applied on every sampler swap.

    Used for upstream mismatches — e.g. NoobAI Vpred is v-prediction but
    ships ``prediction_type="epsilon"`` in its bundled config. Merged with
    sampler-specific kwargs (sampler wins on conflict).
    """
    score_tags: str = ""
    """Auto-prepended to the user's prompt unless ``/raw`` is used."""

    load: Callable[[str], Any] = field(default=lambda _d: None)
    """``load(device) -> diffusers pipeline``."""

    tokenize_report: Callable[[Any, str], None] = field(default=lambda _p, _t: None)
    """``tokenize_report(pipe, text)`` — prints to stdout."""

    @property
    def default_h(self) -> int:
        return self.resolutions[0][1]

    @property
    def default_w(self) -> int:
        return self.resolutions[0][0]
