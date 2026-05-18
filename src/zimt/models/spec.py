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
from typing import Any, Callable

from ..buckets import Resolution, SDXL_BUCKETS


@dataclass
class ModelSpec:
    name: str
    description: str
    default_steps: int
    default_cfg: float
    default_negative: str
    resolutions: list[Resolution] = field(default_factory=lambda: list(SDXL_BUCKETS))
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
