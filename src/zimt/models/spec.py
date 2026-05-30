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
from ..memory import MemStrategy
from ..samplers import SDXL_SAMPLERS, SamplerEntry

Family = Literal["sdxl", "zimage", "flux", "flux2"]

# LoRA application differs by family because of how the transformer loads:
#
#  * LIVE  (`sdxl`, `zimage`) — unquantized, so adapters inject cleanly via
#    load_lora_weights + set_adapters and can be stacked/reweighted live,
#    per-generate (see generate._apply_lora_stack).
#  * FUSE  (`flux`, `flux2`)  — loaded fp8-quantized; PEFT can't inject into
#    quantized Linear layers. Instead the loader bakes the LoRA(s) into the
#    bf16 transformer (fuse_lora) BEFORE quantizing, so the stack is fixed at
#    model-load time. Changing it requires a reload (see the /lora handlers).
LIVE_LORA_FAMILIES: frozenset[str] = frozenset({"sdxl", "zimage"})
FUSE_LORA_FAMILIES: frozenset[str] = frozenset({"flux", "flux2"})
# Any family that can apply LoRAs at all (either mechanism).
LORA_FAMILIES: frozenset[str] = LIVE_LORA_FAMILIES | FUSE_LORA_FAMILIES


@dataclass
class ModelSpec:
    name: str
    description: str
    default_steps: int
    default_cfg: float
    default_negative: str
    repo_id: str = ""
    """HuggingFace repository id (``"owner/name"``).

    Source of truth for the loader, install-status scan, and the
    ``repo_id`` / ``repo_url`` PNG metadata fields. Empty string means
    "no associated HF repo" (shouldn't happen for any registered model).
    """
    compatibility_tags: list[str] = field(default_factory=list)
    """Free-form tags advertised by this base model.

    A :class:`LoraSpec` whose ``compatible_with`` shares at least one tag
    with this list is considered loadable on top of this pipeline. By
    convention every entry includes its architecture (``"sdxl"`` /
    ``"zimage"``) plus narrower fine-tune-family tags such as
    ``"pony"`` or ``"illustrious"``.
    """
    is_builtin: bool = True
    """``False`` for entries loaded from ``CUSTOM_DIR`` at startup.

    The UI uses this to surface a "remove" affordance only for custom
    entries — built-ins live in the source tree and can't be deleted
    from the running server.
    """
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

    load: Callable[..., Any] = field(
        default=lambda _d, _m, _loras=(): None
    )
    """``load(device, mem, loras=()) -> diffusers pipeline``.

    ``loras`` is the fuse-at-load stack — a tuple of ``(name, weight)`` — used
    only by FUSE-family loaders (flux/flux2), which bake the adapters into the
    bf16 transformer before quantizing. LIVE-family loaders ignore it (they
    apply LoRAs live, post-load)."""

    tokenize_report: Callable[[Any, str], None] = field(default=lambda _p, _t: None)
    """``tokenize_report(pipe, text)`` — prints to stdout."""

    @property
    def default_h(self) -> int:
        return self.resolutions[0][1]

    @property
    def default_w(self) -> int:
        return self.resolutions[0][0]


@dataclass
class LoraSpec:
    """A LoRA adapter loadable on top of a compatible base pipeline.

    Diffusers loads these via :meth:`load_lora_weights` (with an
    ``adapter_name``), then :meth:`set_adapters` activates one or more
    at once with per-adapter weights. The set of currently-active LoRAs
    is the :class:`zimt.generate.GenConfig` ``lora_stack``.
    """

    name: str
    description: str
    repo_id: str
    family: Family = "sdxl"
    compatible_with: list[str] = field(default_factory=list)
    """Tag set this LoRA needs. Matches against the active base model's
    :attr:`ModelSpec.compatibility_tags` — any-overlap is sufficient."""
    weight_name: str = ""
    """Optional file name inside the repo when the repo carries several
    LoRA variants. Empty means "let diffusers pick the canonical file"."""
    default_weight: float = 1.0
    """Suggested adapter weight when the user doesn't supply one to
    ``/lora <name>``."""
    trigger_tags: str = ""
    """Optional trigger words the LoRA was trained against — surfaced in
    the UI so the user knows what to put in the prompt to activate it.
    Not auto-injected (this is a hint, not a side effect)."""
    is_builtin: bool = True
