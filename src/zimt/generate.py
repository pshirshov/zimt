"""Core generation logic — used by both the REPL and the web UI.

The :func:`generate` function is the single point where a diffusers pipeline
is actually called. Both the CLI loop and the web worker thread funnel
through it so the prompt-composition, seeding, and metadata-recording rules
stay identical.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

_log = logging.getLogger(__name__)

from PIL.PngImagePlugin import PngInfo

from . import ansi
from .backend import (
    Backend,
    CancelledByUser,
    LocalBackend,
    RemoteBackend,
    RenderRequest,
    UninstalledLoraError,
    _apply_lora_stack,
    unload,
)
from .device import DEVICE
from .dynamics import DynamicsSyntaxError, expand, has_dynamics
from .memory import MemStrategy
from .models.spec import LORA_FAMILIES, ModelSpec
from .paths import OUT_DIR

# Re-exported for callers that import these from the orchestration module
# (and for tests that patch them here). The implementations live in
# :mod:`zimt.backend`, which this module sits on top of.
__all__ = [
    "Backend", "CancelledByUser", "UninstalledLoraError", "GenConfig",
    "compose_prompt", "generate", "load_spec", "unload", "_apply_lora_stack",
]


@dataclass
class GenConfig:
    """Mutable per-session generation knobs scoped to one loaded model.

    Recreated whenever the model changes — its defaults come from the spec.
    """
    spec: ModelSpec
    cfg: float
    negative_prompt: str
    height: int
    width: int
    steps: int
    sampler: str = ""
    """Currently-active sampler name. Empty == use the spec's default."""
    clip_skip: int = 0
    """For SDXL only. 0 means "don't pass clip_skip" (diffusers default).
    Pony was trained with clip_skip=2; many SDXL fine-tunes work well there."""
    lora_stack: list[tuple[str, float]] = field(default_factory=list)
    """Active LoRA adapters as ``(name, weight)`` pairs.

    ``name`` is a key in :data:`zimt.models.loras.LORAS`. The order is
    significant only for the UI (insertion order = display order);
    diffusers blends adapters linearly with the supplied weights.
    Adapters are lazily loaded on the pipe and activated via
    ``set_adapters``; an empty stack triggers ``disable_lora``.
    """
    aspect_ratio: str = ""
    """Remote-family only — the API ``aspect_ratio`` enum value. Empty for
    local models (they use ``width``/``height`` instead)."""
    resolution_tier: str = ""
    """Remote-family only — the API ``resolution`` tier (e.g. ``1k``/``2k``);
    the closest the hosted API has to a "quality" knob. Empty for local."""

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *,
        lora_stack: list[tuple[str, float]] | None = None,
    ) -> "GenConfig":
        """Build a fresh config from a spec's defaults.

        Branches on backend family: remote specs have no pixel-space defaults
        (steps/cfg/size are zero) and instead seed ``aspect_ratio`` /
        ``resolution_tier`` from :class:`RemoteImageConfig`.
        """
        if spec.remote is not None:
            return cls(
                spec=spec, cfg=0.0, negative_prompt="",
                height=0, width=0, steps=0,
                aspect_ratio=spec.remote.default_aspect_ratio,
                resolution_tier=spec.remote.default_resolution,
                lora_stack=list(lora_stack or []),
            )
        return cls(
            spec=spec, cfg=spec.default_cfg,
            negative_prompt=spec.default_negative,
            height=spec.default_h, width=spec.default_w,
            steps=spec.default_steps,
            lora_stack=list(lora_stack or []),
        )


def compose_prompt(spec: ModelSpec, raw_prompt: str, *, raw: bool) -> str:
    """Auto-prepend the model's score-tag prefix unless ``raw`` is set."""
    if raw or not spec.score_tags:
        return raw_prompt
    return f"{spec.score_tags}, {raw_prompt}"


def _pnginfo(
    g: GenConfig,
    full_prompt: str,
    raw_prompt: str,
    expanded_prompt: str,
    seed: int,
    render_meta: dict[str, str],
    command_line: str = "",
) -> PngInfo:
    """Build the PNG tEXt chunks recorded with every saved image.

    Readable by PIL (``Image.open(...).info``), exiftool, or
    ``identify -verbose``.

    ``raw_prompt`` is what the user typed for the prompt portion (may
    contain ``{a|b}`` template syntax). ``expanded_prompt`` is the
    post-template-expansion text actually fed to ``compose_prompt`` —
    recorded as a distinct field only when it differs from
    ``raw_prompt``. ``command_line`` is the FULL untouched line the
    user submitted, including any ``/cmd`` parts (same value the web UI
    stores in its "recent prompts" list). Recorded as ``command_line``
    so a future generation can round-trip exactly what was typed,
    /commands and template syntax included.
    """
    info = PngInfo()
    info.add_text("model", g.spec.name)
    if g.spec.repo_id:
        info.add_text("repo_id", g.spec.repo_id)
        info.add_text("repo_url", f"https://huggingface.co/{g.spec.repo_id}")
    if command_line:
        info.add_text("command_line", command_line)
    info.add_text("raw_prompt", raw_prompt)
    if expanded_prompt != raw_prompt:
        info.add_text("expanded_prompt", expanded_prompt)
    info.add_text("prompt", full_prompt)
    info.add_text("negative_prompt", g.negative_prompt or "")
    info.add_text("seed", str(seed))
    if g.spec.remote is None:
        # Local diffusers knobs. Remote models have no analogue, so we record
        # their own controls below instead of writing zeroed-out fields.
        info.add_text("steps", str(g.steps))
        info.add_text("cfg", str(g.cfg))
        info.add_text("sampler", g.sampler or g.spec.default_sampler)
        info.add_text("clip_skip", str(g.clip_skip))
        info.add_text("width", str(g.width))
        info.add_text("height", str(g.height))
    # Backend-supplied provenance: device/dtype for local; api_model +
    # aspect_ratio + resolution + revised_prompt for remote.
    for key, value in render_meta.items():
        info.add_text(key, value)
    # Only LoRA-capable families actually apply the stack (see
    # backend._apply_lora_stack); other families skip the LoRAs, so the PNG
    # must not claim they were used. (PR-10-D02)
    if g.spec.family in LORA_FAMILIES and g.lora_stack:
        info.add_text("loras", ",".join(f"{n}:{w}" for n, w in g.lora_stack))
    return info


def generate(
    backend: Backend,
    g: GenConfig,
    raw_prompt: str,
    seed: int,
    *,
    raw: bool,
    on_step: Callable[[int, int], None] | None = None,
    command_line: str = "",
    out_dir: str = OUT_DIR,
) -> str:
    """Run a single generation, save the PNG, return its path.

    Owns the shared concerns — dynamic-template expansion, score-tag
    composition, the negative-prompt rule, PNG metadata — and delegates the
    actual image production to ``backend`` (local diffusers or a remote API).

    ``on_step``, if given, is invoked once per progress step with
    ``(step_one_based, total_steps)``. For local models it fires from the
    pipeline's ``callback_on_step_end`` hook (and may raise
    :class:`CancelledByUser` to abort); for remote models it fires once on
    completion. ``command_line`` / ``out_dir`` are recorded / used as before.
    """
    # Dynamic-prompt expansion happens here — once the seed is known and
    # before compose_prompt prepends any model score-tag prefix. That
    # ordering matters: score_tags must never be template-expanded by
    # accident, and the same seed must always yield the same expansion.
    expanded_prompt = raw_prompt
    if has_dynamics(raw_prompt):
        expanded_prompt = expand(raw_prompt, random.Random(seed))

    full_prompt = compose_prompt(g.spec, expanded_prompt, raw=raw)
    neg = g.negative_prompt if g.cfg > 0 else None

    if expanded_prompt != raw_prompt:
        print(f"{ansi.DIM}template:{ansi.RESET} {raw_prompt!r}")
    print(f"{ansi.DIM}positive:{ansi.RESET} {full_prompt!r}")
    print(f"{ansi.DIM}negative:{ansi.RESET} {neg!r}")

    req = RenderRequest(
        spec=g.spec, g=g, full_prompt=full_prompt, negative=neg,
        seed=seed, on_step=on_step,
    )
    t0 = time.time()
    result = backend.render(req)
    dt = time.time() - t0

    if result.revised_prompt and result.revised_prompt != full_prompt:
        print(f"{ansi.DIM}revised:{ansi.RESET} {result.revised_prompt!r}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = os.path.join(out_dir, f"{ts}-{g.spec.name}-seed{seed}.png")
    result.image.save(
        out,
        pnginfo=_pnginfo(
            g, full_prompt, raw_prompt, expanded_prompt, seed,
            result.metadata, command_line,
        ),
    )
    print(f"generated in {dt:.1f}s -> {out}")
    return out


def load_spec(
    spec: ModelSpec, mem: MemStrategy,
    loras: tuple[tuple[str, float], ...] = (),
) -> Backend:
    """Print a banner and build the backend for ``spec``.

    For local models this dispatches to the spec's ``load`` callable and wraps
    the returned diffusers pipeline in a :class:`LocalBackend`. For remote
    models (``spec.remote`` set) it constructs a :class:`RemoteBackend` from
    the environment — ``mem`` and ``loras`` are no-ops there.

    ``mem`` controls device placement for local pipelines (see
    :mod:`zimt.memory`); the active strategy is shown in the banner. ``loras``
    is the fuse-at-load stack passed to FUSE-family loaders (flux/flux2);
    other loaders ignore it.
    """
    print(f"loading {spec.name}: {spec.description}  [mem={mem.describe()}]")
    t0 = time.time()
    if spec.remote is not None:
        backend: Backend = RemoteBackend.from_spec(spec)
    else:
        pipe = spec.load(DEVICE, mem, loras)
        backend = LocalBackend(pipe, spec)
    print(f"loaded in {time.time() - t0:.1f}s")
    return backend
