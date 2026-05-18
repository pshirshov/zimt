"""Core generation logic — used by both the REPL and the web UI.

The :func:`generate` function is the single point where a diffusers pipeline
is actually called. Both the CLI loop and the web worker thread funnel
through it so the prompt-composition, seeding, and metadata-recording rules
stay identical.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import torch
from PIL.PngImagePlugin import PngInfo

from . import ansi
from .device import DEVICE
from .models.spec import ModelSpec
from .paths import OUT_DIR
from .samplers import apply_sampler
from .weighting import encode_sdxl, has_weighting


class CancelledByUser(Exception):
    """Raised inside a pipeline callback to abort an in-flight generation."""


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


def compose_prompt(spec: ModelSpec, raw_prompt: str, *, raw: bool) -> str:
    """Auto-prepend the model's score-tag prefix unless ``raw`` is set."""
    if raw or not spec.score_tags:
        return raw_prompt
    return f"{spec.score_tags}, {raw_prompt}"


def _pnginfo(g: GenConfig, full_prompt: str, raw_prompt: str, seed: int) -> PngInfo:
    """Build the PNG tEXt chunks recorded with every saved image.

    Readable by PIL (``Image.open(...).info``), exiftool, or
    ``identify -verbose``.
    """
    info = PngInfo()
    info.add_text("model", g.spec.name)
    info.add_text("raw_prompt", raw_prompt)
    info.add_text("prompt", full_prompt)
    info.add_text("negative_prompt", g.negative_prompt or "")
    info.add_text("seed", str(seed))
    info.add_text("steps", str(g.steps))
    info.add_text("cfg", str(g.cfg))
    info.add_text("sampler", g.sampler or g.spec.default_sampler)
    info.add_text("clip_skip", str(g.clip_skip))
    info.add_text("width", str(g.width))
    info.add_text("height", str(g.height))
    info.add_text("dtype", "bfloat16")
    info.add_text("device", DEVICE)
    return info


def _ensure_sampler(pipe: Any, g: GenConfig) -> None:
    """Apply the configured sampler if it differs from what's on the pipe.

    Cached via a sentinel attribute on the pipe so back-to-back generations
    with the same sampler skip the swap (which would otherwise rebuild the
    scheduler each call).
    """
    target = g.sampler or g.spec.default_sampler
    if getattr(pipe, "_zimt_sampler", None) == target:
        return
    apply_sampler(pipe, g.spec.samplers, target)
    pipe._zimt_sampler = target


def _maybe_encode_with_compel(
    pipe: Any, g: GenConfig, full_prompt: str, neg: str | None,
) -> dict[str, Any] | None:
    """If the prompt or negprompt use weighting syntax and the model is
    SDXL, run them through compel and return the SDXL embed-kwargs dict.
    Otherwise return ``None`` and let the caller use the plain string
    interface.
    """
    if g.spec.family != "sdxl":
        return None
    if not (has_weighting(full_prompt) or (neg and has_weighting(neg))):
        return None
    # Build (and cache) the Compel instance on the pipe so we don't pay the
    # construction cost on every weighted generate.
    compel = getattr(pipe, "_zimt_compel", None)
    if compel is None:
        from .weighting import build_compel
        compel = build_compel(pipe, g.spec.family)
        if compel is None:
            print(f"{ansi.DIM}(compel not installed; weighting ignored){ansi.RESET}")
            return None
        pipe._zimt_compel = compel
    return encode_sdxl(compel, full_prompt, neg)


def generate(
    pipe: Any,
    g: GenConfig,
    raw_prompt: str,
    seed: int,
    *,
    raw: bool,
    on_step: Callable[[int, int], None] | None = None,
) -> str:
    """Run a single generation, save the PNG, return its path.

    ``on_step``, if given, is invoked once per scheduler step with
    ``(step_one_based, total_steps)`` from inside the pipeline's
    ``callback_on_step_end`` hook. It may raise (typically
    :class:`CancelledByUser`) to abort; the exception propagates out so the
    caller can mark the run.
    """
    full_prompt = compose_prompt(g.spec, raw_prompt, raw=raw)
    neg = g.negative_prompt if g.cfg > 0 else None

    print(f"{ansi.DIM}positive:{ansi.RESET} {full_prompt!r}")
    print(f"{ansi.DIM}negative:{ansi.RESET} {neg!r}")

    _ensure_sampler(pipe, g)

    extra: dict[str, Any] = {}

    # Prompt weighting takes over the prompt-vs-prompt_embeds slot.
    compel_kwargs = _maybe_encode_with_compel(pipe, g, full_prompt, neg)
    if compel_kwargs is not None:
        extra.update(compel_kwargs)
        prompt_arg: str | None = None
        neg_arg: str | None = None
    else:
        prompt_arg = full_prompt
        neg_arg = neg

    # clip_skip is SDXL-only — Z-Image's pipeline doesn't accept the kwarg.
    if g.clip_skip > 0 and g.spec.family == "sdxl":
        extra["clip_skip"] = g.clip_skip

    if on_step is not None:
        total_steps = g.steps
        cb = on_step

        def _on_step_end(
            _pipeline: Any, i: int, _t: Any, kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            cb(i + 1, total_steps)
            return kwargs

        extra["callback_on_step_end"] = _on_step_end

    t0 = time.time()
    image = pipe(
        prompt=prompt_arg,
        negative_prompt=neg_arg,
        height=g.height,
        width=g.width,
        num_inference_steps=g.steps,
        guidance_scale=g.cfg,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
        **extra,
    ).images[0]
    dt = time.time() - t0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_DIR, f"{ts}-{g.spec.name}-seed{seed}.png")
    image.save(out, pnginfo=_pnginfo(g, full_prompt, raw_prompt, seed))
    print(f"generated in {dt:.1f}s -> {out}")
    return out


def unload(pipe: Any) -> None:
    """Release a pipeline's references and try to free device memory."""
    del pipe
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_spec(spec: ModelSpec) -> Any:
    """Print a banner and dispatch to the spec's ``load`` callable."""
    print(f"loading {spec.name}: {spec.description}")
    t0 = time.time()
    pipe = spec.load(DEVICE)
    print(f"loaded in {time.time() - t0:.1f}s")
    return pipe
