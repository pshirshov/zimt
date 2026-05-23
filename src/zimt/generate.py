"""Core generation logic — used by both the REPL and the web UI.

The :func:`generate` function is the single point where a diffusers pipeline
is actually called. Both the CLI loop and the web worker thread funnel
through it so the prompt-composition, seeding, and metadata-recording rules
stay identical.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

_log = logging.getLogger(__name__)

import torch
from PIL.PngImagePlugin import PngInfo

from . import ansi
from .device import DEVICE
from .dynamics import DynamicsSyntaxError, expand, has_dynamics
from .memory import MemStrategy
from .models.loras import LORAS
from .models.spec import ModelSpec
from .paths import OUT_DIR
from .samplers import apply_sampler
from .weighting import encode_sdxl, has_weighting


class CancelledByUser(Exception):
    """Raised inside a pipeline callback to abort an in-flight generation."""


class UninstalledLoraError(Exception):
    """Raised when generation requests a LoRA whose HF repo is not in the
    local cache. Fail-fast — we refuse to initiate an untracked download.
    Resolution: install the LoRA via the Models tab first."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(
            f"LoRA(s) not installed: {', '.join(names)}. "
            f"Install via the Models tab before generating."
        )
        self.names = names


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
) -> PngInfo:
    """Build the PNG tEXt chunks recorded with every saved image.

    Readable by PIL (``Image.open(...).info``), exiftool, or
    ``identify -verbose``.

    ``raw_prompt`` is what the user typed (may contain ``{a|b}`` syntax).
    ``expanded_prompt`` is the post-template-expansion text actually fed
    to ``compose_prompt``. Recorded as a distinct field only when it
    differs from ``raw_prompt`` — that way images generated from plain
    prompts have unchanged PNG schema, and template users can recover
    both the original and the resolved text from the file.
    """
    info = PngInfo()
    info.add_text("model", g.spec.name)
    if g.spec.repo_id:
        info.add_text("repo_id", g.spec.repo_id)
        info.add_text("repo_url", f"https://huggingface.co/{g.spec.repo_id}")
    info.add_text("raw_prompt", raw_prompt)
    if expanded_prompt != raw_prompt:
        info.add_text("expanded_prompt", expanded_prompt)
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
    # Only SDXL actually applies the stack (see _apply_lora_stack); for
    # non-SDXL families the LoRAs are skipped, so the PNG must not claim
    # they were used. (PR-10-D02)
    if g.spec.family == "sdxl" and g.lora_stack:
        info.add_text("loras", ",".join(f"{n}:{w}" for n, w in g.lora_stack))
    return info


def _apply_lora_stack(pipe: Any, g: GenConfig) -> None:
    """Lazily load every adapter in ``g.lora_stack`` then activate the set.

    Diffusers' ``load_lora_weights`` is slow (it materializes the weight
    deltas into the UNet), so we track which adapter names are already
    loaded on this pipe via ``pipe._zimt_loras_loaded`` and short-circuit
    repeats. ``set_adapters`` is cheap by comparison — it just rewrites
    the blending weights — so we always call it on every generate.

    LoRAs are *pipe-scoped*: a model swap installs a fresh pipe and the
    loaded set resets implicitly with it. We also cache the last applied
    stack so we can skip set_adapters when the stack hasn't changed.
    """
    if g.spec.family != "sdxl":
        # ZImagePipeline doesn't currently expose set_adapters.
        if g.lora_stack:
            _log.warning(
                "lora: family=%s does not support LoRA stacking; "
                "%d adapter(s) in stack will be ignored: %s",
                g.spec.family,
                len(g.lora_stack),
                ", ".join(name for name, _ in g.lora_stack),
            )
        return
    if g.lora_stack:
        from .webui.models_info import is_installed
        missing: list[str] = []
        for name, _w in g.lora_stack:
            spec = LORAS.get(name)
            if spec is None:
                # Unknown names fall through to the existing ValueError
                # in the load loop below — that's a separate concern.
                continue
            if spec.repo_id and not is_installed(spec.repo_id):
                missing.append(name)
        if missing:
            raise UninstalledLoraError(missing)
    loaded: set[str] = getattr(pipe, "_zimt_loras_loaded", set())
    for name, _weight in g.lora_stack:
        if name in loaded:
            continue
        spec = LORAS.get(name)
        if spec is None:
            raise ValueError(f"unknown lora {name!r}")
        kwargs: dict[str, Any] = {"adapter_name": name}
        if spec.weight_name:
            kwargs["weight_name"] = spec.weight_name
        pipe.load_lora_weights(spec.repo_id, **kwargs)
        loaded.add(name)
    pipe._zimt_loras_loaded = loaded

    desired = tuple(g.lora_stack)
    if getattr(pipe, "_zimt_lora_active", None) == desired:
        return
    if desired:
        pipe.set_adapters(
            [n for n, _ in desired], adapter_weights=[w for _, w in desired],
        )
    else:
        try:
            pipe.disable_lora()
        except Exception:
            # Some pipelines no-op disable_lora when nothing's loaded; fine.
            pass
    pipe._zimt_lora_active = desired


def _ensure_sampler(pipe: Any, g: GenConfig) -> None:
    """Apply the configured sampler if it differs from what's on the pipe.

    Cached via a sentinel attribute on the pipe so back-to-back generations
    with the same sampler skip the swap (which would otherwise rebuild the
    scheduler each call). Model-level ``scheduler_overrides`` are merged in
    every time — see ``apply_sampler``.
    """
    target = g.sampler or g.spec.default_sampler
    if getattr(pipe, "_zimt_sampler", None) == target:
        return
    apply_sampler(pipe, g.spec.samplers, target, g.spec.scheduler_overrides)
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

    _ensure_sampler(pipe, g)
    _apply_lora_stack(pipe, g)

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

    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = os.path.join(OUT_DIR, f"{ts}-{g.spec.name}-seed{seed}.png")
    image.save(
        out,
        pnginfo=_pnginfo(g, full_prompt, raw_prompt, expanded_prompt, seed),
    )
    print(f"generated in {dt:.1f}s -> {out}")
    return out


def _release_pipe_components(pipe: Any) -> None:
    # Wrapped in its own function so the loop locals (`sub`, `mover`,
    # `components`) die at return — leaving no stray references that
    # would keep submodule tensors alive past the caller's gc pass.
    # Strip accelerate dispatch hooks first if the pipe is hook-managed
    # (model/sequential CPU offload or device_map). Without this, the
    # subsequent .to("cpu") emits "you shouldn't move a model that is
    # dispatched using accelerate hooks" and may leave the hook closures
    # holding references to submodule weights.
    remover = getattr(pipe, "remove_all_hooks", None)
    if callable(remover):
        try:
            remover()
        except Exception as e:
            _log.debug("unload: remove_all_hooks failed: %r", e)
    components: dict[str, Any] = getattr(pipe, "components", {}) or {}
    for name in list(components.keys()):
        sub = getattr(pipe, name, None)
        if sub is None:
            continue
        mover = getattr(sub, "to", None)
        if callable(mover):
            try:
                sub.to("cpu")
            except Exception as e:
                _log.debug("unload: failed to move %s to cpu: %r", name, e)
        try:
            setattr(pipe, name, None)
        except Exception as e:
            _log.debug("unload: failed to clear pipe.%s: %r", name, e)


def unload(pipe: Any) -> None:
    """Release a diffusers pipeline's submodules and try to free device memory.

    `del pipe` on a parameter only drops the local binding; if the
    caller still references the wrapper, the UNet/VAE/text-encoder
    submodules — which actually own the VRAM — stay alive and
    ``empty_cache`` returns nothing useful. So we walk the pipe's
    registered components, move each ``nn.Module`` to CPU and null
    the wrapper's slot. The device tensors then become unreachable
    even if the caller's wrapper reference outlives this call.

    Note: ``empty_cache`` only returns *unused* allocator-cached blocks
    to the driver. The torch allocator may keep an arena reserved per
    process; that reservation only fully releases on process exit.
    """
    if pipe is not None:
        _release_pipe_components(pipe)
    pipe = None  # noqa: F841 — drop our parameter binding before gc
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_spec(spec: ModelSpec, mem: MemStrategy) -> Any:
    """Print a banner and dispatch to the spec's ``load`` callable.

    ``mem`` controls device placement (see :mod:`zimt.memory`). The active
    strategy is included in the banner so the user can see at a glance
    whether they're on a low-VRAM mode.
    """
    print(f"loading {spec.name}: {spec.description}  [mem={mem.describe()}]")
    t0 = time.time()
    pipe = spec.load(DEVICE, mem)
    print(f"loaded in {time.time() - t0:.1f}s")
    return pipe
