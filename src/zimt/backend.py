"""Rendering backends — the one place an image is actually produced.

``generate()`` (in :mod:`zimt.generate`) owns the *shared* concerns of a
generation: dynamic-prompt expansion, score-tag composition, seed handling,
and writing the PNG. The act of turning a fully-resolved request into pixels
is delegated to a :class:`Backend`:

  * :class:`LocalBackend` wraps a diffusers pipeline. All the pixel-space
    machinery — sampler swap, LoRA stack, compel weighting, clip_skip, the
    ``torch.Generator`` seed, the per-family ``__call__`` quirks — lives here.
  * :class:`RemoteBackend` calls a hosted image API (xAI / Grok Imagine). It
    has no concept of steps/cfg/sampler/seed/negative/clip_skip/LoRA; it maps
    the request onto ``model`` + ``prompt`` + ``aspect_ratio`` + ``resolution``
    and decodes the returned image.

This module is the lower layer: it imports nothing from :mod:`zimt.generate`
at runtime (only a ``TYPE_CHECKING`` annotation), so ``generate`` can import
*it* without a cycle and re-export the back-compat helper names.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch
from PIL import Image

from . import ansi
from .device import DEVICE
from .models.loras import LORAS
from .models.spec import LIVE_LORA_FAMILIES, LORA_FAMILIES, ModelSpec
from .samplers import apply_sampler
from .weighting import encode_sdxl, has_weighting

if TYPE_CHECKING:
    from .generate import GenConfig

_log = logging.getLogger(__name__)


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


class RemoteBackendError(RuntimeError):
    """A hosted image API returned a non-success response. Carries the
    provider's status code and message so the surfaces can show it verbatim."""


# --------------------------------------------------------------------------
# Shared render request / result
# --------------------------------------------------------------------------

@dataclass
class RenderRequest:
    """A single generation, fully resolved by ``generate()`` and handed to a
    backend. ``full_prompt`` already includes any score-tag prefix and
    dynamic-template expansion; ``negative`` is ``None`` when CFG is off (or
    the model has no negative-prompt concept). ``g`` carries the remaining
    knobs — each backend reads the subset it understands."""

    spec: ModelSpec
    g: "GenConfig"
    full_prompt: str
    negative: str | None
    seed: int
    on_step: Callable[[int, int], None] | None = None


@dataclass
class RenderResult:
    """Output of a backend render. ``metadata`` is merged into the PNG tEXt
    chunks — it carries backend-specific provenance (device/dtype for local;
    api_model/aspect_ratio/resolution/revised_prompt for remote)."""

    image: Image.Image
    revised_prompt: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class Backend(ABC):
    """Produces images for one loaded model. Implementations hold whatever
    resources the model needs (a diffusers pipe, an HTTP client) and release
    them in :meth:`unload`."""

    spec: ModelSpec

    @abstractmethod
    def render(self, req: RenderRequest) -> RenderResult:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    def tokenize_report(self, text: str) -> None:
        """Print a per-encoder token analysis. Default: not supported."""
        print(f"{ansi.DIM}(tokenization not available for this model){ansi.RESET}")


# --------------------------------------------------------------------------
# Local diffusers backend + its pixel-space helpers
# --------------------------------------------------------------------------

def _apply_lora_stack(pipe: Any, g: "GenConfig") -> None:
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
    if g.spec.family not in LIVE_LORA_FAMILIES:
        # FUSE families (flux/flux2) have their LoRAs baked into the quantized
        # transformer at load time, so there's nothing to apply live here —
        # return quietly. A family in neither set genuinely can't take LoRAs,
        # so warn if a stack was somehow set.
        if g.lora_stack and g.spec.family not in LORA_FAMILIES:
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


def _ensure_sampler(pipe: Any, g: "GenConfig") -> None:
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
    pipe: Any, g: "GenConfig", full_prompt: str, neg: str | None,
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


class LocalBackend(Backend):
    """A loaded diffusers pipeline."""

    def __init__(self, pipe: Any, spec: ModelSpec) -> None:
        self.pipe = pipe
        self.spec = spec

    def tokenize_report(self, text: str) -> None:
        self.spec.tokenize_report(self.pipe, text)

    def unload(self) -> None:
        unload(self.pipe)
        self.pipe = None

    def render(self, req: RenderRequest) -> RenderResult:
        pipe, g = self.pipe, req.g
        _ensure_sampler(pipe, g)
        _apply_lora_stack(pipe, g)

        extra: dict[str, Any] = {}

        # Prompt weighting takes over the prompt-vs-prompt_embeds slot.
        compel_kwargs = _maybe_encode_with_compel(
            pipe, g, req.full_prompt, req.negative)
        if compel_kwargs is not None:
            extra.update(compel_kwargs)
            prompt_arg: str | None = None
            neg_arg: str | None = None
        else:
            prompt_arg = req.full_prompt
            neg_arg = req.negative

        # clip_skip is SDXL-only — Z-Image's pipeline doesn't accept the kwarg.
        if g.clip_skip > 0 and g.spec.family == "sdxl":
            extra["clip_skip"] = g.clip_skip

        if req.on_step is not None:
            total_steps = g.steps
            cb = req.on_step

            def _on_step_end(
                _pipeline: Any, i: int, _t: Any, kwargs: dict[str, Any]
            ) -> dict[str, Any]:
                cb(i + 1, total_steps)
                return kwargs

            extra["callback_on_step_end"] = _on_step_end

        # Flux.2's pipeline has no `negative_prompt` parameter (it is purely
        # guidance-distilled), so passing the kwarg at all would raise
        # TypeError. Flux.1 keeps it (a no-op unless true_cfg_scale > 1).
        if g.spec.family != "flux2":
            extra["negative_prompt"] = neg_arg

        image = pipe(
            prompt=prompt_arg,
            height=g.height,
            width=g.width,
            num_inference_steps=g.steps,
            guidance_scale=g.cfg,
            generator=torch.Generator(device=DEVICE).manual_seed(req.seed),
            **extra,
        ).images[0]
        return RenderResult(
            image=image,
            metadata={"device": DEVICE, "dtype": "bfloat16"},
        )


# --------------------------------------------------------------------------
# Remote (hosted image API) backend
# --------------------------------------------------------------------------

# xAI / Grok Imagine image-generations endpoint. Overridable for tests or a
# self-hosted proxy.
XAI_API_KEY_ENV = "XAI_API_KEY"
XAI_BASE_URL_ENV = "XAI_BASE_URL"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
_REMOTE_TIMEOUT_S = 180.0


def remote_api_key() -> str:
    """The hosted-API key from the environment, or ``""`` when unset.

    Remote-model discovery and backend construction are both gated on this:
    no key → no remote models registered → remote logic entirely off.
    """
    return os.environ.get(XAI_API_KEY_ENV, "").strip()


def remote_base_url() -> str:
    return os.environ.get(XAI_BASE_URL_ENV, "").strip() or DEFAULT_XAI_BASE_URL


class RemoteBackend(Backend):
    """A hosted image-generation model (xAI Grok Imagine).

    The diffusers knobs (steps/cfg/sampler/seed/negative/clip_skip/LoRA) have
    no analogue in the REST API and are silently dropped — the surfaces reject
    or no-op those commands for remote models, so by the time a request
    reaches here only ``prompt`` + ``aspect_ratio`` + ``resolution`` matter.
    """

    def __init__(self, spec: ModelSpec, client: Any) -> None:
        assert spec.remote is not None, "RemoteBackend requires spec.remote"
        self.spec = spec
        self._client = client

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "RemoteBackend":
        import httpx
        key = remote_api_key()
        if not key:
            raise RemoteBackendError(
                f"{XAI_API_KEY_ENV} is not set; remote model "
                f"{spec.name!r} cannot be loaded")
        client = httpx.Client(
            base_url=remote_base_url(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=_REMOTE_TIMEOUT_S,
        )
        return cls(spec, client)

    def unload(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()
        self._client = None

    def render(self, req: RenderRequest) -> RenderResult:
        rc = self.spec.remote
        assert rc is not None
        g = req.g
        aspect = g.aspect_ratio or rc.default_aspect_ratio
        resolution = g.resolution_tier or rc.default_resolution
        payload: dict[str, Any] = {
            "model": rc.api_model,
            "prompt": req.full_prompt,
            "n": 1,
            "response_format": "b64_json",
            "aspect_ratio": aspect,
            "resolution": resolution,
        }
        resp = self._client.post("/images/generations", json=payload)
        if resp.status_code != 200:
            raise RemoteBackendError(_format_api_error(resp))
        body = resp.json()
        image, revised = _decode_image_response(body)
        if req.on_step is not None:
            # Remote generation is a single blocking call; surface it to the
            # progress UI as a one-step job that has just completed.
            req.on_step(1, 1)
        meta = {
            "device": "remote:xai",
            "dtype": "api",
            "api_model": rc.api_model,
            "aspect_ratio": aspect,
            "resolution": resolution,
        }
        if revised:
            meta["revised_prompt"] = revised
        return RenderResult(image=image, revised_prompt=revised, metadata=meta)


def _format_api_error(resp: Any) -> str:
    """Best-effort extraction of the provider's error message for display."""
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = str(body.get("error") or body.get("message") or body)
    except Exception:
        detail = (resp.text or "").strip()
    code = getattr(resp, "status_code", "?")
    return f"remote API HTTP {code}: {detail}" if detail else f"remote API HTTP {code}"


def _decode_image_response(body: dict[str, Any]) -> tuple[Image.Image, str | None]:
    """Decode the first image from an OpenAI-style images response.

    Accepts either ``b64_json`` (preferred — we request it) or a ``url`` we
    then fetch. ``revised_prompt`` may live at the top level or per-item.
    """
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise RemoteBackendError(f"remote API returned no images: {body!r}")
    item = data[0]
    revised = body.get("revised_prompt") or item.get("revised_prompt")
    b64 = item.get("b64_json")
    if b64:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB"), revised
    url = item.get("url")
    if url:
        import httpx
        r = httpx.get(url, timeout=_REMOTE_TIMEOUT_S)
        if r.status_code != 200:
            raise RemoteBackendError(f"image fetch HTTP {r.status_code} for {url}")
        return Image.open(io.BytesIO(r.content)).convert("RGB"), revised
    raise RemoteBackendError(f"remote API item has neither b64_json nor url: {item!r}")
