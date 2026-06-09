"""Discovery of hosted (remote-API) image models at startup.

When ``XAI_API_KEY`` is set, :func:`merge_remote_into_registry` queries the
provider's model-listing endpoint and registers one :class:`ModelSpec` per
returned image-generation model — keyed by the exact API model id, so the
name the user types after ``/model`` is the name sent to the API
(pass-through). When the env var is missing, or the listing call fails (e.g.
a credit-blocked team returns 403), nothing is registered and remote logic
stays entirely off; the caller logs the reason.

Per-model option sets (aspect ratios, resolution tiers) are not part of the
listing response — they are documented capabilities of the Grok Imagine image
endpoint — so every discovered model is given the documented option sets via
:data:`XAI_ASPECT_RATIOS` / :data:`XAI_RESOLUTIONS`.
"""

from __future__ import annotations

import logging
from typing import Any

from ..backend import (
    RemoteBackendError,
    remote_api_key,
    remote_base_url,
)
from .registry import MODELS
from .spec import ModelSpec, RemoteImageConfig

_log = logging.getLogger(__name__)

# Documented Grok Imagine image-generation option sets (xAI docs, 2026-06).
# `auto` lets the model pick; `1k`/`2k` are the resolution tiers (~1024² /
# ~2048², exact dims depend on aspect ratio).
XAI_ASPECT_RATIOS: tuple[str, ...] = (
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2",
    "19.5:9", "9:19.5", "20:9", "9:20", "auto",
)
XAI_DEFAULT_ASPECT_RATIO = "1:1"
XAI_RESOLUTIONS: tuple[str, ...] = ("1k", "2k")
XAI_DEFAULT_RESOLUTION = "1k"

_MODELS_ENDPOINT = "/image-generation-models"
_DISCOVERY_TIMEOUT_S = 30.0


def _extract_model_list(body: Any) -> list[dict[str, Any]]:
    """Pull the per-model dicts out of a listing response.

    Tolerant of both the OpenAI-style ``{"data": [...]}`` envelope and a
    ``{"models": [...]}`` shape, and of a bare top-level list.
    """
    if isinstance(body, list):
        candidates = body
    elif isinstance(body, dict):
        candidates = body.get("models") or body.get("data") or []
    else:
        candidates = []
    return [c for c in candidates if isinstance(c, dict)]


def _price_label(item: dict[str, Any]) -> str:
    """Best-effort price string from whatever field the listing used."""
    for key in ("image_price", "price_per_image", "price", "image_token_price"):
        val = item.get(key)
        if val:
            return str(val)
    return ""


def _build_remote_spec(model_id: str, item: dict[str, Any]) -> ModelSpec:
    price = _price_label(item)
    desc = item.get("description") or f"Hosted image model {model_id!r} (xAI Grok Imagine)"
    if price:
        desc = f"{desc} — {price}/image"
    return ModelSpec(
        name=model_id,
        description=desc,
        repo_id="",
        family="remote",
        # No pixel-space defaults; the remote knobs live on `remote` below.
        default_steps=0,
        default_cfg=0.0,
        default_negative="",
        resolutions=[],
        samplers={},
        default_sampler="",
        score_tags="",
        is_builtin=True,
        remote=RemoteImageConfig(
            api_model=model_id,
            aspect_ratios=XAI_ASPECT_RATIOS,
            default_aspect_ratio=XAI_DEFAULT_ASPECT_RATIO,
            resolutions=XAI_RESOLUTIONS,
            default_resolution=XAI_DEFAULT_RESOLUTION,
            price_per_image=price,
        ),
    )


def discover_remote_models(client: Any | None = None) -> dict[str, ModelSpec]:
    """Query the provider listing and return ``{model_id: ModelSpec}``.

    Returns ``{}`` when no API key is set. ``client`` may be supplied (an
    httpx-compatible client) for tests; otherwise a real one is built from the
    environment and closed before returning. Raises
    :class:`RemoteBackendError` on a non-200 listing response so the caller
    can log the provider's message.
    """
    own_client = False
    if client is None:
        key = remote_api_key()
        if not key:
            return {}
        import httpx
        client = httpx.Client(
            base_url=remote_base_url(),
            headers={"Authorization": f"Bearer {key}"},
            timeout=_DISCOVERY_TIMEOUT_S,
        )
        own_client = True
    try:
        resp = client.get(_MODELS_ENDPOINT)
        if resp.status_code != 200:
            from ..backend import _format_api_error
            raise RemoteBackendError(_format_api_error(resp))
        items = _extract_model_list(resp.json())
        out: dict[str, ModelSpec] = {}
        for item in items:
            model_id = item.get("id") or item.get("name")
            if not isinstance(model_id, str) or not model_id:
                continue
            out[model_id] = _build_remote_spec(model_id, item)
        return out
    finally:
        if own_client:
            client.close()


def merge_remote_into_registry(client: Any | None = None) -> dict[str, list[str]]:
    """Refresh the ``family == "remote"`` entries of :data:`MODELS` in place.

    Idempotent: removes previously-merged remote entries first, then re-adds
    whatever the listing returns. Returns ``{"added": [...], "errors": [...]}``
    so startup can log the outcome. Never raises — a failed discovery is
    reported via ``errors`` and leaves remote support off.
    """
    report: dict[str, list[str]] = {"added": [], "errors": []}
    for name in [k for k, v in MODELS.items() if v.family == "remote"]:
        del MODELS[name]
    if not remote_api_key():
        return report
    try:
        specs = discover_remote_models(client)
    except Exception as e:  # network error, non-200, malformed body
        report["errors"].append(f"{type(e).__name__}: {e}")
        _log.warning("remote model discovery failed: %s", report["errors"][-1])
        return report
    for name, spec in specs.items():
        existing = MODELS.get(name)
        if existing is not None and existing.family != "remote":
            report["errors"].append(
                f"remote model {name!r} collides with a local model; skipped")
            continue
        MODELS[name] = spec
        report["added"].append(name)
    return report
