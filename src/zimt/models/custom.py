"""User-supplied model + LoRA descriptors loaded from disk.

Two on-disk directories scanned at server startup (and after every
``custom_add`` / ``custom_remove`` RPC):

* ``CUSTOM_DIR/bases/<slug>.json`` — additional :class:`ModelSpec`
  entries merged into :data:`zimt.models.registry.MODELS`. Only SDXL
  bases are supported via the JSON schema below; for anything that
  needs a hand-written loader, add a module like
  :mod:`zimt.models.zimage` instead.
* ``CUSTOM_DIR/loras/<slug>.json`` — :class:`LoraSpec` entries merged
  into :data:`zimt.models.loras.LORAS`.

JSON shape (per file):

  Base:  ``{"name", "description", "repo_id",
            "compatibility_tags": [str], "family"?: "sdxl",
            "default_steps"?: int, "default_cfg"?: float,
            "default_negative"?: str, "default_sampler"?: str,
            "score_tags"?: str}``
  LoRA:  ``{"name", "description", "repo_id",
            "family"?: "sdxl"|"zimage",
            "compatible_with": [str],
            "weight_name"?: str, "default_weight"?: float,
            "trigger_tags"?: str}``

Validation errors are logged and the offending file is skipped — one
bad descriptor must not poison the rest of the registry.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..paths import CUSTOM_BASES_DIR, CUSTOM_LORAS_DIR, ensure_dirs
from .loras import LORAS
from .registry import MODELS
from .sdxl_factory import make_sdxl_loader
from .sdxl_common import tokenize_report_sdxl
from .spec import LoraSpec, ModelSpec

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class CustomDescriptorError(ValueError):
    """Raised when a user-supplied JSON descriptor fails validation."""


def _require(d: dict[str, Any], key: str, kind: type) -> Any:
    if key not in d:
        raise CustomDescriptorError(f"missing required field {key!r}")
    val = d[key]
    if not isinstance(val, kind):
        raise CustomDescriptorError(f"field {key!r} must be {kind.__name__}")
    return val


def _list_of_str(d: dict[str, Any], key: str, *, required: bool = False) -> list[str]:
    if key not in d:
        if required:
            raise CustomDescriptorError(f"missing required field {key!r}")
        return []
    val = d[key]
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise CustomDescriptorError(f"field {key!r} must be list[str]")
    return list(val)


def _check_slug(name: str) -> None:
    if not _SLUG_RE.match(name):
        raise CustomDescriptorError(
            f"name {name!r} must match [a-z0-9][a-z0-9_-]* (≤63 chars)")


def parse_base(data: dict[str, Any]) -> ModelSpec:
    name = _require(data, "name", str)
    _check_slug(name)
    description = _require(data, "description", str)
    repo_id = _require(data, "repo_id", str)
    family = data.get("family", "sdxl")
    if family != "sdxl":
        # Non-SDXL custom bases need a hand-written loader (z-image, …).
        raise CustomDescriptorError(
            "only family='sdxl' is supported for custom bases via JSON")
    tags = _list_of_str(data, "compatibility_tags") or ["sdxl"]
    return ModelSpec(
        name=name,
        description=description,
        repo_id=repo_id,
        compatibility_tags=tags,
        family="sdxl",
        default_steps=int(data.get("default_steps", 28)),
        default_cfg=float(data.get("default_cfg", 6.0)),
        default_negative=str(data.get("default_negative", "")),
        default_sampler=str(data.get("default_sampler", "euler-a")),
        score_tags=str(data.get("score_tags", "")),
        load=make_sdxl_loader(repo_id),
        tokenize_report=tokenize_report_sdxl,
        is_builtin=False,
    )


def parse_lora(data: dict[str, Any]) -> LoraSpec:
    name = _require(data, "name", str)
    _check_slug(name)
    description = _require(data, "description", str)
    repo_id = _require(data, "repo_id", str)
    family = data.get("family", "sdxl")
    if family not in ("sdxl", "zimage"):
        raise CustomDescriptorError("family must be 'sdxl' or 'zimage'")
    compat = _list_of_str(data, "compatible_with", required=True)
    return LoraSpec(
        name=name,
        description=description,
        repo_id=repo_id,
        family=family,
        compatible_with=compat,
        weight_name=str(data.get("weight_name", "")),
        default_weight=float(data.get("default_weight", 1.0)),
        trigger_tags=str(data.get("trigger_tags", "")),
        is_builtin=False,
    )


def _load_descriptor(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise CustomDescriptorError("top-level JSON must be an object")
    return data


def reload_custom_into_registries() -> dict[str, list[str]]:
    """Wipe previously-loaded custom entries from MODELS/LORAS, then re-scan.

    Built-in entries are preserved (we keep them by ``is_builtin``).
    Returns a report of loaded names + per-file errors so the caller can
    surface them.
    """
    ensure_dirs()
    # Drop existing custom entries before reloading.
    for n in [k for k, v in MODELS.items() if not v.is_builtin]:
        del MODELS[n]
    for n in [k for k, v in LORAS.items() if not v.is_builtin]:
        del LORAS[n]

    bases_loaded: list[str] = []
    loras_loaded: list[str] = []
    errors: list[str] = []

    def _iter_jsons(d: str) -> list[str]:
        if not os.path.isdir(d):
            return []
        return sorted(
            os.path.join(d, n) for n in os.listdir(d)
            if n.endswith(".json") and not n.startswith(".")
        )

    for path in _iter_jsons(CUSTOM_BASES_DIR):
        try:
            spec = parse_base(_load_descriptor(path))
            if spec.name in MODELS:
                raise CustomDescriptorError(
                    f"name collides with existing model {spec.name!r}")
            MODELS[spec.name] = spec
            bases_loaded.append(spec.name)
        except (CustomDescriptorError, json.JSONDecodeError, OSError) as e:
            errors.append(f"{os.path.basename(path)}: {e}")

    for path in _iter_jsons(CUSTOM_LORAS_DIR):
        try:
            spec = parse_lora(_load_descriptor(path))
            if spec.name in LORAS:
                raise CustomDescriptorError(
                    f"name collides with existing lora {spec.name!r}")
            LORAS[spec.name] = spec
            loras_loaded.append(spec.name)
        except (CustomDescriptorError, json.JSONDecodeError, OSError) as e:
            errors.append(f"{os.path.basename(path)}: {e}")

    return {"bases": bases_loaded, "loras": loras_loaded, "errors": errors}


def write_custom(kind: str, data: dict[str, Any]) -> ModelSpec | LoraSpec:
    """Validate ``data`` for the given ``kind`` and atomically write it to disk.

    Returns the parsed spec on success. Raises :class:`CustomDescriptorError`
    on validation failure or filename collision with a built-in.
    """
    if kind == "base":
        spec = parse_base(data)
        if spec.name in MODELS and MODELS[spec.name].is_builtin:
            raise CustomDescriptorError(
                f"name {spec.name!r} collides with a built-in base model")
        target_dir = CUSTOM_BASES_DIR
    elif kind == "lora":
        spec = parse_lora(data)
        if spec.name in LORAS and LORAS[spec.name].is_builtin:
            raise CustomDescriptorError(
                f"name {spec.name!r} collides with a built-in LoRA")
        target_dir = CUSTOM_LORAS_DIR
    else:
        raise CustomDescriptorError(f"unknown kind {kind!r}")

    ensure_dirs()
    path = os.path.join(target_dir, f"{spec.name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return spec


def delete_custom(kind: str, name: str) -> None:
    """Remove a custom entry's JSON file. Built-ins cannot be deleted."""
    _check_slug(name)
    if kind == "base":
        target_dir = CUSTOM_BASES_DIR
        registry: dict[str, Any] = MODELS
    elif kind == "lora":
        target_dir = CUSTOM_LORAS_DIR
        registry = LORAS
    else:
        raise CustomDescriptorError(f"unknown kind {kind!r}")
    existing = registry.get(name)
    if existing is not None and getattr(existing, "is_builtin", False):
        raise CustomDescriptorError(
            f"{name!r} is a built-in {kind} and cannot be deleted")
    path = os.path.join(target_dir, f"{name}.json")
    if not os.path.isfile(path):
        raise CustomDescriptorError(f"no custom {kind} {name!r}")
    os.remove(path)
