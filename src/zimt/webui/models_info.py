"""Install-status + size reporting for the model registry.

Uses :func:`huggingface_hub.scan_cache_dir` as the source of truth — it
honours ``HF_HOME`` and reports per-repo sizes built from the on-disk
blobs that are still pinned by at least one revision. Repos that exist
in the cache but have been fully GC'd by ``huggingface-cli delete-cache``
will not appear here.

The scan walks every cached repo (not just zimt's models), so we filter
by ``repo_id`` to the entries we care about. Cost is one ``stat()`` per
blob; on a warm filesystem this is a few ms per repo, which is fine to
recompute on demand whenever the UI asks.
"""

from __future__ import annotations

import time
from typing import Any

from ..models.loras import LORAS
from ..models.registry import MODELS

# Short TTL cache around scan_cache_dir(). A single /many 16 with a
# 3-LoRA stack would otherwise stat()-walk every cached repo 48 times
# for state that cannot change inside a single second. Invalidated
# explicitly by callers (prefetch_model on successful download).
_CACHE_TTL_S = 1.0
_cache_value: dict[str, dict[str, Any]] | None = None
_cache_ts: float = 0.0


def _invalidate_cache() -> None:
    """Drop the TTL cache; callers use this when a prefetch finishes."""
    global _cache_value, _cache_ts
    _cache_value = None
    _cache_ts = 0.0


def _scan_cache() -> dict[str, dict[str, Any]]:
    """Return ``repo_id -> {size_bytes, last_modified}`` for cached repos.

    Returns an empty dict when huggingface_hub is unavailable or the
    cache hasn't been initialised — both are non-fatal: the UI shows
    "not installed" for everything in that case.
    """
    global _cache_value, _cache_ts
    now = time.monotonic()
    if _cache_value is not None and (now - _cache_ts) < _CACHE_TTL_S:
        return _cache_value
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        _cache_value = {}
        _cache_ts = now
        return _cache_value
    try:
        info = scan_cache_dir()
    except Exception:
        _cache_value = {}
        _cache_ts = now
        return _cache_value
    out: dict[str, dict[str, Any]] = {}
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        out[repo.repo_id] = {
            "size_bytes": int(repo.size_on_disk),
            "last_modified": float(repo.last_modified),
        }
    _cache_value = out
    _cache_ts = now
    return out


def is_installed(repo_id: str) -> bool:
    """True if *repo_id* is present in the local HF cache.

    Single source of truth; callers must not re-implement cache scanning.
    Returns False when ``huggingface_hub`` is missing or the cache cannot
    be scanned — the safer default so generation fails fast rather than
    trusting an empty cache.
    """
    if not repo_id:
        return False
    return repo_id in _scan_cache()


def _entry_for(name: str, repo_id: str, cache: dict[str, dict[str, Any]],
               **extra: Any) -> dict[str, Any]:
    cached = cache.get(repo_id) if repo_id else None
    return {
        "name": name,
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}" if repo_id else "",
        "installed": cached is not None,
        "size_bytes": cached["size_bytes"] if cached else 0,
        "last_modified": cached["last_modified"] if cached else 0.0,
        **extra,
    }


def models_info() -> dict[str, Any]:
    """Return ``{bases:[...], loras:[...]}`` with install status + repo metadata."""
    cache = _scan_cache()
    bases = [
        _entry_for(
            name, spec.repo_id, cache,
            description=spec.description,
            family=spec.family,
            compatibility_tags=list(spec.compatibility_tags),
            is_builtin=spec.is_builtin,
        )
        for name, spec in MODELS.items()
    ]
    loras = [
        _entry_for(
            name, spec.repo_id, cache,
            description=spec.description,
            family=spec.family,
            compatible_with=list(spec.compatible_with),
            default_weight=spec.default_weight,
            trigger_tags=spec.trigger_tags,
            weight_name=spec.weight_name,
            is_builtin=spec.is_builtin,
        )
        for name, spec in LORAS.items()
    ]
    return {"bases": bases, "loras": loras}
