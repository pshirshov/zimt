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

from typing import Any

from ..models.loras import LORAS
from ..models.registry import MODELS


def _scan_cache() -> dict[str, dict[str, Any]]:
    """Return ``repo_id -> {size_bytes, last_modified}`` for cached repos.

    Returns an empty dict when huggingface_hub is unavailable or the
    cache hasn't been initialised — both are non-fatal: the UI shows
    "not installed" for everything in that case.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return {}
    try:
        info = scan_cache_dir()
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        out[repo.repo_id] = {
            "size_bytes": int(repo.size_on_disk),
            "last_modified": float(repo.last_modified),
        }
    return out


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
