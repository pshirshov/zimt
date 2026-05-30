"""Filesystem helpers for the output directory.

Scans both :data:`OUT_DIR` and :data:`FAV_DIR`, reads PNG ``tEXt`` metadata,
and guards against path-traversal when resolving by name. Pagination is
caller-driven (``page`` is 1-indexed, ``per_page`` 1..200).
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from ..paths import profile_fav_dir, profile_out_dir, valid_profile_name


def read_png_meta(path: str) -> dict[str, str]:
    # Filter both key and value: Pillow types `info` as
    # ``dict[str | tuple[int, int], str]`` because some PNG ancillary chunks
    # use tuple keys. We only want the tEXt scalars.
    try:
        with Image.open(path) as im:
            return {k: v for k, v in im.info.items()
                    if isinstance(k, str) and isinstance(v, str)}
    except Exception:
        return {}


def safe_name(name: str) -> bool:
    """True iff ``name`` is a bare filename safe to use under our dirs."""
    return bool(name) and not name.startswith(".") \
        and "/" not in name and "\\" not in name


def resolve_output(profile: str, name: str) -> tuple[str, bool] | None:
    """Return ``(absolute_path, is_fav)`` within ``profile`` or ``None``."""
    if not valid_profile_name(profile) or not safe_name(name):
        return None
    main = os.path.join(profile_out_dir(profile), name)
    if os.path.isfile(main):
        return (main, False)
    fav = os.path.join(profile_fav_dir(profile), name)
    if os.path.isfile(fav):
        return (fav, True)
    return None


def _scan_dir(directory: str, *, is_fav: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if not name.lower().endswith(".png") or name.startswith("."):
            continue
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        try:
            st = os.stat(full)
        except OSError:
            continue
        out.append({
            "name": name,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "fav": is_fav,
            "metadata": read_png_meta(full),
        })
    return out


def list_outputs(profile: str, tab: str = "all",
                 page: int = 1, per_page: int = 60) -> dict[str, Any]:
    """Return one paginated page of a profile's outputs, newest first."""
    per_page = max(1, min(per_page, 200))
    page = max(1, page)
    main_dir = profile_out_dir(profile)
    fav_dir = profile_fav_dir(profile)
    if tab == "favs":
        entries = _scan_dir(fav_dir, is_fav=True)
    else:
        entries = _scan_dir(main_dir, is_fav=False) + _scan_dir(fav_dir, is_fav=True)
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    total = len(entries)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "entries": entries[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
        "has_more": end < total,
    }
