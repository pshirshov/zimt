"""Filesystem locations.

Resolution order for the output directory:
  1. ``$ZIMT_OUT_DIR`` (also set by ``--out-dir`` via :mod:`zimt.cli`).
  2. ``$PROJECT_ROOT/out`` — the dev default when running from a working tree.
  3. ``$XDG_DATA_HOME/zimt/out`` (or ``~/.local/share/zimt/out``) — fallback
     when ``PROJECT_ROOT`` is read-only (i.e. installed from /nix/store).

``HF_HOME`` is honored by huggingface_hub and respected throughout — set via
``--hf-cache`` or the env var of the same name.
"""

from __future__ import annotations

import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_out_dir() -> str:
    env = os.environ.get("ZIMT_OUT_DIR")
    if env:
        return env
    # When installed from /nix/store, PROJECT_ROOT is read-only — fall back
    # to the user's data dir so first-run doesn't ENOENT.
    if PROJECT_ROOT.startswith("/nix/store/"):
        xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        return os.path.join(xdg, "zimt", "out")
    return os.path.join(PROJECT_ROOT, "out")


def _default_history_path() -> str:
    if PROJECT_ROOT.startswith("/nix/store/"):
        xdg = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
        return os.path.join(xdg, "zimt", "prompt_history")
    return os.path.join(PROJECT_ROOT, ".prompt_history")


def _default_static_dir() -> str:
    # The Nix package symlinks `static/` into the python module's parent
    # directory so this path lookup works the same way in both layouts.
    return os.path.join(PROJECT_ROOT, "static")


def _default_custom_dir() -> str:
    env = os.environ.get("ZIMT_CUSTOM_DIR")
    if env:
        return env
    if PROJECT_ROOT.startswith("/nix/store/"):
        xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        return os.path.join(xdg, "zimt", "custom")
    return os.path.join(PROJECT_ROOT, "custom")


OUT_DIR: str = _default_out_dir()
# Legacy top-level favourites dir. Outputs are now grouped per profile under
# OUT_DIR/<profile>/ (+ /fav); FAV_DIR is retained only as the source for the
# one-time migration in migrate_outputs_layout().
FAV_DIR: str = os.path.join(OUT_DIR, "fav")
# Profiles / prompt-history / globals live in a SQLite DB alongside the
# generated images. It never shows up in the thumbnail browser because
# zimt.webui.outputs only scans for *.png.
DB_PATH: str = os.path.join(OUT_DIR, "zimt.db")
HISTORY_PATH: str = _default_history_path()
STATIC_DIR: str = _default_static_dir()
CUSTOM_DIR: str = _default_custom_dir()
CUSTOM_BASES_DIR: str = os.path.join(CUSTOM_DIR, "bases")
CUSTOM_LORAS_DIR: str = os.path.join(CUSTOM_DIR, "loras")

# Profile whose folder receives migrated pre-profile images + localStorage data.
DEFAULT_PROFILE = "Default"
# Sentinel marking that migrate_outputs_layout() has run for this OUT_DIR.
_LAYOUT_MARKER: str = os.path.join(OUT_DIR, ".profile-layout")
# Allowed profile-name characters. The name doubles as a filesystem directory
# component (OUT_DIR/<name>/) and a URL path segment, so it must be safe for
# both — a conservative subset, no separators.
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]+$")


def valid_profile_name(name: str) -> bool:
    """True iff ``name`` is safe as a directory component + URL path segment."""
    if not isinstance(name, str) or not name:
        return False
    if name in (".", "..") or name.startswith(".") or name.endswith(" "):
        return False
    return bool(_PROFILE_NAME_RE.match(name))


def profile_out_dir(name: str) -> str:
    """Main image directory for a profile: ``OUT_DIR/<name>``."""
    return os.path.join(OUT_DIR, name)


def profile_fav_dir(name: str) -> str:
    """Favourites directory for a profile: ``OUT_DIR/<name>/fav``."""
    return os.path.join(OUT_DIR, name, "fav")


def ensure_dirs() -> None:
    """Create OUT_DIR / the history dir + custom dirs if missing."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    os.makedirs(CUSTOM_BASES_DIR, exist_ok=True)
    os.makedirs(CUSTOM_LORAS_DIR, exist_ok=True)


def migrate_outputs_layout() -> None:
    """Move pre-profile images into the Default profile folder, once.

    Before profiles, images lived directly in ``OUT_DIR`` and favourites in
    ``OUT_DIR/fav``. This relocates loose top-level PNGs into
    ``OUT_DIR/Default/`` and the legacy ``OUT_DIR/fav/*.png`` into
    ``OUT_DIR/Default/fav/``, then drops the empty legacy ``fav`` dir.
    Idempotent via the ``.profile-layout`` sentinel.
    """
    if os.path.exists(_LAYOUT_MARKER):
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    dst_main = profile_out_dir(DEFAULT_PROFILE)
    dst_fav = profile_fav_dir(DEFAULT_PROFILE)
    os.makedirs(dst_main, exist_ok=True)
    os.makedirs(dst_fav, exist_ok=True)

    def _move_pngs(src_dir: str, dst_dir: str) -> None:
        if not os.path.isdir(src_dir):
            return
        for entry in os.listdir(src_dir):
            if not entry.lower().endswith(".png") or entry.startswith("."):
                continue
            src = os.path.join(src_dir, entry)
            # Real regular files only — never follow a symlink out of OUT_DIR.
            if os.path.islink(src) or not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, entry)
            if not os.path.exists(dst):
                os.replace(src, dst)

    _move_pngs(OUT_DIR, dst_main)
    _move_pngs(FAV_DIR, dst_fav)
    # Drop the now-empty legacy favourites dir so it isn't mistaken for a
    # profile named "fav".
    if os.path.isdir(FAV_DIR):
        try:
            os.rmdir(FAV_DIR)
        except OSError:
            pass  # not empty (non-PNG leftovers) — leave it be
    with open(_LAYOUT_MARKER, "w") as f:
        f.write("1\n")
