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


OUT_DIR: str = _default_out_dir()
FAV_DIR: str = os.path.join(OUT_DIR, "fav")
HISTORY_PATH: str = _default_history_path()
STATIC_DIR: str = _default_static_dir()


def ensure_dirs() -> None:
    """Create OUT_DIR / FAV_DIR / the history dir if missing (idempotent)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FAV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
