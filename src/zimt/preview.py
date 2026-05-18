"""Inline image preview in the terminal (kitty / iTerm2, tmux passthrough).

Detection is based on env vars set by the terminal emulator. ``ZIMT_GRAPHICS``
forces one of ``kitty`` / ``iterm`` / ``none``. Inside tmux every escape
sequence is wrapped in DCS passthrough — requires ``set -g allow-passthrough
on`` in tmux.conf, otherwise tmux silently drops the sequence.
"""

from __future__ import annotations

import base64
import os
import sys
from typing import Literal

Protocol = Literal["kitty", "iterm", "none"]

IN_TMUX: bool = bool(os.environ.get("TMUX"))


def detect_protocol() -> Protocol:
    forced = os.environ.get("ZIMT_GRAPHICS")
    if forced in ("kitty", "iterm", "none"):
        return forced  # type: ignore[return-value]
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    if (os.environ.get("KITTY_WINDOW_ID")
            or "kitty" in term
            or term_program == "ghostty"
            or os.environ.get("GHOSTTY_RESOURCES_DIR")
            or os.environ.get("GHOSTTY_BIN_DIR")):
        return "kitty"
    if term_program in ("iTerm.app", "WezTerm") or os.environ.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    return "none"


def _wrap_tmux(seq: str) -> str:
    return f"\x1bPtmux;{seq.replace(chr(0x1b), chr(0x1b) * 2)}\x1b\\"


def _emit(seq: str) -> None:
    sys.stdout.write(_wrap_tmux(seq) if IN_TMUX else seq)


def _show_kitty(path: str) -> None:
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    chunk_size = 4096
    sys.stdout.write("\n")
    if len(data) <= chunk_size:
        _emit(f"\x1b_Ga=T,f=100;{data}\x1b\\")
    else:
        first = True
        i = 0
        while i < len(data):
            chunk = data[i:i + chunk_size]
            i += chunk_size
            more = 1 if i < len(data) else 0
            ctrl = "a=T,f=100," if first else ""
            _emit(f"\x1b_G{ctrl}m={more};{chunk}\x1b\\")
            first = False
    sys.stdout.write("\n")
    sys.stdout.flush()


def _show_iterm(path: str) -> None:
    with open(path, "rb") as f:
        raw = f.read()
    data = base64.standard_b64encode(raw).decode("ascii")
    name = base64.standard_b64encode(os.path.basename(path).encode()).decode("ascii")
    sys.stdout.write("\n")
    _emit(
        f"\x1b]1337;File=name={name};size={len(raw)};inline=1;preserveAspectRatio=1:{data}\x07"
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def preview(path: str, proto: Protocol) -> None:
    if proto == "kitty":
        _show_kitty(path)
    elif proto == "iterm":
        _show_iterm(path)
    else:
        print(f"(no inline-image protocol detected; open {path})")
