"""Persistent readline history + Tab completion wiring.

Importing :mod:`readline` already hooks ``input()`` — this module just
sets up the on-disk history file and installs a completer that knows the
command set + the currently-registered model names.
"""

from __future__ import annotations

import atexit
import readline

from ..models.registry import MODELS
from ..paths import HISTORY_PATH
from .commands import COMMANDS


def _save_history() -> None:
    try:
        readline.write_history_file(HISTORY_PATH)
    except OSError:
        pass


def _completer(text: str, state: int) -> str | None:
    """Cycle through commands at the start, model names after ``/model``."""
    line = readline.get_line_buffer()
    begidx = readline.get_begidx()
    prefix = line[:begidx]

    if not prefix.strip():
        if not text.startswith("/"):
            return None
        opts = [c for c in COMMANDS if c.startswith(text)]
    else:
        cmd = prefix.split()[0]
        if cmd == "/model":
            opts = [m for m in MODELS if m.startswith(text)]
        else:
            opts = []

    return opts[state] if state < len(opts) else None


def init_readline() -> None:
    try:
        readline.read_history_file(HISTORY_PATH)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"(history disabled: {e})")
    readline.set_history_length(2000)
    atexit.register(_save_history)

    # Treat '/' as part of a token so '/m' completes to '/model'.
    readline.set_completer_delims(" \t\n")
    readline.set_completer(_completer)
    # GNU readline binding (Python on Linux uses real readline, not libedit).
    readline.parse_and_bind("tab: complete")
