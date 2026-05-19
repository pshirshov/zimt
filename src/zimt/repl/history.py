"""Persistent readline history + Tab completion wiring.

Importing :mod:`readline` already hooks ``input()`` — this module just
sets up the on-disk history file and installs a completer that knows the
command set + finite argument sets such as models, samplers, and resolutions.
"""

from __future__ import annotations

import atexit
import readline

from ..models.loras import LORAS
from ..models.registry import MODELS
from ..paths import HISTORY_PATH
from .commands import COMMANDS


_ORIENTATION_RESOLUTIONS = ("square", "landscape", "portrait")


def _save_history() -> None:
    try:
        readline.write_history_file(HISTORY_PATH)
    except OSError:
        pass


def _model_for_context(tokens: list[str]) -> str | None:
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] != "/model":
            continue
        name = tokens[i + 1]
        if name in MODELS:
            return name
    return None


def _sampler_options(tokens: list[str], text: str) -> list[str]:
    model = _model_for_context(tokens)
    if model is not None:
        opts = sorted(MODELS[model].samplers)
    else:
        seen: set[str] = set()
        for spec in MODELS.values():
            seen.update(spec.samplers)
        opts = sorted(seen)
    return [s for s in opts if s.startswith(text)]


def _lora_options(tokens: list[str], text: str) -> list[str]:
    """LoRA names compatible with the model in this command-line context.

    ``/lora`` is greedy, so ``/lora foo:0.8 ba<TAB>`` should still complete
    LoRA names. We do that by walking the tokens backwards from the
    cursor and treating "most recent /cmd is /lora" as the trigger.
    """
    base_name = _model_for_context(tokens)
    base_tags: set[str] = set()
    if base_name is not None:
        base_tags = set(MODELS[base_name].compatibility_tags)
    prefix = text
    if prefix.startswith("-"):
        prefix = prefix[1:]
    if ":" in prefix:
        # Completing a weight after `name:`; nothing useful to suggest.
        return []
    names: list[str] = []
    for name, spec in LORAS.items():
        if not name.startswith(prefix):
            continue
        if base_tags and not any(t in base_tags for t in spec.compatible_with):
            continue
        names.append(name)
    return sorted(names)


def _resolution_options(tokens: list[str], text: str) -> list[str]:
    model = _model_for_context(tokens)
    specs = [MODELS[model]] if model is not None else list(MODELS.values())
    seen: set[str] = set(_ORIENTATION_RESOLUTIONS)
    for spec in specs:
        for w, h, _label in spec.resolutions:
            seen.add(f"{w}x{h}")
    return [r for r in sorted(seen) if r.startswith(text)]


def _completion_options(line: str, begidx: int, text: str) -> list[str]:
    prefix = line[:begidx]
    tokens = prefix.split()
    prev = tokens[-1] if tokens else ""

    if text.startswith("/"):
        return [c for c in COMMANDS if c.startswith(text)]
    if prev == "/model":
        return [m for m in MODELS if m.startswith(text)]
    if prev == "/sampler":
        return _sampler_options(tokens, text)
    if prev == "/res":
        return _resolution_options(tokens, text)
    if prev == "/lora":
        return _lora_options(tokens, text)
    return []


def _completer(text: str, state: int) -> str | None:
    """Cycle through commands or finite argument values at the cursor."""
    line = readline.get_line_buffer()
    begidx = readline.get_begidx()
    opts = _completion_options(line, begidx, text)

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
