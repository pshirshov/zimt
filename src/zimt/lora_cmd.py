"""Parsing for the ``/lora`` REPL/web command.

Syntax accepted in the GREEDY arg of ``/lora``:

* ``""``          — return the current stack unchanged, request a "show me"
* ``-``           — clear all
* ``-<name>``     — remove one by name
* ``<name>``      — add (or update weight to spec.default_weight)
* ``<name>:0.7``  — add (or update weight to 0.7)

Multiple tokens are accepted in one call; they apply left-to-right onto
the working stack. Compatibility with the base model is enforced — an
incompatible LoRA produces a message and is skipped, leaving the rest
of the call to apply.
"""

from __future__ import annotations

from typing import Iterable

from .models.loras import LORAS
from .models.spec import ModelSpec


class LoraCmdError(ValueError):
    """Raised on syntactic or semantic errors in ``/lora`` input."""


def _parse_weight(s: str) -> float:
    try:
        w = float(s)
    except ValueError as e:
        raise LoraCmdError(f"invalid weight {s!r}") from e
    if w != w or w < -4.0 or w > 4.0:
        raise LoraCmdError(f"weight {w} out of range (-4..4)")
    return w


def _is_compatible(spec_tags: list[str], base: ModelSpec) -> bool:
    return any(t in base.compatibility_tags for t in spec_tags)


def apply_lora_args(
    stack: list[tuple[str, float]],
    args: Iterable[str],
    base: ModelSpec,
) -> list[str]:
    """Mutate ``stack`` per ``args``. Returns log messages."""
    log: list[str] = []
    for tok in args:
        if not tok:
            continue
        if tok == "-":
            if stack:
                log.append(f"lora: cleared {len(stack)} adapters")
            stack.clear()
            continue
        if tok.startswith("-"):
            name = tok[1:]
            for i, (n, _w) in enumerate(stack):
                if n == name:
                    del stack[i]
                    log.append(f"lora: removed {name}")
                    break
            else:
                log.append(f"lora: {name!r} not in active stack")
            continue
        name, sep, wstr = tok.partition(":")
        spec = LORAS.get(name)
        if spec is None:
            log.append(f"lora: unknown {name!r}")
            continue
        if base.family != "sdxl":
            log.append(
                f"lora: {base.family!r} family does not yet support LoRAs; "
                f"skipping {name!r}"
            )
            continue
        if not _is_compatible(spec.compatible_with, base):
            log.append(
                f"lora: {name!r} (compat={spec.compatible_with}) is not "
                f"compatible with base {base.name} (tags={base.compatibility_tags})"
            )
            continue
        if spec.repo_id:
            from .webui.models_info import is_installed
            if not is_installed(spec.repo_id):
                log.append(
                    f"lora: {name!r} is not installed locally; "
                    f"install it via the Models tab before adding"
                )
                continue
        weight = _parse_weight(wstr) if sep else spec.default_weight
        for i, (n, _w) in enumerate(stack):
            if n == name:
                stack[i] = (name, weight)
                log.append(f"lora: {name} weight={weight}")
                break
        else:
            stack.append((name, weight))
            log.append(f"lora: +{name} weight={weight}")
    return log


def format_stack(stack: list[tuple[str, float]]) -> str:
    if not stack:
        return "(no active loras)"
    return ", ".join(f"{n}:{w}" for n, w in stack)
