"""Memory placement strategies for pipeline loading.

Four strategies, three orthogonal mechanisms:
  * ``off``            — ``pipe.to(device)`` after construction. Default and
    fastest steady-state; the whole pipeline lives in VRAM.
  * ``max``            — ``from_pretrained(device_map="balanced",
    max_memory={0: <cap>, "cpu": "60GiB"})``. Accelerate splits submodules
    across device and CPU to honour the cap. Note: on Intel XPU,
    accelerate/diffusers' dispatch is incomplete (forward-pass crosses
    device boundaries without an alignment hook on the conv path), so
    this mode currently throws at inference time. CUDA users get a
    working knob today; XPU users get a placeholder for the day the
    upstream gap closes.
  * ``cpuoffload``     — ``pipe.enable_model_cpu_offload(device=...)``.
    Each submodule is pulled to device on first forward and pushed back
    when the next submodule runs. Coarse-grained.
  * ``cpuoffload-seq`` — ``pipe.enable_sequential_cpu_offload(device=...)``.
    Same idea but per ``nn.Module`` rather than per top-level submodel.
    Lowest VRAM, slowest.

The ``max`` strategy needs kwargs at ``from_pretrained`` time; the offload
strategies need a post-construction method call. Split helpers below so a
model loader can apply either without conditional knowledge.

Measured cost (SDXL @ 768x768, 8 steps, Intel Arc B70):

  off              warm  2.0s   peak 7808 MiB
  max  <cap>       (broken on XPU at time of writing; works on CUDA)
  cpuoffload       warm  5.1s   peak 5179 MiB (~34% VRAM saving)
  cpuoffload-seq   warm 10.6s   peak 1085 MiB (~86% VRAM saving)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Mode = Literal["off", "max", "cpuoffload", "cpuoffload-seq"]
MODES: tuple[Mode, ...] = ("off", "max", "cpuoffload", "cpuoffload-seq")


@dataclass(frozen=True)
class MemStrategy:
    """Immutable description of how to place a pipeline.

    ``max_size`` is meaningful only when ``mode == "max"``; for the other
    modes it's left empty.
    """
    mode: Mode = "off"
    max_size: str = ""

    def describe(self) -> str:
        if self.mode == "max":
            return f"max {self.max_size}"
        return self.mode


DEFAULT = MemStrategy(mode="off")


class MemArgError(ValueError):
    """User typed ``/mem <stuff>`` that doesn't parse."""


def parse_mem_args(args: list[str]) -> MemStrategy:
    """Parse the args from a ``/mem`` invocation.

    Accepts a list of whitespace-split tokens, e.g. ``["max", "6GiB"]``.
    Raises :class:`MemArgError` with a help-shaped message on bad input;
    the caller surfaces the message verbatim to the user.
    """
    if not args:
        raise MemArgError(
            "usage: /mem <off|max <size>|cpuoffload|cpuoffload-seq>"
        )
    mode = args[0]
    if mode not in MODES:
        raise MemArgError(
            f"unknown mode {mode!r}; expected one of {', '.join(MODES)}"
        )
    if mode == "max":
        if len(args) < 2 or not args[1].strip():
            raise MemArgError("usage: /mem max <size>, e.g. /mem max 6GiB")
        if len(args) > 2:
            raise MemArgError(
                f"/mem max: expected one size argument, got {args[1:]}"
            )
        return MemStrategy(mode="max", max_size=args[1].strip())
    if len(args) > 1:
        raise MemArgError(
            f"/mem {mode}: no extra args expected, got {args[1:]}"
        )
    return MemStrategy(mode=mode)  # type: ignore[arg-type]


def from_pretrained_kwargs(mem: MemStrategy) -> dict[str, Any]:
    """Extra kwargs to thread into ``from_pretrained`` for this strategy.

    Only ``mode == "max"`` adds kwargs; the rest are post-construction.
    """
    if mem.mode == "max":
        # Integer device key — accelerate maps ``0`` to the active
        # accelerator's index. String keys like ``"xpu:0"`` are rejected
        # by accelerate.utils.modeling.get_max_memory's validator.
        return {
            "device_map": "balanced",
            "max_memory": {0: mem.max_size, "cpu": "60GiB"},
        }
    return {}


def finalize_pipe(pipe: Any, device: str, mem: MemStrategy) -> None:
    """Apply post-construction device placement.

    For ``mode == "max"`` this is a no-op because ``device_map`` already
    placed submodules during ``from_pretrained``. Calling ``.to(device)``
    on a hook-dispatched pipeline would clobber the dispatch hooks and
    accelerate would warn — so we skip it.
    """
    if mem.mode == "off":
        pipe.to(device)
    elif mem.mode == "cpuoffload":
        pipe.enable_model_cpu_offload(device=device)
    elif mem.mode == "cpuoffload-seq":
        pipe.enable_sequential_cpu_offload(device=device)
    # "max": already placed by from_pretrained.
