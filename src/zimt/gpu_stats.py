"""Lightweight GPU memory polling for the web UI top bar.

Each backend exposes memory differently. We surface a small common shape::

    {
      "device": "Intel(R) Arc(TM) Pro B70 Graphics",  # human label
      "backend": "xpu",                                # xpu / cuda / cpu
      "allocated_bytes": 1234567890,
      "reserved_bytes":  9876543210,
      "total_bytes":    17179869184,
    }

Utilization isn't exposed by torch for any backend and needs vendor tools
(intel_gpu_top, nvidia-smi, rocm-smi) — skipped here. The web UI just
shows memory; users who want util can read it in their terminal.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

import torch


class GpuStats(TypedDict):
    backend: str
    device: str
    allocated_bytes: int
    reserved_bytes: int
    total_bytes: int


def _xpu_stats() -> GpuStats | None:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return None
    try:
        props = torch.xpu.get_device_properties(0)
        name = getattr(props, "name", "Intel XPU")
        total = int(getattr(props, "total_memory", 0))
        return GpuStats(
            backend="xpu",
            device=name,
            allocated_bytes=int(torch.xpu.memory_allocated(0)),
            reserved_bytes=int(torch.xpu.memory_reserved(0)),
            total_bytes=total,
        )
    except Exception:
        return None


def _cuda_stats() -> GpuStats | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return GpuStats(
            backend="cuda",
            device=getattr(props, "name", "CUDA device"),
            allocated_bytes=int(torch.cuda.memory_allocated(0)),
            reserved_bytes=int(torch.cuda.memory_reserved(0)),
            total_bytes=int(getattr(props, "total_memory", 0)),
        )
    except Exception:
        return None


def _cpu_stats() -> GpuStats:
    # Last-resort: report the python process RSS so the bar isn't blank.
    rss = 0
    try:
        rss = int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()) * 1024
    except Exception:
        pass
    return GpuStats(
        backend="cpu",
        device="CPU",
        allocated_bytes=rss,
        reserved_bytes=rss,
        total_bytes=0,
    )


def current() -> GpuStats:
    """Best-effort one-shot stats sample. Tries XPU, then CUDA, then CPU."""
    for fn in (_xpu_stats, _cuda_stats):
        s = fn()
        if s is not None:
            return s
    return _cpu_stats()
