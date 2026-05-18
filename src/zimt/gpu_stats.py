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
import re
from typing import Any, TypedDict

import torch

# Friendly names for Intel discrete-GPU PCI IDs. intel-compute-runtime
# emits e.g. "Intel(R) Graphics [0xe223]" for IDs it doesn't recognise,
# which is uninformative in the top bar. Update this map as Intel ships
# silicon — only entries we've personally observed (or confirmed via
# kernel xe_pci.c) belong here.
INTEL_GPU_NAMES: dict[int, str] = {
    # Battlemage Arc Pro B-series (project's daily driver)
    0xE223: "Intel Arc Pro B70",
    # Battlemage consumer Arc B-series
    0xE20B: "Intel Arc B580",
    0xE20C: "Intel Arc B570",
    # Alchemist Arc A-series — selected most-common IDs
    0x56A0: "Intel Arc A770",
    0x56A1: "Intel Arc A750",
    0x56A5: "Intel Arc A380",
    0x56A6: "Intel Arc A310",
}

_PCI_RE = re.compile(r"\[0x([0-9a-fA-F]+)\]")


def _friendly_intel_name(raw: str) -> str:
    """Normalise an intel-compute-runtime device name string.

    "Intel(R) Graphics [0xe223]"  →  "Intel Arc Pro B70"     (known ID)
    "Intel(R) Graphics [0xe2ff]"  →  "Intel Graphics [0xe2ff]"  (unknown)
    "Intel(R) Arc(TM) A770 ..."   →  "Intel Arc A770 ..."       (no ID)
    """
    m = _PCI_RE.search(raw)
    if m:
        pci = int(m.group(1), 16)
        if pci in INTEL_GPU_NAMES:
            return INTEL_GPU_NAMES[pci]
    # Generic cleanup: drop the "(R)" / "(TM)" registered-marks noise.
    cleaned = raw.replace("(R)", "").replace("(TM)", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
        raw_name = getattr(props, "name", "Intel XPU")
        total = int(getattr(props, "total_memory", 0))
        return GpuStats(
            backend="xpu",
            device=_friendly_intel_name(raw_name),
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
