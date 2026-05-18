"""Pick the torch device to run pipelines on.

The default is auto-detect: prefer accelerators in this order — CUDA / ROCm
(both report as ``cuda`` from PyTorch), Intel XPU, Apple MPS, then CPU.
Override via the ``ZIMT_DEVICE`` env var.
"""

from __future__ import annotations

import os

import torch


def detect_device() -> str:
    forced = os.environ.get("ZIMT_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():           # CUDA builds + ROCm builds
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = detect_device()
