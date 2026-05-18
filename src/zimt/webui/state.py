"""Process-level state for the web server.

One global :data:`STATE` is the source of truth for the loaded pipeline,
its mutable :class:`zimt.generate.GenConfig`, the WebSocket client set, and
the per-job records used by the UI. Concurrency is governed by:
  * :data:`PIPE_LOCK` — held by anything that touches the pipeline (load,
    generate). One job at a time on a single GPU.
  * :data:`EXECUTOR` — single-worker thread pool so the asyncio event loop
    stays responsive while a pipeline call blocks.
  * :data:`CANCEL_EVENTS` — per-job ``threading.Event``s set by
    ``/api/jobs/{id}/cancel`` and polled by the pipeline callback.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import WebSocket

from ..generate import GenConfig
from ..models.registry import MODELS

EXECUTOR = ThreadPoolExecutor(max_workers=1)
PIPE_LOCK = asyncio.Lock()
CANCEL_EVENTS: dict[str, threading.Event] = {}


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error | canceled
    raw_prompt: str = ""
    full_prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    model: str = ""
    path: str | None = None
    error: str | None = None
    ts_queued: float = field(default_factory=lambda: datetime.now().timestamp())
    ts_done: float | None = None
    # Progress: ``step`` is 1-based; 0 means not started yet.
    step: int = 0
    total_steps: int = 0


@dataclass
class AppState:
    """Per-process state — singleton :data:`STATE` below."""
    pipe: Any | None = None
    g: GenConfig | None = None
    jobs: dict[str, Job] = field(default_factory=dict)
    clients: set[WebSocket] = field(default_factory=set)

    def state_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.pipe is not None,
            "model": self.g.spec.name if self.g else None,
            "models": [
                {"name": n, "description": m.description,
                 "family": m.family,
                 "score_tags": m.score_tags,
                 "default_steps": m.default_steps,
                 "default_cfg": m.default_cfg,
                 "default_sampler": m.default_sampler,
                 "samplers": sorted(m.samplers),
                 "resolutions": [
                     {"w": w, "h": h, "label": label}
                     for w, h, label in m.resolutions
                 ]}
                for n, m in MODELS.items()
            ],
            "settings": {
                "cfg": self.g.cfg if self.g else None,
                "steps": self.g.steps if self.g else None,
                "width": self.g.width if self.g else None,
                "height": self.g.height if self.g else None,
                "negative_prompt": self.g.negative_prompt if self.g else None,
                "score_tags": self.g.spec.score_tags if self.g else None,
                "sampler": (self.g.sampler or self.g.spec.default_sampler) if self.g else None,
                "clip_skip": self.g.clip_skip if self.g else None,
            },
        }


STATE = AppState()
