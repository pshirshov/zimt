"""WebSocket broadcast primitives.

``broadcast`` is the only place we touch the client set, so dead-socket
pruning happens consistently. Higher-level helpers (:func:`emit_state`,
:func:`emit_job`, :func:`emit_log`) wrap the common event shapes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from fastapi import WebSocket

from .state import Job, STATE


async def broadcast(event: dict[str, Any]) -> None:
    payload = json.dumps(event)
    dead: list[WebSocket] = []
    for ws in list(STATE.clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.clients.discard(ws)


async def emit_state() -> None:
    await broadcast({"type": "state", "state": STATE.state_dict()})


async def emit_job(job: Job) -> None:
    await broadcast({"type": "job", "job": asdict(job)})


async def emit_log(msg: str, level: str = "info") -> None:
    await broadcast({"type": "log", "level": level, "message": msg})
