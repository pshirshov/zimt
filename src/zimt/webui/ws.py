"""WebSocket broadcast primitives.

``broadcast`` is the only place we touch the client set, so dead-socket
pruning happens consistently. Higher-level helpers (:func:`emit_state`,
:func:`emit_job`, :func:`emit_log`) wrap the common event shapes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

from fastapi import WebSocket

from .state import Job, STATE


async def broadcast(event: dict[str, Any]) -> None:
    # Per-event isolation: a single non-JSON-serializable payload must
    # not tear down the broadcast loop for every client. default=str is
    # the last-resort coercion for stray Path / datetime / numpy values.
    try:
        payload = json.dumps(event, default=str)
    except (TypeError, ValueError) as e:
        # Guard the diagnostic itself: event repr() can also raise.
        try:
            evt_repr = repr(event)
        except Exception:
            evt_repr = "<unrepr-able>"
        print(f"zimt: dropped non-serializable broadcast: {e!r} event={evt_repr}",
              file=sys.stderr)
        return
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
    if job.kind == "download":
        # Read download_* fields under the same lock the tqdm worker
        # holds when writing them — see downloads.ProgressTqdm._emit.
        from .downloads import _active_lock
        with _active_lock:
            payload = asdict(job)
    else:
        payload = asdict(job)
    await broadcast({"type": "job", "job": payload})


async def emit_log(msg: str, level: str = "info") -> None:
    await broadcast({"type": "log", "level": level, "message": msg})
