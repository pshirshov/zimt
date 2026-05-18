"""Async model load + unload, gated by :data:`PIPE_LOCK`."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException

from ..generate import GenConfig, load_spec, unload
from ..models.registry import MODELS
from .state import EXECUTOR, PIPE_LOCK, STATE
from .ws import broadcast, emit_log, emit_state


def _has_active_jobs() -> bool:
    return any(job.status in ("queued", "running") for job in STATE.jobs.values())


def _do_load_sync(name: str) -> None:
    """Executor-thread payload — unload current (if any), load new, reset cfg."""
    if STATE.pipe is not None:
        unload(STATE.pipe)
        STATE.pipe = None
    STATE.g = None
    spec = MODELS[name]
    pipe = load_spec(spec)
    STATE.pipe = pipe
    STATE.g = GenConfig(
        spec=spec,
        cfg=spec.default_cfg,
        negative_prompt=spec.default_negative,
        height=spec.default_h,
        width=spec.default_w,
        steps=spec.default_steps,
    )


async def load_model(name: str) -> None:
    """No-ops when the requested model is already loaded."""
    if name not in MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model {name!r}")
    if STATE.pipe is not None and STATE.g is not None and STATE.g.spec.name == name:
        await emit_log(f"{name} is already loaded")
        return

    while STATE.loading_model is not None:
        await asyncio.sleep(0.25)

    STATE.loading_model = name
    await emit_state()
    try:
        logged_wait = False
        while _has_active_jobs():
            if not logged_wait:
                await emit_log(f"waiting for active generations before loading {name}")
                logged_wait = True
            await asyncio.sleep(0.25)

        async with PIPE_LOCK:
            if STATE.pipe is not None and STATE.g is not None and STATE.g.spec.name == name:
                await emit_log(f"{name} is already loaded")
                return
            await broadcast({"type": "model_loading", "model": name})
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(EXECUTOR, _do_load_sync, name)
            except Exception as e:
                STATE.pipe = None
                STATE.g = None
                await broadcast({"type": "model_error", "error": repr(e)})
                await emit_state()
                raise HTTPException(status_code=500, detail=repr(e)) from e
        await emit_state()
        await broadcast({"type": "model_loaded", "model": name})
    finally:
        if STATE.loading_model == name:
            STATE.loading_model = None
            await emit_state()
