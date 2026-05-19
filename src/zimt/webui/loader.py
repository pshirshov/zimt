"""Async model load + unload, gated by :data:`PIPE_LOCK`.

Model loads can take several minutes the first time a checkpoint is
fetched from HuggingFace. We surface that to the UI as a
:class:`zimt.webui.state.Job` of ``kind == "download"`` so the queue
shows a live row with the current file + bytes — see
:mod:`zimt.webui.downloads` for the tqdm bridge that drives the
``download_*`` fields.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from ..generate import GenConfig, load_spec, unload
from ..models.registry import MODELS
from .downloads import set_active_download
from .state import EXECUTOR, Job, PIPE_LOCK, STATE
from .ws import broadcast, emit_job, emit_log, emit_state


class ModelLoadError(Exception):
    """Raised when the requested model name doesn't exist or load fails."""


def _has_active_generation_jobs() -> bool:
    return any(
        job.status in ("queued", "running") and job.kind == "generate"
        for job in STATE.jobs.values()
    )


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
    """No-ops when the requested model is already loaded.

    Raises :class:`ModelLoadError` on unknown name or load failure.
    """
    if name not in MODELS:
        raise ModelLoadError(f"unknown model {name!r}")
    if STATE.pipe is not None and STATE.g is not None and STATE.g.spec.name == name:
        await emit_log(f"{name} is already loaded")
        return

    while STATE.loading_model is not None:
        await asyncio.sleep(0.25)

    # Create a download Job for the UI queue. It tracks bytes via the
    # tqdm bridge installed at startup. We mark it ``running`` because
    # the load starts immediately — there's no queued state for loads.
    job = Job(
        id=uuid.uuid4().hex,
        kind="download",
        target_kind="base",
        status="running",
        model=name,
    )
    STATE.jobs[job.id] = job
    STATE.loading_model = name
    await emit_job(job)
    await emit_state()
    set_active_download(job.id)
    try:
        logged_wait = False
        while _has_active_generation_jobs():
            if not logged_wait:
                await emit_log(f"waiting for active generations before loading {name}")
                logged_wait = True
            await asyncio.sleep(0.25)

        async with PIPE_LOCK:
            if STATE.pipe is not None and STATE.g is not None and STATE.g.spec.name == name:
                await emit_log(f"{name} is already loaded")
                job.status = "done"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                return
            await broadcast({"type": "model_loading", "model": name})
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(EXECUTOR, _do_load_sync, name)
            except Exception as e:
                STATE.pipe = None
                STATE.g = None
                job.status = "error"
                job.error = repr(e)
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await broadcast({"type": "model_error", "error": repr(e)})
                await emit_state()
                raise ModelLoadError(repr(e)) from e
        job.status = "done"
        job.ts_done = datetime.now().timestamp()
        await emit_job(job)
        await emit_state()
        await broadcast({"type": "model_loaded", "model": name})
    finally:
        set_active_download(None)
        if STATE.loading_model == name:
            STATE.loading_model = None
            await emit_state()
