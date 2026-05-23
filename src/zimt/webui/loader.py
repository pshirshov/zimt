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
import contextvars
import threading
import uuid
from datetime import datetime

from ..generate import GenConfig, load_spec, unload
from ..models.registry import MODELS
from .downloads import (
    DownloadCanceled, clear_active_download, download_context, set_active_download,
)
from .models_info import _invalidate_cache as _invalidate_models_cache
from .state import CANCEL_EVENTS, EXECUTOR, Job, LOADING_LOCK, PIPE_LOCK, STATE
from .ws import broadcast, emit_job, emit_log, emit_state


class ModelLoadError(Exception):
    """Raised when the requested model name doesn't exist or load fails."""


def _has_active_generation_jobs() -> bool:
    return any(
        job.status in ("queued", "running") and job.kind == "generate"
        for job in STATE.jobs.values()
    )


def _do_load_sync(name: str) -> None:
    """Executor-thread payload — unload current (if any), load new, reset cfg.

    The active memory strategy lives on STATE.mem; if the user changed it
    via ``/mem``, the next load picks it up here.
    """
    if STATE.pipe is not None:
        unload(STATE.pipe)
        STATE.pipe = None
    STATE.g = None
    spec = MODELS[name]
    pipe = load_spec(spec, STATE.mem)
    STATE.pipe = pipe
    STATE.g = GenConfig(
        spec=spec,
        cfg=spec.default_cfg,
        negative_prompt=spec.default_negative,
        height=spec.default_h,
        width=spec.default_w,
        steps=spec.default_steps,
    )


async def load_model(name: str, *, force: bool = False) -> None:
    """No-ops when the requested model is already loaded.

    ``force=True`` skips the "already loaded" short-circuits so the model
    is unloaded and re-loaded — used when STATE.mem changes and the
    currently-resident pipeline needs to pick up the new strategy.

    Raises :class:`ModelLoadError` on unknown name or load failure.
    """
    if name not in MODELS:
        raise ModelLoadError(f"unknown model {name!r}")
    if (not force
            and STATE.pipe is not None and STATE.g is not None
            and STATE.g.spec.name == name):
        await emit_log(f"{name} is already loaded")
        return

    # Atomic check-and-set of STATE.loading_model. Without LOADING_LOCK,
    # two concurrent load_model() coroutines could both observe
    # loading_model is None across the await asyncio.sleep boundary and
    # both create download jobs / set STATE.loading_model.
    async with LOADING_LOCK:
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
        CANCEL_EVENTS[job.id] = threading.Event()
        STATE.loading_model = name
    await emit_job(job)
    await emit_state()
    try:
        owns_progress = set_active_download(job.id)
        job.progress_owner = owns_progress
        if not owns_progress:
            await emit_log(f"download progress slot busy; {name} load progress will not be broadcast")
            await emit_job(job)
        logged_wait = False
        while _has_active_generation_jobs():
            if CANCEL_EVENTS[job.id].is_set():
                job.status = "canceled"
                job.error = "canceled before load"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await emit_state()
                return
            if not logged_wait:
                await emit_log(f"waiting for active generations before loading {name}")
                logged_wait = True
            await asyncio.sleep(0.25)

        async with PIPE_LOCK:
            if (not force
                    and STATE.pipe is not None and STATE.g is not None
                    and STATE.g.spec.name == name):
                await emit_log(f"{name} is already loaded")
                job.status = "done"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                return
            await broadcast({"type": "model_loading", "model": name})
            loop = asyncio.get_running_loop()
            try:
                with download_context(job.id):
                    ctx = contextvars.copy_context()
                    await loop.run_in_executor(
                        EXECUTOR, lambda: ctx.run(_do_load_sync, name)
                    )
            except DownloadCanceled:
                # User-initiated cancel — not a failure. Don't raise
                # ModelLoadError; just leave the pipe unloaded and mark
                # the job canceled.
                STATE.pipe = None
                STATE.g = None
                job.status = "canceled"
                job.error = "canceled during download"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await broadcast({"type": "model_load_canceled", "model": name})
                await emit_state()
                return
            except Exception as e:
                STATE.pipe = None
                STATE.g = None
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await broadcast({"type": "model_error", "error": f"{type(e).__name__}: {e}"})
                await emit_state()
                raise ModelLoadError(f"{type(e).__name__}: {e}") from e
        _invalidate_models_cache()
        job.status = "done"
        job.ts_done = datetime.now().timestamp()
        await emit_job(job)
        await emit_state()
        await broadcast({"type": "model_loaded", "model": name})
    finally:
        clear_active_download(job.id)
        CANCEL_EVENTS.pop(job.id, None)
        if STATE.loading_model == name:
            STATE.loading_model = None
            await emit_state()
