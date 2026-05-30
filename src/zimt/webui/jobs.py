"""Job lifecycle workers.

The async :func:`run_job` is fired off via :func:`asyncio.create_task` from
:func:`zimt.webui.exec_api.api_exec`. It serializes against the global
:data:`PIPE_LOCK` so only one pipeline call is in flight at a time, then
runs :func:`zimt.generate.generate` inside the single-worker
:data:`EXECUTOR` thread pool. Progress + cancellation are routed through
the pipeline's ``callback_on_step_end`` hook.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any

import random as _random

from ..dynamics import DynamicsSyntaxError, expand, has_dynamics
from ..generate import CancelledByUser, UninstalledLoraError, generate
from ..generate import GenConfig
from ..paths import OUT_DIR
from .outputs import read_png_meta
from .state import CANCEL_EVENTS, EXECUTOR, Job, PIPE_LOCK, STATE
from .ws import broadcast, emit_job, emit_log


def _run_generate_sync(job: Job, pipe: Any, g: GenConfig,
                       raw_prompt: str, seed: int, raw: bool,
                       loop: asyncio.AbstractEventLoop,
                       command_line: str, out_dir: str) -> str:
    """Worker-thread entry. Builds the cancel/progress callback then dispatches."""
    ev = CANCEL_EVENTS.get(job.id)

    def _on_step(step: int, total: int) -> None:
        if ev is not None and ev.is_set():
            raise CancelledByUser()
        job.step = step
        job.total_steps = total
        # Fire-and-forget: schedule the WS broadcast on the event loop.
        asyncio.run_coroutine_threadsafe(emit_job(job), loop)

    return generate(
        pipe, g, raw_prompt, seed, raw=raw, on_step=_on_step,
        command_line=command_line, out_dir=out_dir,
    )


async def run_job(job: Job, raw_prompt: str, seed: int, raw: bool,
                  g: GenConfig, command_line: str = "",
                  out_dir: str = OUT_DIR, profile: str = "") -> None:
    """Acquire the pipeline lock, run one generation, broadcast events."""
    # Pre-expand the template (if any) BEFORE acquiring PIPE_LOCK so a
    # syntax error surfaces immediately and so the WS log can show both
    # the original template and the resolved text. The same seed +
    # text yields the same expansion inside generate() — that re-run
    # is the one that actually feeds the encoder, the call here only
    # exists to drive the UI log and report syntax errors early.
    if has_dynamics(raw_prompt):
        try:
            expanded = expand(raw_prompt, _random.Random(seed))
        except DynamicsSyntaxError as e:
            job.status = "error"
            job.error = f"template: {e}"
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_log(f"template: {e}", level="error")
            await emit_job(job)
            return
        if expanded != raw_prompt:
            await emit_log(f"template: {raw_prompt!r} → {expanded!r}")

    async with PIPE_LOCK:
        ev = CANCEL_EVENTS.get(job.id)
        if ev is not None and ev.is_set():
            job.status = "canceled"
            job.error = "canceled before start"
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_job(job)
            return

        pipe = STATE.pipe
        if pipe is None:
            job.status = "error"
            job.error = "no model loaded"
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_job(job)
            return
        if STATE.g is None or STATE.g.spec.name != g.spec.name:
            job.status = "error"
            job.error = f"loaded model changed before generation: expected {g.spec.name}"
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_job(job)
            return

        from ..generate import compose_prompt
        job.status = "running"
        job.model = g.spec.name
        job.full_prompt = compose_prompt(g.spec, raw_prompt, raw=raw)
        job.negative_prompt = g.negative_prompt if g.cfg > 0 else ""
        await emit_job(job)

        try:
            loop = asyncio.get_running_loop()
            path = await loop.run_in_executor(
                EXECUTOR, _run_generate_sync,
                job, pipe, g, raw_prompt, seed, raw, loop, command_line, out_dir,
            )
        except CancelledByUser:
            job.status = "canceled"
            job.error = "canceled during generation"
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_job(job)
            return
        except UninstalledLoraError as e:
            job.status = "error"
            job.error = str(e)
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_job(job)
            return
        except Exception as e:
            job.status = "error"
            # class-name + message; repr(e) would render as
            # "RuntimeError('CUDA out of memory')" which the UI surfaces verbatim.
            job.error = f"{type(e).__name__}: {e}"
            job.ts_done = datetime.now().timestamp()
            CANCEL_EVENTS.pop(job.id, None)
            await emit_job(job)
            return

        job.path = os.path.relpath(path, OUT_DIR)
        job.status = "done"
        job.ts_done = datetime.now().timestamp()
        CANCEL_EVENTS.pop(job.id, None)

    # Outside the lock: announce result + new output entry. The entry name is
    # the bare filename; `profile` lets each client ignore additions for a
    # profile other than the one its tab is currently viewing.
    await emit_job(job)
    full = os.path.join(OUT_DIR, job.path)
    try:
        st_size = os.path.getsize(full)
    except OSError:
        st_size = 0
    await broadcast({
        "type": "output_added",
        "profile": profile,
        "entry": {
            "name": os.path.basename(job.path),
            "mtime": job.ts_done,
            "size": st_size,
            "fav": False,
            "metadata": read_png_meta(full),
        },
    })
