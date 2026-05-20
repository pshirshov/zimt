"""HuggingFace prefetch — download a model snapshot without loading it.

Companion to :mod:`zimt.webui.loader`. ``load_model`` is the
download-then-instantiate path used by ``/model``; ``prefetch_model`` is
the pure-download path used by the "download" button in the models tab,
so the user can warm the cache without kicking out whatever pipeline is
currently in memory.

Concurrency model:
  * a dedicated single-worker ``ThreadPoolExecutor`` so prefetches don't
    contend with the inference :data:`zimt.webui.state.EXECUTOR`,
  * a process-wide :class:`asyncio.Lock` so the tqdm bridge in
    :mod:`zimt.webui.downloads` (which has one global "active download"
    slot) can route progress unambiguously.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from ..models.loras import LORAS
from ..models.registry import MODELS
from .downloads import (
    DownloadCanceled, clear_active_download, download_context, set_active_download,
)
from .state import CANCEL_EVENTS, Job, STATE
from .ws import broadcast, emit_job, emit_log

_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zimt-prefetch")
_PREFETCH_LOCK = asyncio.Lock()


class PrefetchError(Exception):
    """Raised when prefetch fails (unknown model, missing huggingface_hub, etc.)."""


def _snapshot_download_sync(repo_id: str) -> None:
    """Executor-thread payload — pure download, no diffusers instantiation."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id)


async def prefetch_model(name: str, *, kind: str = "base") -> str:
    """Start a background prefetch for the registered base or LoRA ``name``.

    Returns the Job id. Raises :class:`PrefetchError` if the entry isn't
    registered or has no ``repo_id``.
    """
    if kind == "base":
        registry: dict[str, Any] = MODELS
    elif kind == "lora":
        registry = LORAS
    else:
        raise PrefetchError(f"unknown kind {kind!r}")
    if name not in registry:
        raise PrefetchError(f"unknown {kind} {name!r}")
    spec = registry[name]
    if not spec.repo_id:
        raise PrefetchError(f"{kind} {name!r} has no associated HF repo")

    # PR-06: idempotent — a duplicate download request for the same asset
    # returns the existing active job rather than queueing a second download.
    for existing in STATE.jobs.values():
        if (
            existing.kind == "download"
            and existing.target_kind == kind
            and existing.model == name
            and existing.status in {"queued", "running"}
        ):
            ev = CANCEL_EVENTS.get(existing.id)
            if ev is not None and ev.is_set():
                # cancel-pending: the cancel event is set synchronously by
                # job_cancel; the status transition lags until the runner
                # reaches its checkpoint, so we skip these and let the
                # user's retry land on a fresh Job.
                continue
            return existing.id

    job = Job(
        id=uuid.uuid4().hex,
        kind="download",
        target_kind=kind,
        status="queued",
        model=name,
    )
    STATE.jobs[job.id] = job
    CANCEL_EVENTS[job.id] = threading.Event()
    await emit_job(job)

    async def _run() -> None:
        async with _PREFETCH_LOCK:
            # Cancellation can arrive while we were parked on the lock; the
            # controlled boundary for queued downloads is right here, before
            # we flip status to running.
            if CANCEL_EVENTS[job.id].is_set():
                job.status = "canceled"
                job.error = "canceled before start"
                job.ts_done = datetime.now().timestamp()
                CANCEL_EVENTS.pop(job.id, None)
                await emit_job(job)
                await emit_log(f"prefetch canceled before start: {name}")
                return
            job.status = "running"
            await emit_job(job)
            try:
                owns_progress = set_active_download(job.id)
                if not owns_progress:
                    await emit_log(f"download progress slot busy; prefetch of {name} progress will not be broadcast")
                loop = asyncio.get_running_loop()
                with download_context(job.id):
                    ctx = contextvars.copy_context()
                    repo_id = spec.repo_id
                    await loop.run_in_executor(
                        _PREFETCH_EXECUTOR,
                        lambda: ctx.run(_snapshot_download_sync, repo_id),
                    )
                job.status = "done"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await emit_log(f"prefetched {name} ({spec.repo_id})")
                await broadcast({"type": "model_prefetched",
                                 "kind": kind, "model": name,
                                 "repo_id": spec.repo_id})
            except DownloadCanceled:
                job.status = "canceled"
                job.error = "canceled during download"
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await emit_log(f"prefetch canceled: {name}")
            except Exception as e:
                job.status = "error"
                job.error = repr(e)
                job.ts_done = datetime.now().timestamp()
                await emit_job(job)
                await emit_log(f"prefetch failed for {name}: {e!r}", "error")
            finally:
                clear_active_download(job.id)
                CANCEL_EVENTS.pop(job.id, None)

    asyncio.create_task(_run())
    return job.id
