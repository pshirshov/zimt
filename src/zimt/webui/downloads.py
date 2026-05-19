"""HuggingFace download progress → WebSocket bridge.

``huggingface_hub`` reports per-file download progress through a tqdm
subclass (``huggingface_hub.utils.tqdm.tqdm``). We replace it with a
subclass that, while a model load is in progress (i.e. a
:class:`zimt.webui.state.Job` of ``kind == "download"`` is active),
mirrors each ``__init__`` / ``update`` / ``close`` into that Job's
``download_*`` fields and broadcasts a job update over the WebSocket.

The hook is installed once at server startup via :func:`install`.
Outside of an active download context (``set_active_download(None)``)
the patched tqdm is a no-op pass-through to its base — diffusers,
transformers etc. also use tqdm but we don't want to spam the UI with
their bars.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import asdict
from typing import Any, Optional

from .state import Job, STATE

# The asyncio loop running the FastAPI app. Captured at startup so the
# tqdm hook (which runs on a worker thread inside the HF download stack)
# can schedule broadcasts back onto the main loop.
_loop: Optional[asyncio.AbstractEventLoop] = None

# id of the Job that should receive download progress events, plus its
# guard. tqdm callbacks run in worker threads; mutating STATE.jobs from
# there is fine because we only update fields the asyncio side reads,
# and the actual ws send is scheduled back on _loop.
_active_lock = threading.Lock()
_active_job_id: Optional[str] = None


def set_active_download(job_id: Optional[str]) -> None:
    """Mark which Job receives the next tqdm events. Pass ``None`` to
    detach (idiomatic at the end of :func:`zimt.webui.loader.load_model`).

    Caller is responsible for creating / finalizing the Job — this only
    routes intermediate progress.
    """
    global _active_job_id
    with _active_lock:
        _active_job_id = job_id


def install(loop: asyncio.AbstractEventLoop) -> None:
    """Patch ``huggingface_hub.utils.tqdm.tqdm`` to broadcast progress.

    Must run before any HF download starts (we replace symbols that
    other HF modules may already have imported, so we also walk
    ``sys.modules`` and re-bind by identity). Idempotent.
    """
    global _loop
    _loop = loop

    try:
        # huggingface_hub's __init__ shadows the ``tqdm`` submodule name
        # with the class of the same name, so a plain
        # ``import huggingface_hub.utils.tqdm as m`` rebinds ``m`` to the
        # class. Grab the actual module from sys.modules instead.
        import huggingface_hub.utils.tqdm  # noqa: F401 (force import)
        hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
        OriginalTqdm = hf_tqdm_mod.tqdm
    except (ImportError, KeyError, AttributeError):
        # No HF in the env: nothing to patch (CPU/CUDA backends still
        # work — they get their torch from nixpkgs and may not pull
        # huggingface_hub during normal operation).
        return

    if getattr(OriginalTqdm, "_zimt_patched", False):
        return  # already installed

    class ProgressTqdm(OriginalTqdm):  # type: ignore[misc, valid-type]
        _zimt_patched = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._emit()

        def update(self, n: int = 1) -> Any:  # noqa: D401
            r = super().update(n)
            self._emit()
            return r

        def close(self) -> None:
            super().close()
            # Bump files_done on close so the UI can show "k files
            # finished" even when total is unknown.
            with _active_lock:
                jid = _active_job_id
            if jid is not None:
                job = STATE.jobs.get(jid)
                if job is not None and job.kind == "download":
                    job.download_files_done += 1
                    _schedule_emit(job)

        def _emit(self) -> None:
            with _active_lock:
                jid = _active_job_id
            if jid is None:
                return
            job = STATE.jobs.get(jid)
            if job is None or job.kind != "download":
                return
            desc = (getattr(self, "desc", "") or "").strip()
            job.download_file = desc or job.download_file
            try:
                job.download_n = int(self.n or 0)
                job.download_total = int(self.total or 0)
            except (TypeError, ValueError):
                pass
            _schedule_emit(job)

    hf_tqdm_mod.tqdm = ProgressTqdm

    # huggingface_hub modules that imported the symbol before this hook
    # ran still hold the old class; rebind by identity.
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        attr = getattr(mod, "tqdm", None)
        if attr is OriginalTqdm:
            try:
                setattr(mod, "tqdm", ProgressTqdm)
            except Exception:
                pass


def _schedule_emit(job: Job) -> None:
    """Schedule ``emit_job`` on the asyncio loop from a worker thread."""
    if _loop is None or _loop.is_closed():
        return
    # Avoid recursive imports at module load time.
    from .ws import emit_job
    asyncio.run_coroutine_threadsafe(emit_job(job), _loop)


# Optional debug aid: returns a snapshot of the currently-tracked job
# so tests / log lines can inspect state without grabbing the lock.
def current_download() -> dict[str, Any] | None:
    with _active_lock:
        jid = _active_job_id
    if jid is None:
        return None
    job = STATE.jobs.get(jid)
    return asdict(job) if job else None
