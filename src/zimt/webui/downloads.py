"""HuggingFace download progress → WebSocket bridge.

``huggingface_hub`` reports per-file download progress through a tqdm
subclass (``huggingface_hub.utils.tqdm.tqdm``). We replace it with a
subclass that, while a model load is in progress (i.e. a
:class:`zimt.webui.state.Job` of ``kind == "download"`` is active),
mirrors each ``__init__`` / ``update`` / ``close`` into that Job's
``download_*`` fields and broadcasts a job update over the WebSocket.

The hook is installed once at server startup via :func:`install`.
Outside of an active download context the patched tqdm is a no-op
pass-through to its base — diffusers, transformers etc. also use tqdm
but we don't want to spam the UI with their bars.

Ownership invariant: at most one HuggingFace download owns the progress
slot at a time. Ownership is acquired with :func:`set_active_download`
(returns ``True`` on success) and released by the *same* job id via
:func:`clear_active_download`. A concurrent download that cannot acquire
the slot proceeds normally but its tqdm progress is not broadcast.

Per-bar ownership: callers must wrap their executor invocations with
both :func:`download_context` **and** an explicit ``copy_context``
call — ``loop.run_in_executor`` in CPython does **not** copy the
current ``contextvars.Context`` into the worker thread (only
``asyncio.Task`` creation does). The required pattern is::

    with download_context(job.id):
        ctx = contextvars.copy_context()
        await loop.run_in_executor(executor, ctx.run, func, *args)

Without the explicit ``ctx.run`` wrapper ``_owner_var`` will be
``None`` inside the worker, every ``ProgressTqdm`` bar will silently
gate as non-owning, and no progress will be broadcast. Adding a new
download path without this pattern is a silent regression.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import re
import sys
import threading
from dataclasses import asdict
from typing import Any, Optional

from .state import Job, STATE

# HF snapshot_download sets tqdm_desc to "Fetching N files" (numeric N) for the
# normal case, "Fetching ... files" for the size-unknown case, and
# "[dry-run] Fetching ... files" for dry runs.  All three should be treated as
# the file-count bar regardless of their unit (which is tqdm's default 'it').
# The regex is intentionally broad: match anything starting with an optional
# "[dry-run] " prefix followed by the word "Fetching".
_HF_FETCHING_FILES_RE = re.compile(r"^(?:\[dry-run\]\s*)?Fetching\b")


def _classify(raw_unit: str, desc: str) -> str:
    """Map HF tqdm's (unit, desc) pair to a UI-stable category.

    HF snapshot_download uses two ProgressTqdm-subclass bars:
      * unit='B' for the aggregate bytes bar (our "bytes" category).
      * unit='it' (tqdm default) with desc='Fetching N files' for the
        outer file-count thread_map bar — classified by desc, not unit.
    Anything else is opaque "items".
    """
    if raw_unit == "B":
        return "bytes"
    if raw_unit in ("file", "files"):
        return "files"
    if _HF_FETCHING_FILES_RE.match(desc):
        return "files"
    return "items"


class DownloadCanceled(Exception):
    """Raised inside the HF download stack to abort at the next tqdm boundary
    when the owning job's cancel event is set. Propagates up to the executor
    and is caught in loader.load_model / prefetch.prefetch_model."""


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

# Per-bar identity: set by download_context() before run_in_executor so
# the ContextVar propagates into the worker thread. Each ProgressTqdm bar
# captures the value at construction time and gates all events on it.
_owner_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "zimt_download_owner", default=None
)


@contextlib.contextmanager
def download_context(job_id: str):
    """Mark the current asyncio/thread context as belonging to *job_id*.

    Does **not** propagate automatically through ``loop.run_in_executor``.
    Callers must use ``ctx = contextvars.copy_context()`` after entering
    this context manager and dispatch via ``ctx.run``. tqdm bars created
    inside the worker then capture *job_id* and only emit if it matches
    the current slot owner. See :func:`set_active_download`.
    """
    token = _owner_var.set(job_id)
    try:
        yield
    finally:
        _owner_var.reset(token)


def set_active_download(job_id: str) -> bool:
    """Attempt to claim the progress slot for *job_id*.

    Returns ``True`` if the slot was free (or already owned by this job)
    and ownership is now held. Returns ``False`` if another job owns the
    slot; the caller may continue its work but tqdm progress will not be
    broadcast.

    Release ownership with :func:`clear_active_download`.
    """
    global _active_job_id
    with _active_lock:
        if _active_job_id is None or _active_job_id == job_id:
            _active_job_id = job_id
            return True
        return False


def clear_active_download(job_id: str) -> None:
    """Release the progress slot — but only if *job_id* is the current owner.

    Owner-scoped: a job that never acquired the slot (set_active_download
    returned False) will not accidentally clear another job's ownership.
    """
    global _active_job_id
    with _active_lock:
        if _active_job_id == job_id:
            _active_job_id = None


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
            # Capture the owning job id before super().__init__ in case
            # the base ever calls overridden methods during construction.
            self._zimt_owner = _owner_var.get()
            super().__init__(*args, **kwargs)
            # tqdm callbacks are the only controlled cancellation boundary
            # inside HF's snapshot_download — raise here so the executor
            # surfaces DownloadCanceled before any further bytes are read.
            self._zimt_check_cancel()
            self._emit()

        def update(self, n: int = 1) -> Any:  # noqa: D401
            # Mirror the check at update time so a cancel arriving mid-file
            # takes effect at the next tqdm event rather than waiting for
            # the next file's __init__.
            self._zimt_check_cancel()
            r = super().update(n)
            self._emit()
            return r

        def _zimt_check_cancel(self) -> None:
            jid = self._zimt_owner
            if jid is None:
                return
            from .state import CANCEL_EVENTS
            ev = CANCEL_EVENTS.get(jid)
            if ev is not None and ev.is_set():
                raise DownloadCanceled()

        def close(self) -> None:
            super().close()
            with _active_lock:
                jid = _active_job_id
            if jid is None or jid != self._zimt_owner:
                return
            job = STATE.jobs.get(jid)
            if job is not None and job.kind == "download":
                # download_files_done is tracked exclusively via the
                # file-count bar's _emit (its current `n` value); close()
                # does not mutate it. See _classify.
                _schedule_emit(job)

        def _emit(self) -> None:
            with _active_lock:
                jid = _active_job_id
            if jid is None or jid != getattr(self, "_zimt_owner", None):
                return
            job = STATE.jobs.get(jid)
            if job is None or job.kind != "download":
                return
            desc = (getattr(self, "desc", "") or "").strip()
            job.download_file = desc or job.download_file
            cat = _classify(getattr(self, "unit", "") or "", desc)
            job.download_unit = cat
            try:
                job.download_n = int(self.n or 0)
                job.download_total = int(self.total or 0)
            except (TypeError, ValueError):
                pass
            if cat == "files":
                try:
                    job.download_files_done = int(self.n or 0)
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
