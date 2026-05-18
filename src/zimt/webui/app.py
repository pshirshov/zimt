"""FastAPI app + route handlers + entry point :func:`run_web`."""

from __future__ import annotations

import io
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .. import gpu_stats
from ..models.registry import MODELS
from ..paths import FAV_DIR, OUT_DIR, STATIC_DIR, ensure_dirs
from .exec_api import ExecBody, api_exec as exec_handler
from .loader import load_model
from .outputs import list_outputs, read_png_meta, resolve_output, safe_name
from .state import CANCEL_EVENTS, EXECUTOR, STATE
from .ws import broadcast, emit_job

app = FastAPI()


class _NoCacheStaticFiles(StaticFiles):
    """Static files served with ``Cache-Control: no-store``.

    We iterate on the UI a lot; stale cached CSS/JS caused at least one bug
    that wasn't actually a bug. The cost is negligible on localhost.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


# Mount static after STATIC_DIR is set up via paths.py.
app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# GPU stats: a small background task broadcasts memory usage every ~2s
# (only while at least one WebSocket client is connected — no cost when
# nobody's looking).
# ---------------------------------------------------------------------------

async def _gpu_stats_loop() -> None:
    import asyncio
    while True:
        try:
            if STATE.clients:
                await broadcast({"type": "gpu_stats", "stats": gpu_stats.current()})
        except Exception:
            pass
        await asyncio.sleep(2.0)


@app.on_event("startup")
async def _start_gpu_stats() -> None:
    import asyncio
    asyncio.create_task(_gpu_stats_loop())


@app.on_event("shutdown")
async def _graceful_shutdown() -> None:
    """Cancel in-flight generations + drain the executor on systemd SIGTERM.

    Without this, uvicorn's graceful-shutdown only awaits HTTP requests —
    the pipeline call runs in an executor thread that uvicorn doesn't know
    about, so SIGKILL hits mid-step once systemd's TimeoutStopSec elapses.
    Here we fire every per-job cancel event (the pipeline's
    ``callback_on_step_end`` raises CancelledByUser at the next step, ~1-3s)
    then explicitly drain the executor so the pipeline call has time to
    return before the process exits.
    """
    print("zimt: shutdown — signaling in-flight generations to cancel")
    for ev in list(CANCEL_EVENTS.values()):
        ev.set()
    # cancel_futures=True drops jobs that haven't started; in-flight ones
    # are honored. wait=True blocks until the executor thread is idle.
    EXECUTOR.shutdown(wait=True, cancel_futures=True)
    print("zimt: shutdown complete")


@app.get("/api/gpu_stats")
async def api_gpu_stats() -> dict[str, Any]:
    return dict(gpu_stats.current())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    return STATE.state_dict()


class ModelSwitchBody(BaseModel):
    name: str


@app.post("/api/model")
async def api_model(body: ModelSwitchBody) -> dict[str, Any]:
    await load_model(body.name)
    return STATE.state_dict()


@app.post("/api/exec")
async def api_exec(body: ExecBody) -> dict[str, Any]:
    return await exec_handler(body)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str) -> dict[str, Any]:
    from dataclasses import asdict
    job = STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    return asdict(job)


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel(job_id: str) -> dict[str, Any]:
    job = STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    if job.status in ("done", "error", "canceled"):
        return {"ok": False, "reason": f"already {job.status}"}
    ev = CANCEL_EVENTS.get(job_id)
    if ev is None:
        return {"ok": False, "reason": "no cancel event"}
    ev.set()
    return {"ok": True, "status": job.status}


@app.post("/api/jobs/cancel-all")
async def api_cancel_all() -> dict[str, Any]:
    count = 0
    for job_id, job in STATE.jobs.items():
        if job.status in ("queued", "running"):
            ev = CANCEL_EVENTS.get(job_id)
            if ev is not None and not ev.is_set():
                ev.set()
                count += 1
    return {"canceled": count}


@app.post("/api/jobs/clear-completed")
async def api_clear_completed() -> dict[str, Any]:
    cleared: list[str] = []
    for jid in list(STATE.jobs.keys()):
        if STATE.jobs[jid].status in ("done", "error", "canceled"):
            del STATE.jobs[jid]
            CANCEL_EVENTS.pop(jid, None)
            cleared.append(jid)
    if cleared:
        await broadcast({"type": "jobs_cleared", "ids": cleared})
    return {"cleared": len(cleared)}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@app.get("/api/outputs")
async def api_outputs(
    tab: str = Query(default="all"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=60, ge=1, le=200),
) -> dict[str, Any]:
    if tab not in ("all", "favs"):
        raise HTTPException(status_code=400, detail="tab must be 'all' or 'favs'")
    return list_outputs(tab=tab, page=page, per_page=per_page)


@app.get("/api/outputs/{name}")
async def api_output(name: str, thumb: int | None = Query(default=None)) -> Response:
    found = resolve_output(name)
    if not found:
        raise HTTPException(status_code=404, detail="not found")
    path, _ = found

    if thumb is None:
        with open(path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    side = max(32, min(thumb, 1024))
    with Image.open(path) as im:
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=82)
        return Response(content=buf.getvalue(), media_type="image/jpeg")


class FavoriteBody(BaseModel):
    favorite: bool


@app.post("/api/outputs/{name}/favorite")
async def api_favorite(name: str, body: FavoriteBody) -> dict[str, Any]:
    found = resolve_output(name)
    if not found:
        raise HTTPException(status_code=404, detail="not found")
    src, is_fav = found
    if is_fav == body.favorite:
        return {"name": name, "fav": is_fav}
    dst_dir = FAV_DIR if body.favorite else OUT_DIR
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        raise HTTPException(status_code=409, detail="destination already exists")
    os.replace(src, dst)
    new_meta = read_png_meta(dst)
    try:
        st = os.stat(dst)
        mtime, size = st.st_mtime, st.st_size
    except OSError:
        mtime, size = 0.0, 0
    entry = {"name": name, "mtime": mtime, "size": size,
             "fav": body.favorite, "metadata": new_meta}
    await broadcast({"type": "favorite_changed", "entry": entry})
    return entry


@app.post("/api/outputs/cleanup")
async def api_outputs_cleanup() -> dict[str, Any]:
    """Delete every PNG directly in OUT_DIR. Files in OUT_DIR/fav/ are spared."""
    deleted = 0
    if os.path.isdir(OUT_DIR):
        for name in os.listdir(OUT_DIR):
            if name.startswith(".") or not name.lower().endswith(".png"):
                continue
            path = os.path.join(OUT_DIR, name)
            if not os.path.isfile(path):
                continue  # don't recurse into fav/
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                pass
    await broadcast({"type": "outputs_cleared"})
    return {"deleted": deleted}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    STATE.clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "state", "state": STATE.state_dict()}))
        while True:
            # No client→server messages defined; just keep the socket open.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        STATE.clients.discard(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_web(host: str, port: int) -> int:
    import uvicorn

    ensure_dirs()
    print(f"zimt web ui → http://{host}:{port}")
    print(f"  output dir: {OUT_DIR}")
    print(f"  models: {', '.join(MODELS)}  (none loaded; pick one in the UI)")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


# Silence pyright on the unused symbols imported for re-export side effects.
__all__ = [
    "app", "api_state", "api_model", "api_exec", "api_job", "api_cancel",
    "api_cancel_all", "api_clear_completed", "api_outputs", "api_output",
    "api_favorite", "api_outputs_cleanup", "ws_endpoint", "emit_job",
    "safe_name", "run_web",
]
