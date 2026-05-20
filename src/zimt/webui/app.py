"""FastAPI app + entry point :func:`run_web`.

Almost all client↔server communication runs over the single
``/ws`` WebSocket — see :mod:`zimt.webui.rpc` for the request/response
envelope and :mod:`zimt.webui.ws` for the server→client event broadcasts.

What's still HTTP, and why:
  * ``GET /``                   — initial page load.
  * ``GET /static/*``           — JS / CSS / images for the page itself.
  * ``GET /api/outputs/{name}`` — generated PNGs served to ``<img src>``;
    browsers can't load binary into ``<img>`` from a WebSocket without
    blob-URL gymnastics that defeat caching.

Auth is enforced once at the WS handshake and HTTP middleware below; the
RPC layer itself doesn't re-check (a connected socket is by definition
authenticated for the life of the connection).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import json
import os
import secrets
import time
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .. import gpu_stats
from ..models.registry import MODELS
from ..paths import FAV_DIR, OUT_DIR, STATIC_DIR, ensure_dirs
from .downloads import install as install_download_hook
from .exec_api import ExecBody, api_exec as exec_handler
from .loader import ModelLoadError, load_model
from ..models.custom import (
    CustomDescriptorError, delete_custom, reload_custom_into_registries,
    write_custom,
)
from .models_info import models_info
from .outputs import list_outputs, read_png_meta, resolve_output, safe_name
from .prefetch import PrefetchError, prefetch_model
from .rpc import dispatch, method, rpc_error
from .state import CANCEL_EVENTS, EXECUTOR, STATE, register_task
from .ws import broadcast, emit_job, emit_state

app = FastAPI()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _web_auth_token() -> str:
    return os.environ.get("ZIMT_AUTH_TOKEN", "")


def _auth_matches(auth_header: str, token: str) -> bool:
    if not token:
        return True
    if auth_header.startswith("Bearer "):
        return hmac.compare_digest(auth_header.removeprefix("Bearer ").strip(), token)
    if not auth_header.startswith("Basic "):
        return False
    encoded = auth_header.removeprefix("Basic ").strip()
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    _user, sep, password = decoded.partition(":")
    if not sep:
        return False
    return hmac.compare_digest(password, token)


def _origin_allowed(origin: str, host: str, *, require_origin: bool = False) -> bool:
    if not origin:
        # Browsers omit Origin on top-level navigations (GET /, static files).
        # Non-browser RPC callers must always send Origin; when require_origin
        # is True (WS handshake) an empty Origin is rejected.
        return not require_origin
    same_host = {f"http://{host}", f"https://{host}"}
    configured = os.environ.get("ZIMT_ALLOWED_ORIGINS", "")
    if configured:
        allowed = {item.strip() for item in configured.split(",") if item.strip()}
        return origin in allowed or origin in same_host
    return origin in same_host


def _request_permitted(headers: Any, *, require_origin: bool = False) -> bool:
    host = headers.get("host", "")
    origin = headers.get("origin", "")
    if not _origin_allowed(origin, host, require_origin=require_origin):
        return False
    return _auth_matches(headers.get("authorization", ""), _web_auth_token())


@app.middleware("http")
async def _require_auth(request: Request, call_next: Any) -> Response:
    if _request_permitted(request.headers):
        return await call_next(request)
    return Response(
        content="authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="zimt", charset="UTF-8"'},
    )


# ---------------------------------------------------------------------------
# Static + image routes (the only remaining HTTP endpoints)
# ---------------------------------------------------------------------------

class _NoCacheStaticFiles(StaticFiles):
    """Static files served with ``Cache-Control: no-store``.

    We iterate on the UI a lot; stale cached CSS/JS caused at least one bug
    that wasn't actually a bug. The cost is negligible on localhost.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store"},
    )


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


# ---------------------------------------------------------------------------
# Background loops + lifecycle hooks
# ---------------------------------------------------------------------------

async def _gpu_stats_loop() -> None:
    while True:
        try:
            if STATE.clients:
                await broadcast({"type": "gpu_stats", "stats": gpu_stats.current()})
        except Exception:
            pass
        await asyncio.sleep(2.0)


@app.on_event("startup")
async def _on_startup() -> None:
    # Wire the HF download progress hook so it sees every download from
    # this point on. Capture the running loop so the tqdm subclass (which
    # is called from a worker thread) can schedule WS broadcasts here.
    install_download_hook(asyncio.get_running_loop())
    # Load user-supplied LoRA / base descriptors from disk.
    report = reload_custom_into_registries()
    if report["errors"]:
        for err in report["errors"]:
            print(f"zimt: custom descriptor error: {err}")
    if report["bases"] or report["loras"]:
        print(f"zimt: loaded custom bases={report['bases']} loras={report['loras']}")
    register_task(_gpu_stats_loop())


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
    EXECUTOR.shutdown(wait=True, cancel_futures=True)
    # Drain the prefetch executor too; it runs HF snapshot_download on a
    # separate single-worker pool, and without an explicit shutdown a
    # SIGTERM during prefetch is left to systemd's KILL.
    from .prefetch import _PREFETCH_EXECUTOR
    _PREFETCH_EXECUTOR.shutdown(wait=True, cancel_futures=True)
    print("zimt: shutdown complete")


# ---------------------------------------------------------------------------
# RPC methods (replaces the old HTTP API)
# ---------------------------------------------------------------------------

class _ModelSwitchParams(BaseModel):
    name: str


@method("state")
async def _rpc_state(_params: dict[str, Any]) -> dict[str, Any]:
    return STATE.state_dict()


@method("gpu_stats")
async def _rpc_gpu_stats(_params: dict[str, Any]) -> dict[str, Any]:
    return dict(gpu_stats.current())


@method("model_switch")
async def _rpc_model_switch(params: dict[str, Any]) -> dict[str, Any]:
    try:
        body = _ModelSwitchParams.model_validate(params)
    except Exception as e:
        raise rpc_error(str(e))
    try:
        await load_model(body.name)
    except ModelLoadError as e:
        raise rpc_error(str(e))
    return STATE.state_dict()


@method("models_info")
async def _rpc_models_info(_params: dict[str, Any]) -> dict[str, Any]:
    return models_info()


@method("model_download")
async def _rpc_model_download(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    if not isinstance(name, str):
        raise rpc_error("missing model name")
    kind = params.get("kind", "base")
    if kind not in ("base", "lora"):
        raise rpc_error("kind must be 'base' or 'lora'")
    try:
        job_id = await prefetch_model(name, kind=kind)
    except PrefetchError as e:
        raise rpc_error(str(e))
    return {"job_id": job_id}


@method("custom_add")
async def _rpc_custom_add(params: dict[str, Any]) -> dict[str, Any]:
    kind = params.get("kind")
    descriptor = params.get("descriptor")
    if kind not in ("base", "lora"):
        raise rpc_error("kind must be 'base' or 'lora'")
    if not isinstance(descriptor, dict):
        raise rpc_error("descriptor must be an object")
    try:
        write_custom(kind, descriptor)
        report = reload_custom_into_registries()
    except CustomDescriptorError as e:
        raise rpc_error(str(e))
    await broadcast({"type": "registries_changed"})
    await emit_state()
    return {"loaded": report}


@method("custom_remove")
async def _rpc_custom_remove(params: dict[str, Any]) -> dict[str, Any]:
    kind = params.get("kind")
    name = params.get("name")
    if kind not in ("base", "lora") or not isinstance(name, str):
        raise rpc_error("need kind=base|lora and name")
    try:
        delete_custom(kind, name)
        report = reload_custom_into_registries()
    except CustomDescriptorError as e:
        raise rpc_error(str(e))
    await broadcast({"type": "registries_changed"})
    await emit_state()
    return {"loaded": report}


@method("exec")
async def _rpc_exec(params: dict[str, Any]) -> dict[str, Any]:
    try:
        body = ExecBody.model_validate(params)
    except Exception as e:
        raise rpc_error(str(e))
    return await exec_handler(body)


@method("job_get")
async def _rpc_job_get(params: dict[str, Any]) -> dict[str, Any]:
    job_id = params.get("id")
    if not isinstance(job_id, str):
        raise rpc_error("missing job id")
    job = STATE.jobs.get(job_id)
    if not job:
        raise rpc_error("no such job")
    return asdict(job)


@method("job_cancel")
async def _rpc_job_cancel(params: dict[str, Any]) -> dict[str, Any]:
    job_id = params.get("id")
    if not isinstance(job_id, str):
        raise rpc_error("missing job id")
    job = STATE.jobs.get(job_id)
    if not job:
        raise rpc_error("no such job")
    if job.status in ("done", "error", "canceled"):
        return {"ok": False, "reason": f"already {job.status}"}
    ev = CANCEL_EVENTS.get(job_id)
    if ev is None:
        return {"ok": False, "reason": "no cancel event"}
    ev.set()
    return {"ok": True, "status": job.status}


@method("jobs_cancel_all")
async def _rpc_jobs_cancel_all(_params: dict[str, Any]) -> dict[str, Any]:
    count = 0
    for job_id, job in STATE.jobs.items():
        if job.kind in ("generate", "download") and job.status in ("queued", "running"):
            ev = CANCEL_EVENTS.get(job_id)
            if ev is not None and not ev.is_set():
                ev.set()
                count += 1
    return {"canceled": count}


@method("jobs_clear_completed")
async def _rpc_jobs_clear_completed(_params: dict[str, Any]) -> dict[str, Any]:
    cleared: list[str] = []
    for jid in list(STATE.jobs.keys()):
        if STATE.jobs[jid].status in ("done", "error", "canceled"):
            del STATE.jobs[jid]
            CANCEL_EVENTS.pop(jid, None)
            cleared.append(jid)
    if cleared:
        await broadcast({"type": "jobs_cleared", "ids": cleared})
    return {"cleared": len(cleared)}


@method("outputs_list")
async def _rpc_outputs_list(params: dict[str, Any]) -> dict[str, Any]:
    tab = params.get("tab", "all")
    if tab not in ("all", "favs"):
        raise rpc_error("tab must be 'all' or 'favs'")
    page = params.get("page", 1)
    per_page = params.get("per_page", 60)
    try:
        page = int(page)
        per_page = int(per_page)
    except (TypeError, ValueError):
        raise rpc_error("page and per_page must be integers")
    if page < 1:
        raise rpc_error("page must be >= 1")
    if per_page < 1 or per_page > 200:
        raise rpc_error("per_page must be in 1..200")
    return list_outputs(tab=tab, page=page, per_page=per_page)


@method("output_favorite")
async def _rpc_output_favorite(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    favorite = params.get("favorite")
    if not isinstance(name, str) or not isinstance(favorite, bool):
        raise rpc_error("missing name or favorite flag")
    found = resolve_output(name)
    if not found:
        raise rpc_error("not found")
    src, is_fav = found
    if is_fav == favorite:
        return {"name": name, "fav": is_fav}
    dst_dir = FAV_DIR if favorite else OUT_DIR
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        raise rpc_error("destination already exists")
    os.replace(src, dst)
    new_meta = read_png_meta(dst)
    try:
        st = os.stat(dst)
        mtime, size = st.st_mtime, st.st_size
    except OSError:
        mtime, size = 0.0, 0
    entry = {"name": name, "mtime": mtime, "size": size,
             "fav": favorite, "metadata": new_meta}
    await broadcast({"type": "favorite_changed", "entry": entry})
    return entry


@method("outputs_cleanup")
async def _rpc_outputs_cleanup(_params: dict[str, Any]) -> dict[str, Any]:
    """Delete every PNG directly in OUT_DIR. Files in OUT_DIR/fav/ are spared."""
    deleted = 0
    if os.path.isdir(OUT_DIR):
        for name in os.listdir(OUT_DIR):
            if name.startswith(".") or not name.lower().endswith(".png"):
                continue
            path = os.path.join(OUT_DIR, name)
            if os.path.islink(path) or not os.path.isfile(path):
                # Skip symlinks: os.path.isfile follows symlinks and would
                # allow os.remove to delete a target outside OUT_DIR. We only
                # want to delete real regular files.
                continue
            try:
                os.unlink(path)
                deleted += 1
            except OSError:
                pass
    await broadcast({"type": "outputs_cleared"})
    return {"deleted": deleted}


# ---------------------------------------------------------------------------
# WebSocket — RPC dispatch + event sink
# ---------------------------------------------------------------------------

# Heartbeat timings. The server sends a ping every PING_INTERVAL and
# closes the socket with code 1012 (retriable) if no matching pong comes
# back within PONG_TIMEOUT. The client mirrors the protocol; either side
# detecting a stalled peer takes the connection down so the client can
# reconnect via :mod:`resilient-ws-ui`-style overlapping recovery.
#
# Reverse proxies (nginx) typically idle out WS at 60s; 15s pings keep
# the channel warm well under that. PONG_TIMEOUT is generous because
# the application is single-loop and a slow ``exec`` handler must not
# starve heartbeats — we run them as a separate task per socket.
PING_INTERVAL_S = 15.0
PONG_TIMEOUT_S = 10.0


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    # WS handshakes are RPC entry-points; require an Origin header so that
    # non-browser clients without Origin are rejected (empty-Origin bypass fix).
    if not _request_permitted(ws.headers, require_origin=True):
        await ws.close(code=1008)
        return
    await ws.accept()
    STATE.clients.add(ws)

    pending_pings: dict[str, float] = {}
    hb_task: asyncio.Task[None] | None = None

    async def _send(payload: dict[str, Any]) -> bool:
        try:
            await ws.send_text(json.dumps(payload))
            return True
        except Exception:
            return False

    async def _watchdog(nonce: str) -> None:
        await asyncio.sleep(PONG_TIMEOUT_S)
        if nonce in pending_pings:
            # No matching pong — peer is unresponsive. 1012 = service
            # restart, which our retriable-close list (R7) treats as
            # reconnectable on the client.
            try:
                await ws.close(code=1012, reason="ping timeout")
            except Exception:
                pass

    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL_S)
                nonce = secrets.token_hex(8)
                pending_pings[nonce] = time.monotonic()
                if not await _send({"type": "ping", "nonce": nonce,
                                    "ts": time.time() * 1000.0}):
                    return
                register_task(_watchdog(nonce))
        except asyncio.CancelledError:
            pass

    try:
        # Send the current state as a "hello" so the UI can render
        # immediately on connect without an extra round-trip.
        await ws.send_text(json.dumps({"type": "state", "state": STATE.state_dict()}))
        # Re-emit any in-flight jobs so a reconnecting client sees the
        # queue rather than waiting for the next event.
        for job in STATE.jobs.values():
            await ws.send_text(json.dumps({"type": "job", "job": asdict(job)}))

        hb_task = asyncio.create_task(_heartbeat())

        while True:
            raw = await ws.receive_text()
            # Cheap parse to intercept heartbeat frames before we
            # spawn a dispatch task. Unknown frames fall through to
            # the RPC dispatcher, which itself ignores non-``req``.
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict):
                t = msg.get("type")
                if t == "pong":
                    nonce = msg.get("nonce")
                    if isinstance(nonce, str):
                        pending_pings.pop(nonce, None)
                    continue
                if t == "ping":
                    # Reply on the same socket. Echo the client nonce
                    # so the client can correlate, and pass-through ts
                    # so the client can compute RTT.
                    await _send({
                        "type": "pong",
                        "nonce": msg.get("nonce"),
                        "ts": msg.get("ts"),
                    })
                    continue
            # Each request is dispatched as its own task so a slow
            # handler (e.g. ``exec`` that awaits a model download)
            # doesn't block other messages on this socket.
            register_task(dispatch(ws, raw))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if hb_task is not None:
            hb_task.cancel()
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


__all__ = [
    "app", "api_output", "ws_endpoint", "emit_job",
    "safe_name", "run_web",
]
