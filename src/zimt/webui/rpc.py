"""WebSocket-based RPC.

Replaces the previous HTTP API. The wire format is JSON, one message per
frame:

  client → server   ``{"type":"req","id":"<uuid>","method":"<name>",
                       "params":{...}}``
  server → client   ``{"type":"resp","id":"<uuid>","ok":true,"result":{...}}``
                    ``{"type":"resp","id":"<uuid>","ok":false,"error":"..."}``

Event broadcasts (``state``, ``job``, ``log``, ``gpu_stats``, …) keep
their existing shape — see :mod:`zimt.webui.ws`.

The ``id`` is opaque to the server; the client uses it to correlate the
response with the original request. Unknown methods come back as
``{"ok":false,"error":"unknown method: …"}``.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Awaitable, Callable

from fastapi import WebSocket

# Method handler signature: takes raw params dict, returns a JSON-able
# result. Handlers may raise — the dispatcher converts that into an
# error response without dropping the connection.
Handler = Callable[[dict[str, Any]], Awaitable[Any]]

_METHODS: dict[str, Handler] = {}


def method(name: str) -> Callable[[Handler], Handler]:
    """Decorator to register a handler under a method name."""
    def deco(fn: Handler) -> Handler:
        if name in _METHODS:
            raise RuntimeError(f"duplicate RPC method registration: {name}")
        _METHODS[name] = fn
        return fn
    return deco


async def dispatch(ws: WebSocket, raw: str) -> None:
    """Parse one inbound frame and send back exactly one response.

    Errors during parse/dispatch are reported back on the same socket
    rather than closing it — a single bad request shouldn't tear down
    the whole UI session.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        await _send(ws, {"type": "resp", "id": None, "ok": False,
                         "error": f"invalid json: {e}"})
        return

    if not isinstance(msg, dict) or msg.get("type") != "req":
        # Ignore non-request frames (the WS is otherwise server→client only).
        return

    req_id = msg.get("id")
    method_name = msg.get("method")
    params = msg.get("params") or {}

    if not isinstance(method_name, str):
        await _send(ws, {"type": "resp", "id": req_id, "ok": False,
                         "error": "missing method"})
        return
    if not isinstance(params, dict):
        await _send(ws, {"type": "resp", "id": req_id, "ok": False,
                         "error": "params must be an object"})
        return

    handler = _METHODS.get(method_name)
    if handler is None:
        await _send(ws, {"type": "resp", "id": req_id, "ok": False,
                         "error": f"unknown method: {method_name}"})
        return

    try:
        result = await handler(params)
    except _RpcError as e:
        # Expected, user-facing error — no traceback.
        await _send(ws, {"type": "resp", "id": req_id, "ok": False,
                         "error": str(e)})
        return
    except Exception as e:
        # Unexpected — log to stdout so it lands in journalctl, return
        # the repr so the UI can show *something*.
        traceback.print_exc()
        await _send(ws, {"type": "resp", "id": req_id, "ok": False,
                         "error": repr(e)})
        return

    await _send(ws, {"type": "resp", "id": req_id, "ok": True,
                     "result": result})


class _RpcError(Exception):
    """Raise this from a handler to send a clean error back to the
    client without a server-side traceback."""


def rpc_error(msg: str) -> _RpcError:
    return _RpcError(msg)


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        # If the socket is dead the broadcast loop will reap it; nothing
        # for us to do here beyond not crashing the dispatch task.
        pass
