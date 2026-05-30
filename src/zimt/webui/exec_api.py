"""``POST /api/exec`` — runs a multi-command line like the REPL would.

Order of operations:
  1. Walk the parsed commands in input order, applying settings/model swap.
  2. After all commands, if a non-empty ``/_prompt`` remains, enqueue
     generation(s); ``count`` from a prior ``/many`` and ``seed`` from
     ``/seed`` apply. ``/raw`` toggles raw-prompt mode for the upcoming gen.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import os
import random
import re
import threading
import uuid
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from ..buckets import parse_res
from ..dynamics import DynamicsSyntaxError, validate as validate_dynamics
from ..lora_cmd import LoraCmdError, apply_lora_args, format_stack
from ..memory import MemArgError, parse_mem_args
from ..models.registry import MODELS
from ..paths import DEFAULT_PROFILE, profile_fav_dir, profile_out_dir, valid_profile_name
from ..repl.commands import parse_commands
from .jobs import run_job
from .loader import ModelLoadError, load_model
from .state import CANCEL_EVENTS, Job, STATE, register_task
from .ws import emit_job, emit_log, emit_state


class ExecBody(BaseModel):
    line: str
    # Active profile name (per browser tab). Generated images are written to
    # OUT_DIR/<profile>/. Defaults to the migration profile so non-web callers
    # and older clients keep working.
    profile: str = DEFAULT_PROFILE


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MIN_STEPS = 1
_MAX_STEPS = 100
_MIN_DIMENSION = 64
_MAX_DIMENSION = 2048
_MAX_CFG = 30.0
_MAX_PIXELS_BY_FAMILY = {
    "sdxl": 1536 * 1536,
    "zimage": 2048 * 2048,
    "flux": 2048 * 2048,
    "flux2": 2048 * 2048,
}


def _validate_size(width: int, height: int) -> str | None:
    assert STATE.g is not None
    if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
        return f"width and height must be at least {_MIN_DIMENSION}"
    if width > _MAX_DIMENSION or height > _MAX_DIMENSION:
        return f"width and height must be at most {_MAX_DIMENSION}"
    if width % 16 or height % 16:
        return "width and height must be multiples of 16"
    max_pixels = _MAX_PIXELS_BY_FAMILY[STATE.g.spec.family]
    if width * height > max_pixels:
        return f"total pixels must be at most {max_pixels} for {STATE.g.spec.family}"
    return None


async def _need_pipe(action: str, log: list[str]) -> bool:
    if STATE.loading_model is not None:
        msg = f"{action}: model {STATE.loading_model} is loading"
        log.append(msg)
        await emit_log(msg, level="error")
        return False
    if STATE.g is None or STATE.pipe is None:
        msg = f"{action}: no model loaded"
        log.append(msg)
        await emit_log(msg, level="error")
        return False
    return True


async def _exec_setting(
    cmd: str, args: list[str], log: list[str]
) -> None:
    """Mutate STATE.g for one of the settings commands."""
    if STATE.g is None:
        return
    g = STATE.g
    if cmd == "/cfg":
        try:
            cfg = float(args[0])
            if not math.isfinite(cfg) or cfg < 0.0 or cfg > _MAX_CFG:
                log.append(f"/cfg: expected finite float 0..{_MAX_CFG:g}")
                return
            g.cfg = cfg
            log.append(f"cfg = {cfg}")
            await emit_state()
        except (ValueError, IndexError):
            log.append(f"/cfg: expected finite float 0..{_MAX_CFG:g}")
    elif cmd == "/steps":
        try:
            steps = int(args[0])
            if steps < _MIN_STEPS or steps > _MAX_STEPS:
                log.append(f"/steps: expected int {_MIN_STEPS}..{_MAX_STEPS}")
                return
            g.steps = steps
            log.append(f"steps = {steps}")
            await emit_state()
        except (ValueError, IndexError):
            log.append(f"/steps: expected int {_MIN_STEPS}..{_MAX_STEPS}")
    elif cmd == "/size":
        try:
            w, h = int(args[0]), int(args[1])
            err = _validate_size(w, h)
            if err is not None:
                log.append(f"/size: {err}")
                return
            g.width, g.height = w, h
            log.append(f"size = {w}x{h}")
            await emit_state()
        except (ValueError, IndexError):
            log.append("/size: expected W H")
    elif cmd == "/res":
        picked = parse_res(args[0] if args else "", g.spec.resolutions, warn=False)
        if picked is None:
            log.append("/res: expected <N> or WxH")
        else:
            err = _validate_size(picked[0], picked[1])
            if err is not None:
                log.append(f"/res: {err}")
                return
            g.width, g.height = picked
            log.append(f"size = {picked[0]}x{picked[1]}")
            await emit_state()
    elif cmd == "/sampler":
        name = (args[0] if args else "").strip()
        if not name:
            log.append(
                f"sampler = {g.sampler or g.spec.default_sampler}  "
                f"(available: {', '.join(sorted(g.spec.samplers))})"
            )
        elif name not in g.spec.samplers:
            log.append(f"/sampler: unknown {name!r}; "
                       f"available: {', '.join(sorted(g.spec.samplers))}")
        else:
            g.sampler = name
            log.append(f"sampler = {name}")
            await emit_state()
    elif cmd == "/clip_skip":
        try:
            n = int(args[0])
            if n < 0 or n > 12:
                log.append("/clip_skip: expected 0..12")
            else:
                g.clip_skip = n
                log.append(f"clip_skip = {n}")
                if g.spec.family != "sdxl":
                    log.append(f"(clip_skip is SDXL-only; ignored for family={g.spec.family})")
                await emit_state()
        except (ValueError, IndexError):
            log.append("/clip_skip: expected int")
    elif cmd == "/negprompt":
        text = args[0] if args else ""
        if text == "" or text == "-":
            g.negative_prompt = ""
            log.append("negprompt cleared")
        else:
            g.negative_prompt = text
            log.append(f"negprompt = {text!r}")
        await emit_state()


def _tokenize_text(text: str) -> str:
    """Capture the tokenize-report output and strip ANSI for web display."""
    assert STATE.pipe is not None and STATE.g is not None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            STATE.g.spec.tokenize_report(STATE.pipe, text)
        except Exception as e:
            print(f"error: {e}", file=buf)
    return _ANSI_RE.sub("", buf.getvalue().rstrip())


# The web reference now lives in the help modal (static/index.html
# #help-modal). The REPL keeps its own text-based help_() in
# zimt/repl/main.py. /help in the web exec just emits a pointer to the
# ? button — see the handler below.


async def api_exec(body: ExecBody) -> dict[str, Any]:
    cmds = parse_commands(body.line)
    if not cmds:
        return {"job_ids": [], "log": ["empty input"]}

    profile = body.profile or DEFAULT_PROFILE
    if not valid_profile_name(profile):
        msg = f"invalid profile name: {profile!r}"
        await emit_log(msg, level="error")
        return {"job_ids": [], "log": [msg]}

    log: list[str] = []
    raw_flag = False
    prompt_text = ""
    next_seed: int | None = None
    count = 1

    for cmd, args in cmds:
        if cmd == "/_prompt":
            prompt_text = args[0] if args else ""
        elif cmd == "/model":
            if not args:
                log.append("/model: missing name")
                continue
            name = args[0]
            if name not in MODELS:
                msg = f"/model: unknown model {name!r}"
                log.append(msg)
                await emit_log(msg, level="error")
                continue
            # No "loading model: {name}" emit_log here — load_model() broadcasts
            # `model_loading`, which the UI renders as "loading {name}…" in the
            # log row. Emitting here too would produce two near-identical lines
            # for what is, from the user's perspective, a single event.
            try:
                await load_model(name)
            except ModelLoadError as e:
                msg = f"/model: {e}"
                log.append(msg)
                await emit_log(msg, level="error")
                continue
        elif cmd == "/seed":
            try:
                next_seed = int(args[0])
                log.append(f"next seed = {next_seed}")
            except (ValueError, IndexError):
                log.append("/seed: expected int")
        elif cmd == "/raw":
            raw_flag = True
            log.append("/raw enabled for next gen")
        elif cmd == "/many":
            try:
                n = int(args[0])
                # Only override prompt_text if /many has its own greedy
                # prompt arg (non-empty). Otherwise we'd erase a prompt
                # the user typed before /many on the same line.
                if len(args) > 1 and args[1]:
                    prompt_text = args[1]
                if 1 <= n <= 256:
                    count = n
                    log.append(f"batch count = {n}")
                else:
                    log.append("/many: N must be 1..256")
            except (ValueError, IndexError):
                log.append("/many: expected N <prompt>")
        elif cmd == "/lora":
            if not await _need_pipe("/lora", log):
                continue
            assert STATE.g is not None
            if not args:
                msg = f"active loras: {format_stack(STATE.g.lora_stack)}"
                log.append(msg)
                await emit_log(msg)
                continue
            try:
                messages = apply_lora_args(STATE.g.lora_stack, [args[0]], STATE.g.spec)
            except LoraCmdError as e:
                log.append(f"/lora: {e}")
                await emit_log(f"/lora: {e}", level="error")
                continue
            for ln in messages:
                log.append(ln)
                await emit_log(ln)
            await emit_state()
        elif cmd == "/mem":
            tokens = (args[0].split() if args else [])
            if not tokens:
                msg = f"mem = {STATE.mem.describe()}"
                log.append(msg)
                await emit_log(msg)
                continue
            try:
                new_mem = parse_mem_args(tokens)
            except MemArgError as e:
                msg = f"/mem: {e}"
                log.append(msg)
                await emit_log(msg, level="error")
                continue
            if new_mem == STATE.mem:
                msg = f"mem = {STATE.mem.describe()} (unchanged)"
                log.append(msg)
                await emit_log(msg)
                continue
            STATE.mem = new_mem
            msg = f"mem = {STATE.mem.describe()}"
            log.append(msg)
            await emit_log(msg)
            await emit_state()
            # If a model is loaded, reload it so the new strategy takes
            # effect now (matches the REPL's auto-reload behaviour). With
            # no model loaded, the new strategy is sticky and applies on
            # the next /model load.
            if STATE.pipe is not None and STATE.g is not None:
                current = STATE.g.spec.name
                await emit_log(
                    f"reloading {current} with mem={STATE.mem.describe()}"
                )
                try:
                    await load_model(current, force=True)
                except ModelLoadError as e:
                    msg = f"/mem reload: {e}"
                    log.append(msg)
                    await emit_log(msg, level="error")
        elif cmd == "/tokenize":
            if not await _need_pipe("/tokenize", log):
                continue
            text = args[0] if args else ""
            if not text:
                log.append("/tokenize: empty input")
                continue
            output = _tokenize_text(text)
            log.append(output)
            await emit_log(output)
        elif cmd in ("/help", "/?"):
            # The web UI hosts the full reference behind the topbar `?`
            # button (id="help-btn") — emit_log is too narrow a surface
            # for the multi-section content. We acknowledge the command
            # so users who type it from muscle memory get a pointer
            # rather than silence.
            msg = "see the ? button in the top bar for the full reference"
            log.append(msg)
            await emit_log(msg)
        elif cmd in ("/quit", "/exit", "/q"):
            log.append("(ignored — web UI; close the browser tab to leave)")
        else:
            # Settings commands all require a loaded pipeline.
            if not await _need_pipe(cmd, log):
                continue
            await _exec_setting(cmd, args, log)

    # After parsing: if a prompt remains, enqueue generation(s).
    job_ids: list[str] = []
    if not prompt_text.strip():
        return {"job_ids": job_ids, "log": log}
    if STATE.loading_model is not None:
        msg = f"generate: model {STATE.loading_model} is loading"
        log.append(msg)
        await emit_log(msg, level="error")
        return {"job_ids": job_ids, "log": log}
    if STATE.pipe is None or STATE.g is None:
        msg = "generate: no model loaded"
        log.append(msg)
        await emit_log(msg, level="error")
        return {"job_ids": job_ids, "log": log}
    # Validate template syntax once, before enqueuing any /many batch.
    # Without this a `/many 8 {red|blue` would enqueue 8 jobs that all
    # die in run_job with the same syntax error.
    try:
        validate_dynamics(prompt_text)
    except DynamicsSyntaxError as e:
        msg = f"template: {e}"
        log.append(msg)
        await emit_log(msg, level="error")
        return {"job_ids": job_ids, "log": log}

    # Ensure the profile's output dirs exist before the first job writes.
    out_dir = profile_out_dir(profile)
    os.makedirs(profile_fav_dir(profile), exist_ok=True)

    for i in range(max(1, count)):
        seed = ((next_seed + i) if next_seed is not None
                else random.randint(0, 2**31 - 1))
        job = Job(
            id=uuid.uuid4().hex,
            raw_prompt=prompt_text,
            seed=seed,
            model=STATE.g.spec.name,
        )
        # dataclasses.replace is a shallow copy; lora_stack is a list, so
        # without an explicit copy the snapshot would alias the live
        # STATE.g.lora_stack and observe any post-enqueue /lora mutation.
        g_snapshot = replace(STATE.g, lora_stack=list(STATE.g.lora_stack))
        STATE.jobs[job.id] = job
        CANCEL_EVENTS[job.id] = threading.Event()
        job_ids.append(job.id)
        await emit_job(job)
        register_task(run_job(
            job, prompt_text, seed, raw_flag, g_snapshot,
            command_line=body.line, out_dir=out_dir, profile=profile,
        ))
    return {"job_ids": job_ids, "log": log}
