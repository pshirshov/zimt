"""Interactive REPL — drives the same command set as the web ``/api/exec``."""

from __future__ import annotations

import random
from typing import Any

import torch

from ..buckets import parse_res
from ..generate import GenConfig, generate, load_spec, unload
from ..models.registry import MODELS
from ..preview import IN_TMUX, detect_protocol, preview
from .commands import parse_commands
from .history import init_readline


def _help() -> None:
    print("commands:")
    print("  <prompt>             generate one image with a random seed")
    print("  /raw <prompt>        skip the model's auto-prefix")
    print("  /many N <prompt>     generate N images, each with a fresh random seed")
    print("  /seed N              pin seed for the next single generation")
    print("  /negprompt [text]    set negative prompt; bare shows; '-' clears")
    print("  /cfg [f]             set guidance scale")
    print("  /steps [N]           set num inference steps")
    print("  /size [W H]          set output (width height) — free form")
    print("  /res [N | WxH]       pick a model-preset resolution by index or set explicitly")
    print("  /sampler <name>      switch scheduler (model-specific; tab-complete)")
    print("  /clip_skip N         SDXL only — skip top N CLIP layers (0=off, 2=Pony default)")
    print("  /model [name]        switch model; bare lists available")
    print("  /tokenize <text>     show per-encoder tokenization heuristics")
    print("  /help                show this")
    print("  /quit | /exit | ^D   leave")
    print("note: negative prompt is only consulted when cfg > 0.")


def _print_state(g: GenConfig) -> None:
    preset = next(
        (label for w, h, label in g.spec.resolutions if (w, h) == (g.width, g.height)),
        "custom",
    )
    sampler = g.sampler or g.spec.default_sampler
    print(f"  model={g.spec.name}  cfg={g.cfg}  steps={g.steps}  "
          f"size={g.width}x{g.height} [{preset}]")
    print(f"  sampler={sampler}  clip_skip={g.clip_skip}")
    print(f"  negprompt={g.negative_prompt!r}")
    if g.spec.score_tags:
        print(f"  auto-prefix={g.spec.score_tags!r}")


def _list_models(current: str) -> None:
    print("available models:")
    for name, m in MODELS.items():
        marker = "*" if name == current else " "
        print(f" {marker} {name:<16}  {m.description}")


def _require_pipe(pipe: Any) -> bool:
    if pipe is None:
        print("no model loaded — use `/model <name>` first (try /model<TAB>)")
        return False
    return True


def _new_config(name: str) -> GenConfig:
    spec = MODELS[name]
    return GenConfig(
        spec=spec, cfg=spec.default_cfg,
        negative_prompt=spec.default_negative,
        height=spec.default_h, width=spec.default_w,
        steps=spec.default_steps,
    )


def repl_main() -> int:
    init_readline()
    print(f"torch={torch.__version__}  xpu={torch.xpu.is_available() if hasattr(torch, 'xpu') else False}")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"xpu[0]={torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        print(f"cuda[0]={torch.cuda.get_device_name(0)}")

    pipe: Any = None
    g: GenConfig | None = None

    proto = detect_protocol()
    tmux_note = " (in tmux: needs `set -g allow-passthrough on`)" if IN_TMUX else ""
    print(f"terminal protocol: {proto}{tmux_note}")
    _help()
    print()
    _list_models(current="")
    print("\nno model loaded — pick one with `/model <name>` to start.")

    next_seed: int | None = None

    while True:
        try:
            line = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line in ("/help", "/?"):
            _help()
            if g is not None:
                _print_state(g)
            continue

        # Use the same parser as the web layer so behaviour matches exactly.
        cmds = parse_commands(line)
        if not cmds:
            continue

        # First pass: settings / state-mutating commands.
        raw_flag = False
        prompt_text = ""
        count = 1
        for cmd, args in cmds:
            if cmd == "/_prompt":
                prompt_text = args[0] if args else ""
            elif cmd == "/model":
                if not args:
                    _list_models(g.spec.name if g else "")
                    continue
                name = args[0]
                if name not in MODELS:
                    print(f"unknown model {name!r}")
                    continue
                if g is not None and g.spec.name == name:
                    print(f"{name} is already loaded")
                    continue
                if pipe is not None:
                    print(f"unloading {g.spec.name} ...")  # type: ignore[union-attr]
                    unload(pipe)
                    pipe = None
                pipe = load_spec(MODELS[name])
                g = _new_config(name)
                _print_state(g)
            elif cmd == "/seed":
                try:
                    next_seed = int(args[0])
                    print(f"next seed = {next_seed}")
                except (ValueError, IndexError):
                    print("usage: /seed <int>")
            elif cmd == "/raw":
                raw_flag = True
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
                    else:
                        print("/many: N must be 1..256")
                except (ValueError, IndexError):
                    print("usage: /many <N> <prompt>")
            elif cmd == "/help" or cmd == "/?":
                _help()
                if g is not None:
                    _print_state(g)
            elif cmd == "/tokenize":
                if not _require_pipe(pipe) or g is None:
                    continue
                text = args[0] if args else ""
                if not text:
                    print("usage: /tokenize <text>")
                    continue
                try:
                    g.spec.tokenize_report(pipe, text)
                except Exception as e:
                    print(f"error: {e}")
            else:
                # remaining settings commands require a loaded pipeline
                if not _require_pipe(pipe) or g is None:
                    continue
                _apply_setting(cmd, args, g)

        # Second pass: maybe generate.
        if not prompt_text:
            continue
        if not _require_pipe(pipe) or g is None:
            continue
        for i in range(max(1, count)):
            seed = (next_seed + i) if next_seed is not None else random.randint(0, 2**31 - 1)
            print(f"seed={seed}")
            try:
                out = generate(pipe, g, prompt_text, seed, raw=raw_flag)
            except Exception as e:
                print(f"error: {e}")
                continue
            preview(out, proto)
        next_seed = None


def _apply_setting(cmd: str, args: list[str], g: GenConfig) -> None:
    """Apply a settings-mutating command to ``g``. Returns silently on success."""
    if cmd == "/cfg":
        try:
            g.cfg = float(args[0])
            print(f"cfg = {g.cfg}")
        except (ValueError, IndexError):
            print("usage: /cfg <float>")
    elif cmd == "/steps":
        try:
            g.steps = int(args[0])
            print(f"steps = {g.steps}")
        except (ValueError, IndexError):
            print("usage: /steps <int>")
    elif cmd == "/size":
        try:
            w, h = int(args[0]), int(args[1])
            g.width, g.height = w, h
            print(f"size = {w}x{h}")
        except (ValueError, IndexError):
            print("usage: /size <W> <H>")
    elif cmd == "/res":
        picked = parse_res(args[0] if args else "", g.spec.resolutions)
        if picked is None:
            print("usage: /res <N> (preset index) or /res WxH or /res W H")
        else:
            g.width, g.height = picked
            print(f"size = {g.width}x{g.height}")
    elif cmd == "/sampler":
        name = (args[0] if args else "").strip()
        if not name:
            print(f"current sampler = {g.sampler or g.spec.default_sampler}")
            print(f"available for {g.spec.name}: {', '.join(sorted(g.spec.samplers))}")
            return
        if name not in g.spec.samplers:
            print(f"unknown sampler {name!r}; available: {', '.join(sorted(g.spec.samplers))}")
            return
        g.sampler = name
        print(f"sampler = {name}")
    elif cmd == "/clip_skip":
        try:
            n = int(args[0])
            if n < 0 or n > 12:
                print("/clip_skip: expected 0..12")
                return
            g.clip_skip = n
            if g.spec.family != "sdxl":
                print(f"(clip_skip is SDXL-only; ignored for family={g.spec.family})")
            print(f"clip_skip = {g.clip_skip}")
        except (ValueError, IndexError):
            print("usage: /clip_skip <int>")
    elif cmd == "/negprompt":
        text = args[0] if args else ""
        if text == "" or text == "-":
            g.negative_prompt = ""
            print("negprompt cleared")
        else:
            g.negative_prompt = text
            print(f"negprompt = {g.negative_prompt!r}")
        if g.cfg == 0:
            print("(cfg=0 — negprompt is dormant; /cfg <float> to enable)")
