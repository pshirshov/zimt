"""Top-level entry point — argparse + dispatch between REPL and web mode.

The env-affecting flags (``--out-dir`` / ``--hf-cache``) must take effect
*before* the heavy imports because ``huggingface_hub.constants.HF_HOME`` is
computed at module load time. :func:`zimt.cli.early_env_args` runs in
:mod:`zimt.__main__` before anything else.
"""

from __future__ import annotations

import argparse
import os
import sys


def early_env_args() -> None:
    """Sniff env-affecting flags from ``sys.argv`` and set the corresponding
    env vars *before* anything from torch / diffusers / huggingface_hub
    imports."""
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--hf-cache" and i + 1 < len(args):
            os.environ["HF_HOME"] = os.path.abspath(args[i + 1])
            i += 2; continue
        if a.startswith("--hf-cache="):
            os.environ["HF_HOME"] = os.path.abspath(a.split("=", 1)[1])
            i += 1; continue
        if a == "--out-dir" and i + 1 < len(args):
            os.environ["ZIMT_OUT_DIR"] = os.path.abspath(args[i + 1])
            i += 2; continue
        if a.startswith("--out-dir="):
            os.environ["ZIMT_OUT_DIR"] = os.path.abspath(a.split("=", 1)[1])
            i += 1; continue
        i += 1


def main() -> int:
    # The env-vars have already been applied by zimt.__main__; importing
    # the heavy bits is safe here.
    from .paths import OUT_DIR, ensure_dirs
    from .repl import repl_main
    from .webui import run_web

    ensure_dirs()

    parser = argparse.ArgumentParser(
        prog="zimt",
        description="Multi-model image generation REPL + web UI.",
    )
    parser.add_argument(
        "--web", metavar="HOST:PORT", default=None,
        help="Start the HTTP/WebSocket UI on HOST:PORT instead of the CLI REPL.",
    )
    parser.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help=f"Output directory for generated images. Default: ./out  "
             f"(also: $ZIMT_OUT_DIR). Currently: {OUT_DIR}",
    )
    parser.add_argument(
        "--hf-cache", metavar="DIR", default=None,
        help="HuggingFace model cache directory (sets $HF_HOME). "
             f"Currently: {os.environ.get('HF_HOME') or '<default>'}",
    )
    args = parser.parse_args()

    if args.web:
        host, _, port_s = args.web.rpartition(":")
        if not host or not port_s.isdigit():
            parser.error(f"--web expects HOST:PORT, got {args.web!r}")
        return run_web(host=host, port=int(port_s))

    return repl_main()
