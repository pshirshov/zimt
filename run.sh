#!/usr/bin/env bash
# Dev entry point for the venv-based workflow. Runs `python -m zimt` with
# the local source tree on PYTHONPATH and HF cache redirected into the
# project directory.
set -euo pipefail

cd "$(dirname "$0")"

# libstdc++ for the manylinux torch wheel (NixOS python wrapper doesn't add it).
export LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib

# HF cache lives in the project dir (sandbox can only write here / /tmp/exchange).
export HF_HOME=/home/pavel/work/safe/zimt/hf_cache
export HF_XET_HIGH_PERFORMANCE=1

exec .venv/bin/python -m zimt "$@"
