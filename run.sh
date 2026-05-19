#!/usr/bin/env bash
# Dev entry point for the venv-based workflow. Runs `python -m zimt` with
# the local source tree on PYTHONPATH and HF cache redirected into the
# project directory.
set -euo pipefail

cd "$(dirname "$0")"

# libstdc++ for the manylinux torch wheel (NixOS python wrapper doesn't add
# it) + the NixOS GPU driver path. Level Zero's loader is in the pip
# wheels (intel-sycl-rt), but the actual *driver* (libze_intel_gpu.so,
# libze_loader.so) ships with intel-compute-runtime installed via
# ``hardware.graphics.extraPackages`` and lands at
# ``/run/opengl-driver/lib/``. Without that on LD_LIBRARY_PATH the
# loader fails with ``ZE_RESULT_ERROR_UNINITIALIZED`` and
# ``torch.xpu.device_count()`` silently reports 0.
export LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib
export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors

# HF cache lives in the project dir (sandbox can only write here / /tmp/exchange).
export HF_HOME=/home/pavel/work/safe/zimt/hf_cache
export HF_XET_HIGH_PERFORMANCE=1

exec .venv/bin/python -m zimt "$@"
