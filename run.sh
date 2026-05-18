#!/usr/bin/env bash
# Reproduce Z-Image-Turbo image generation on Intel Arc Pro B70 (Battlemage).
#
# Battlemage's L0 path is broken in NEO (intel/compute-runtime#922) — first
# GEM_USERPTR allocation aborts. We route torch.xpu through the OpenCL UR
# adapter instead, using the LD_PRELOAD shim and three env vars from
# 7mind/nix-config:modules/nixos/intel-gpu.nix.
set -euo pipefail

cd "$(dirname "$0")"

# libstdc++ for the manylinux torch wheel (NixOS python wrapper doesn't add it).
export LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib

# Battlemage L0 → OpenCL bypass.
export LD_PRELOAD=/nix/store/8q8cn3zr406qaz0llfdsjm3cy5vnk028-sycl-force-platform-l0-1/lib/libsycl_force_platform_l0.so
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors

# HF cache lives in the project dir (sandbox can only write here / /tmp/exchange).
export HF_HOME=/home/pavel/work/safe/zimt/hf_cache
export HF_XET_HIGH_PERFORMANCE=1

exec .venv/bin/python -m zimt "$@"
