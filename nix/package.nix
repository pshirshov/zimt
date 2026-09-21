# zimt application package.
#
# The Python environment is built from one of two sources depending on backend:
#   * cuda / rocm / cpu — uses `python.pkgs.torch` (or whatever the caller
#     passes in via `pytorchPackage`). nixpkgs already builds and maintains
#     these so there's nothing to vendor.
#   * xpu — Intel Arc / Battlemage. nixpkgs doesn't ship torch+xpu so we
#     load a wheel set from ./wheels-xpu.nix (hashes seeded by
#     scripts/seed-wheel-hashes.sh).
#
# Callers can override every dependency by passing a custom `pytorchPackage`
# / `torchvisionPackage` / etc., which makes the module composable with
# whatever overlays the host already has (eg. cudaPackages_12, rocmPackages).
{ lib
, python
, runCommand
, makeWrapper
, fetchurl
, fetchPypi
, fetchFromGitHub
, stdenv
, autoPatchelfHook
, unzip
, zlib
, zimtSrc
, backend ? "xpu"

# Each of these defaults to nixpkgs's variant for the chosen backend (set
# below in `pickDefault`). Pass `null` to disable — useful for the xpu
# branch, which uses vendored wheels instead of nixpkgs.
, pytorchPackage      ? null
, torchvisionPackage  ? null
, diffusersPackage    ? null
, transformersPackage ? null
, acceleratePackage   ? null

# Optional extra Python packages to layer on top.
, extraPythonPackages ? (_ps: [])
}:

let
  inherit (lib) optional optionalString optionals;

  withoutPythonPackages = names: packages:
    builtins.filter (pkg: !(builtins.elem (pkg.pname or (pkg.name or "")) names)) packages;

  # For the XPU backend we strip torch out of every nixpkgs Python
  # package that declares it as a dependency. The wheel set in
  # ./wheels-xpu.nix provides torch+xpu (plus the Intel SYCL/oneAPI
  # runtime), and the nixpkgs `torch-2.11.0` ships overlapping files
  # (`functorch/dim/_order.py`, …) under the same site-packages
  # subpath. `buildEnv` refuses to merge those conflicting subpaths.
  #
  # The override MUST live on the Python package set itself, not on
  # ad-hoc rebindings — otherwise transitive consumers pull in the
  # unmodified package via `propagatedBuildInputs` and the conflict
  # re-surfaces one layer further out.
  #
  # We strip torch from: accelerate, peft. (transformers declares
  # torch only as an optional extra so it doesn't reach this list.)
  stripTorch = pkg: pkg.overridePythonAttrs (old: {
    dependencies = withoutPythonPackages [ "torch" ] (old.dependencies or []);
    propagatedBuildInputs = withoutPythonPackages [ "torch" ] (old.propagatedBuildInputs or []);
    dontCheckRuntimeDeps = true;
    pythonImportsCheck = [];
    # Upstream tests import torch directly; with torch removed the
    # collection phase blows up. The wheel-set torch is layered in
    # at the env level, not at per-package build time.
    doCheck = false;
    doInstallCheck = false;
  });
  pythonForBackend = python.override {
      packageOverrides = pySelf: pySuper:
        let runtime = import ./model-runtime.nix {
            inherit lib fetchPypi stdenv autoPatchelfHook;
            blackForChecks = python.pkgs.black;
          } pySelf pySuper;
        in runtime // lib.optionalAttrs (backend == "xpu") {
        accelerate = stripTorch runtime.accelerate;
        peft = stripTorch runtime.peft;
      };
    };

  pyPkgs = pythonForBackend.pkgs;

  # Resolve a default for the named role per backend. Callers can still
  # override by passing the argument explicitly.
  pickDefault = arg: cudaVariant: rocmVariant: cpuVariant: xpuVariant:
    if arg != null then arg
    else if backend == "cuda" then cudaVariant
    else if backend == "rocm" then rocmVariant
    else if backend == "cpu"  then cpuVariant
    else if backend == "xpu"  then xpuVariant
    else null;

  resolvedTorch       = pickDefault pytorchPackage
                          (pyPkgs.torchWithCuda or pyPkgs.torch)
                          (pyPkgs.torchWithRocm or pyPkgs.torch)
                          pyPkgs.torch
                          null;  # xpu: comes from wheels-xpu.nix
  resolvedTorchvision = pickDefault torchvisionPackage
                          pyPkgs.torchvision
                          pyPkgs.torchvision
                          pyPkgs.torchvision
                          null;
  resolvedTransformers = pickDefault transformersPackage
                          pyPkgs.transformers pyPkgs.transformers
                          pyPkgs.transformers pyPkgs.transformers;
  resolvedAccelerate  = pickDefault acceleratePackage
                          pyPkgs.accelerate pyPkgs.accelerate
                          pyPkgs.accelerate pyPkgs.accelerate;

  # Qwen-Image 2.1 requires unreleased Diffusers code. Keep pyproject.toml
  # on this exact revision; Krea 2 and Ideogram 4 are included here too.
  diffusersFromGit = pyPkgs.buildPythonPackage rec {
    pname = "diffusers";
    version = "0.41.0.dev0";
    format = "pyproject";
    src = fetchFromGitHub {
      owner = "huggingface";
      repo = "diffusers";
      rev = "cc8644b447d8f11074d3df06d0ee0e3e7c91bf75";
      hash = "sha256-fmYms6h5u0Er3wKOhd9EhnkNWxQAB2VnZzfH7E5PuN4=";
    };
    nativeBuildInputs = [ pyPkgs.setuptools ];
    propagatedBuildInputs = with pyPkgs; [
      filelock huggingface-hub importlib-metadata numpy
      pillow regex requests safetensors
    ];
    doCheck = false;
    pythonImportsCheck = [ "diffusers" ];
  };

  # optimum-quanto: fp8 (qfloat8) quantization backend for the FLUX.1 and
  # FLUX.2-klein transformers. Not in nixpkgs, so pin the universal wheel.
  # torch is intentionally NOT a propagated dep — the xpu wheel set layers it
  # into the env; declaring nixpkgs torch here would re-introduce the
  # site-packages conflict that stripTorch exists to avoid.
  optimumQuanto = pyPkgs.buildPythonPackage rec {
    pname = "optimum-quanto";
    version = "0.2.7";
    format = "wheel";
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/8d/33/4ad914b0ae7e46296fe00d76d084be351fef69816b3498ed32a178471c8a/optimum_quanto-0.2.7-py3-none-any.whl";
      hash = "sha256-E2mx2aShl/iMDRxn6NlQaU5bhs5Mnzh44XjVvjUzn2E=";
    };
    # ninja is omitted on purpose: it's only needed to JIT-compile CUDA
    # kernels (never on XPU — qfloat8 dequantizes via torch ops), and pulling
    # the python ninja package injects a build-phase hook that breaks the
    # wheel install.
    propagatedBuildInputs = with pyPkgs; [ numpy safetensors huggingface-hub ];
    # torch>=2.6.0 from the wheel METADATA is satisfied at the env layer (xpu
    # wheels), not here — skip the runtime-deps + import checks (importing
    # optimum.quanto needs torch present).
    dontCheckRuntimeDeps = true;
    doCheck = false;
    pythonImportsCheck = [ ];
  };
  resolvedDiffusers = pickDefault diffusersPackage
                        diffusersFromGit diffusersFromGit
                        diffusersFromGit diffusersFromGit;

  # XPU-only wheel set: torch+xpu, torchvision+xpu, triton-xpu, Intel
  # runtime. Returned as a single-element list (one combined derivation
  # containing every wheel unpacked into a unified prefix) — see
  # ``./wheels-xpu.nix`` for why a single derivation is needed.
  xpuWheels =
    if backend == "xpu"
    then import (./. + "/wheels-xpu.nix") {
      python = pythonForBackend;
      inherit fetchurl lib stdenv autoPatchelfHook unzip zlib;
    }
    else [];

  # Keep the tested oneAPI 2026 NF4 kernels; CUDA/ROCm and oneAPI 2025
  # binaries in the same wheel would require unrelated runtime libraries.
  bitsandbytesXpu = pyPkgs.buildPythonPackage {
    pname = "bitsandbytes";
    version = "0.50.2";
    format = "wheel";
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/a5/6e/e4e8b75716dbe5e50964f070266e06f4e6806ce051bfb97f52ee162b9310/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl";
      sha256 = "55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283";
    };
    nativeBuildInputs = [ autoPatchelfHook ];
    buildInputs = [ stdenv.cc.cc.lib ] ++ xpuWheels;
    dependencies = [ pyPkgs.numpy pyPkgs.packaging ] ++ xpuWheels;
    postInstall = ''
      rm "$out/${pythonForBackend.sitePackages}/bitsandbytes/"libbitsandbytes_cuda*.so
      rm "$out/${pythonForBackend.sitePackages}/bitsandbytes/"libbitsandbytes_rocm*.so
      rm "$out/${pythonForBackend.sitePackages}/bitsandbytes/libbitsandbytes_xpu2025.so"
    '';
    doCheck = false;
    pythonImportsCheck = [ "bitsandbytes" ];
  };

  pythonEnv = pythonForBackend.withPackages (ps: with ps;
    # Always-on common deps.
    [ fastapi
      uvicorn
      # uvicorn needs a ws backend (websockets or wsproto) to serve the
      # WebSocket upgrade — otherwise it 404s every WS request. We
      # carry the whole UI over WS, so this is load-bearing.
      websockets
      pydantic
      huggingface-hub
      safetensors
      sentencepiece
      protobuf
      tokenizers
      pillow
      numpy
      # peft is the backend diffusers' load_lora_weights / set_adapters
      # delegate to in v0.27+. Without it, any /lora invocation blows up
      # with "PEFT backend is required for this method.".
      peft
    ]
    # fp8 quantization backend for the FLUX loaders (see flux.py).
    ++ [ optimumQuanto ]
    ++ optional (backend == "xpu") ps.pyelftools
    ++ optional (backend == "xpu") bitsandbytesXpu
    ++ optional (backend == "cuda") (ps.bitsandbytes.override { torch = resolvedTorch; })
    ++ optional (resolvedTorch != null) resolvedTorch
    ++ optional (resolvedTorchvision != null) resolvedTorchvision
    ++ optional (resolvedTransformers != null) resolvedTransformers
    ++ optional (resolvedAccelerate != null) resolvedAccelerate
    ++ optional (resolvedDiffusers != null) resolvedDiffusers
    ++ xpuWheels
    ++ extraPythonPackages ps
  );

  deviceEnv =
    if backend == "xpu" then "xpu"
    else if backend == "cuda" || backend == "rocm" then "cuda"
    else "cpu";

  src = zimtSrc;
in

runCommand "zimt-${backend}-${python.version}" {
  nativeBuildInputs = [ makeWrapper ];
  meta = with lib; {
    description = "Multi-model image generation REPL + web UI (${backend} backend)";
    license = licenses.mit;
    platforms = platforms.linux;
    mainProgram = "zimt";
  };
  passthru = {
    inherit pythonEnv backend;
    inherit resolvedTorch resolvedTorchvision resolvedDiffusers
            resolvedTransformers resolvedAccelerate;
  };
} ''
  mkdir -p $out/{bin,lib/python}

  # Copy the package source. We deliberately do NOT use makeWrapper to
  # install at $out/lib/python3.X — instead we ship just the zimt module
  # tree and let PYTHONPATH find it. That keeps the install layout flat
  # and easy to override.
  cp -r ${src}/src/zimt $out/lib/python/zimt

  # paths.py computes ``PROJECT_ROOT = dirname(__file__)/../..`` which from
  # ``$out/lib/python/zimt/paths.py`` resolves to ``$out/lib/``. STATIC_DIR
  # is then ``$out/lib/static`` — keep this in sync with paths.py if the
  # python module's install depth changes.
  cp -r ${src}/static $out/lib/static

  makeWrapper ${pythonEnv}/bin/python3 $out/bin/zimt \
    --add-flags "-m zimt" \
    --set PYTHONPATH "$out/lib/python" \
    --set ZIMT_BACKEND "${backend}" \
    --set ZIMT_DEVICE "${deviceEnv}" ${lib.optionalString (backend == "xpu") ''\
    --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib" \
    --set-default OCL_ICD_VENDORS "/run/opengl-driver/etc/OpenCL/vendors"''}
''
