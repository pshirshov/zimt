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
  pythonForBackend =
    if backend == "xpu"
    then python.override {
      packageOverrides = pySelf: pySuper: {
        accelerate = stripTorch pySuper.accelerate;
        peft = stripTorch pySuper.peft;
      };
    }
    else python;

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

  # diffusers needs to be newer than nixpkgs for Z-Image / Pony / Illustrious
  # support. v0.37.1 keeps safetensors compatible with nixpkgs 0.7.0.
  diffusersFromGit = pyPkgs.buildPythonPackage rec {
    pname = "diffusers";
    version = "0.37.1";
    format = "pyproject";
    src = fetchFromGitHub {
      owner = "huggingface";
      repo = "diffusers";
      rev = "ad3a3afc3a4d3068bbb12f58129c855087ffc6d6";
      hash = "sha256-PKVzByWR6VjtD6ZE+/Uc52Xv+As2OzIPJcQK9vj6sXo=";
    };
    nativeBuildInputs = [ pyPkgs.setuptools ];
    propagatedBuildInputs = with pyPkgs; [
      filelock huggingface-hub importlib-metadata numpy
      pillow regex requests safetensors
    ];
    doCheck = false;
    pythonImportsCheck = [ "diffusers" ];
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
      inherit python fetchurl lib stdenv autoPatchelfHook unzip zlib;
    }
    else [];

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
