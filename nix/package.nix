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
  pyPkgs = python.pkgs;

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

  # diffusers needs to be from git main for Z-Image / Pony / Illustrious
  # support — nixpkgs's pinned diffusers is usually too old.
  diffusersFromGit = pyPkgs.buildPythonPackage rec {
    pname = "diffusers";
    version = "0.39.0.dev0";
    format = "pyproject";
    src = fetchFromGitHub {
      owner = "huggingface";
      repo = "diffusers";
      rev = "main";
      hash = lib.fakeHash;          # populate via scripts/seed-wheel-hashes.sh
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

  # XPU-only wheel set: torch+xpu, torchvision+xpu, triton-xpu, Intel runtime.
  xpuWheels =
    if backend == "xpu"
    then import (./. + "/wheels-xpu.nix") {
      inherit python fetchurl;
      inherit (pyPkgs) buildPythonPackage;
    }
    else [];

  pythonEnv = python.withPackages (ps: with ps;
    # Always-on common deps.
    [ fastapi
      uvicorn
      pydantic
      huggingface-hub
      safetensors
      sentencepiece
      protobuf
      tokenizers
      pillow
      numpy
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

  # static/ must sit next to the python module so paths.py:STATIC_DIR
  # resolves correctly (`PROJECT_ROOT/static` where PROJECT_ROOT computes to
  # the parent of the zimt package, ie $out/lib/python here).
  cp -r ${src}/static $out/lib/python/static

  makeWrapper ${pythonEnv}/bin/python3 $out/bin/zimt \
    --add-flags "-m zimt" \
    --set PYTHONPATH "$out/lib/python" \
    --set ZIMT_BACKEND "${backend}" \
    --set ZIMT_DEVICE "${deviceEnv}"
''
