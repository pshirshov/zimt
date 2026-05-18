# CPU backend: nixpkgs's `python.pkgs.torch` is the CPU-only variant by
# default, so no vendored wheels are needed. This file exists only because
# `package.nix` imports `./wheels-<backend>.nix` uniformly.
_args: []
