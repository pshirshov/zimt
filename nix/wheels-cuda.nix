# CUDA backend: nixpkgs provides `python.pkgs.torchWithCuda`. No vendored
# wheels needed; the consuming flake's overlay (nixpkgs config
# `cudaSupport = true;` + the relevant cudaPackages_* version) controls
# which CUDA toolkit the torch build is linked against.
_args: []
