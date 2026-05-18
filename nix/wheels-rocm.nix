# ROCm backend: nixpkgs provides `python.pkgs.torchWithRocm`. Like the
# cuda case, the actual ROCm toolkit version is selected by the consuming
# flake's nixpkgs configuration (`rocmSupport = true;` + the rocmPackages
# version).
_args: []
