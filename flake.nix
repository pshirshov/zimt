{
  description = "zimt — multi-model image-generation REPL + web UI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Pick a Python version. cp313 wheels are pinned in nix/wheels-*.nix
        # so this must match.
        python = pkgs.python313;
        callPackage = pkgs.lib.callPackageWith (pkgs // {
          inherit python;
          zimtSrc = self;
        });
      in {
        packages = rec {
          # XPU build (Intel Arc / Battlemage). This is the default because
          # the project was developed and is being daily-driven on Arc Pro B70.
          zimt-xpu = callPackage ./nix/package.nix { backend = "xpu"; };

          # Other backends — wheel sets scaffolded in nix/wheels-<backend>.nix,
          # hashes need populating via scripts/seed-wheel-hashes.sh.
          zimt-cpu  = callPackage ./nix/package.nix { backend = "cpu"; };
          zimt-cuda = callPackage ./nix/package.nix { backend = "cuda"; };
          zimt-rocm = callPackage ./nix/package.nix { backend = "rocm"; };

          default = zimt-xpu;
          zimt    = zimt-xpu;
        };

        # Dev shell: full Python env + pyright + supporting tools.
        # `nix develop` then `./run.sh` or `python -m zimt`.
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            python313Packages.pip
            python313Packages.virtualenv
            pyright
            curl
            jq
            nix-prefetch
          ];
          shellHook = ''
            echo "zimt dev shell"
            echo "  bootstrap a venv with: ./scripts/bootstrap-venv.sh"
            echo "  run pyright:           pyright src/zimt"
          '';
        };

        # Flake checks: pyright must be green on every commit.
        checks.pyright = pkgs.stdenv.mkDerivation {
          name = "zimt-pyright";
          src = self;
          nativeBuildInputs = [ pkgs.pyright python ];
          # We deliberately skip the venv here — pyright's `useLibraryCodeForTypes`
          # plus our `reportMissingImports = "warning"` keeps a missing torch
          # from breaking the check. For full-fat checking, run inside `./run.sh`.
          buildPhase = ''
            export HOME=$TMPDIR
            pyright src/zimt > $out 2>&1 || (cat $out && exit 1)
          '';
          installPhase = "true";
        };
      }) // {

      # NixOS module — wrapped in a self-closure so the module can default
      # `package` from this flake's per-backend outputs without the consumer
      # threading them through explicitly.
      nixosModules.default = { pkgs, ... }@args: import ./nix/module.nix (args // {
        zimtPackages = self.packages.${pkgs.stdenv.hostPlatform.system};
      });
      nixosModules.zimt = self.nixosModules.default;
    };
}
