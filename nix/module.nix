# NixOS module for the zimt image-generation web service.
#
# Provides ``smind.services.zimt.*`` options. The ``gpuSupport`` enum
# selects the package variant (xpu / cuda / rocm / cpu).
#
# Secret handling: the HuggingFace token is loaded via systemd
# ``LoadCredential`` so the file path itself stays in the unit but the
# token value never lands in the unit's static environment. A small
# ExecStart wrapper reads the credential at start and exports HF_TOKEN.
{ config, lib, pkgs, zimtPackages, ... }:

let
  cfg = config.smind.services.zimt;

  defaultPackage =
    if cfg.gpuSupport == "xpu"  then zimtPackages.zimt-xpu
    else if cfg.gpuSupport == "cuda" then zimtPackages.zimt-cuda
    else if cfg.gpuSupport == "rocm" then zimtPackages.zimt-rocm
    else                             zimtPackages.zimt-cpu;

  hasTokenFile = cfg.hfTokenFile != null;
  hasWebAuthTokenFile = cfg.webAuthTokenFile != null;

  startScript = pkgs.writeShellScript "zimt-start" ''
    set -euo pipefail
    ${lib.optionalString hasTokenFile ''
      # Token is mounted by systemd LoadCredential under $CREDENTIALS_DIRECTORY.
      if [ -r "$CREDENTIALS_DIRECTORY/hf-token" ]; then
        export HF_TOKEN="$(cat "$CREDENTIALS_DIRECTORY/hf-token")"
      else
        echo "warning: hfTokenFile is set but credential not readable" >&2
      fi
    ''}
    ${lib.optionalString hasWebAuthTokenFile ''
      # Browser/API auth token. Sent as the Basic-auth password or a Bearer token.
      if [ -r "$CREDENTIALS_DIRECTORY/web-auth-token" ]; then
        export ZIMT_AUTH_TOKEN="$(cat "$CREDENTIALS_DIRECTORY/web-auth-token")"
      else
        echo "warning: webAuthTokenFile is set but credential not readable" >&2
      fi
    ''}
    exec ${cfg.package}/bin/zimt \
      --web ${cfg.listenAddress}:${toString cfg.port} \
      --out-dir ${cfg.outDir} \
      --hf-cache ${cfg.hfCacheDir}
  '';
in
{
  options.smind.services.zimt = {
    enable = lib.mkEnableOption "zimt image-generation web service";

    gpuSupport = lib.mkOption {
      type = lib.types.enum [ "xpu" "cuda" "rocm" "cpu" ];
      default = "cpu";
      description = ''
        Which torch backend to run.
          * ``xpu``  — Intel Arc / Battlemage. Vendored torch+xpu wheel set
            (see ``nix/wheels-xpu.nix``).
          * ``cuda`` — NVIDIA. Uses ``python.pkgs.torchWithCuda`` from
            nixpkgs; host must allow unfree.
          * ``rocm`` — AMD. Uses ``python.pkgs.torchWithRocm``.
          * ``cpu``  — fallback. Slow but useful for verification.
      '';
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression
        "zimt.packages.<system>.zimt-<gpuSupport>";
      description = ''
        zimt package to run. Defaults to the right per-backend variant from
        this flake's outputs. Override to inject custom Python deps via
        ``zimt.packages.<system>.zimt-<backend>.override``.
      '';
    };

    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address the HTTP / WebSocket server binds to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the configured TCP port in the host firewall.";
    };

    outDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/zimt/out";
      description = ''
        Directory for generated PNGs. ``<outDir>/fav/`` is used for
        favorites. Must be writable by the service user.
      '';
    };

    hfCacheDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/zimt/hf_cache";
      description = ''
        HuggingFace model cache. Maps to the ``HF_HOME`` env var so
        diffusers / transformers / huggingface_hub all share it.
      '';
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/zimt-hf-token";
      description = ''
        Path to a file containing the HuggingFace API token. Loaded via
        systemd ``LoadCredential`` so the path itself is the only thing in
        the unit; the value isn't exposed in the static environment block.
        Read at start and exported as ``HF_TOKEN``. ``null`` to disable.
      '';
    };

    webAuthTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/zimt-web-token";
      description = ''
        Path to a file containing the zimt web/API auth token. Loaded via
        systemd ``LoadCredential`` and exported as ``ZIMT_AUTH_TOKEN`` at
        start. When set, browsers authenticate with Basic auth using any
        username and this token as the password; API clients may also send
        ``Authorization: Bearer <token>``.
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "zimt";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "zimt";
    };

    extraEnvironment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {};
      example = lib.literalExpression ''
        { ZIMT_DEVICE = "xpu"; HF_HUB_OFFLINE = "0"; }
      '';
      description = ''
        Additional env vars for the systemd unit. Merged on top of the
        zimt-managed env (``ZIMT_OUT_DIR``, ``HF_HOME``); user values win.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = lib.mkIf (cfg.user == "zimt") {
      isSystemUser = true;
      group = cfg.group;
      home = "/var/lib/zimt";
      description = "zimt image generation service user";
      # Render group: /dev/dri/renderD*. Video group: legacy fallback.
      extraGroups = [ "render" "video" ];
    };
    users.groups.${cfg.group} = lib.mkIf (cfg.group == "zimt") {};

    networking.firewall.allowedTCPPorts =
      lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.zimt = {
      description = "zimt image-generation web UI";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        ZIMT_OUT_DIR = cfg.outDir;
        HF_HOME = cfg.hfCacheDir;
      } // cfg.extraEnvironment;

      serviceConfig = {
        Type = "exec";
        User = cfg.user;
        Group = cfg.group;
        SupplementaryGroups = [ "render" "video" ];

        # /var/lib/zimt and subdirs; mode 0700 keeps the cache private.
        StateDirectory = "zimt";
        StateDirectoryMode = "0700";

        # ReadWritePaths is needed when outDir / hfCacheDir live outside
        # /var/lib/zimt (e.g. on a separate ZFS dataset).
        ReadWritePaths = [ cfg.outDir cfg.hfCacheDir ];

        LoadCredential =
          lib.optional hasTokenFile "hf-token:${toString cfg.hfTokenFile}"
          ++ lib.optional hasWebAuthTokenFile
            "web-auth-token:${toString cfg.webAuthTokenFile}";

        ExecStart = startScript;
        Restart = "on-failure";
        RestartSec = "5s";

        # On stop, systemd sends SIGTERM. uvicorn's signal handler triggers
        # graceful shutdown, which fires our @app.on_event("shutdown") that
        # cancels in-flight generations (the pipeline callback aborts at
        # the next step, ~1-3s) and drains the executor. 60s is well above
        # the worst-case path: scheduler-step latency × max steps_setting
        # we ever ship. systemd's default 90s would also work but we make
        # the budget explicit.
        KillSignal = "SIGTERM";
        TimeoutStopSec = "60s";

        # Hardening — kept compatible with GPU device access (which needs
        # ``/dev/dri/*``, hence ``DeviceAllow = "char-drm rw"`` rather than
        # ``PrivateDevices = true``).
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectHome = true;
        ProtectSystem = "strict";
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        SystemCallArchitectures = "native";
        # Memory-exec must be left on: PyTorch/Triton JIT mmaps
        # PROT_WRITE|PROT_EXEC pages.
        MemoryDenyWriteExecute = false;

        PrivateDevices = false;
        DeviceAllow = [ "char-drm rw" ];
      };
    };
  };
}
