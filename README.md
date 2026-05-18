# zimt

Minimalistic and powerful multi-model image-generation REPL + web UI, tuned for Intel Arc GPUs but
backend-agnostic (XPU / CUDA / ROCm / CPU). 

Graphs? Workflows? Repositories full of shit? You don't need that anymore.

Need a customization? A new model? A your own, very specific workflow? Ask Claude, it will bake the shit in for you.

<img width="1954" height="1678" alt="image" src="https://github.com/user-attachments/assets/eff91206-9793-4860-9f40-0c58a9f143ad" />

Models currently supported:

| name | source | notes |
|---|---|---|
| `z-image-turbo` | `Tongyi-MAI/Z-Image-Turbo` | 6B DiT, Qwen3-4B text encoder, CFG=0 |
| `pony-v6-xl` | `kitty7779/ponyDiffusionV6XL` | SDXL fine-tune, score-tag prefix, fp16-fix VAE |
| `illustrious-xl-v1` | `WhiteAiZ/Illustrious-xl-v1.0` | anime-focused SDXL fine-tune |
| `hassaku-xl-illustrious` | `John6666/hassaku-xl-illustrious-v31-sdxl` | Illustrious-family, anime-realistic |
| `wai-nsfw-illustrious` | `John6666/wai-nsfw-illustrious-v80-sdxl` | Illustrious-family, NSFW-focused |
| `wai-nsfw-illustrious-v110` | `John6666/wai-nsfw-illustrious-v110-sdxl` | Illustrious-family, newer NSFW-focused checkpoint |
| `wai-mature-illustrious` | `John6666/wai-mature-illustrious-v20-sdxl` | Illustrious-family, mature/body/style focus |
| `noobai-xl-vpred` | `Laxhar/noobai-XL-Vpred-1.0` | Illustrious-family, **v-prediction** — needs `scheduler_overrides` (handled) |
| `animagine-xl-4` | `cagliostrolab/animagine-xl-4.0` | Cagliostro's flagship anime SDXL |
| `cyberrealistic-pony` | `John6666/cyberrealistic-pony-v85-sdxl` | Pony-family, photorealism focus |
| `pony-realism-v23` | `John6666/pony-realism-v23-sdxl` | Pony-family, photorealism focus |
| `spicy-realism-nsfw-mix` | `John6666/spicy-realism-nsfw-mix-v30-sdxl` | Pony-family, adult photorealism focus |

The fp16-fix VAE is wired through a shared `make_sdxl_loader` factory
(`src/zimt/models/sdxl_factory.py`), so adding another SDXL fine-tune is
typically two lines (a `make_sdxl_loader("hf/repo")` plus a `ModelSpec`
entry).

The codebase is a single Python package (`src/zimt/`) with two entry modes
sharing the same command parser, so anything you can do in the CLI
(`/model`, `/cfg`, `/res`, `/many`, `/tokenize`, …) works verbatim in the
web UI's prompt box too.

## Quick start (dev tree)

`run.sh` is the development entry point. It activates the project venv
and execs the Python entrypoint:

```sh
./run.sh                       # CLI REPL
./run.sh --web 127.0.0.1:8000  # web UI on http://127.0.0.1:8000
./run.sh --help                # full flag list
```

Useful CLI flags:

* `--out-dir DIR` — where PNGs are saved (default `./out`; falls back to
  `$XDG_DATA_HOME/zimt/out` when installed from `/nix/store`).
* `--hf-cache DIR` — `HF_HOME` for the diffusers / transformers cache.
* `ZIMT_AUTH_TOKEN` — optional web/API token. Browsers can use Basic auth
  with any username and this token as the password; API clients can send
  `Authorization: Bearer <token>`.
* `ZIMT_ALLOWED_ORIGINS` — optional comma-separated origin allow-list. If
  unset, requests with an `Origin` header must match the request host.

## REPL / web command surface

Multi-command lines compose left-to-right; greedy commands (`/negprompt`,
`/tokenize`, `/many`) stop at the next `/cmd`:

```
/model pony-v6-xl /cfg 5 /steps 25 /res 1216x832 cute anime girl
/many 8 /seed 42 a forest
/tokenize 西安大雁塔 ⚡️ supercalifragilistic
```

| command | effect |
|---|---|
| `<prompt>` | generate one image with a random seed |
| `/raw <prompt>` | skip the model's auto-prefix (score-tags etc.) |
| `/many N <prompt>` | generate N images; seeds increment if `/seed` is also set |
| `/seed N` | pin seed for the next generation |
| `/cfg X` / `/steps N` / `/size W H` | numeric settings |
| `/res N` / `/res WxH` | pick a model-preset resolution or set explicitly |
| `/sampler <name>` | swap scheduler. SDXL: euler, euler-a, dpmpp-2m, dpmpp-2m-karras, dpmpp-sde, dpmpp-sde-karras, heun, unipc, lms, ddim. Z-Image: flow-match-euler. |
| `/clip_skip N` | SDXL only — skip top N CLIP layers (0=off; Pony was trained with 2) |
| `/negprompt …` / `/negprompt -` | set / clear negative prompt |
| `/model <name>` | swap the loaded model (no-op if already loaded) |
| `/tokenize <text>` | per-encoder token analysis + budget headroom |
| `/help` / `/quit` | help / leave |

Prompts may use **A1111/compel weighting syntax** on SDXL models —
`(red hair:1.4)`, `(detailed)+`, `[loose]-`. zimt auto-detects the
syntax and routes through compel for SDXL; Z-Image falls back to plain
strings (compel doesn't have a Qwen3 adapter).

Tab-completion works in both modes — `/m<TAB>` cycles `/model`/`/many`,
`/model <TAB>` cycles registered model names. Up/Down navigates prompt
history.

## Web UI features

* Two-tab thumbnail browser (`all` / `favs`) with `★` per-thumb favorite
  toggle and modal preview (full image + every PNG metadata field + per-row
  copy button + Restore-to-prompt that reproduces the run byte-for-byte).
* Resizable splitter; thumbnail grid uses `auto-fill` so a new column
  snaps in as you widen the panel.
* WebSocket-driven job queue with per-step progress bars (`callback_on_step_end`
  hook), cancel-one + cancel-all + clear-completed.
* Three section-header buttons:
  * `outputs` → `clean` (deletes non-favorite PNGs server-side)
  * `queue` → `cancel all` + `clear` (drop done/error/canceled jobs)
  * `recent prompts` → `clear` (localStorage-only)
* `Cache-Control: no-store` on every static asset so dev iterations land
  without forced reloads.

## Nix flake

```sh
nix build .#zimt-xpu          # Intel Arc (default)
nix build .#zimt-cpu          # CPU fallback
NIXPKGS_ALLOW_UNFREE=1 \
nix build .#zimt-cuda --impure
nix build .#zimt-rocm

nix flake check               # pyright + per-backend eval
nix develop                   # full dev shell
```

* CPU / CUDA / ROCm variants pull `torch` / `torchWithCuda` /
  `torchWithRocm` from **nixpkgs** — nothing vendored.
* XPU pulls 24 wheels from PyTorch's xpu index + PyPI (`intel-*`, `mkl`,
  `onemkl-*`, `triton-xpu`, …) since nixpkgs doesn't carry a torch-xpu
  build.

### Populating the XPU wheel hashes

`nix/wheels-xpu.nix` ships with `lib.fakeHash` placeholders. Two ways to
fill them in:

```sh
# 1. If you've already pip-installed the wheels at least once
#    (./run.sh or an earlier nix build has primed ~/.cache/pip):
.venv/bin/python scripts/seed-from-pip-cache.py

# 2. Otherwise, fetch each URL with nix-prefetch-url:
./scripts/seed-wheel-hashes.sh
```

The pip-cache seeder reads each cached wheel's `dist-info/METADATA` to
match `(pname, version)` and computes the SRI sha256 in-place — no
network.

## NixOS module

The flake exposes `nixosModules.default`. Minimal use:

```nix
{
  inputs.zimt.url = "github:user/zimt";
  outputs = { self, nixpkgs, zimt, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      modules = [
        zimt.nixosModules.default
        ({ ... }: {
          smind.services.zimt = {
            enable = true;
            gpuSupport = "xpu";
            listenAddress = "0.0.0.0";
            port = 8000;
            openFirewall = true;
            hfTokenFile = "/run/secrets/zimt-hf-token";
          };
        })
      ];
    };
  };
}
```

Options:

| option | type | default | notes |
|---|---|---|---|
| `enable` | bool | false | |
| `gpuSupport` | enum | `cpu` | `xpu` / `cuda` / `rocm` / `cpu` |
| `package` | derivation | auto from `gpuSupport` | override to pass a custom Python env |
| `listenAddress` | str | `127.0.0.1` | |
| `port` | port | `8000` | |
| `openFirewall` | bool | false | |
| `outDir` | path | `/var/lib/zimt/out` | |
| `hfCacheDir` | path | `/var/lib/zimt/hf_cache` | maps to `HF_HOME` |
| `hfTokenFile` | nullable path | `null` | loaded via systemd `LoadCredential`; never appears in the unit's static env |
| `user` / `group` | str | `zimt` / `zimt` | system user with `render` / `video` groups for `/dev/dri/*` |
| `extraEnvironment` | attrs of str | `{}` | merged on top of zimt-managed env |

The XPU path requires the Intel Graphics Compiler (IGC) to be reachable
from `libze_intel_gpu.so.1`. On NixOS this means either having IGC on
`libze_intel_gpu.so.1`'s RPATH (set in the `intel-compute-runtime`
derivation's `postFixup`) or on the loader search path (e.g. via
`/run/opengl-driver/lib`). Without it, NEO's Level Zero driver aborts
during eager device init (the failure surfaces in
`gmm_helper/resource_info.cpp`).

## Repository layout

```
zimt/
├── flake.nix
├── pyproject.toml          # pyright in basic mode, venv-aware, 0/0/0
├── run.sh                  # dev entry: sets XPU env + execs `python -m zimt`
├── nix/
│   ├── package.nix         # backend-aware build, accepts overrides
│   ├── module.nix          # NixOS module
│   └── wheels-{xpu,cuda,rocm,cpu}.nix
├── scripts/
│   ├── seed-from-pip-cache.py    # no-network hash seeder
│   └── seed-wheel-hashes.sh      # network fallback via nix-prefetch-url
├── src/zimt/
│   ├── __main__.py / cli.py      # argparse, env-var pre-pass
│   ├── paths.py                  # OUT_DIR / HF_HOME / Nix-store fallback
│   ├── device.py                 # auto-detect cuda / xpu / mps / cpu
│   ├── buckets.py                # SDXL_BUCKETS + ZIMAGE_BUCKETS + /res parser
│   ├── tokenize_report.py        # per-encoder analysis
│   ├── generate.py               # GenConfig, generate(), CancelledByUser
│   ├── preview.py                # kitty / iTerm inline + tmux passthrough
│   ├── models/
│   │   ├── spec.py               # ModelSpec dataclass
│   │   ├── sdxl_common.py        # shared SDXL two-encoder tokenize
│   │   ├── {zimage,pony,illustrious}.py
│   │   └── registry.py           # MODELS = { … }
│   ├── repl/
│   │   ├── commands.py           # parse_commands + COMMAND_ARITY
│   │   ├── history.py            # readline + Tab completion
│   │   └── main.py               # repl_main()
│   └── webui/
│       ├── app.py                # FastAPI routes + run_web()
│       ├── state.py              # AppState, Job, locks
│       ├── ws.py                 # broadcast helpers
│       ├── loader.py             # async model swap
│       ├── jobs.py               # run_job() worker + cancel/progress
│       ├── outputs.py            # listing + favorite + cleanup
│       └── exec_api.py           # /api/exec multi-command executor
└── static/                       # vanilla HTML / CSS / JS, no build step
```

## Troubleshooting

* **Pony or Illustrious output looks pastel / washed-out** — fixed (the
  bundled SDXL VAE has the SD 1.x `scaling_factor` 0.18215 and isn't
  fp16-stable). `models/pony.py` and `models/illustrious.py` substitute
  `madebyollin/sdxl-vae-fp16-fix` automatically.
* **NEO aborts at `gmm_helper/resource_info.cpp` on first XPU op** —
  Level Zero is failing to dlopen the Intel Graphics Compiler. Ensure IGC
  is on `libze_intel_gpu.so.1`'s RPATH (patch `intel-compute-runtime`'s
  `postFixup` in your nixpkgs overlay) or on the loader search path
  (e.g. `LD_LIBRARY_PATH=${intel-graphics-compiler}/lib`). Confirm with
  `zeInit` returning `0x0` and `torch.xpu.device_count() >= 1`.
* **`nix flake check` fails on `zimt-cuda`** — expected without
  `NIXPKGS_ALLOW_UNFREE=1`; `cuda_nvcc` is unfree.
* **Inline preview doesn't appear in tmux** — add
  `set -g allow-passthrough on` to `~/.tmux.conf` and reattach.
* **HF rate-limit / token errors** — set `HF_TOKEN` (or `--hf-cache` to a
  pre-populated cache).
