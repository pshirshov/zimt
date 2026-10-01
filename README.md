# zimt

Minimalistic and powerful multi-model image-generation REPL + web UI, tuned for Intel Arc GPUs but
backend-agnostic (XPU / CUDA / ROCm / CPU). 

Graphs? Workflows? Repositories full of shit? You don't need that anymore.

Need a customization? A new model? A your own, very specific workflow? Ask Claude, it will bake the shit in for you.

<img width="1954" height="1678" alt="image" src="https://github.com/user-attachments/assets/eff91206-9793-4860-9f40-0c58a9f143ad" />

Models currently supported:

| name | source | notes |
|---|---|---|
| `qwen-image-2.1` | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | bf16, 40 steps, CFG=1, native RGBA and 2K presets; Qwen Research License |
| `krea-2-turbo` | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | bf16, 8 steps, CFG=0; gated, Krea community license |
| `krea-2-raw` | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) | bf16, 52 steps, CFG=3.5; gated, Krea community license |
| `ideogram-4` | [ideogram-ai/ideogram-4-nf4-diffusers](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers) | NF4, NVIDIA CUDA / Intel XPU; gated, non-commercial license; local caption expansion |
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

Qwen 2.1 and Krea 2 use the released bf16 weights. Their transformer and
text encoder can exceed the available inference memory on a 32 GiB
GPU; select `/mem cpuoffload` before loading. Defaults start at 1024×1024.
For example:

```
/mem cpuoffload /model qwen-image-2.1 /seed 42 a ceramic teapot
/mem cpuoffload /model krea-2-turbo /seed 42 a fox in the snow
```

These entries appear in the Models tab and use its existing download button.
For Krea and Ideogram, accept access on the linked Hugging Face model pages
and provide `HF_TOKEN` to the Zimt process. The model licenses apply separately
from Zimt's license; see the respective model cards before use.

Ideogram's download also fetches
[`diffusers/qwen3-vl-8b-instruct-lm-head`](https://huggingface.co/diffusers/qwen3-vl-8b-instruct-lm-head).
Ordinary prompts are expanded locally to its structured caption format;
the expanded caption is saved as `revised_prompt` in PNG metadata. A JSON
caption can be supplied directly to skip expansion. `/cfg` selects constant
guidance (7 by default), overriding upstream's 7→3 guidance schedule. Negative
prompts are not supported. The NF4 checkpoint needs `bitsandbytes` (included
in the Nix CUDA and XPU packages; install the `ideogram` extra for a venv).
The XPU package pins bitsandbytes 0.50.2 with native oneAPI 2026 kernels;
ZLUDA is not required. On the 32 GiB B70, use `/mem off` at 1024×1024:

```
/mem off /model ideogram-4 /seed 42 a ceramic teapot
```

Prompt expansion runs unconstrained when optional upstream `outlines` is
absent; generated captions can contain duplicate JSON keys. Supply a structured
JSON caption directly when exact caption structure matters. The upstream FP8
release uses a different runtime and is not included in this integration.

These integrations require the Diffusers revision pinned in both
`pyproject.toml` and `nix/package.nix`, plus Transformers ≥5.17, Accelerate ≥1.15,
PEFT ≥0.21, Hub ≥1.32, and safetensors ≥0.8.
The Nix package includes these dependency updates. LoRAs and image editing
for these new families are not exposed by Zimt yet.

Boogu Image 0.1 is deferred: its
[`boogu-image` package](https://github.com/boogu-project/Boogu-Image/blob/main/pyproject.toml)
requires Python <3.13, Diffusers <0.39, and PyTorch <2.12, conflicting with
Zimt's Python 3.13, newer Diffusers, and XPU torch 2.14 runtime. Its
[inference guide](https://github.com/boogu-project/Boogu-Image/blob/main/INFERENCE_GUIDE.md)
documents CPU/CUDA devices, not Intel XPU. Supporting it needs a compatible
port or an isolated runtime; it is not registered as a working model.

Hardware verification uses `scripts/verify-model-xpu.py`, separately from the
automated test suite. It loads real cached checkpoints through Zimt, generates
two images with the same seed, checks PNG dimensions/provenance and nonblank
pixels, and writes timings, peak allocated/reserved GPU memory, versions,
checkpoint revision, and pixel hashes to `report.json` alongside the images.
Download the model first; verification itself can run entirely offline:

```bash
ZIMT_PYTHON_ENV=$(nix build .#zimt-xpu.pythonEnv --no-link --print-out-paths)
LD_LIBRARY_PATH=/run/opengl-driver/lib \
OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors \
HF_HOME=/srv/nvme/zimt/hf_cache HF_HUB_OFFLINE=1 \
PYTHONPATH=src ZIMT_DEVICE=xpu \
"$ZIMT_PYTHON_ENV/bin/python" scripts/verify-model-xpu.py qwen-image-2.1 \
  --mem cpuoffload --output-dir out/b70-verification/qwen-image-2.1
```

For Ideogram, select `ideogram-4 --mem off` and a separate output directory.

Verified on 2026-09-21 with an Intel Arc Pro B70 (31.89 GiB), torch
2.14.0+xpu, the pinned Diffusers revision, and the dependency versions above.
These are real-checkpoint runs through Zimt at 1024×1024 with
`/mem cpuoffload`, seed 42, and each model's default steps/CFG:

| model | steps / CFG | generation 1 / 2 | peak allocated GPU memory | output |
|---|---|---|---|---|
| Qwen Image 2.1 | 40 / 1 | 188.3 / 348.9 s | 16.38 GiB | RGBA |
| Krea 2 Turbo | 8 / 0 | 148.4 / 247.9 s | 24.56 GiB | RGB |
| Krea 2 Raw | 52 / 3.5 | 270.9 / 395.2 s | 24.59 GiB | RGB |

All three passed two runs, with identical pixel hashes within each same-seed
pair. They produced nonblank images with the requested dimensions, correct
PNG provenance, and the prompted teapot and text card on visual inspection.
Host paging was observed; these timings are not isolated performance
benchmarks. This does not verify 2K generation or other memory strategies.

Ideogram 4 NF4 was also verified through the packaged Nix XPU environment and
production loader, using bitsandbytes 0.50.2, `/mem off`, 1024×1024, 48 steps,
CFG 7, and seed 42. Both runs passed: 171.3 / 168.1 seconds, 20.32 GiB peak
allocated and 21.44 GiB peak reserved GPU memory. Both produced coherent teapot
and text-card images. Prompt expansion was identical, but pixels were not
bit-exact (mean absolute channel difference 0.31 on a 0–255 scale). Ideogram's
2K, CPU-offload, and CUDA generation remain unverified.

`run.sh` also defaults to `/srv/nvme/zimt/hf_cache`, while preserving an explicit
`HF_HOME` override. In the sandbox this dataset is read-only: populate it from
the host, not by creating a second model cache in the checkout.

Built-in SDXL LoRAs (for SDXL base models):

| name | source | trigger | notes |
|---|---|---|---|
| `pixel-art-xl` | `nerijs/pixel-art-xl` | `pixel art` | Nerijs's pixel-art style |
| `ascii-art` | `CiroN2022/ascii-art` | `ascii_art` | ASCII-art style |
| `studio-ghibli-style` | `KappaNeuro/studio-ghibli-style` | `Studio Ghibli Style` | Ghibli-look fine-tune |

The fp16-fix VAE is wired through a shared `make_sdxl_loader` factory
(`src/zimt/models/sdxl_factory.py`), so adding another SDXL fine-tune is
typically two lines (a `make_sdxl_loader("hf/repo")` plus a `ModelSpec`
entry).

LoRAs ship as a built-in starter set (`pixel-art-xl`, `ascii-art`,
`studio-ghibli-style` — all SDXL) and can be stacked at runtime via
`/lora <name>[:weight]`. Compatibility is enforced through tag overlap
between the base's `compatibility_tags` and the LoRA's `compatible_with`.

You can also drop your own bases and LoRAs in as JSON descriptors under
`./custom/{bases,loras}/*.json` (or `$XDG_DATA_HOME/zimt/custom/` for
installed builds) — see `src/zimt/models/custom.py` for the schema, or
use the "+ add" form in the web UI's `models` tab. No restart needed:
they reload on every add/remove.

The codebase is a single Python package (`src/zimt/`) with two entry modes
sharing the same command parser, so anything you can do in the CLI
(`/model`, `/cfg`, `/res`, `/many`, `/lora`, `/tokenize`, …) works
verbatim in the web UI's prompt box too.

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
/lora pixel-art-xl:0.8 /lora ascii-art:0.5 a knight at sunset
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
| `/lora <name>[:w]` | add or update one LoRA (A1111-style stacking by repetition). `-<name>` removes one, bare `-` clears, bare `/lora` lists active. |
| `/tokenize <text>` | per-encoder token analysis + budget headroom |
| `/help` / `/quit` | help / leave |

Prompts may use **A1111/compel weighting syntax** on SDXL models —
`(red hair:1.4)`, `(detailed)+`, `[loose]-`. zimt auto-detects the
syntax and routes through compel for SDXL; Z-Image falls back to plain
strings (compel doesn't have a Qwen3 adapter).

Tab-completion works in both modes — `/m<TAB>` cycles `/model`/`/many`,
`/model <TAB>` cycles registered model names, `/lora <TAB>` cycles
LoRAs compatible with the currently-loaded base. Up/Down navigates
prompt history.

## Web UI features

* Top-bar tab switcher between `inference` (prompt + state + queue +
  recent) and `models` (per-entry install status / size / HF link /
  prefetch / load / per-row remove for custom entries; separate
  sections for base models and LoRAs).
* `+ add` button per section opens a form that writes a JSON descriptor
  under `custom/{bases,loras}/` and hot-reloads it.
* Active LoRA stack is reflected in state + image PNG metadata, and is
  reproduced verbatim by the modal `Restore to prompt` button.
* "Recent prompts" surfaces a showcase set when empty so new users see
  the command shape (`/model`, `/lora`, `/res`, weighting, `/many`, …)
  at a glance.
* Two-tab thumbnail browser (`all` / `favs`) with `★` per-thumb favorite
  toggle and modal preview (full image + every PNG metadata field + per-row
  copy button + Restore-to-prompt that reproduces the run byte-for-byte).
* Resizable splitter; thumbnail grid uses `auto-fill` so a new column
  snaps in as you widen the panel.
* WebSocket-driven job queue with per-step progress bars (`callback_on_step_end`
  hook), cancel-one + cancel-all + clear-completed. Model downloads
  (load OR prefetch) appear as their own job rows with per-file byte
  progress streamed from HF's tqdm.
* Section-header action buttons:
  * `outputs` → `clean` (deletes non-favorite PNGs server-side)
  * `queue` → `cancel all` + `clear` (drop done/error/canceled jobs)
  * `recent prompts` → `clear` (localStorage-only)
  * `base models` / `loras` → `+ add` + `refresh`
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
| `xaiApiKeyFile` | nullable path | `null` | file holding the bare xAI API key; loaded via `LoadCredential` and exported as `XAI_API_KEY` to enable the hosted Grok Imagine models |
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
│   ├── paths.py                  # OUT_DIR / HF_HOME / CUSTOM_DIR / Nix-store fallback
│   ├── device.py                 # auto-detect cuda / xpu / mps / cpu
│   ├── buckets.py                # SDXL_BUCKETS + ZIMAGE_BUCKETS + /res parser
│   ├── tokenize_report.py        # per-encoder analysis
│   ├── generate.py               # GenConfig (incl. lora_stack), generate()
│   ├── lora_cmd.py               # shared /lora arg parser + compat check
│   ├── preview.py                # kitty / iTerm inline + tmux passthrough
│   ├── models/
│   │   ├── spec.py               # ModelSpec + LoraSpec dataclasses
│   │   ├── sdxl_common.py        # shared SDXL two-encoder tokenize
│   │   ├── sdxl_factory.py       # make_sdxl_loader (fp16-fix VAE wired in)
│   │   ├── {zimage,pony,illustrious,noobai,community}.py
│   │   ├── loras.py              # built-in LoRA registry (LORAS = { … })
│   │   ├── custom.py             # JSON descriptors → MODELS / LORAS at runtime
│   │   └── registry.py           # MODELS = { … }
│   ├── repl/
│   │   ├── commands.py           # parse_commands + COMMAND_ARITY
│   │   ├── history.py            # readline + Tab completion (incl. /lora)
│   │   └── main.py               # repl_main()
│   └── webui/
│       ├── app.py                # FastAPI routes + run_web() + RPC handlers
│       ├── rpc.py                # WebSocket JSON-RPC dispatcher
│       ├── state.py              # AppState, Job, locks
│       ├── ws.py                 # broadcast helpers
│       ├── loader.py             # async model swap (with Job-tracked download)
│       ├── prefetch.py           # download-only model_download RPC
│       ├── downloads.py          # huggingface_hub tqdm → WS bridge
│       ├── models_info.py        # install/size scan against HF cache
│       ├── jobs.py               # run_job() worker + cancel/progress
│       ├── outputs.py            # listing + favorite + cleanup
│       └── exec_api.py           # /api/exec multi-command executor
├── custom/                       # user-supplied descriptors (gitignored)
│   ├── bases/<slug>.json
│   └── loras/<slug>.json
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
