# zimt — project notes for Claude

Multi-model image-generation REPL + web UI. Same command grammar drives
both surfaces; the parser (`src/zimt/repl/commands.py`) is the contract
both layers share.

## Running

The venv has manylinux torch wheels that need `libstdc++` from the nix
gcc-lib, and the GPU driver path from `/run/opengl-driver`. Use `run.sh`
or replicate its env:

```
GCC_LIB=$(nix eval --raw nixpkgs#gcc.cc.lib.outPath)
LD_LIBRARY_PATH="${GCC_LIB}/lib:/run/opengl-driver/lib" \
OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors \
HF_HOME=/srv/nvme/zimt/hf_cache \
.venv/bin/python -m zimt
```

Models live in `/srv/nvme/zimt/hf_cache` (writable to the user, read-only
to the sandbox). `HF_HOME` must be set or HuggingFace will scan
`~/.cache` which isn't bound into the sandbox.

## Adding a new `/foo` command

The parser is single-source-of-truth; both REPL and web exec consume the
same `parse_commands()` output. Adding a command means wiring all the
surfaces below — the completion-contract test will fail if Python's
`COMMANDS` and the JS `COMMANDS` drift apart, but everything else is on
you.

Concrete walk-through: search `git log -p -- src/zimt/memory.py` for the
`/mem` change — every file listed below was touched there.

### 1. Parser registration — `src/zimt/repl/commands.py`

Add `"/foo"` to `COMMAND_ARITY` with an arity:

| arity         | shape                              | when to use |
|---------------|------------------------------------|-------------|
| `0`           | `/foo`                             | no args (toggles, info dumps) |
| `N` (int)     | `/foo a b ...` (exactly N tokens)  | fixed positional, no whitespace in any arg |
| `"GREEDY"`    | `/foo <rest until next /cmd>`      | free-form trailing text OR variable arg count |
| `"MANY"`      | `/foo N <rest until next /cmd>`    | leading int + greedy tail (only `/many` uses this) |

GREEDY collects everything up to the next known `/cmd` token into a
single joined string `args[0]`. If you need multi-token validation (like
`/mem max 6GiB`), use GREEDY and `args[0].split()` in your handler.

### 2. REPL handler — `src/zimt/repl/main.py`

Add a branch in the `for cmd, args in cmds:` loop inside `repl_main()`.
Pure settings commands go in `_apply_setting()`; commands that mutate
session state (model swap, mem strategy) stay in `repl_main` so they can
touch the local `pipe` / `g` / `mem` bindings.

If your command can affect the loaded pipeline (e.g. reload required),
follow `/mem`: parse → compare to current → if changed, `unload(pipe)` +
`pipe = load_spec(MODELS[name], mem)`. Print what you did.

### 3. REPL completion — `src/zimt/repl/history.py`

If `/foo` takes finite-set args, add a `prev == "/foo"` branch in
`_completion_options()`. For commands whose finite set lives on the
loaded model (samplers, resolutions), use `_model_for_context(tokens)` —
it walks back to the most recent `/model X` arg in the line so
`/model pony /sampler <TAB>` completes correctly even before the load.

### 4. REPL help — `src/zimt/repl/main.py` `_help()`

One line under the existing list, matching the column width.

### 5. Web exec handler — `src/zimt/webui/exec_api.py`

Add a branch in the `for cmd, args in cmds:` loop in `api_exec()`. The
mutable state lives on `STATE` (singleton), not on locals. Use
`emit_log()` / `emit_state()` so the WebSocket clients see the change in
real time. Wrap pipe-touching work in `await _need_pipe(...)` if the
command requires a loaded model.

Pipeline reload from a settings command: call
`await load_model(current_name, force=True)` — the `force` flag bypasses
the "already loaded" short-circuit. Without it your reload silently no-ops.

### 6. Web exec help — `src/zimt/webui/exec_api.py` `_HELP_LINES`

One line, matching the existing format.

### 7. Web state surface (only if stateful) — `src/zimt/webui/state.py`

If `/foo` changes server-side state the UI needs to display, add a field
to `AppState` and surface it in `state_dict()`. The frontend
auto-broadcasts on `emit_state()`.

### 8. Web JS completion — `static/completion.js`

This file is the **completion contract** — Python's `COMMANDS` and JS's
`COMMANDS` must match (enforced by `test_completion_contracts.py`).
Three sub-edits:

- Add `"/foo"` to the `COMMANDS` array (it's `.sort()`ed; order doesn't
  matter).
- Add `"/foo": "one-line description"` to `COMMAND_HELP`.
- If `/foo` takes finite-set args, add a `prev === "/foo"` branch in
  `completionItems()`. For finite sets that depend on the active model,
  use `modelForContext(before, appState)` — JS analogue of the Python
  helper.

### 9. Web JS syntax highlight — `static/app.js` `HL_CMDS`

Without an entry here, a typed `/foo` renders as `hl-unknown-cmd`
(underlined red). Pick a shape:

| spec                                   | render |
|----------------------------------------|--------|
| `{n: 0}`                               | command alone, no args |
| `{n: 1, cls: "hl-num"}`                | one numeric arg |
| `{n: 1, cls: "hl-model"}`              | one identifier arg (use existing class) |
| `{greedy: true}`                       | greedy with default-coloured args |
| `{greedy: true, cls: "hl-neg"}`        | greedy with all args coloured |

Available classes are in `static/style.css` under "Syntax-highlight token
colours": `hl-cmd`, `hl-model`, `hl-lora`, `hl-sampler`, `hl-num`,
`hl-flag`, `hl-neg`. Prefer reuse over adding new colours.

### 10. Tests

Always run after wiring:
```
PYTHONPATH=src .venv/bin/python -m unittest tests.test_completion_contracts
node --test tests/completion.test.js
```

The completion-contract test catches Python/JS command-list drift but
**does not** verify that your handler does the right thing. If `/foo`
has interesting semantics (state mutation, validation, reload),
write a focused unit test next to similar ones in
`tests/test_webui_service.py`.

## Dynamic prompt syntax

`src/zimt/dynamics.py` implements A1111/ComfyUI-style alternation +
variables: `{a|b|c}`, weighted (`{2::a|1::b}`), nesting, empty options
(`{|a}`), variables (`${c=red|green}` then `${c}`), escapes (`\{`), and
HTML-style comments (`<!-- foo -->`). Expansion runs once per
`generate()` call, after the seed is picked and before `compose_prompt`
prepends the score-tag prefix — so a model's score tags can never be
template-expanded accidentally.

Variables bind silently — the definition site renders to "" so the
prompt reads cleanly. Inside `${name=...}`, `|` acts as an implicit
choice separator (sugar). The variable environment is fresh per
`expand()` call; nothing leaks between successive generations. Forward
references raise `DynamicsSyntaxError`. A reference to a variable that
was only defined inside a Choice branch the seed didn't pick also
raises — the error message says so explicitly.

Determinism is anchored to the same `seed` that drives image sampling.
A `random.Random(seed)` instance is created inside `expand()`; the torch
generator inside `pipe(...)` uses the same seed via its own
`torch.Generator`. The two RNG streams are independent.

PNG metadata records `raw_prompt` (the original user text) and adds
`expanded_prompt` only when expansion ran. Plain prompts have unchanged
PNG schema.

Both surfaces validate template syntax once at submission time
(`dynamics.validate(text)`), so a malformed `{red|blue` on a `/many 8`
fails fast with one error rather than eight identical errors per job.

When adding more syntax (variables, wildcard files, ...) the natural
seam is in `dynamics.py` only — `generate.py` and the surface wirings
don't need changes as long as `expand(text, rng) -> str` stays the
public API.

## Memory placement (`/mem`)

`src/zimt/memory.py` defines four placement strategies. `MemStrategy` is
threaded through `ModelSpec.load(device, mem)` and applied via two
helpers — `from_pretrained_kwargs(mem)` for the construction-time
`device_map` path, `finalize_pipe(pipe, device, mem)` for the
post-construction `.to(device)` / `enable_*_cpu_offload(...)` path. Both
helpers are no-ops in the modes that don't apply.

Steady-state cost (SDXL @ 768x768, 8 steps, Intel Arc B70):

| mode             | warm infer | peak VRAM |
|------------------|-----------|-----------|
| `off`            | 2.0s      | 7808 MiB  |
| `max <cap>`      | broken on XPU (works on CUDA) |
| `cpuoffload`     | 5.1s      | 5179 MiB  |
| `cpuoffload-seq` | 10.6s     | 1085 MiB  |

The `max` mode currently fails at inference on XPU — accelerate's
`device_map="balanced"` doesn't install device-alignment hooks for the
conv path. Ships anyway as forward-compatible UX; the failure is a
clean `RuntimeError`, not silent corruption.

## Adding a new model

`src/zimt/models/registry.py` is the single registry. For SDXL-family
fine-tunes, reuse `make_sdxl_loader(repo_id)` from `sdxl_factory.py`.
For new architectures, write a sibling module with
`load(device, mem) -> pipeline` (and ideally `tokenize_report`), then
add a `ModelSpec` entry. Every loader **must** accept `mem` — see
`zimage.py` for the non-SDXL example.

## Outputs gallery modal

The image-preview modal in the web UI supports ←/→ + on-screen prev/next
overlay buttons. Navigation uses the live `outputs` array — already
filtered by the active tab (all/fav) — so it respects the user's filter
implicitly without needing a separate "filter mode" argument.

The pure neighbor lookup lives in `static/modal_nav.js`
(`neighborInOutputs(outputs, currentName, direction)`) so it can be
unit-tested without a DOM. The two test harnesses that `vm.runInNewContext`
the bulk `static/app.js` (`tests/frontend_state.test.js`,
`tests/model_download_state.test.js`) stub the helper as a no-op since
they don't exercise the modal navigation path.

Keyboard shortcuts are bound at `document` level but gated on
`#modal.open` AND the active element NOT being an INPUT/TEXTAREA/
contenteditable — so the prompt editor's arrow-key behaviour is
unchanged when the modal is closed.

## Debugging

- One-shot scripts go under `./debug/{YYYYMMDD-HHMMSS}-{name}.py`. They
  must set the `LD_LIBRARY_PATH` + `HF_HOME` env from `run.sh` to import
  torch successfully.
- `unload()` walks `pipe.components` and nulls each slot so VRAM is
  released even if the caller keeps a wrapper reference. Hook-dispatched
  pipes (cpuoffload modes) need `pipe.remove_all_hooks()` first or you
  get the "you shouldn't move a model that is dispatched using
  accelerate hooks" warning.

## Sandbox

`SMIND_SANDBOXED=1` is set inside the bubblewrap wrapper. Writes only
persist inside the project directory and `/tmp/exchange`. The model
cache at `/srv/nvme/zimt` is read-only from the sandbox. Use the
exchange-script workflow (see the `environment` skill) for anything
outside.

The user's hostname is `vm` — use that literal where scripts reference
the current host; `$HOSTNAME` is not exported by zsh.
