# zimt — Task Ledger

Authoritative ledger for the requested model-tab download fixes and whole-codebase review. Detailed plan: `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`.

Status: `[ ]` planned · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Milestones (high-level)

- [x] **M1** — Resolve known model-tab/download correctness defects with regression tests.
- [~] **M2** — Perform whole-codebase adversarial review and execute follow-up fixes for confirmed defects.
- [x] **M3** — Apply Firefox WebSocket quirks per the `/resilient-ws-ui` skill.

---

## Milestone 1 — PR breakdown

Detail in `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`. One line per PR here; sub-task detail stays in the plan doc.

- [x] **PR-01** — Introduce base/LoRA download target identity.
- [x] **PR-02** — Make download progress ownership job-scoped.
- [x] **PR-03** — Add cancellation semantics for download jobs.
- [x] **PR-04** — Normalize Hugging Face progress units.
- [x] **PR-05** — Prevent untracked LoRA downloads during generation.
- [x] **PR-06** — Consolidate model-tab download state transitions.

---

## Milestone 2 — PR breakdown

Detail in `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`.

- [x] **PR-07** — Whole-codebase review inventory and defect triage.
- [x] **PR-08** — Backend concurrency and lifecycle follow-up fixes.
- [x] **PR-09** — Frontend state and API-contract follow-up fixes.
- [ ] **PR-10** — Filesystem, configuration, and external-boundary follow-up fixes.
- [ ] **PR-11** — Final adversarial review and release verification.

---

## Milestone 3 — PR breakdown

Scope: front-end WebSocket transport must remain reliable on Firefox per the `/resilient-ws-ui` skill (Firefox treats unclean closes differently than Chromium-based browsers; heartbeats, reconnect backoff, and connection-state surfacing all need explicit handling). Detail plan to be written when work begins.

- [x] **PR-12** — Apply Firefox WebSocket quirks (heartbeat / reconnect backoff / connection-state UX) per the `/resilient-ws-ui` skill. Added 2026-05-20 at user request.

---

## Cross-cutting architectural notes (locked)

- [x] Download jobs need asset identity separate from queue job type: `kind="download"` remains the queue discriminator; `target_kind` distinguishes base-model assets from LoRA assets. Lands in PR-01.
- [x] Download progress ownership invariant — at most one HF download owns the progress slot at a time, acquired by job id via `set_active_download` and released by the same id via `clear_active_download`. Per-bar identity is propagated to executor workers via a `contextvars.ContextVar` (`download_context(job_id)`); callers MUST use `ctx = contextvars.copy_context()` and dispatch via `loop.run_in_executor(executor, lambda: ctx.run(func, *args))` — `run_in_executor` does NOT propagate contextvars on its own (CPython 3.13). tqdm bars constructed in the worker capture the owner id and silently drop events when it does not match the slot. Loser callers receive a passive busy state via `emit_log`; a row-level "busy" indicator is deferred to PR-06. Lands in PR-02 (D10 fix landed in PR-03 to satisfy the production assumption).
- [x] Download cancellation invariant — each download job (`load_model` and `prefetch_model`) registers a `threading.Event` in `CANCEL_EVENTS` at Job creation and pops it on exit. Cancellation observed at controlled boundaries: loader's pre-load polling loop, prefetch's post-`_PREFETCH_LOCK` checkpoint, and `ProgressTqdm.__init__` / `update` (raises `DownloadCanceled` which is caught in both callers and ends the job with status `canceled`). `jobs_cancel_all` RPC covers both `generate` and `download` kinds. The UI renders a cancel button on queued and running download rows. Lands in PR-03.
- [x] Progress unit invariant — every progress event carries an explicit unit category (`"bytes"` | `"files"` | `"items"` | `""`) on `Job.download_unit`. Classification: HF's byte aggregate bar (`unit='B'`) is `"bytes"`; HF's outer file-count bar (`unit='it'` default, identified by `desc` regex `^(?:\[dry-run\]\s*)?Fetching\b`) is `"files"`; any other tqdm event is `"items"`. The UI never claims byte semantics without `unit=='B'` evidence. `download_files_done` is driven exclusively by the file-count bar's current `n` (overwrite in `_emit`, never mutated by `close()`); without a file-count bar it stays 0. Lands in PR-04.
- [x] Download state-machine invariant — duplicate `model_download` requests for the same `(target_kind, name)` while a job is `queued` or `running` (and not cancel-pending) collapse to the existing job id; cancel-pending jobs are excluded from the collapse so user-initiated retries during the cancel window land on a fresh Job. Frontend renders three button states via `_modelDlBtnState({dlJob, installed})`: "downloading [N%]" disabled while a job is active; "redownload" for installed assets; "download" otherwise. Active-job state takes precedence over the installed/uninstalled distinction. Lands in PR-06.
- [x] LoRA dependency invariant — generation must not initiate implicit HF downloads. `is_installed(repo_id)` in `webui.models_info` is the single oracle for cache status; it scans `huggingface_hub.scan_cache_dir()` on each call and returns False on any error (safer default). `generate._apply_lora_stack` raises `UninstalledLoraError(missing)` before any `pipe.load_lora_weights(...)` call when any LoRA in the stack has an uninstalled repo. `lora_cmd.apply_lora_args` rejects uninstalled LoRAs at add time. `webui.jobs.run_job` catches `UninstalledLoraError` separately and reports a clean `str(e)`. Frontend disables the LoRA "add" button when `!installed` (active stack entries remain removable). Lands in PR-05.

---

## Completed

- **PR-01** (2026-05-19) — Download queue jobs now keep queue type and
  downloadable asset identity separate: download jobs still use
  `kind="download"`, and base/LoRA identity travels as `target_kind`. Base
  model load jobs set `target_kind="base"`; model-tab prefetch jobs set
  `target_kind` from the requested target kind; model-tab active download
  lookup filters by both `target_kind` and model name so same-named base and
  LoRA entries do not share progress state.
  Reproduction before fix:
  - `node --test tests/model_download_state.test.js` failed because
    `_downloadingByName("lora")` returned the active base download for the
    shared model name.
  - `LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors HF_HOME=/home/pavel/work/safe/zimt/hf_cache HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadIdentityTests`
    failed with missing `Job.target_kind` / unexpected `target_kind`
    constructor argument.
  Verification:
  - `node --test tests/model_download_state.test.js` → pass, 1 test.
  - `LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors HF_HOME=/home/pavel/work/safe/zimt/hf_cache HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadIdentityTests`
    → pass, 3 tests.
  - `node --test tests/*.test.js` → pass, 10 tests.
  Review:
  - Adversarial review reported no PR-01 defects. Residual low-risk gaps:
    serialization coverage uses `asdict` directly rather than `emit_job`, and
    frontend coverage checks lookup separation rather than full rendered
    same-name base/LoRA row state.
  Notes / constraints:
  - `pytest` is not installed in the virtualenv; backend tests were run via
    `unittest`.
  - `tests.test_webui_service` still has an unrelated pre-existing failure:
    `LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads`
    expects `HTTPException` while `loader.load_model()` raises
    `ModelLoadError`.
  - `nix develop --command pyright src/zimt` still reports an unrelated
    pre-existing `ModuleType.tqdm` monkey-patch error in
    `src/zimt/webui/downloads.py:120`.

- **PR-02** (2026-05-20) — Download progress ownership is now job-scoped at
  two layers: the slot itself (`_active_job_id` acquired by
  `set_active_download(job_id) -> bool`, released by
  `clear_active_download(job_id)` only when the caller is the current owner)
  and the per-tqdm-bar ownership (a `contextvars.ContextVar` propagated
  through `loop.run_in_executor` via a new `download_context(job_id)` context
  manager; `ProgressTqdm.__init__` captures the owner id, and `_emit` / `close`
  early-return when the captured id does not match `_active_job_id`).
  Net behavioral change: a `load_model` that races a running prefetch no
  longer steals the slot, no longer clears another job's slot, and its
  HF tqdm events do not corrupt the prefetch's progress fields — they are
  silent no-ops at the bar level. A symmetric `emit_log` on each caller
  surfaces the busy-slot condition for the user. Both callers' acquires
  now sit inside the `try:` paired with the finally's owner-scoped clear,
  so an exception in the busy-log path cannot leak the slot.
  Reproduction before fix:
  - New `DownloadOwnershipTests` test (added in the PR-02 work-in-progress
    state) failed: during a load while a prefetch held the slot,
    `current_download()` inside the load reported the load's job (slot was
    stolen) and after `load_model` returned reported `None` (slot was
    cleared while prefetch still ran). Verified with
    `.venv/bin/python -m unittest tests.test_webui_service.DownloadOwnershipTests`.
  Verification:
  - `LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors HF_HOME=/home/pavel/work/safe/zimt/hf_cache HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadOwnershipTests`
    → pass, 1 test.
  - `LD_LIBRARY_PATH=...same... .venv/bin/python -m unittest tests.test_webui_service.DownloadIdentityTests`
    → pass, 3 tests (no regression to PR-01).
  - `LD_LIBRARY_PATH=...same... .venv/bin/python -m unittest discover -s tests`
    → 17 tests, 16 pass, 1 pre-existing failure unchanged
    (`LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads`,
    expects `HTTPException`, raises `ModelLoadError` — flagged in PR-01's
    Completed entry, still out of scope).
  - `node --test tests/*.test.js` → 10/10 pass.
  - `nix develop --command pyright src/zimt` → only the pre-existing
    `downloads.py:181` tqdm monkey-patch finding remains; no new findings.
  Review:
  - Three rounds of adversarial review. Round 1 confirmed the slot-level
    fix worked but flagged cross-attribution of tqdm events (D01–D03).
    Round 2 fix introduced per-bar ContextVar gating; review then flagged
    acquire-outside-try (D08) and an unnecessary `# type: ignore[return]`
    (D09). Round 3 review found no merge-blocking defects.
  Notes / constraints:
  - The "loser" download proceeds without broadcasting progress and is
    signaled only via an `emit_log` line ("download progress slot busy;
    ... will not be broadcast"); a row-level "busy" UI indicator is
    intentionally deferred to PR-06 (D04 closed with a deferral note).
  - Correction (added 2026-05-20, during PR-03): the PR-02 design relied
    on the assumption that `loop.run_in_executor` propagates the current
    `contextvars.Context` to the worker thread. **That assumption is
    false** in CPython 3.13 — `run_in_executor` does not copy context
    (only `asyncio.Task` creation does). All three rounds of PR-02
    adversarial review missed this because every PR-02 test patched the
    executor payload to a function that never constructs a real tqdm bar.
    PR-03's execution surfaced the latent defect (now recorded as
    PR-02-D10) and fixed it: callers now explicitly
    `ctx = contextvars.copy_context()` inside `with download_context(...)`
    and dispatch via `loop.run_in_executor(executor, lambda: ctx.run(func, *args))`.
    Without that wrapper, `_owner_var` is `None` in the worker and the
    per-bar gating drops all events. New `DownloadContextPropagationTests`
    locks in the correct mechanism.
  - `StateCase.setUp/tearDown` now snapshot and restore
    `downloads._active_job_id` to keep module-global slot ownership from
    leaking across tests. New test
    `DownloadOwnershipApiTests.test_set_active_download_is_idempotent_for_same_owner`
    locks in the documented `set_active_download` idempotent re-acquire
    contract.
  - Pre-existing `LoaderTests` failure (HTTPException vs ModelLoadError)
    remains unchanged; not in PR-02's scope.

- **PR-03** (2026-05-20) — Download jobs now support per-job and
  cancel-all cancellation. Each `kind="download"` Job (from `load_model`
  or `prefetch_model`) registers a `threading.Event` in `CANCEL_EVENTS`
  at creation time and pops it on every exit path. Cancellation is
  observed at three controlled boundaries: (a) the loader's
  pre-load polling loop checks each iteration and exits with status
  `canceled` if the event is set; (b) `prefetch._run()` checks the event
  right after acquiring `_PREFETCH_LOCK` (queued-cancel boundary, before
  flipping status to `running`); (c) the executor-thread tqdm hook
  (`ProgressTqdm.__init__` and `update`) calls `_zimt_check_cancel`
  which raises a new `DownloadCanceled` exception that propagates up
  through the HF stack and is caught in both callers' `_run` /
  `load_model`. The `jobs_cancel_all` RPC was updated to include
  `kind="download"` (single-character change at `app.py:337`). The
  frontend renders a cancel button on `kind === "download"` rows in
  `queued` / `running` status, mirroring the generate row's button. The
  PR also fixed PR-02-D10 (see correction in PR-02's Completed notes):
  contextvars don't propagate through `run_in_executor` by default, so
  both callers now explicitly `ctx = contextvars.copy_context()` and
  dispatch via `lambda: ctx.run(func, *args)`. The PR-03 cancel path
  works because `_zimt_owner` is now actually populated in worker
  threads. The earlier PR-03 attempt used a fallback to `_active_job_id`
  inside `_zimt_check_cancel`; the fallback was removed because it
  would re-introduce PR-02-D01 (cross-attribution) and misattribute
  cancellation across concurrent downloads.
  Reproduction before fix:
  - New `DownloadCancelTests` tests failed: queued-prefetch cancellation
    did not prevent execution (no cancel event existed for downloads);
    running-prefetch cancellation had no boundary to fire at; and
    `jobs_cancel_all` returned 0 for download jobs (filter was
    `kind == "generate"`).
  Verification:
  - `LD_LIBRARY_PATH=...HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadCancelTests`
    → pass, 3 tests.
  - `.venv/bin/python -m unittest tests.test_webui_service.DownloadContextPropagationTests`
    → pass, 2 tests.
  - `.venv/bin/python -m unittest discover -s tests` → 22 tests, 21 pass,
    1 pre-existing failure unchanged (`LoaderTests.test_failed_load_...`).
  - `node --test tests/*.test.js` → 11/11 pass (1 new frontend test
    asserts download rows render a cancel-btn for running status and
    none for done status).
  - `nix develop --command pyright src/zimt` → only the pre-existing
    tqdm monkey-patch finding remains; no new findings.
  Review:
  - One round of adversarial review on the combined PR-03 + D10 fix.
    No merge-blocking defects. Three nits noted and discarded as future
    polish (cosmetic emit_state flicker on cancel; queued-cancel
    latency when another download holds `_PREFETCH_LOCK`; coverage gap
    on `__init__`-time cancel-already-set).
  Notes / constraints:
  - HF `snapshot_download` does not natively cancel; the controlled
    boundary is the tqdm callback. Cancel during a single file in flight
    takes effect at the next tqdm `update()` (typically every chunk),
    bounded by file size. The PR-04 progress-unit work and any future
    chunked-download work should preserve this raise-in-tqdm boundary.
  - Cancel of a queued prefetch that is parked on `_PREFETCH_LOCK`
    behind a running prefetch will not take effect until the running
    prefetch finishes (could be minutes). UI still shows the queued
    row's cancel button disabled (the click handler disables it). This
    is structurally correct (no execution occurs) but the latency may
    surprise the user; not a defect against the plan, deferred as UX
    follow-up.
  - `asyncio.create_task(_run())` in `prefetch.py:128` still does not
    retain a strong task reference (pre-existing antipattern, untouched
    by PR-03).
  - `_zimt_check_cancel` does NOT fall back to `_active_job_id`. Any
    future caller that creates a `ProgressTqdm` outside `download_context`
    or without `ctx = copy_context(); ctx.run(...)` will have a bar with
    `_zimt_owner = None`, and neither cancellation nor progress will
    fire for that bar. Documented in `src/zimt/webui/downloads.py`
    module docstring.
  - Pre-existing `LoaderTests` failure (HTTPException vs ModelLoadError)
    remains unchanged; not in PR-03's scope.

- **PR-04** (2026-05-20) — HuggingFace progress now carries an explicit
  unit category so the UI never claims byte semantics for file-count
  progress. New `Job.download_unit` field with values `"bytes"` |
  `"files"` | `"items"` | `""`. `ProgressTqdm._emit` classifies each
  event via `_classify(raw_unit, desc)`: `unit=='B'` → `"bytes"`;
  hand-built `unit in {'file','files'}` → `"files"`; otherwise if `desc`
  matches `_HF_FETCHING_FILES_RE` (`^(?:\[dry-run\]\s*)?Fetching\b`,
  covering all four HF `tqdm_desc` variants) → `"files"`; else
  `"items"`. The byte-unit check runs first so a hypothetical byte bar
  with a "Fetching…" desc still classifies as bytes. `download_files_done`
  is now driven by the file-count bar's current `n` via overwrite in
  `_emit`; `close()` was changed to no longer mutate the counter at all
  (the pre-PR-04 code incremented on every close, which was wrong both
  for snapshot_download — exactly one byte-bar close per snapshot — and
  conceptually). The frontend `fmtDownloadCounter(j)` formats per unit:
  bytes via `fmtBytes`, files as `"N / T files"`, items as bare numbers,
  no-unit as bare numbers (never bytes).
  Reproduction before fix:
  - PR-04 work-in-progress introduced `_normalize_unit` keyed only on
    the tqdm `unit` attribute. Adversarial review verified against the
    installed `huggingface_hub`'s `_snapshot_download.py` that the outer
    file-count bar HF emits is `unit='it'` (tqdm default) — so PR-04's
    initial test that constructed `unit='file'` passed but never
    exercised the production path. Defect logged as PR-04-D01.
  Verification:
  - `LD_LIBRARY_PATH=...HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.ProgressUnitTests`
    → pass, 4 tests (including the new combined-bar integration test).
  - `.venv/bin/python -m unittest discover -s tests` → 26 tests, 25 pass,
    1 pre-existing failure unchanged (`LoaderTests.test_failed_load_...`).
  - `node --test tests/*.test.js` → 16/16 pass (5 new frontend
    `fmtDownloadCounter` cases covering bytes / files / items / no-unit
    with and without counters).
  - `nix develop --command pyright src/zimt` → only the pre-existing
    tqdm monkey-patch finding remains; no new findings.
  Review:
  - Two rounds of adversarial review. Round 1 flagged PR-04-D01:
    `_normalize_unit` keyed on the wrong attribute and `close()`
    incremented `download_files_done` on a bar that does not represent
    one file (the snapshot-aggregate byte bar). Round 2 verified the
    fix — `_classify` by desc, files_done driven by the file-count
    bar's `n`, close() no longer mutating files_done.
  Notes / constraints:
  - The `_HF_FETCHING_FILES_RE` regex is deliberately broad (matches
    `[dry-run]` prefix and `Fetching …` without explicit digits). If a
    future HF release changes the desc wording, the bar will silently
    fall back to `"items"` and `download_files_done` will read 0 even
    when files are completing. Maintainer should re-verify the regex
    against new HF versions when bumping the `huggingface_hub` pin.
  - During a real `snapshot_download` both the file-count bar and the
    byte bar fire alternately. `Job` carries a single `download_n` /
    `download_total` / `download_unit` slot, so the UI flickers between
    file-mode and byte-mode labels and the progress bar fill width
    oscillates between file-percent and byte-percent. PR-06 (Consolidate
    model-tab download state transitions) is scoped to introduce a
    coherent state model for this; PR-04 leaves it as-is per scope.
  - Pre-existing `LoaderTests` failure (HTTPException vs ModelLoadError)
    remains unchanged; not in PR-04's scope.

- **PR-05** (2026-05-20) — Generation can no longer initiate an
  untracked HF download. The defect was that `generate._apply_lora_stack`
  called `pipe.load_lora_weights(spec.repo_id, ...)` for any LoRA in
  `g.lora_stack`; if the repo wasn't in the local HF cache, diffusers
  would silently spawn a network download outside the project's queue,
  with no progress, no cancel, and no Job entry. PR-05 adds three
  layers of guard: (1) `webui.models_info.is_installed(repo_id)` —
  authoritative cache-status oracle backed by `huggingface_hub.scan_cache_dir`
  (returns False on any error, fail-safe); (2) `_apply_lora_stack`
  raises a new `UninstalledLoraError` listing every missing LoRA name
  *before* the existing load loop runs; (3) `lora_cmd.apply_lora_args`
  rejects uninstalled LoRAs at add time with a "not installed locally"
  log line, so they never enter the stack in the first place. The
  webui `jobs.run_job` adds a dedicated `except UninstalledLoraError`
  branch before the generic catch so `job.error` reads as
  `str(e)` ("LoRA(s) not installed: …. Install via the Models tab…")
  rather than `repr(e)`. The frontend disables the LoRA "add" button
  when the LoRA is uninstalled (active stack entries remain removable
  to avoid stranding the user) via a new `_loraAddBtnState` helper.
  Reproduction before fix:
  - `UninstalledLoraGuardTests` failing first against the unmodified
    `_apply_lora_stack`: with `is_installed` patched to False and a
    LoRA in the stack, `pipe.load_lora_weights` was called (no
    exception). Demonstrates the untracked-download surface.
  Verification:
  - `LD_LIBRARY_PATH=...HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.UninstalledLoraGuardTests`
    → pass, 4 tests including the run_job end-to-end which asserts
    both `pipe.load_lora_weights.assert_not_called()` and the bare
    `STATE.pipe.assert_not_called()` (no pipeline invocation).
  - `.venv/bin/python -m unittest discover -s tests` → 30 tests, 29 pass,
    1 pre-existing failure unchanged (`LoaderTests.test_failed_load_...`).
  - `node --test tests/*.test.js` → 20/20 pass (4 new
    `_loraAddBtnState` tests covering installed, uninstalled,
    active-uninstalled, and incompatible cases).
  - `nix develop --command pyright src/zimt` → only the pre-existing
    tqdm monkey-patch finding remains; no new findings.
  Review:
  - One round of adversarial review on the implementation. Two
    coverage gaps flagged (PR-05-D01: incompatible-LoRA frontend test
    missing; PR-05-D02: run_job test lacked the plan-required network
    sentinel). Both closed in a single follow-up; no source code
    changes, only test additions.
  Notes / constraints:
  - `lora_cmd.py` must lazy-import `is_installed` from
    `webui.models_info` because `webui.exec_api` top-imports
    `lora_cmd` (a top-level reverse import would create a cycle).
    Same lazy-import pattern in `generate._apply_lora_stack`.
  - `_apply_lora_stack`'s `family != "sdxl"` early return continues
    to skip the install check — by inspection no non-SDXL family
    triggers `load_lora_weights` today, so non-SDXL is structurally
    safe. If a future model family begins calling `load_lora_weights`
    the guard must move above the family check.
  - `is_installed` calls `_scan_cache()` per invocation. Cost is the
    same "few ms per repo" that PR-01's notes already document. No
    in-process caching was added because cache state can change
    between calls (a parallel prefetch can install a LoRA mid-session),
    and a stale cache would re-introduce the untracked-download path
    we just closed.
  - The user-visible error message format (`"LoRA(s) not installed:
    {names}. Install via the Models tab before generating."`) is
    deterministic; the existing `run_job` test asserts the exact
    string. If the wording changes, update the test.
  - Pre-existing `LoaderTests` failure (HTTPException vs
    ModelLoadError) remains unchanged; not in PR-05's scope.

- **PR-06** (2026-05-20) — Final PR of Milestone 1. Consolidates the
  model-tab download state transitions: (a) backend idempotency — a
  duplicate `model_download` request for the same `(target_kind, name)`
  while an earlier job is queued or running returns the existing
  job_id, with one exception: jobs whose cancel event is set (but
  whose status hasn't yet transitioned through the runner's checkpoint)
  are *excluded* from the scan so the user's retry during the cancel
  window lands on a fresh Job; (b) frontend — extracted
  `_modelDlBtnState({dlJob, installed})` pure helper returning
  `{text, disabled, title}` for the three branches (in-progress /
  installed / fresh), mirroring PR-05's `_loraAddBtnState` pattern.
  The retry-after-terminal paths (done / error / canceled) already
  produced fresh job ids because each `prefetch_model` call generates
  a new UUID; PR-06 adds explicit regression tests confirming that.
  FIFO order across multiple queued downloads already held via
  `_PREFETCH_LOCK`'s asyncio.Lock wait queue; unchanged.
  Reproduction before fix:
  - New `DownloadStateMachineTests` failed against the unmodified
    `prefetch_model`: duplicate calls for the same `(kind, name)`
    created multiple Jobs, and the new frontend tests failed because
    `_modelDlBtnState` did not exist yet.
  Verification:
  - `LD_LIBRARY_PATH=...HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadStateMachineTests`
    → pass, 6 tests (running-duplicate, queued-duplicate, retry after
    done/error/canceled, retry-during-cancel-pending).
  - `.venv/bin/python -m unittest discover -s tests` → 36 tests, 35
    pass, 1 pre-existing failure unchanged
    (`LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads`).
  - `node --test tests/*.test.js` → 25/25 pass (5 new
    `_modelDlBtnState` branch tests).
  - `nix develop --command pyright src/zimt` → only the pre-existing
    tqdm monkey-patch finding; no new findings.
  Review:
  - Round 1 found three nits and two real but minor concerns
    (PR-06-D01 cancel-pending window leak; PR-06-D02 misplaced lock
    workaround). Both fixed.
  - Round 2 confirmed both fixes correct, the D01 test runs 5/5 across
    repeated invocations (not flaky), and no regressions to PR-03
    cancellation tests when the `_PREFETCH_LOCK._loop` reset moved into
    `StateCase`. Clean.
  Notes / constraints:
  - The asyncio.Lock cross-loop binding workaround
    (`prefetch._PREFETCH_LOCK._loop = None` in `StateCase.setUp`) is a
    necessary CPython quirk: `asyncio.Lock` caches `_loop` as an
    instance attribute on its contended path; subsequent
    `IsolatedAsyncioTestCase` runs would otherwise raise
    `RuntimeError: <Lock [locked]> is bound to a different event
    loop`. Production runs one persistent event loop, so the
    workaround is purely test-infrastructure.
  - The plan's PR-06 "Cover the complete model-tab state machine"
    scope was honoured for state-transitions and retry semantics. The
    deferred row-level "busy" indicator for the load-loser
    (PR-02-D04) and the bar-flicker between file-count and byte
    counters (PR-04 follow-up) both require a `Job` model
    refactor (new field for `progress_owner: bool`, splitting
    `download_n/total` into per-unit slots). They were intentionally
    left out of PR-06's scope and remain as future work — see the
    cross-cutting notes for the deferral rationale.
  - Cancel-pending detection in idempotency: the scan checks
    `CANCEL_EVENTS.get(existing.id).is_set()`. If a future code path
    forgets to register a cancel event for a download Job, the scan
    treats it as "not cancel-pending" and may collapse retries into
    it. PR-03's loader/prefetch both create cancel events at Job
    creation, so this is currently safe. A regression test would be
    valuable if a future PR adds another download path.
  - Pre-existing `LoaderTests` failure (HTTPException vs
    ModelLoadError) remains unchanged; not in PR-06's scope.

---

**Milestone 1 complete (2026-05-20).** All six known model-tab/download
correctness defects have regression tests and shipped fixes.
Cross-cutting invariants are locked in tasks.md's "Cross-cutting
architectural notes (locked)" section. Two design choices were
intentionally deferred to future work (and recorded in defects.md as
PR-02-D04 + the PR-04 flicker note): a Job-level `progress_owner` flag
and per-unit `download_*` slots; both require a Job dataclass refactor.
The whole-codebase review (M2) will pick up next.

- **PR-12** (2026-05-20) — Closed the Firefox-pertinent gaps from the
  `/resilient-ws-ui` skill. `static/connection.js` is refactored into a
  per-socket `Connection` class and a pool-orchestrator
  `ConnectionManager`. Three patterns shipped: **(R6) overlapping
  connection failover** — when the active socket goes STALE the
  manager spawns a replacement in parallel; whichever reaches ALIVE
  first wins and the loser is closed with the application-private
  code 4002 "superseded"; pool capped at `MAX_LIVE_CONNECTIONS = 3`.
  This is the canonical mitigation for Firefox bug 920074 (silent
  NAT drops with no `close` event). **(R9) BFCache wiring** —
  `pagehide(persisted=true)` closes all sockets with code 1001 and
  parks reconnect via `_bfcacheParked` so the page is BFCache-eligible;
  `pageshow(persisted=true)` clears the flag, resets attempts to 0,
  and reconnects immediately. **(R9) Network Information API** — the
  `change` listener on `navigator.connection` is now actually attached
  and pings the active socket on path change. The public `stats()`
  shape preserves all keys consumed by `app.js:deriveWidget`
  (`state`/`attempt`/`maxAttempts`/`isTerminal`/`deferredOnVisible`/
  `pendingPings`/`nextReconnectInMs`/`lastCloseCode`/`lastCloseReason`)
  while adding `connections[]`, `activeConnectionId`, `pool`,
  `frozen`. `app.js` required zero changes — the widget renders
  correctly off the preserved keys, with state-fallback ranked
  ALIVE > NEW > STALE > DEAD when no active is set so the bootstrap
  pill shows "connecting…" rather than "disconnected".
  Reproduction before fix:
  - The two PR-12 tests for overlapping failover and BFCache failed
    against the pre-refactor single-socket manager: STALE didn't
    spawn a replacement, and `pagehide(persisted)` did nothing
    useful.
  Verification:
  - `node --test tests/*.test.js` → 33/33 pass (25 → 33; 8 new
    tests in `tests/connection_manager.test.js` covering single-
    connection happy path, STALE-triggers-replacement, late-pong
    promotes-back-and-supersedes-replacement, replacement-wins-
    and-supersedes-stale, MAX_LIVE_CONNECTIONS cap actually hits
    the rejection branch, `pagehide(persisted)` closes-without-
    reconnect, `pageshow(persisted)` reconnects-immediately, and
    `request()` routes through the new active after promotion).
  - `.venv/bin/python -m unittest discover -s tests` → 36 tests,
    35 pass, 1 pre-existing failure unchanged
    (`LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads`).
  - `nix develop --command pyright src/zimt` → only the pre-existing
    tqdm monkey-patch finding; no new findings.
  Review:
  - One round of adversarial review found five issues, two of
    which were closed in this PR (PR-12-D01 listener leak on
    `destroy()`, PR-12-D02 weak cap test). The other three (F-3
    `pendingPings` aggregation, F-4 stale `lastCloseCode` on
    pageshow, F-5 dead-code `_pickActiveFromPool`) were considered
    and discarded as informational. Promotion scenarios (a)/(b)/(c)/
    (d) all audit correctly; superseded-close path is race-free
    because `_supersededClose` is set synchronously before
    `ws.close()`. BFCache + time-jump detector interaction is
    benign — on `pageshow` there is no active connection, so the
    post-resume tick is a no-op and the pageshow-scheduled
    immediate reconnect runs without a duplicate race.
  Notes / constraints:
  - Server side (`src/zimt/webui/app.py:444-534`) was not touched —
    it already does nonce-correlated ping/pong with watchdog;
    the skill's R11 (Node-specific `setImmediate` ordering) does
    not apply to Python asyncio.
  - Intentional gaps remaining (per the file's updated docblock):
    R14 main-thread heartbeat — Chrome throttles main-thread
    timers in heavily-backgrounded tabs and the heartbeats stop
    until visible; a dedicated Web Worker would fix this but is
    out of scope for a single-tab tool. Session resumption — a
    reconnect is still a fresh logical session; the server's
    "hello" state + in-flight Jobs is sufficient for zimt today.
  - Test harness uses a `FakeWS` injected via the `opts.WebSocket`
    constructor option and a virtual-time `setTimeout`/`setInterval`
    harness; no real WebSocket connections are opened in tests.
  - Pre-existing `LoaderTests` failure (HTTPException vs
    ModelLoadError) remains unchanged; not in PR-12's scope.

---

**Milestone 3 complete (2026-05-20).** The Firefox WebSocket
quirks the user flagged are now closed: silent NAT drops have
zero-gap failover, BFCache is supported, and Network Information
API changes are observed. M2 (whole-codebase adversarial review)
remains planned.

- **PR-07** (2026-05-20) — Whole-codebase adversarial review pass.
  Read-only inventory of every subsystem (backend webui, generation
  core, frontend, tests, config/packaging, filesystem/external
  boundaries). No source code changes. Produced a review memo at
  `docs/drafts/20260520-1014-pr07-codebase-review.md` and 19 defect
  entries in `defects.md` under `## PR-07` (PR-07-D01..PR-07-D19).
  Severity split: 5 major, 13 minor, 1 nit. Notable finds:
  `STATE.loading_model` check-then-set race across an await (D03);
  unretained `asyncio.create_task` background tasks vulnerable to
  GC (D05); `dataclasses.replace(STATE.g)` shares `lora_stack` by
  reference with the live config (D08); `outputs_cleanup` follows
  symlinks (D04); empty-Origin CSRF bypass (D06); `is_installed`
  full HF cache scan on every generation (D13). Two pre-existing
  items M1 explicitly named are now logged as D01 (LoaderTests
  HTTPException/ModelLoadError mismatch) and D02 (pyright tqdm
  monkey-patch finding). Follow-up fixes are routed by domain:
  PR-08 backend (D01, D03, D05, D07–D13); PR-09 frontend (D15–D18);
  PR-10 boundaries (D02, D04, D06, D14, D19). PR-11 is the final
  release-verification pass.
  Verification: this PR has no source code changes; the verification
  is the integrity of the review itself. Memo cites file:line for
  every defect; each defect carries Description + Root cause +
  Suggested fix per the ledger schema.
  Notes / constraints:
  - The memo's "Tests" section notes coverage gaps (`_gpu_stats_loop`,
    heartbeat watchdog, `outputs_cleanup` symlinks) that were not
    raised as separate defects per the brief's "quality not
    coverage" framing. PR-08/PR-09 fix work should add regression
    tests for any defect whose fix lands.

- **PR-08** (2026-05-20) — Backend concurrency and lifecycle fixes for
  PR-07-D01, D03, D05, D07, D08, D09, D10, D11, D12, D13. Headline
  changes: new `LOADING_LOCK = asyncio.Lock()` in `state.py` wraps the
  loader's gate-then-set so two concurrent loads can't race through
  the polling window (D03); `_BACKGROUND_TASKS: set[asyncio.Task]` +
  `register_task(coro)` helper in `state.py` is now used at every
  prior `asyncio.create_task` site except the heartbeat (which is
  already bound to a local), eliminating the GC-of-pending-task class
  (D05); `_graceful_shutdown` now drains `_PREFETCH_EXECUTOR` after
  `EXECUTOR` (D07); `exec_api.api_exec`'s `replace(STATE.g)` now
  passes `lora_stack=list(STATE.g.lora_stack)` so the job snapshot
  doesn't alias the live stack (D08); `jobs.run_job`'s generic
  exception path now writes `f"{type(e).__name__}: {e}"` rather than
  `repr(e)`, matching the cleaner `UninstalledLoraError` rendering
  (D09); `_active_lock`'s hold extends across the field writes in
  `_emit` and across the `asdict(job)` snapshot in `ws.emit_job` for
  download jobs, so the asyncio reader can't observe a torn
  download_* tuple (D10); `ws.broadcast` wraps the `json.dumps` in
  try/except with a `default=str` fallback and an inner guard against
  pathological `__repr__`s, so a single bad event no longer takes the
  broadcast loop down (D11); the `StateCase.setUp` cross-loop reset
  now also targets `PIPE_LOCK` and `LOADING_LOCK` in addition to
  `_PREFETCH_LOCK` (D12); `models_info` got a `_CACHE_TTL_S = 1.0`
  TTL cache + `_invalidate_cache()` invalidated on prefetch success
  AND on base-model load success, with two regression tests covering
  TTL expiry and explicit invalidation (D13 + PR-08-D03 follow-up).
  D01 (pre-existing `LoaderTests` mismatch) was closed by aligning
  the assertion to expect `ModelLoadError`. Three follow-ups
  surfaced by the adversarial review were closed in the same PR:
  PR-08-D01 (the D03 regression test was structurally unable to
  exercise the race), PR-08-D02 (`loader.load_model` still used
  `repr(e)`, asymmetric with the D09 fix in `jobs.py`), and
  PR-08-D03 (cache invalidation skipped the base-load path).
  Verification:
  - `.venv/bin/python -m unittest discover -s tests` → 46 tests, all
    pass (was 35 pass + 1 fail). The previously-failing
    `LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads`
    now passes after the D01 fix. +11 net tests: 8 new in PR-08
    proper, 3 new added by the PR-08 follow-ups (TTL-expiry test,
    explicit-invalidate test, restructured concurrent-load test
    replaces but doesn't add).
  - `node --test tests/*.test.js` → 33/33 unchanged.
  - `nix develop --command pyright src/zimt` → only the pre-existing
    `downloads.py:246` tqdm monkey-patch finding (PR-10/D02).
  Review:
  - One adversarial review round flagged three follow-up gaps; all
    closed in this PR. The reviewer's verdict was "ship with
    caveats"; the caveats were addressed before commit.
  Notes / constraints:
  - The `_active_lock` extension over `asdict(job)` for download jobs
    means a brief asyncio-side hold of a `threading.Lock`. Audit found
    no nested-lock acquisition path that could deadlock — the executor
    thread acquires `_active_lock` only inside `_emit`/`close`, the
    asyncio thread only inside `emit_job`/`current_download`.
  - The TTL cache for `_scan_cache` is intentionally short (1.0s).
    Long-running test suites that need fresh scans should call
    `_invalidate_cache()` between assertions; the new
    `ModelsInfoCacheTests` document this pattern.
  - `register_task` is intentionally not used for the per-WS
    heartbeat task — that task is retained by its local `hb_task`
    binding in the `ws_endpoint` coroutine, so a separate set entry
    is redundant.

- **PR-09** (2026-05-20) — Frontend state and API-contract fixes for
  PR-07-D15, D16, D17, D18. Headline changes: the queue's
  `MAX_QUEUE_SHOWN` LRU eviction in `static/app.js` now scans for the
  oldest completed (`done`/`error`/`canceled`) entry and falls back to
  oldest-active only when no completed entry exists, with a
  `console.warn` documenting the degradation (D15). `_modelDlBtnState`
  surfaces the unit alongside the percentage (`"downloading N% bytes"`
  / `"downloading N% files"` / `"downloading N%"` when no unit), so
  the model-tab button no longer flickers between byte-percent and
  file-percent during a snapshot — note that the underlying
  single-`download_n/total` Job-model issue is unchanged (PR-04 carry-
  over); the mitigation lives in the helper (D16). The download
  button's `onclick` now forces a `renderBases()` / `renderLoras()`
  re-render after the await (and restores `dlBtn.textContent` from
  the closure-captured `dlState.text` on the catch branch), so the
  optimistic `"starting…"` is cleared even when the request collapses
  into an existing job via PR-06 idempotency (D17). `buildRestorePromptLine`
  emits one `/lora` command per LoRA in the active stack rather than a
  single `/lora foo:0.7,bar:0.3` token that the arity-1 parser would
  truncate (D18). New `tests/frontend_state.test.js` houses 8 tests
  covering each defect's contract.
  Verification:
  - `node --test tests/*.test.js` → 41/41 pass (was 33; +8 new).
  - `.venv/bin/python -m unittest discover -s tests` → 46/46 unchanged.
  - `nix develop --command pyright src/zimt` → only the pre-existing
    `downloads.py:246` finding (PR-10/D02).
  Review:
  - One adversarial review round. Verdict: clean accept. Five nits
    considered and discarded (no Python round-trip for D18; D17
    `notEqual` assertion is lenient; D16 button format diverges
    from queue-row format intentionally; `console.warn` noise in D15
    fallback; pyright finding pre-existing).
  Notes / constraints:
  - D16's fix mitigates the user-visible flicker but does not refactor
    the `Job.download_n / download_total / download_unit` triple into
    per-unit slots — that's the deferred PR-04 follow-up. The button
    text and the queue-row text are unit-consistent at the same
    instant; the helper's percentage is computed from the same
    `n/total` regardless of which bar is currently firing.
  - D18's emitted format relies on `repl.commands.parse_commands`'s
    arity-1 contract for `/lora`. If that contract changes, the
    restore-prompt format must be revisited.
  - Pre-existing pyright `downloads.py:246` finding remains for
    PR-10/D02.
