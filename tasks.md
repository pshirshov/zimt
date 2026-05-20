# zimt — Task Ledger

Authoritative ledger for the requested model-tab download fixes and whole-codebase review. Detailed plan: `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`.

Status: `[ ]` planned · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Milestones (high-level)

- [~] **M1** — Resolve known model-tab/download correctness defects with regression tests.
- [ ] **M2** — Perform whole-codebase adversarial review and execute follow-up fixes for confirmed defects.

---

## Milestone 1 — PR breakdown

Detail in `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`. One line per PR here; sub-task detail stays in the plan doc.

- [x] **PR-01** — Introduce base/LoRA download target identity.
- [x] **PR-02** — Make download progress ownership job-scoped.
- [x] **PR-03** — Add cancellation semantics for download jobs.
- [x] **PR-04** — Normalize Hugging Face progress units.
- [ ] **PR-05** — Prevent untracked LoRA downloads during generation.
- [ ] **PR-06** — Consolidate model-tab download state transitions.

---

## Milestone 2 — PR breakdown

Detail in `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`.

- [ ] **PR-07** — Whole-codebase review inventory and defect triage.
- [ ] **PR-08** — Backend concurrency and lifecycle follow-up fixes.
- [ ] **PR-09** — Frontend state and API-contract follow-up fixes.
- [ ] **PR-10** — Filesystem, configuration, and external-boundary follow-up fixes.
- [ ] **PR-11** — Final adversarial review and release verification.

---

## Cross-cutting architectural notes (locked)

- [x] Download jobs need asset identity separate from queue job type: `kind="download"` remains the queue discriminator; `target_kind` distinguishes base-model assets from LoRA assets. Lands in PR-01.
- [x] Download progress ownership invariant — at most one HF download owns the progress slot at a time, acquired by job id via `set_active_download` and released by the same id via `clear_active_download`. Per-bar identity is propagated to executor workers via a `contextvars.ContextVar` (`download_context(job_id)`); callers MUST use `ctx = contextvars.copy_context()` and dispatch via `loop.run_in_executor(executor, lambda: ctx.run(func, *args))` — `run_in_executor` does NOT propagate contextvars on its own (CPython 3.13). tqdm bars constructed in the worker capture the owner id and silently drop events when it does not match the slot. Loser callers receive a passive busy state via `emit_log`; a row-level "busy" indicator is deferred to PR-06. Lands in PR-02 (D10 fix landed in PR-03 to satisfy the production assumption).
- [x] Download cancellation invariant — each download job (`load_model` and `prefetch_model`) registers a `threading.Event` in `CANCEL_EVENTS` at Job creation and pops it on exit. Cancellation observed at controlled boundaries: loader's pre-load polling loop, prefetch's post-`_PREFETCH_LOCK` checkpoint, and `ProgressTqdm.__init__` / `update` (raises `DownloadCanceled` which is caught in both callers and ends the job with status `canceled`). `jobs_cancel_all` RPC covers both `generate` and `download` kinds. The UI renders a cancel button on queued and running download rows. Lands in PR-03.
- [x] Progress unit invariant — every progress event carries an explicit unit category (`"bytes"` | `"files"` | `"items"` | `""`) on `Job.download_unit`. Classification: HF's byte aggregate bar (`unit='B'`) is `"bytes"`; HF's outer file-count bar (`unit='it'` default, identified by `desc` regex `^(?:\[dry-run\]\s*)?Fetching\b`) is `"files"`; any other tqdm event is `"items"`. The UI never claims byte semantics without `unit=='B'` evidence. `download_files_done` is driven exclusively by the file-count bar's current `n` (overwrite in `_emit`, never mutated by `close()`); without a file-count bar it stays 0. Lands in PR-04.
- [ ] LoRA dependency invariant — decide and encode in PR-05: generation must not initiate implicit Hugging Face downloads.

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
