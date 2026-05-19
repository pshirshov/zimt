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
- [ ] **PR-03** — Add cancellation semantics for download jobs.
- [ ] **PR-04** — Normalize Hugging Face progress units.
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
- [x] Download progress ownership invariant — at most one HF download owns the progress slot at a time, acquired by job id via `set_active_download` and released by the same id via `clear_active_download`. Per-bar identity is propagated to executor workers via a `contextvars.ContextVar` (`download_context(job_id)`); tqdm bars created outside their owning context (or in a context whose id no longer matches the slot) are silent no-ops rather than misattributing progress. Loser callers receive a passive busy state via `emit_log` ("download progress slot busy; ... will not be broadcast"); a future row-level "busy" indicator is deferred to PR-06. Lands in PR-02.
- [ ] Download cancellation invariant — decide and encode in PR-03: queued downloads must be cancelable before start; running downloads must enter a terminal canceled state at a controlled boundary if immediate upstream abort is unavailable.
- [ ] Progress unit invariant — decide and encode in PR-04: counters render only with units proven by the progress event.
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
  - `loop.run_in_executor` copies the current `contextvars.Context` to the
    worker thread. Callers must wrap their `run_in_executor` invocations in
    `with download_context(job.id):` for the per-bar gating to work. Both
    `loader.py` and `prefetch.py` do this correctly today; future callers
    that bypass `download_context` will not be gated and their tqdm bars
    will misattribute. Documented in `src/zimt/webui/downloads.py` module
    docstring.
  - `StateCase.setUp/tearDown` now snapshot and restore
    `downloads._active_job_id` to keep module-global slot ownership from
    leaking across tests. New test
    `DownloadOwnershipApiTests.test_set_active_download_is_idempotent_for_same_owner`
    locks in the documented `set_active_download` idempotent re-acquire
    contract.
  - Pre-existing `LoaderTests` failure (HTTPException vs ModelLoadError)
    remains unchanged; not in PR-02's scope.
